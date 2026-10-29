#!/usr/bin/env python3
"""Bounded serialized GPU sweep with immutable inputs and preserved raw samples."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/kernel_split_v2.json')
    p.add_argument('--binary',type=Path,default=ROOT/'build/metal/astc-kv-probe')
    p.add_argument('--shader',type=Path,default=ROOT/'metal/attention.metal')
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();cfg=json.loads(args.config.read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    paths=[args.binary,args.shader,args.config,Path(__file__)]
    initial={str(p.resolve()):sha(p) for p in paths}
    rows=[];schedule=[];rng=random.Random(cfg['seed'])
    for session in range(cfg['sessions']):
        jobs=[(tokens,tile) for tokens in cfg['tokens'] for tile in
              cfg.get('tiles_by_tokens',{}).get(str(tokens),cfg.get('tile_tokens',[]))]
        rng.shuffle(jobs);schedule.extend((session,*j) for j in jobs)
    registration={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'config':cfg,'hashes_before':initial,'schedule':schedule}
    (args.out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    if cfg.get('negative_manifest'):
        proc=subprocess.run([str(args.binary.resolve()),str((ROOT/cfg['negative_manifest']).resolve()),
                             str(args.shader.resolve()),'1','1','16'],capture_output=True,text=True,cwd=ROOT,timeout=60)
        (args.out/'negative_manifest.stderr').write_text(proc.stderr)
        assert proc.returncode==1 and 'ASTC height must equal' in proc.stderr and 'released without endEncoding' not in proc.stderr
        (args.out/'negative_manifest.json').write_text(json.dumps({'expected_exit':1,'actual_exit':proc.returncode,'passed':True})+'\n')
    for session,tokens,tile in schedule:
        manifest=args.fixtures/f't{tokens}'/'manifest.json'
        cmd=[str(args.binary.resolve()),str(manifest.resolve()),str(args.shader.resolve()),str(cfg['rounds']),str(cfg['iterations']),str(tile)]
        proc=subprocess.run(cmd,capture_output=True,text=True,cwd=ROOT,timeout=90)
        stem=f's{session}_t{tokens}_tile{tile}'
        (args.out/(stem+'.json')).write_text(proc.stdout)
        (args.out/(stem+'.stderr')).write_text(proc.stderr)
        if proc.returncode:raise RuntimeError(f'{stem} failed exit={proc.returncode}')
        data=json.loads(proc.stdout)
        for variant in sorted({s['variant'] for s in data['raw_samples']}):
            samples=[s for s in data['raw_samples'] if s['variant']==variant]
            gpu=[s['gpu_batch_ms']/s['iterations']*1000 for s in samples if s['gpu_batch_ms'] is not None]
            assert len(gpu)==cfg['rounds'] and all(v>0 for v in gpu)
            rows.append({'session':session,'tokens':tokens,'tile':tile,'variant':variant,
                         'gpu_median_us':statistics.median(gpu),
                         'gpu_min_us':min(gpu),'gpu_max_us':max(gpu),
                         'wall_median_us':statistics.median(s['wall_batch_ms']/s['iterations']*1000 for s in samples),
                         'manifest_sha256':sha(manifest),'raw':stem+'.json','scratch_bytes':data['memory']['attention_scratch_allocated_bytes']})
        print(json.dumps({'completed':stem,'rows':rows[-len({s['variant'] for s in data['raw_samples']}):]}),flush=True)
    final={str(p.resolve()):sha(p) for p in paths};assert initial==final
    result={'schema':'astckv.kernel_sweep.v2','scope':cfg['scope'],'rows':rows,'hashes_after':final}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
