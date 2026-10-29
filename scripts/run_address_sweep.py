#!/usr/bin/env python3
"""Frozen address ablation with matched rounds and runtime provenance."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def environment():
    result = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    for key,cmd in [('battery',['/usr/bin/pmset','-g','batt']),('thermal',['/usr/bin/pmset','-g','therm'])]:
        try:
            run = subprocess.run(cmd,capture_output=True,text=True,timeout=5)
            result[key] = {'exit':run.returncode,'stdout':run.stdout[:4096],'stderr':run.stderr[:1024]}
        except (OSError,subprocess.TimeoutExpired) as error:
            result[key] = {'unavailable':type(error).__name__}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    config = ROOT/'configs/address_v7_20261029.json'
    cfg = json.loads(config.read_text())
    binary = ROOT/'build/metal-atlas-v7/astc-kv-probe-atlas-v7'
    shader = ROOT/'metal/attention_atlas_v7.metal'
    fixture_root = ROOT/'results/kernel_v2/fixtures_page'
    watched = [config,binary,shader,ROOT/'metal/probe_atlas_v7.mm',Path(__file__)]
    for t in [*cfg['tokens'],391]:
        watched.extend(p for p in (fixture_root/f't{t}').iterdir() if p.is_file())
    before = {str(p.relative_to(ROOT)):sha(p) for p in watched}
    schedule = []
    rng = random.Random(cfg['seed'])
    for session in range(cfg['sessions']):
        shapes = cfg['tokens'].copy()
        rng.shuffle(shapes)
        schedule.extend((session,t,cfg['tiles'][str(t)]) for t in shapes)
    (args.out/'registration.json').write_text(json.dumps({'config':cfg,'hashes_before':before,
        'schedule':schedule,'environment':environment()},indent=2)+'\n')

    def execute(stem,t,tile,rounds,iterations,seed):
        cmd = [str(binary),str(fixture_root/f't{t}'/'manifest.json'),str(shader),str(rounds),str(iterations),str(tile),'-',str(seed)]
        env_before = environment()
        run = subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=120)
        (args.out/(stem+'.json')).write_text(run.stdout)
        (args.out/(stem+'.stderr')).write_text(run.stderr)
        (args.out/(stem+'.execution.json')).write_text(json.dumps({'command':cmd,'exit':run.returncode,
            'environment_before':env_before,'environment_after':environment()},indent=2)+'\n')
        if run.returncode:
            raise RuntimeError(f'{stem}: exit={run.returncode}; output retained')
        data = json.loads(run.stdout)
        assert data['status']=='executed_on_metal'
        assert data['correctness']['compact_decode_vs_original_normalized']['passed_bit_exact_float32']
        for item in data['correctness'].values():
            assert item.get('passed_rtol_l2_1e_4',True)
        return data

    execute('ragged',391,13,1,1,cfg['seed'])
    rows = []
    for session,t,tile in schedule:
        stem = f's{session}_t{t}_tile{tile}'
        data = execute(stem,t,tile,cfg['rounds'],cfg['iterations'],cfg['seed']+session*17+t)
        samples = data['raw_samples']
        variants = sorted({s['variant'] for s in samples})
        assert len(variants)==10
        pairs = {}
        for sample in samples:
            assert sample['iterations']==cfg['iterations'] and sample['gpu_batch_ms']>0
            pairs.setdefault(sample['round'],{})[sample['variant']]=sample['gpu_batch_ms']/sample['iterations']*1000
        assert len(pairs)==cfg['rounds'] and all(len(p)==10 for p in pairs.values())
        original = 'astc_compact_factored_v'
        for variant in variants:
            gpu = [p[variant] for p in pairs.values()]
            ratios = [p[variant]/p[original] for p in pairs.values()]
            rows.append({'session':session,'tokens':t,'tile':tile,'variant':variant,'gpu_median_us':statistics.median(gpu),
                'gpu_min_us':min(gpu),'gpu_max_us':max(gpu),'paired_ratio_median_vs_compact':statistics.median(ratios),
                'raw':stem+'.json'})
        print(json.dumps({'completed':stem,'primary_ratio':next(r['paired_ratio_median_vs_compact'] for r in rows
            if r['session']==session and r['tokens']==t and r['variant']=='astc_compact_recurrent_v')}),flush=True)
    after = {str(p.relative_to(ROOT)):sha(p) for p in watched}
    assert before==after
    ratios = [r['paired_ratio_median_vs_compact'] for r in rows if r['tokens']==8192 and r['variant']=='astc_compact_recurrent_v']
    outcome = 'positive' if all(r<=.95 for r in ratios) else ('negative' if all(r>=1.05 for r in ratios) else 'mixed_or_small')
    (args.out/'summary.json').write_text(json.dumps({'rows':rows,'primary_session_ratios':ratios,
        'primary_outcome':outcome,'hashes_after':after,'scope':cfg['scope']},indent=2)+'\n')


if __name__=='__main__':
    main()
