#!/usr/bin/env python3
"""Preregistered fixed-shape pool experiment, serialized GPU processes."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = ROOT/'configs/kernel_pool_v5.json'
    cfg = json.loads(config.read_text())
    binary = ROOT/'build/metal-pool-v5/astc-kv-probe-pool-v5'
    shader = ROOT/'metal/attention_v4.metal'
    manifest = ROOT/'results/kernel_v2/fixtures_page/t8192/manifest.json'
    watched = [config, binary, shader, Path(__file__), ROOT/'metal/probe_pool_v5.mm']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in watched}
    rng = random.Random(cfg['seed'])
    schedule = []
    for session in range(cfg['sessions']):
        pools = cfg['pool_sizes'].copy()
        rng.shuffle(pools)
        schedule.extend((session,n) for n in pools)
    (args.out/'registration.json').write_text(json.dumps({'config':cfg,'hashes_before':hashes,'schedule':schedule},indent=2)+'\n')
    rows = []
    for session,n in schedule:
        command = [str(binary),str(manifest),str(shader),str(cfg['rounds']),str(n),
                   str(cfg['tile_tokens']),'-',str(cfg['seed']+session),str(n)]
        result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=180)
        stem = f's{session}_pool{n}'
        (args.out/(stem+'.json')).write_text(result.stdout)
        (args.out/(stem+'.stderr')).write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f'{stem} failed; retained stdout/stderr')
        data = json.loads(result.stdout)
        assert data['status']=='executed_on_metal'
        for v in ['astc_factored_v','fp16_buffer']:
            samples = [s for s in data['raw_samples'] if s['variant']==v]
            assert len(samples)==cfg['rounds'] and all(s['iterations']==n for s in samples)
            times = [s['gpu_batch_ms']/n*1000 for s in samples]
            rows.append({'session':session,'pool':n,'variant':v,'gpu_median_us':statistics.median(times),
                         'gpu_min_us':min(times),'gpu_max_us':max(times),
                         'wall_amortized_median_us':statistics.median(s['wall_batch_ms']/n*1000 for s in samples),
                         'raw':stem+'.json'})
        print(json.dumps({'completed':stem,'rows':rows[-2:]}),flush=True)
    assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in watched}
    ratios = []
    for session in range(cfg['sessions']):
        r = {x['variant']:x['gpu_median_us'] for x in rows if x['session']==session and x['pool']==64}
        ratios.append(r['astc_factored_v']/r['fp16_buffer'])
    outcome = 'positive' if all(r<=.8 for r in ratios) else ('negative' if all(r>1.05 for r in ratios) else 'inconclusive')
    (args.out/'summary.json').write_text(json.dumps({'rows':rows,'pool64_astc_over_fp16_by_session':ratios,
        'preregistered_outcome':outcome,'hashes_after':hashes,'scope':cfg['scope']},indent=2)+'\n')


if __name__=='__main__':
    main()
