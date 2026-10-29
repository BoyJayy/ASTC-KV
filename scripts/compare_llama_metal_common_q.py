#!/usr/bin/env python3
"""Matched reconstructed-input comparison, synchronized host boundary only."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(cmd, out):
    result = subprocess.run([str(x) for x in cmd], cwd=ROOT, capture_output=True,
                            text=True, timeout=120)
    out.with_suffix('.json').write_text(result.stdout)
    out.with_suffix('.stderr').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f'{out.name}: exit {result.returncode}; retained raw output')
    return json.loads(result.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=ROOT/'configs/llama_comparison_common_q.json')
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    astc = ROOT/'build/metal-v4/astc-kv-probe-v4'
    llama = ROOT/'build/llama-baseline/bin/astc-llama-baseline'
    shader = ROOT/'metal/attention_v4.metal'
    watched = [astc, llama, shader, Path(__file__), args.config,
               ROOT/'metal/probe_v4.mm', ROOT/'baselines/llama/probe.mm']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in watched}
    (args.out/'registration.json').write_text(json.dumps({'config': config, 'hashes': hashes}, indent=2)+'\n')
    fixtures = {}
    for tokens in config['tokens']:
        original = ROOT/f'results/kernel_v2/fixtures_page/t{tokens}/manifest.json'
        source = args.out/'astc_inputs'/f't{tokens}'/'manifest.json'
        shutil.copytree(original.parent,source.parent)
        m = json.loads(source.read_text())
        qpath = source.parent/m['query_file']
        original_q_hash = sha(qpath)
        np.fromfile(qpath,'<f4').astype('<f2').astype('<f4').tofile(qpath)
        m.update(common_query_contract='Both backends receive identical FP16-rounded Q in FP32 storage',
                 original_query_sha256=original_q_hash, common_query_sha256=sha(qpath))
        source.write_text(json.dumps(m,indent=2)+'\n')
        folder = args.out/'matched_inputs'/f't{tokens}'
        folder.mkdir(parents=True)
        execute([astc, source, shader, 1, 1, 32, folder], folder/'readback_validation')
        (folder/'q.f32').write_bytes((source.parent/m['query_file']).read_bytes())
        m.update(query_file='q.f32', baseline_k_f32_file='k_hardware.f32',
                 baseline_v_f32_file='v_hardware.f32',
                 baseline_scope='Hardware ASTC reconstruction rounded to F16, then F16 or Q4: performance/control only, NOT original-data codec quality',
                 baseline_source_manifest_sha256=sha(source))
        target = folder/'manifest.json'
        target.write_text(json.dumps(m, indent=2)+'\n')
        fixtures[tokens] = (source, target)
    jobs = []
    rng = random.Random(config['seed'])
    for session in range(config['sessions']):
        local = []
        for t in config['tokens']:
            local.append((session, 'llama', t, 0))
            local.extend((session, 'astc', t, tile) for tile in config['tiles_by_tokens'][str(t)])
        rng.shuffle(local)
        jobs.extend(local)
    (args.out/'schedule.json').write_text(json.dumps(jobs, indent=2)+'\n')
    rows = []
    for session, backend, t, tile in jobs:
        source, matched = fixtures[t]
        seed = config['seed'] + session
        stem = args.out/f's{session}_{backend}_t{t}_tile{tile}'
        cmd = ([astc, source, shader, config['rounds'], 1, tile, '-', seed]
               if backend == 'astc' else [llama, matched, config['rounds'], seed])
        data = execute(cmd, stem)
        assert data['status'] == 'executed_on_metal'
        if backend == 'llama':
            assert data['timing_valid'] and data['timing']['new_pipelines_during_timing'] == 0
            for side in ['k', 'v']:
                represented = np.fromfile(matched.parent/f'{side}_hardware.f32', '<f4').astype('<f2')
                assert hashlib.sha256(represented.tobytes()).hexdigest() == data['input_provenance'][f'{side}_base_f16_sha256']
        for variant in sorted({s['variant'] for s in data['raw_samples']}):
            samples = [s for s in data['raw_samples'] if s['variant'] == variant]
            times = [(s['wall_batch_ms'] if backend == 'astc' else s['host_sync_ms'])*1000 for s in samples]
            assert len(times) == config['rounds'] and all(x > 0 for x in times)
            row = dict(session=session, backend=backend, tokens=t, tile=tile, variant=variant,
                       host_sync_median_us=statistics.median(times), min_us=min(times), max_us=max(times),
                       raw=stem.name+'.json')
            rows.append(row)
        print(json.dumps({'completed': stem.name}), flush=True)
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in watched}
    (args.out/'summary.json').write_text(json.dumps({'rows': rows, 'scope': config['scope'],
        'hashes_after': hashes, 'matched_f16_input_hashes_passed': True}, indent=2)+'\n')


if __name__ == '__main__':
    main()
