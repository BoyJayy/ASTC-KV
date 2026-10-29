#!/usr/bin/env python3
"""Recompute the v8/v9/v10 evidence and verify current code/input identities."""
import argparse
import importlib
import json
import math
from pathlib import Path
import statistics
from run_address_sweep import sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research_v4_20261029'
STAGES = [('gather', 'run_gather_v8'), ('exchange', 'run_gather_exchange_v9'), ('gqa', 'run_gqa_v10')]


def calculate():
    result = {'status': 'PASS', 'stages': {}, 'totals': {'gpu_processes': 0, 'attention_checks': 0,
              'gather_scalar_checks': 0, 'main_blocks': 0, 'main_samples': 0, 'max_relative_l2': 0}}
    for stage, module in STAGES:
        validator = importlib.import_module(module)
        stage_result = {}
        for kind in ('smoke', 'main'):
            folder = OUT / f'{stage}_{kind}'
            registration = json.loads((folder / 'registration.json').read_text())
            summary = json.loads((folder / 'summary.json').read_text())
            cfg = registration['config']
            before = registration['hashes_before']
            assert before == summary['hashes_after']
            assert all(sha(ROOT / name) == expected for name, expected in before.items())
            assert registration['smoke'] == (kind == 'smoke') and summary['smoke'] == (kind == 'smoke')
            rows, memory = [], []
            jobs = [(f'ragged_t{t}_tile{tile}', t, tile, None) for t, tile in cfg['correctness_cases']]
            jobs += [(f's{s}_t{t}_tile{tile}', t, tile, s) for s, t, tile in registration['schedule']]
            assert len(jobs) == (5 if kind == 'smoke' else 19)
            if kind == 'main':
                assert {(s, t) for s, t, tile in registration['schedule']} == {(s, t) for s in range(5) for t in (384, 2048, 8192)}
            for stem, tokens, tile, session in jobs:
                data = json.loads((folder / (stem + '.json')).read_text())
                validator.validate(data, tokens, tile)
                execution = json.loads((folder / (stem + '.execution.json')).read_text())
                assert execution['exit'] == 0
                checks = data['correctness']
                exact = checks['gather_vs_direct']
                assert exact['gather_sha256'] == exact['direct_sha256'] and exact['errors']['max_abs'] == 0
                exact = checks['compact_decode_vs_original_normalized']
                assert exact['compact_sha256'] == exact['original_sha256'] and exact['errors']['max_abs'] == 0
                errors = [v['vs_same_values_cpu_fp64_accumulation']['relative_l2'] for v in checks.values() if 'passed_rtol_l2_1e_4' in v]
                totals = result['totals']
                totals['gpu_processes'] += 1
                totals['attention_checks'] += len(errors)
                totals['gather_scalar_checks'] += checks['gather_vs_direct']['checked_scalar_values']
                totals['max_relative_l2'] = max(totals['max_relative_l2'], *errors)
                blocks, iterations = (1, 1) if kind == 'smoke' or session is None else (cfg['blocks'], cfg['iterations'])
                measured = validator.extract_blocks(data, session, tokens, tile, iterations, blocks, stem + '.json')
                if session is not None:
                    rows.extend(measured)
                    m = data['memory']
                    memory.append({'session': session, 'tokens': tokens,
                        'astc_bytes': m['compact_persistent_k_v_metadata_allocated_bytes'],
                        'fp16_bytes': m['fp16_persistent_k_v_allocated_bytes'],
                        'ratio': m['compact_persistent_k_v_metadata_allocated_bytes'] / m['fp16_persistent_k_v_allocated_bytes']})
                    if kind == 'main':
                        totals['main_blocks'] += len(measured)
                        totals['main_samples'] += len(data['raw_samples'])
            assert rows == summary['rows']
            if kind == 'smoke':
                assert summary['primary_passed'] is None
                continue
            table = []
            for tokens in cfg['tokens']:
                for pair in validator.PAIRS:
                    subset = [r for r in rows if r['tokens'] == tokens and r['pair'] == pair]
                    ratios = [statistics.median(r['gpu_ratio_b_over_a'] for r in subset if r['session'] == s) for s in range(5)]
                    aa = [statistics.median(r['a_us'] for r in subset if r['session'] == s) for s in range(5)]
                    bb = [statistics.median(r['b_us'] for r in subset if r['session'] == s) for s in range(5)]
                    table.append({'tokens': tokens, 'pair': pair, 'session_ratios': ratios,
                        'median_session_ratio': statistics.median(ratios), 'a_us_median_of_session_medians': statistics.median(aa),
                        'b_us_median_of_session_medians': statistics.median(bb),
                        'all_block_ratio_range': [min(r['gpu_ratio_b_over_a'] for r in subset), max(r['gpu_ratio_b_over_a'] for r in subset)]})
            if stage == 'gqa':
                pairs = ('fused_fp16_to_astc', 'fp16_to_gqa_astc')
                primary = {pair: next(r['session_ratios'] for r in table if r['tokens'] == 8192 and r['pair'] == pair) for pair in pairs}
                assert memory == summary['memory_rows']
                passed = all(x <= cfg['primary_max_ratio'] for values in primary.values() for x in values)
                passed = passed and all(r['ratio'] <= cfg['max_persistent_memory_ratio'] for r in memory if r['tokens'] == 8192)
            else:
                pair = 'hoisted_to_gather' if stage == 'gather' else 'hoisted_to_exchange'
                primary = next(r['session_ratios'] for r in table if r['tokens'] == 8192 and r['pair'] == pair)
                passed = all(x <= cfg['primary_max_ratio'] for x in primary)
            assert primary == summary['primary_session_medians'] and passed == summary['primary_passed']
            stage_result = {'primary_passed': passed, 'primary_session_medians': primary, 'comparisons': table, 'memory_rows': memory}
        result['stages'][stage] = stage_result
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write', action='store_true', help='Create a new derived summary; never overwrite')
    args = ap.parse_args()
    result = calculate()
    target = OUT / 'derived_summary.json'
    if args.write:
        with target.open('x') as f:
            json.dump(result, f, indent=2); f.write('\n')
    else:
        assert result == json.loads(target.read_text())
    print(json.dumps({'status': 'PASS', 'totals': result['totals'], 'primary_results': {k: v['primary_passed'] for k, v in result['stages'].items()}}))


if __name__ == '__main__':
    main()
