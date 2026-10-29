#!/usr/bin/env python3
"""Recompute the selected kernel summaries from immutable session files."""
import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]/'results/kernel_v2'


def summarize():
    result = {'scope': 'Observed GPU medians; original query at T384, repeated-block traffic controls at T2048 and T8192. Per-shape minima over exploratory tiles are selected, not confirmation.'}
    for phase in ['split_v1_page', 'attribution', 'factored', 'wave']:
        folder = ROOT/phase
        rows = json.loads((folder/'summary.json').read_text())['rows']
        groups = []
        for tokens, variant in sorted({(r['tokens'], r['variant']) for r in rows}):
            candidates = []
            for tile in sorted({r['tile'] for r in rows if r['tokens']==tokens and r['variant']==variant}):
                medians = [r['gpu_median_us'] for r in rows
                           if (r['tokens'],r['variant'],r['tile']) == (tokens,variant,tile)]
                candidates.append({'tile':tile, 'median_us':statistics.median(medians), 'session_medians_us':medians})
            groups.append({'tokens':tokens, 'variant':variant,
                           'best':min(candidates,key=lambda x:x['median_us']), 'all_tiles':candidates})
        errors = []
        raws = sorted(folder.glob('s*_t*_tile*.json'))
        for path in raws:
            raw = json.loads(path.read_text())
            for item in raw['correctness'].values():
                if 'vs_same_values_cpu_fp64_accumulation' in item:
                    e = item['vs_same_values_cpu_fp64_accumulation']
                    assert e['finite'] and e['relative_l2'] <= 1e-4
                    errors.append(e['relative_l2'])
            # Check each saved session median independently against raw rounds.
            for row in (r for r in rows if r['raw']==path.name):
                samples = [s for s in raw['raw_samples'] if s['variant']==row['variant']]
                med = statistics.median(s['gpu_batch_ms']/s['iterations']*1000 for s in samples)
                assert med == row['gpu_median_us']
        result[phase] = {'groups':groups, 'processes':len(raws),
                         'max_own_reference_rel_l2':max(errors), 'variant_session_rows':len(rows)}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT/'derived_kernel_summary.json')
    args = parser.parse_args()
    result = summarize()
    if args.verify:
        assert result == json.loads(args.out.read_text())
        print('PASS: raw medians, correctness errors, selected tiles and session summaries')
    else:
        with args.out.open('x') as output:
            json.dump(result, output, indent=2)
            output.write('\n')
