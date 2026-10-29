#!/usr/bin/env python3
"""Frozen paired-gather experiment; all raw ABBA samples are retained."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
from run_address_sweep import sha, environment

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    'astc_texture', 'uncompressed_texture', 'fp16_buffer', 'normalized_fp16_buffer',
    'uncompressed_rgba_texture', 'astc_factored_v', 'normalized_buffer_factored_v',
    'astc_compact_factored_v', 'astc_compact_hoisted_v', 'astc_compact_recurrent_v',
    'compact_paired_direct_v', 'compact_paired_gather_v',
}
PAIRS = {
    'hoisted_to_gather': ('astc_compact_hoisted_v', 'compact_paired_gather_v'),
    'paired_direct_to_gather': ('compact_paired_direct_v', 'compact_paired_gather_v'),
    'fp16_to_gather': ('fp16_buffer', 'compact_paired_gather_v'),
}


def validate(data, tokens, tile):
    assert data['status'] == 'executed_on_metal'
    checks = data['correctness']
    assert checks['compact_decode_vs_original_normalized']['passed_bit_exact_float32'] is True
    assert checks['gather_vs_direct']['passed_bit_exact_float32'] is True
    attention = {k: v for k, v in checks.items() if 'passed_rtol_l2_1e_4' in v}
    assert set(attention) == VARIANTS
    for item in attention.values():
        assert item['passed_rtol_l2_1e_4'] is True
        errors = item['vs_same_values_cpu_fp64_accumulation']
        assert errors['finite'] is True and math.isfinite(errors['relative_l2']) and 0 <= errors['relative_l2'] <= 1e-4
    for item in checks.values():
        for key, value in item.items():
            if key.startswith('passed'):
                assert value is True
    shape = data['shape']
    assert shape['tokens'] == tokens and shape['tile_tokens'] == tile
    assert (shape['head_dim'], shape['kv_heads'], shape['query_heads']) == (64, 3, 9)


def extract_blocks(data, session, tokens, tile, iterations, expected_blocks, raw):
    blocks = {}
    for sample in data['raw_samples']:
        assert sample['iterations'] == iterations and math.isfinite(sample['gpu_batch_ms']) and sample['gpu_batch_ms'] > 0
        blocks.setdefault((sample['pair_name'], sample['round']), []).append(sample)
    assert set(blocks) == {(pair, i) for pair in PAIRS for i in range(expected_blocks)}
    rows = []
    for (pair, round_id), samples in sorted(blocks.items()):
        assert len(samples) == 4 and [s['slot_in_block'] for s in samples] == [0, 1, 2, 3]
        a, b = PAIRS[pair]
        order = samples[0]['block_order']
        assert order in ('ABBA', 'BAAB')
        assert all((s['a_variant'], s['b_variant'], s['block_order']) == (a, b, order) for s in samples)
        assert [s['variant'] for s in samples] == ([a, b, b, a] if order == 'ABBA' else [b, a, a, b])
        av = [s['gpu_batch_ms'] for s in samples if s['variant'] == a]
        bv = [s['gpu_batch_ms'] for s in samples if s['variant'] == b]
        rows.append({'session': session, 'tokens': tokens, 'tile': tile, 'pair': pair, 'round': round_id,
                     'block_order': order, 'gpu_ratio_b_over_a': math.sqrt(math.prod(bv) / math.prod(av)),
                     'a_us': math.sqrt(math.prod(av)) * 1000 / iterations,
                     'b_us': math.sqrt(math.prod(bv)) * 1000 / iterations, 'raw': raw})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = ROOT / 'configs/gather_v8_20261029.json'
    cfg = json.loads(config.read_text())
    assert cfg['sessions'] == 5 and cfg['primary_max_ratio'] == .85
    binary = ROOT / 'build/metal-gather-v8/astc-kv-probe-gather-v8'
    shader = ROOT / 'metal/attention_gather_v8.metal'
    fixtures = ROOT / 'results/kernel_v2/fixtures_page'
    watched = [config, binary, shader, ROOT / 'metal/probe_gather_v8.mm', ROOT / 'metal/build_gather_v8.sh',
               ROOT / 'tests/test_gather_pairs_v8.py', ROOT / 'scripts/run_address_sweep.py', Path(__file__)]
    for tokens in [*cfg['tokens'], 391]:
        watched.extend(p for p in (fixtures / f't{tokens}').iterdir() if p.is_file())
    before = {str(p.relative_to(ROOT)): sha(p) for p in watched}
    schedule = []
    rng = random.Random(cfg['seed'])
    for session in range(1 if args.smoke else cfg['sessions']):
        shapes = [384] if args.smoke else cfg['tokens'].copy()
        rng.shuffle(shapes)
        schedule.extend((session, t, cfg['tiles'][str(t)]) for t in shapes)
    registration = {'config': cfg, 'smoke': args.smoke, 'schedule': schedule, 'hashes_before': before, 'environment': environment()}
    (args.out / 'registration.json').write_text(json.dumps(registration, indent=2) + '\n')

    def execute(stem, tokens, tile, blocks, iterations, seed):
        command = [str(binary), str(fixtures / f't{tokens}' / 'manifest.json'), str(shader), str(blocks),
                   str(iterations), str(tile), '-', str(seed)]
        env_before = environment()
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
        (args.out / (stem + '.json')).write_text(run.stdout)
        (args.out / (stem + '.stderr')).write_text(run.stderr)
        (args.out / (stem + '.execution.json')).write_text(json.dumps({'command': command, 'exit': run.returncode,
             'environment_before': env_before, 'environment_after': environment()}, indent=2) + '\n')
        if run.returncode:
            raise RuntimeError(f'{stem} failed; raw output retained')
        data = json.loads(run.stdout)
        validate(data, tokens, tile)
        return data

    for tokens, tile in cfg['correctness_cases']:
        execute(f'ragged_t{tokens}_tile{tile}', tokens, tile, 1, 1, cfg['seed'])
    rows = []
    for session, tokens, tile in schedule:
        stem = f's{session}_t{tokens}_tile{tile}'
        blocks, iterations = (1, 1) if args.smoke else (cfg['blocks'], cfg['iterations'])
        data = execute(stem, tokens, tile, blocks, iterations, cfg['seed'] + session * 23 + tokens)
        rows.extend(extract_blocks(data, session, tokens, tile, iterations, blocks, stem + '.json'))
        current = [r for r in rows if r['session'] == session and r['tokens'] == tokens and r['pair'] == 'hoisted_to_gather']
        print(json.dumps({'completed': stem, 'gather_over_hoisted_median': statistics.median(r['gpu_ratio_b_over_a'] for r in current)}), flush=True)
    after = {str(p.relative_to(ROOT)): sha(p) for p in watched}
    assert before == after
    medians = [statistics.median(r['gpu_ratio_b_over_a'] for r in rows if r['session'] == s and r['tokens'] == 8192 and r['pair'] == 'hoisted_to_gather')
               for s in range(cfg['sessions'])] if not args.smoke else []
    summary = {'rows': rows, 'primary_session_medians': medians,
               'primary_passed': all(r <= cfg['primary_max_ratio'] for r in medians) if medians else None,
               'smoke': args.smoke, 'hashes_after': after, 'scope': cfg['scope']}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
