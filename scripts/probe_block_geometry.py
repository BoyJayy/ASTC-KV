#!/usr/bin/env python3
"""Frozen, CPU-only H1a probe of normalized ASTC block geometry."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from astc_kv.codec import AstcEncoder, encode_tensor, fp16_floor, fp16_ceil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def sse(a):
    return float(np.sum(np.square(a, dtype=np.float64)))


def split_blocks(a, block=6):
    h, t, w = a.shape
    return a.reshape(h, t//block, block, w//block, block).transpose(0, 1, 3, 2, 4).reshape(h, -1, block*block)


def join_blocks(a, tokens, width, block=6):
    return a.reshape(len(a), tokens//block, width//block, block, block).transpose(0, 1, 3, 2, 4).reshape(len(a), tokens, width)


def maps(seed):
    g = np.arange(36).reshape(6, 6)
    result = {
        'identity': g.ravel(), 'rot90': np.rot90(g, 1).ravel(),
        'rot180': np.rot90(g, 2).ravel(), 'rot270': np.rot90(g, 3).ravel(),
        'flip_horizontal': g[:, ::-1].ravel(), 'flip_vertical': g[::-1].ravel(),
        'transpose': g.T.ravel(), 'anti_transpose': g.T[::-1, ::-1].ravel(),
        'random36': np.random.default_rng(seed).permutation(36),
    }
    return result


def transform_blocks(blocks, mapping, block_order=None):
    if block_order is not None:
        return blocks[:, block_order, :]
    return blocks[..., mapping]


def inverse_blocks(blocks, mapping, block_order=None):
    if block_order is not None:
        return blocks[:, np.argsort(block_order), :]
    return blocks[..., np.argsort(mapping)]


def self_test(cfg):
    mm = maps(cfg['seed'])
    assert len({tuple(mm[n]) for n in list(mm)[:8]}) == 8
    original = np.arange(2*12*12, dtype=np.float32).reshape(2, 12, 12)
    blocks = split_blocks(original)
    assert np.array_equal(original, join_blocks(blocks, 12, 12))
    for name, mapping in mm.items():
        assert sorted(mapping) == list(range(36)), name
        shifted = transform_blocks(blocks, mapping)
        assert np.array_equal(blocks, inverse_blocks(shifted, mapping)), name
        assert np.array_equal(np.sort(blocks, -1), np.sort(shifted, -1)), name
    order = np.array([2, 0, 3, 1])
    assert np.array_equal(blocks, inverse_blocks(transform_blocks(blocks, None, order), None, order))
    # Independent explicit causal/GQA expectation, including a zero-weight future token.
    q = np.zeros((4, 1, 2)); k = np.zeros((2, 3, 2)); v = np.arange(12).reshape(2, 3, 2)
    p = probabilities(q, k, np.array([1]))
    assert np.array_equal(p, np.broadcast_to([.5, .5, 0.], p.shape))
    assert np.array_equal(weighted(p, v), np.repeat(v[:, :2].mean(1), 2, axis=0)[:, None])
    # Header decoding check independent of encoder: regular2 partitions and LDR void extent.
    assert mode_header(bytes([0x23, 0x09])+bytes(14)) == ('291', '2')
    assert mode_header(bytes([0xfc, 0x01])+bytes(14)) == ('void_extent', '0')
    return {'passed': True, 'checks': ['block_roundtrip', 'all_9_map_inverses', 'D4_unique', 'multisets', 'whole_block_inverse', 'causal_GQA', 'header_parser']}


def probabilities(q, k, positions):
    hq, _, d = q.shape
    if hq % len(k):
        raise ValueError('Invalid GQA heads')
    kk = k[np.arange(hq)//(hq//len(k))].astype(np.float64)
    logits = np.einsum('hqd,htd->hqt', q.astype(np.float64), kk)/np.sqrt(d)
    logits = np.where(np.arange(k.shape[1])[None, None] <= positions[None, :, None], logits, -np.inf)
    p = np.exp(logits-logits.max(-1, keepdims=True))
    return p/p.sum(-1, keepdims=True)


def weighted(p, v):
    return np.einsum('hqt,htd->hqd', p, v[np.arange(len(p))//(len(p)//len(v))].astype(np.float64))


def mode_header(payload):
    header = int.from_bytes(payload[:2], 'little')
    mode = header & 0x7ff
    if mode & 0x1ff == 0x1fc:
        return 'void_extent', '0'
    return str(mode), str(((header >> 11) & 3)+1)


def describe_modes(payloads, normalized_err, tensor_err):
    mode_hist, partitions = Counter(), Counter()
    stats = defaultdict(lambda: {'blocks': 0, 'normalized_sse': 0., 'tensor_sse': 0.})
    for h, payload in enumerate(payloads):
        for b in range(len(payload)//16):
            mode, count = mode_header(payload[b*16:(b+1)*16])
            mode_hist[mode] += 1; partitions[count] += 1
            st = stats[mode]; st['blocks'] += 1
            st['normalized_sse'] += sse(normalized_err[h, b])
            st['tensor_sse'] += sse(tensor_err[h, b])
    return {'mode_histogram': dict(mode_hist), 'partition_histogram': dict(partitions), 'errors_by_mode': dict(stats)}


def run_pair(cfg, name, layer, output, encoder):
    start = time.perf_counter()
    capture_path = ROOT/'results/captures'/f'{name}.npz'
    capture_sha = sha(capture_path)
    with np.load(capture_path, allow_pickle=False) as capture:
        assert list(capture['layer_ids']) == cfg['layers']
        li = list(capture['layer_ids']).index(layer)
        q, k, v = (capture[n][li] for n in ('q', 'k', 'v'))
    prefix = cfg['prefix_tokens']; positions = np.asarray(cfg['query_positions'])
    assert positions.min() >= prefix and positions.max() < k.shape[1]
    x = v[:, :prefix].astype(np.float32)
    heads, tokens, dim = x.shape; width = ((dim+5)//6)*6
    assert tokens % cfg['page_tokens'] == 0 and cfg['page_tokens'] % 6 == 0
    offset = fp16_floor(x.min(-1, keepdims=True))
    scale = fp16_ceil(np.maximum(x.max(-1, keepdims=True)-offset.astype(np.float32), np.float32(2**-24)))
    normalized = (x-offset.astype(np.float32))/scale.astype(np.float32)
    assert np.isfinite(normalized).all() and normalized.min() >= -1e-5 and normalized.max() <= 1+1e-5
    scalar = np.pad(np.clip(normalized, 0, 1), ((0, 0), (0, 0), (0, width-dim)), mode='edge')
    blocks = split_blocks(scalar)
    block_order = np.random.default_rng(cfg['seed']+1).permutation(blocks.shape[1])
    fixed_maps = maps(cfg['seed'])
    p = probabilities(q[:, positions], k, positions)
    p16 = probabilities(q[:, positions], k.astype(np.float16).astype(np.float32), positions)
    ref = weighted(p, v); ref_sse = sse(ref)
    pair_dir = output/f'{name}_l{layer}'; pair_dir.mkdir()
    with (pair_dir/'fixed_input.npz').open('xb') as f:
        np.savez_compressed(f, original_v_prefix=x, normalized_padded=scalar, fp16_offsets=offset, fp16_scales=scale)
    counts = {'astc_blocks_including_edge_padding': heads*blocks.shape[1]*16, 'fp16_offsets': offset.nbytes, 'fp16_scales': scale.nbytes}
    original_tensor_sse = sse(x)
    reference = encode_tensor(x, {'layout':'scalar', 'block':6, 'normalization':'token', 'profile':'ldr'}, cfg['page_tokens'], cfg['quality'], cfg['seed'])
    identity_payloads = identity_rec = None
    rows = []
    for variant in cfg['variants']:
        order = block_order if variant == 'whole_block_permutation' else None
        mapping = fixed_maps.get(variant)
        transformed = transform_blocks(blocks, mapping, order)
        if order is None:
            assert np.array_equal(np.sort(blocks, -1), np.sort(transformed, -1)), variant
        else:
            assert np.array_equal(blocks, inverse_blocks(transformed, None, order))
        image = np.ones((heads, tokens, width, 4), dtype=np.float32)
        image[..., :3] = join_blocks(transformed, tokens, width)[..., None]
        payloads, recs, encode_s = [], [], 0.
        for im in image:
            payload, rec, elapsed = encoder.roundtrip(im)
            payloads.append(payload); recs.append(rec[..., 0]); encode_s += elapsed
        inverse = inverse_blocks(split_blocks(np.stack(recs)), mapping, order)
        restored = join_blocks(inverse, tokens, width)
        reconstructed = restored[:, :, :dim]*scale.astype(np.float32)+offset.astype(np.float32)
        encoded_payload_sha = hashlib.sha256(b''.join(payloads)).hexdigest()
        if order is not None:
            with (pair_dir/f'{variant}.encoded_order.bin').open('xb') as f: f.write(b''.join(payloads))
            payloads = [np.frombuffer(b, dtype=np.uint8).reshape(-1, 16)[np.argsort(order)].tobytes() for b in payloads]
        if variant == 'identity':
            identity_payloads = payloads; identity_rec = reconstructed.copy()
            assert payloads == reference.payloads, 'Identity payload diverges from existing codec'
            assert np.array_equal(reconstructed, reference.decoded), 'Identity reconstruction diverges'
            assert sum(counts.values()) == reference.logical_bytes
        if variant == 'whole_block_permutation':
            assert payloads == identity_payloads, 'Whole-block ordering changed encoded payload'
            assert np.array_equal(reconstructed, identity_rec), 'Whole-block inverse changed reconstruction'
        assert np.isfinite(reconstructed).all()
        err = reconstructed.astype(np.float64)-x
        delta = weighted(p[:, :, :prefix], err)
        candidate_v = v.astype(np.float16).astype(np.float32); candidate_v[:, :prefix] = reconstructed
        practical_delta = weighted(p16, candidate_v)-ref
        padded_err = np.pad(err, ((0, 0), (0, 0), (0, width-dim)), mode='constant')
        normalized_err = inverse.astype(np.float64)-blocks
        mode = describe_modes(payloads, normalized_err, split_blocks(padded_err))
        map_bytes = 36 if variant == 'random36' else blocks.shape[1]*2 if order is not None else 0
        row = {
            'capture': name, 'layer': int(layer), 'variant': variant,
            'prefix_tokens': prefix, 'query_positions': positions.tolist(),
            'attention_error_sse': sse(delta), 'attention_reference_sse': ref_sse,
            'attention_relative_l2': float(np.sqrt(sse(delta)/ref_sse)),
            'tensor_error_sse': sse(err), 'tensor_reference_sse': original_tensor_sse,
            'tensor_mse': sse(err)/x.size, 'tensor_relative_l2': float(np.sqrt(sse(err)/original_tensor_sse)),
            'normalized_padded_error_sse': sse(normalized_err),
            'practical_v_only_attention_error_sse': sse(practical_delta),
            'practical_v_only_attention_relative_l2': float(np.sqrt(sse(practical_delta)/ref_sse)),
            'memory': {'counts': counts, 'payload_plus_ranges_bytes': sum(counts.values()), 'diagnostic_global_map_bytes': int(map_bytes), 'bytes_with_map': sum(counts.values())+int(map_bytes), 'bits_per_original_v_scalar_without_map': 8*sum(counts.values())/x.size, 'same_cache_bits_candidate': variant in list(fixed_maps)[:8]},
            'astc_modes': mode, 'encode_cpu_s': encode_s,
            'payload_sha256': hashlib.sha256(b''.join(payloads)).hexdigest(),
            'encoded_order_payload_sha256': encoded_payload_sha,
            'controls': {'finite': True, 'same_ranges': True, 'fixed_values_inverse': True, 'identity_reference_exact': True if variant == 'identity' else None, 'whole_block_exact': True if variant == 'whole_block_permutation' else None},
        }
        with (pair_dir/f'{variant}.bin').open('xb') as f: f.write(b''.join(payloads))
        with (pair_dir/f'{variant}.npy').open('xb') as f: np.save(f, reconstructed, allow_pickle=False)
        rows.append(row)
    assert sha(capture_path) == capture_sha, 'Capture changed during pair'
    return rows, {'capture': name, 'layer': layer, 'wall_s': time.perf_counter()-start, 'capture_sha256': capture_sha, 'all_controls_pass': True}


def aggregate(rows):
    sums = defaultdict(lambda: defaultdict(float))
    mode_counts = defaultdict(Counter)
    partitions = defaultdict(Counter)
    for row in rows:
        scopes = ['all', f'capture:{row["capture"]}', f'layer:{row["layer"]}']
        for scope in scopes:
            key = (scope, row['variant']); target = sums[key]
            target['pairs'] += 1
            for field in ('attention_error_sse', 'attention_reference_sse', 'tensor_error_sse', 'tensor_reference_sse', 'practical_v_only_attention_error_sse', 'normalized_padded_error_sse'):
                target[field] += row[field]
            mode_counts[key].update(row['astc_modes']['mode_histogram'])
            partitions[key].update(row['astc_modes']['partition_histogram'])
    result = []
    for (scope, variant), a in sums.items():
        a = dict(a); a['pairs'] = int(a['pairs'])
        a['attention_relative_l2'] = float(np.sqrt(a['attention_error_sse']/a['attention_reference_sse']))
        a['tensor_relative_l2'] = float(np.sqrt(a['tensor_error_sse']/a['tensor_reference_sse']))
        a['practical_v_only_attention_relative_l2'] = float(np.sqrt(a['practical_v_only_attention_error_sse']/a['attention_reference_sse']))
        result.append({'scope': scope, 'variant': variant, **a, 'mode_histogram': dict(mode_counts[(scope, variant)]), 'partition_histogram': dict(partitions[(scope, variant)])})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT/'configs/block_geometry_v2.json')
    ap.add_argument('--out', type=Path, default=ROOT/'results/block_geometry_v2')
    ap.add_argument('--self-test-only', action='store_true')
    args = ap.parse_args(); cfg = json.loads(args.config.read_text())
    checks = self_test(cfg)
    if args.self_test_only:
        print(json.dumps(checks)); return
    for variable in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        if os.environ.get(variable) != '1': raise RuntimeError(f'{variable}=1 required')
    args.out.mkdir(exist_ok=False)
    start = time.perf_counter()
    provenance_paths = [Path(__file__), args.config, ROOT/'astc_kv/codec.py', ROOT/'astc_kv/bridge.cpp', ROOT/'astc_kv/libakv_astc.dylib', ROOT/'third_party/astc-encoder-5.7.0/Source/astcenc_symbolic_physical.cpp']
    provenance = {'sha256': {str(p.relative_to(ROOT)): sha(p) for p in provenance_paths}, 'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(), 'cpu_threads': 1, 'hardware_gpu_executed': False, 'self_tests': checks}
    save_json(args.out/'preregistered_config.json', cfg)
    save_json(args.out/'start_provenance.json', provenance)
    pilot = (cfg['pilot']['capture'], cfg['pilot']['layer'])
    schedule = [pilot]+[(c, l) for c in cfg['captures'] for l in cfg['layers'] if (c, l) != pilot]
    rows, pairs = [], []
    encoder = AstcEncoder(cfg['block'], cfg['quality'], False)
    stop_reason = 'complete'
    try:
        with (args.out/'rows.jsonl').open('x') as stream:
            for index, (capture, layer) in enumerate(schedule):
                if index and time.perf_counter()-start >= cfg['total_max_wall_s']:
                    stop_reason = 'overall_wall_budget'; break
                rr, pair = run_pair(cfg, capture, layer, args.out, encoder)
                rows.extend(rr); pairs.append(pair)
                for row in rr: stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush(); print(json.dumps(pair), flush=True)
                if index == 0 and pair['wall_s'] > cfg['pilot_max_wall_s']:
                    stop_reason = 'pilot_wall_budget'; break
    except Exception as exc:
        save_json(args.out/'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'completed_pairs': pairs, 'quality_interpretation_allowed': False})
        raise
    finally:
        encoder.close()
    unchanged = all(sha(p) == provenance['sha256'][str(p.relative_to(ROOT))] for p in provenance_paths)
    assert unchanged, 'Source/config/encoder changed while running'
    summary = {'schema': cfg['schema'], 'scope': cfg['status'], 'stop_reason': stop_reason, 'complete': len(pairs) == len(schedule), 'pairs': pairs, 'row_count': len(rows), 'wall_s': time.perf_counter()-start, 'all_controls_pass': True, 'sources_unchanged': unchanged, 'aggregates': aggregate(rows), 'provenance': provenance}
    save_json(args.out/'summary.json', summary)
    print(json.dumps({'complete': summary['complete'], 'rows': len(rows), 'wall_s': summary['wall_s'], 'controls': 'PASS'}), flush=True)


if __name__ == '__main__':
    main()
