#!/usr/bin/env python3
"""Fixed transpose confirmation on newly frozen authored texts; CPU quality only."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from astc_kv.codec import AstcEncoder, affine_quantize, encode_tensor, fp16_floor, fp16_ceil
from capture_qkv import NumpyLlama, MODEL_ID, MODEL_REVISION, tokenize, sha256
from evaluate_model import run_teacher_forced, tail_fp16

METHODS = ['fp16', 'astc_identity_V_only', 'astc_transpose_V_only', 'asymmetric_u4_KV']
DOCUMENTS = ['ceramic_workshop', 'coastal_letter', 'bread_notes', 'choir_rehearsal']
CONFIG = ROOT/'configs/geometry_holdout_v2.json'
FIXTURES = ROOT/'fixtures/geometry_holdout_v2'
MODEL = ROOT/'third_party/model'
ASTC = {'layout':'scalar', 'block':6, 'normalization':'token', 'profile':'ldr'}


def save_json(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n')


def source_paths():
    return [Path(__file__).resolve(), ROOT/'scripts/capture_qkv.py', ROOT/'scripts/evaluate_model.py',
            ROOT/'astc_kv/codec.py', ROOT/'astc_kv/bridge.cpp', ROOT/'astc_kv/libakv_astc.dylib']


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in source_paths()}


def transpose_tiles(image):
    h, t, w = image.shape
    if t % 6 or w % 6: raise ValueError('Tile extents must be multiples of six')
    return image.reshape(h, t//6, 6, w//6, 6).transpose(0, 1, 4, 3, 2).reshape(h, t, w)


def normalized_scalar(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or not np.isfinite(x).all(): raise ValueError('Expected finite H,T,D')
    offset = fp16_floor(x.min(-1, keepdims=True))
    scale = fp16_ceil(np.maximum(x.max(-1, keepdims=True)-offset.astype(np.float32), np.float32(2**-24)))
    normalized = (x-offset.astype(np.float32))/scale.astype(np.float32)
    if not np.isfinite(offset).all() or not np.isfinite(scale).all(): raise ValueError('Range overflow')
    if normalized.min() < -1e-5 or normalized.max() > 1+1e-5: raise ValueError('Ranges do not cover input')
    width = ((x.shape[-1]+5)//6)*6
    return np.pad(np.clip(normalized, 0, 1), ((0, 0), (0, 0), (0, width-x.shape[-1])), mode='edge'), offset, scale


def restore_scalar(image, offset, scale, dim, transpose=True):
    restored = transpose_tiles(image) if transpose else image
    return restored[:, :, :dim]*scale.astype(np.float32)+offset.astype(np.float32)


def encode_transpose(x, encoder, transpose=True):
    scalar, offset, scale = normalized_scalar(x)
    transformed = transpose_tiles(scalar) if transpose else scalar
    images = np.ones((*transformed.shape, 4), dtype=np.float32)
    images[..., :3] = transformed[..., None]
    payloads, decoded, elapsed = [], [], 0.
    for image in images:
        payload, rec, seconds = encoder.roundtrip(image)
        payloads.append(payload); decoded.append(rec[..., 0]); elapsed += seconds
    result = restore_scalar(np.stack(decoded), offset, scale, x.shape[-1], transpose)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite reconstruction')
    counts = {'astc_blocks_including_edge_padding': sum(len(p) for p in payloads),
              'fp16_offsets': offset.nbytes, 'fp16_scales': scale.nbytes, 'permutation_map': 0}
    return result, payloads, offset, scale, counts, elapsed


def self_tests():
    # Independent loop implements the intended coordinate swap, including edge tiles.
    source = np.arange(2*12*66, dtype=np.float32).reshape(2, 12, 66)
    explicit = np.empty_like(source)
    for h in range(2):
        for t in range(12):
            for c in range(66): explicit[h, t, c] = source[h, (t//6)*6+c%6, (c//6)*6+t%6]
    assert np.array_equal(transpose_tiles(source), explicit)
    assert np.array_equal(transpose_tiles(explicit), source)
    x = (np.arange(2*12*64, dtype=np.float32).reshape(2, 12, 64)%43-21)
    x *= np.linspace(.02, 3, 12, dtype=np.float32)[None, :, None]
    scalar, offset, scale = normalized_scalar(x)
    transformed = transpose_tiles(scalar)
    recon = restore_scalar(transformed, offset, scale, 64)
    assert np.allclose(recon, x, rtol=2e-6, atol=1e-5)
    for row in range(0, 12, 6):
        for col in range(0, 66, 6):
            assert np.array_equal(np.sort(scalar[:, row:row+6, col:col+6].reshape(2, 36)),
                                  np.sort(transformed[:, row:row+6, col:col+6].reshape(2, 36)))
    class RecordingModel:
        def __init__(self): self.positions = []
        def step(self, token, position, caches, kv_transform):
            assert position == caches[0][0].shape[1]
            self.positions.append(position)
            k, v = caches[0]
            extra = np.full((1, 1, 2), .123456, dtype=np.float32)
            k, v = kv_transform(0, None, np.concatenate((k, extra), 1), np.concatenate((v, extra), 1))
            assert np.array_equal(k[:, :3], original) and np.array_equal(v[:, :3], original)
            assert np.array_equal(k[:, -1], extra.astype(np.float16).astype(np.float32)[:, 0])
            return np.array([0., 1., -1.]), [(k, v)]
    original = np.full((1, 3, 2), .9876543, dtype=np.float32); original.setflags(write=False)
    model = RecordingModel(); ids = np.array([2, 1, 0, 2, 0, 1])
    losses = run_teacher_forced(model, ids, [(original, original)], 3)
    assert model.positions == [3, 4]
    assert [r['target_position'] for r in losses] == [4, 5]
    assert [r['target_id'] for r in losses] == [0, 1]
    assert all(math.isfinite(r['nll']) for r in losses)
    return {'passed':True, 'checks':['explicit_coordinate_oracle', 'transpose_involution', 'same_block_multisets',
                                    'variable_token_ranges_padding_roundtrip', 'causal_target_window', 'tail_only_fp16', 'immutable_prefix']}


def freeze():
    if CONFIG.exists(): raise RuntimeError('Frozen configuration already exists')
    token_paths = [FIXTURES/f'{name}.tokens.json' for name in DOCUMENTS]
    if any(p.exists() for p in token_paths): raise RuntimeError('Token files already exist')
    checks = self_tests(); documents = []
    for name, path in zip(DOCUMENTS, token_paths):
        text_path = FIXTURES/f'{name}.txt'
        ids = tokenize(MODEL, text_path.read_text(), 100000)
        if len(ids) < 384: raise ValueError(f'{name} has fewer than384 tokens')
        selected = ids[:384]
        save_json(path, selected.tolist())
        documents.append({'name':name, 'text':str(text_path.relative_to(ROOT)), 'text_sha256':sha256(text_path),
                          'tokens':str(path.relative_to(ROOT)), 'token_file_sha256':sha256(path),
                          'token_ids_u32le_sha256':hashlib.sha256(selected.astype('<u4').tobytes()).hexdigest(),
                          'full_text_token_count':len(ids), 'selected_tokens':384})
    previous = json.loads((ROOT/'results/revalidation_20261028/model_quality.json').read_text())
    cfg = {
        'schema':'astckv.geometry_holdout.v2', 'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'registered_before_model_inference':True,
        'selection':'Fixed within-block transpose selected on old block_geometry_v2; no further orientation search or fitting on these documents.',
        'data_scope':'Four newly authored English texts, frozen before inference; separate from old four pilot texts. Small controlled confirmation, not representative benchmark or robust-quality proof.',
        'model_id':MODEL_ID, 'model_revision':MODEL_REVISION,
        'expected_weights_sha256':previous['model_weights_sha256'],
        'weight_hash_basis':'Previously verified local original BF16 checkpoint; actual bytes rehashed before run.',
        'model_config_sha256':sha256(MODEL/'config.json'), 'tokenizer_sha256':sha256(MODEL/'tokenizer.json'),
        'tokenization':'Existing checkpoint tokenizer, add_special_tokens=False, no BOS/chat template; first384 tokens of each frozen text.',
        'documents':documents, 'methods':METHODS, 'context_tokens':384, 'prefix_tokens':288, 'page_tokens':96,
        'layers':list(range(30)), 'head_dim':64, 'kv_heads':3, 'quality':98, 'astc':ASTC,
        'K_policy':'Both ASTC methods retain FP16 K. asymmetric_u4_KV uses affine Kchannel group96, Vtoken group64.',
        'V_policy':'ASTC scalar RGB, token ranges computed before transform, edge-pad64to66; transpose all normalized6x6 blocks, inverse before original token denormalization; zero transform metadata.',
        'causality':'Prefill only input0..287, compress prefix once. Consume positions288..382 with genuine per-layer network feedback. New KV FP16. Evaluate targets289..383, exclude288.',
        'primary':{'metric':'exp(meanNLL_transpose-meanNLL_identity), all380 matching targets', 'max_ratio':1.0,
                   'decision':'PASS iff complete finite380targets/method and transpose aggregate PPL<=identity; no posthoc tolerance, per-document gate, or variant selection.'},
        'secondary':'Report all per-document results, ratios versus FP16 and asymmetricU4, and logical prefix+FP16tail memory. No extra acceptance thresholds.',
        'controls':'Pure synthetic coordinate/range/tail/causal tests; actual synthetic identity matches existing codec payload+decode; equal identity/transpose metadata and logical bytes for every real layer; immutable prepared prefix; exact token/hash/target match; finite losses.',
        'budget':{'max_total_wall_s':300, 'check_between_methods':True, 'max_methods':16, 'max_targets_per_method':380,
                  'cpu_threads':1, 'gpu':False, 'new_downloads':False},
        'source_hashes':source_hashes(), 'preflight_tests':checks,
    }
    save_json(CONFIG, cfg)
    print(json.dumps({'stage':'frozen', 'config_sha256':sha256(CONFIG), 'documents':documents}, indent=2))


def verify_config(cfg, include_weights=False):
    assert cfg['methods'] == METHODS and [d['name'] for d in cfg['documents']] == DOCUMENTS
    assert cfg['layers'] == list(range(30)) and cfg['quality'] == 98
    assert cfg['source_hashes'] == source_hashes(), 'Source changed after freezing'
    assert sha256(MODEL/'config.json') == cfg['model_config_sha256']
    assert sha256(MODEL/'tokenizer.json') == cfg['tokenizer_sha256']
    if include_weights: assert sha256(MODEL/'model.safetensors') == cfg['expected_weights_sha256']
    prepared = []
    old_tokens = []
    for name in ('systems', 'story', 'science', 'records'):
        with np.load(ROOT/'results/captures'/f'{name}.npz', allow_pickle=False) as old:
            old_tokens.append(tuple(old['token_ids'][:384]))
    for document in cfg['documents']:
        text_path, token_path = ROOT/document['text'], ROOT/document['tokens']
        assert sha256(text_path) == document['text_sha256']
        assert sha256(token_path) == document['token_file_sha256']
        ids = np.asarray(json.loads(token_path.read_text()), dtype=np.int64)
        assert ids.shape == (384,) and ids.min() >= 0 and ids.max() < 49152
        assert np.array_equal(ids, tokenize(MODEL, text_path.read_text(), 384))
        assert hashlib.sha256(ids.astype('<u4').tobytes()).hexdigest() == document['token_ids_u32le_sha256']
        assert tuple(ids) not in old_tokens, 'Reused old pilot token sequence'
        prepared.append((document, ids))
    assert len({tuple(ids) for _, ids in prepared}) == 4
    return prepared


def actual_codec_control(encoder, cfg):
    # Tiny synthetic completed page; control uses no held-out model state.
    x = np.random.default_rng(414).normal(size=(1, 96, 64)).astype(np.float32)
    base = encode_tensor(x, cfg['astc'], cfg['page_tokens'], cfg['quality'])
    rec, payloads, offset, scale, counts, _ = encode_transpose(x, encoder, transpose=False)
    assert payloads == base.payloads and np.array_equal(rec, base.decoded)
    assert counts == base.byte_counts
    assert np.array_equal(offset, base.offsets) and np.array_equal(scale, base.scales)
    return {'identity_payload_decode_metadata_exact':True}


def prepare_caches(original, cfg, encoder, directory):
    prepared = {name:[] for name in METHODS}
    sizes = {name:{'prefix_k_bytes':0, 'prefix_v_bytes':0} for name in METHODS}
    layer_records = []
    for layer, (k, v) in enumerate(original):
        assert k.shape == v.shape == (3, 288, 64)
        kf, vf = (x.astype(np.float16).astype(np.float32) for x in (k, v))
        ku, kc = affine_quantize(k, 4, 'channel', cfg['page_tokens'])
        vu, vc = affine_quantize(v, 4, 'token', 64)
        identity = encode_tensor(v, cfg['astc'], cfg['page_tokens'], cfg['quality'])
        trans, payloads, offset, scale, counts, encode_s = encode_transpose(v, encoder)
        assert counts == identity.byte_counts
        assert np.array_equal(offset, identity.offsets) and np.array_equal(scale, identity.scales)
        arrays = {'fp16':(kf, vf), 'astc_identity_V_only':(kf, identity.decoded),
                  'astc_transpose_V_only':(kf, trans), 'asymmetric_u4_KV':(ku, vu)}
        byte_pairs = {'fp16':(k.size*2, v.size*2), 'astc_identity_V_only':(k.size*2, identity.logical_bytes),
                      'astc_transpose_V_only':(k.size*2, sum(counts.values())), 'asymmetric_u4_KV':(sum(kc.values()), sum(vc.values()))}
        for name, pair in arrays.items():
            for array in pair: array.setflags(write=False)
            prepared[name].append(pair)
            sizes[name]['prefix_k_bytes'] += byte_pairs[name][0]
            sizes[name]['prefix_v_bytes'] += byte_pairs[name][1]
        for name, parts in [('identity', identity.payloads), ('transpose', payloads)]:
            with (directory/f'layer{layer}_{name}.bin').open('xb') as f: f.write(b''.join(parts))
        with (directory/f'layer{layer}_ranges.npz').open('xb') as f: np.savez(f, offsets=offset, scales=scale)
        layer_records.append({'layer':layer, 'astc_identity_counts':identity.byte_counts, 'astc_transpose_counts':counts,
                              'u4_k_counts':kc, 'u4_v_counts':vc,
                              'identity_payload_sha256':hashlib.sha256(b''.join(identity.payloads)).hexdigest(),
                              'transpose_payload_sha256':hashlib.sha256(b''.join(payloads)).hexdigest(),
                              'identity_encode_cpu_s':identity.encode_cpu_s, 'transpose_encode_cpu_s':encode_s})
    return prepared, sizes, layer_records


def summarize(rows):
    groups = {}
    for scope in ['all']+DOCUMENTS:
        selected = [r for r in rows if scope == 'all' or r['fixture'] == scope]
        means = {method:math.fsum(r['nll'] for r in selected if r['method'] == method)/sum(r['method'] == method for r in selected) for method in METHODS}
        groups[scope] = {}
        for method in METHODS:
            n = sum(r['method'] == method for r in selected)
            groups[scope][method] = {'targets':n, 'mean_nll':means[method], 'perplexity':math.exp(means[method]),
                'ppl_vs_fp16':math.exp(means[method]-means['fp16']),
                'ppl_vs_identity':math.exp(means[method]-means['astc_identity_V_only']),
                'ppl_vs_asymmetric_u4':math.exp(means[method]-means['asymmetric_u4_KV'])}
    return groups


def run(args, cfg):
    for env in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        if os.environ.get(env) != '1': raise RuntimeError(f'{env}=1 required')
    if args.out.exists(): raise RuntimeError('Output exists; preserve prior results and choose a new --out')
    started = time.perf_counter(); documents = verify_config(cfg, include_weights=True)
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out/'frozen_config.json', cfg)
    checks = self_tests(); config_sha = sha256(CONFIG)
    result = {'schema':cfg['schema'], 'status':'running', 'config_sha256':config_sha,
              'source_hashes':cfg['source_hashes'], 'model_weights_sha256':cfg['expected_weights_sha256'],
              'python':sys.version, 'numpy':np.__version__, 'platform':platform.platform(),
              'scope':cfg['data_scope'], 'gpu_executed':False, 'self_tests':checks, 'fixtures':[]}
    save_json(args.out/'start_provenance.json', result)
    model = NumpyLlama(MODEL)
    assert model.num_layers == 30 and model.hkv == 3 and model.head_dim == 64
    encoder = AstcEncoder(6, cfg['quality'], False); rows = []
    try:
        result['actual_codec_control'] = actual_codec_control(encoder, cfg)
        with (args.out/'losses.jsonl').open('x') as raw:
            for document, ids in documents:
                name = document['name']; fixture_start = time.perf_counter()
                directory = args.out/name; directory.mkdir()
                _, original = model.prefill(ids[:cfg['prefix_tokens']])
                prepared, sizes, layer_records = prepare_caches(original, cfg, encoder, directory)
                original = None
                fixture = {'name':name, 'text_sha256':document['text_sha256'], 'token_file_sha256':document['token_file_sha256'],
                           'per_layer':layer_records, 'methods':{}}
                for method in METHODS:
                    if time.perf_counter()-started > cfg['budget']['max_total_wall_s']:
                        raise RuntimeError('Frozen wall-time budget reached; partial result has no primary verdict')
                    assert all(k.shape == v.shape == (3, 288, 64) and not k.flags.writeable and not v.flags.writeable for k, v in prepared[method])
                    then = time.perf_counter(); losses = run_teacher_forced(model, ids, prepared[method], cfg['prefix_tokens'])
                    assert len(losses) == 95 and [r['target_position'] for r in losses] == list(range(289, 384))
                    assert [r['target_id'] for r in losses] == ids[289:384].tolist()
                    assert all(math.isfinite(r['nll']) for r in losses)
                    for loss in losses:
                        row = {'fixture':name, 'method':method, **loss}; rows.append(row)
                        raw.write(json.dumps(row, allow_nan=False)+'\n')
                    raw.flush()
                    tail_bytes = (cfg['context_tokens']-1-cfg['prefix_tokens'])*2*model.num_layers*model.hkv*model.head_dim*2
                    memory = {**sizes[method], 'fp16_tail_bytes':tail_bytes}
                    total = sum(memory.values())
                    fixture['methods'][method] = {'targets':95, 'logical_bytes':total, 'memory_breakdown':memory,
                        'effective_bits_per_kv_scalar':total*8/((cfg['context_tokens']-1)*2*model.num_layers*model.hkv*model.head_dim),
                        'mean_nll':math.fsum(r['nll'] for r in losses)/95, 'cpu_wall_s_diagnostic_only':time.perf_counter()-then}
                    print(json.dumps({'stage':'teacher_forced', 'fixture':name, 'method':method, **fixture['methods'][method]}), flush=True)
                fixture['wall_s'] = time.perf_counter()-fixture_start
                assert fixture['methods']['astc_identity_V_only']['logical_bytes'] == fixture['methods']['astc_transpose_V_only']['logical_bytes']
                save_json(args.out/f'{name}_summary.json', fixture); result['fixtures'].append(fixture)
                prepared = None
        assert len(rows) == 1520 and len({(r['fixture'],r['method'],r['target_position']) for r in rows}) == 1520
        result['aggregates'] = summarize(rows)
        main = result['aggregates']['all']['astc_transpose_V_only']
        assert all(r['targets'] == 380 for r in result['aggregates']['all'].values())
        result['primary'] = {'ppl_transpose_over_identity':main['ppl_vs_identity'], 'max_ratio':cfg['primary']['max_ratio'],
                             'passed':main['ppl_vs_identity'] <= cfg['primary']['max_ratio'],
                             'complete_targets_per_method':380, 'no_posthoc_threshold':True}
        assert sha256(CONFIG) == config_sha and source_hashes() == cfg['source_hashes']
        verify_config(cfg, include_weights=True)
        result.update(status='completed', wall_s=time.perf_counter()-started, raw_sha256=sha256(args.out/'losses.jsonl'), sources_inputs_unchanged=True)
        save_json(args.out/'summary.json', result)
        print(json.dumps({'stage':'completed', 'wall_s':result['wall_s'], 'primary':result['primary'], 'aggregates':result['aggregates']['all']}), flush=True)
    except Exception as exc:
        save_json(args.out/'failure.json', {'status':'incomplete', 'type':type(exc).__name__, 'message':str(exc), 'saved_rows':len(rows), 'primary_verdict':None})
        raise
    finally:
        encoder.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['freeze','check','run'], default='check')
    ap.add_argument('--out', type=Path, default=ROOT/'results/geometry_holdout_v2')
    args = ap.parse_args()
    if args.mode == 'freeze': freeze(); return
    cfg = json.loads(CONFIG.read_text())
    if args.mode == 'check':
        documents = verify_config(cfg); print(json.dumps({'self_tests':self_tests(), 'frozen_documents_verified':len(documents), 'model_inference_executed':False, 'astc_encoder_executed':False})); return
    run(args, cfg)


if __name__ == '__main__': main()
