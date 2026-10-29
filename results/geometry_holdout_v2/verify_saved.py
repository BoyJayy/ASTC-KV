"""Read-only stdlib check of the frozen holdout artifacts; no inference."""
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
cfg = json.loads((ROOT/'configs/geometry_holdout_v2.json').read_text())
summary = json.loads((OUT/'summary.json').read_text())
rows = [json.loads(line) for line in (OUT/'losses.jsonl').read_text().splitlines()]
assert len(rows) == 1520
assert len({(r['fixture'], r['method'], r['target_position']) for r in rows}) == 1520
for scope in ['all']+[d['name'] for d in cfg['documents']]:
    means = {}
    for method in cfg['methods']:
        selected = [r for r in rows if r['method'] == method and (scope == 'all' or r['fixture'] == scope)]
        assert len(selected) == (380 if scope == 'all' else 95)
        assert all(math.isfinite(r['nll']) for r in selected)
        means[method] = math.fsum(r['nll'] for r in selected)/len(selected)
    for method, mean in means.items():
        result = summary['aggregates'][scope][method]
        assert result['mean_nll'] == mean and result['perplexity'] == math.exp(mean)
        for label, reference in [('ppl_vs_fp16', 'fp16'), ('ppl_vs_identity', 'astc_identity_V_only'),
                                 ('ppl_vs_asymmetric_u4', 'asymmetric_u4_KV')]:
            assert result[label] == math.exp(mean-means[reference])
for document in cfg['documents']:
    ids = json.loads((ROOT/document['tokens']).read_text())
    for filename, digest in [(document['text'], document['text_sha256']), (document['tokens'], document['token_file_sha256'])]:
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() == digest
    for method in cfg['methods']:
        selected = [r for r in rows if r['method'] == method and r['fixture'] == document['name']]
        assert [r['target_position'] for r in selected] == list(range(289, 384))
        assert [r['target_id'] for r in selected] == ids[289:384]
for fixture in summary['fixtures']:
    assert [layer['layer'] for layer in fixture['per_layer']] == list(range(30))
    for layer in fixture['per_layer']:
        assert layer['astc_identity_counts'] == layer['astc_transpose_counts'] == {
            'astc_blocks_including_edge_padding': 3*(288//6)*11*16,
            'fp16_offsets': 3*288*2, 'fp16_scales': 3*288*2, 'permutation_map': 0}
        for variant in ['identity', 'transpose']:
            path = OUT/fixture['name']/f'layer{layer["layer"]}_{variant}.bin'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == layer[f'{variant}_payload_sha256']
    tail = 30*2*3*95*64*2
    k16 = 30*3*288*64*2
    astcv = 30*3*((288//6)*11*16+288*4)
    u4k = 30*3*(288*64//2+(288//96)*64*4)
    u4v = 30*3*(288*64//2+288*4)
    expected = {'fp16': 2*k16+tail, 'astc_identity_V_only': k16+astcv+tail,
                'astc_transpose_V_only': k16+astcv+tail, 'asymmetric_u4_KV': u4k+u4v+tail}
    for method, result in fixture['methods'].items():
        assert result['logical_bytes'] == expected[method] == sum(result['memory_breakdown'].values())
for path, digest in cfg['source_hashes'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
assert hashlib.sha256((ROOT/'configs/geometry_holdout_v2.json').read_bytes()).hexdigest() == summary['config_sha256']
assert hashlib.sha256((OUT/'losses.jsonl').read_bytes()).hexdigest() == summary['raw_sha256']
ratio = summary['aggregates']['all']['astc_transpose_V_only']['ppl_vs_identity']
assert summary['primary']['ppl_transpose_over_identity'] == ratio
assert summary['primary']['passed'] == (ratio <= cfg['primary']['max_ratio'])
print(json.dumps({'verification': 'PASS', 'rows': len(rows), 'targets_per_method': 380,
                  'memory': expected, 'primary_ratio': ratio, 'primary_passed': summary['primary']['passed']}))
