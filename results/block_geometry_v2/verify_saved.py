"""Read-only stdlib verification of saved H1a outputs, no codec execution."""
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
summary = json.loads((OUT/'summary.json').read_text())
rows = [json.loads(line) for line in (OUT/'rows.jsonl').read_text().splitlines()]
assert len(rows) == 240
assert len({(r['capture'], r['layer'], r['variant']) for r in rows}) == 240
for aggregate in summary['aggregates']:
    scope, variant = aggregate['scope'], aggregate['variant']
    selected = [r for r in rows if r['variant'] == variant and
                (scope == 'all' or scope == 'capture:'+r['capture'] or scope == 'layer:'+str(r['layer']))]
    assert len(selected) == aggregate['pairs']
    for field in ('attention_error_sse', 'attention_reference_sse', 'tensor_error_sse',
                  'tensor_reference_sse', 'practical_v_only_attention_error_sse', 'normalized_padded_error_sse'):
        total = math.fsum(r[field] for r in selected)
        assert math.isclose(total, aggregate[field], rel_tol=1e-14, abs_tol=1e-14)
    for metric, numerator, denominator in (
        ('attention_relative_l2', 'attention_error_sse', 'attention_reference_sse'),
        ('tensor_relative_l2', 'tensor_error_sse', 'tensor_reference_sse'),
        ('practical_v_only_attention_relative_l2', 'practical_v_only_attention_error_sse', 'attention_reference_sse'),
    ):
        value = math.sqrt(math.fsum(r[numerator] for r in selected)/math.fsum(r[denominator] for r in selected))
        assert math.isclose(value, aggregate[metric], rel_tol=1e-14)
for row in rows:
    modes = row['astc_modes']
    assert sum(modes['mode_histogram'].values()) == 1584
    assert sum(modes['partition_histogram'].values()) == 1584
    for label, field in [('normalized_sse', 'normalized_padded_error_sse'), ('tensor_sse', 'tensor_error_sse')]:
        total = math.fsum(v[label] for v in modes['errors_by_mode'].values())
        assert math.isclose(total, row[field], rel_tol=1e-13, abs_tol=1e-13)
    memory = row['memory']
    assert memory['counts'] == {'astc_blocks_including_edge_padding': 3*48*11*16,
                                'fp16_offsets': 3*288*2, 'fp16_scales': 3*288*2}
    assert sum(memory['counts'].values()) == memory['payload_plus_ranges_bytes'] == 28800
    assert memory['bytes_with_map'] == 28800+memory['diagnostic_global_map_bytes']
    path = OUT/f'{row["capture"]}_l{row["layer"]}'/f'{row["variant"]}.bin'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['payload_sha256']
for pair in summary['pairs']:
    directory = OUT/f'{pair["capture"]}_l{pair["layer"]}'
    for suffix in ('bin', 'npy'):
        assert (directory/f'identity.{suffix}').read_bytes() == (directory/f'whole_block_permutation.{suffix}').read_bytes()
    path = ROOT/'results/captures'/f'{pair["capture"]}.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pair['capture_sha256']
for path, digest in summary['provenance']['sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
print(json.dumps({'verification': 'PASS', 'rows': len(rows), 'aggregates': len(summary['aggregates']),
                  'memory_arithmetic': 'PASS', 'payload_and_whole_block_bytes': 'PASS',
                  'source_and_capture_hashes': 'PASS', 'mode_accounting': 'PASS'}))
