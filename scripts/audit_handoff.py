#!/usr/bin/env python3
"""Independent archive, raw-result and memory arithmetic checks (no model run)."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    local_verified = []
    with zipfile.ZipFile(args.archive) as z:
        manifest = json.loads(z.read('astc-kv/results/file_manifest.json'))
        bad = []
        for item in manifest['files']:
            data = z.read('astc-kv/' + item['path'])
            if len(data) != item['bytes'] or sha(data) != item['sha256']:
                bad.append(item['path'])
            # Bind local inputs used below to the verified archive, independently
            # of any source changes made during the continuation.
            if item['path'].startswith(('results/pilot_', 'results/model_quality')) and item['path'].endswith(('.json', '.jsonl')):
                assert (root/item['path']).read_bytes() == data, item['path']
                local_verified.append(item['path'])
        assert not bad, bad
        assert z.testzip() is None
    rows = []
    for name in ['systems', 'story', 'science', 'records']:
        rows.extend(json.loads(line) for line in (root/'results'/f'pilot_{name}.jsonl').read_text().splitlines())
    assert len(rows) == 936
    assert len({(r['capture'], r['layer'], r['variant'], r['mode']) for r in rows}) == 936
    groups = defaultdict(list)
    split = {'systems': 'tune', 'story': 'development', 'science': 'pilot_holdout', 'records': 'pilot_holdout'}
    for r in rows:
        groups[split[r['capture']], r['variant'], r['mode']].append(r)
        m = r['memory']
        total = m['policy_prefix_k_bytes'] + m['policy_prefix_v_bytes'] + m['fp16_tail_bytes']
        assert total == m['payload_metadata_bytes']
        assert math.isclose(total*8/(2*3*384*64), m['effective_bits_per_original_kv_scalar'], abs_tol=1e-12)
    summary = json.loads((root/'results/pilot_summary.json').read_text())['results']
    max_snapshot_difference = 0.0
    selected = []
    for item in summary:
        rs = groups[item['split'], item['variant'], item['mode']]
        rel = math.sqrt(math.fsum(r['error_sum_squares'] for r in rs)/math.fsum(r['reference_sum_squares'] for r in rs))
        max_snapshot_difference = max(max_snapshot_difference, abs(rel-item['relative_l2']))
        assert abs(rel-item['relative_l2']) < 1e-12
        if item['split'] == 'pilot_holdout' and (item['variant'],item['mode']) in [('scalar6_token','V_only'),('u4_token','V_only'),('u4_asymmetric','KV')]:
            selected.append({'variant':item['variant'],'mode':item['mode'],'relative_l2':rel,'bits_per_kv':item['effective_bits_per_kv_scalar']})
    losses = defaultdict(list)
    primary_rows = [json.loads(line) for line in (root/'results/model_quality.jsonl').read_text().splitlines()]
    assert len(primary_rows) == 2664
    primary = json.loads((root/'results/model_quality.json').read_text())
    assert sha((root/'results/model_quality.jsonl').read_bytes()) == primary['raw_results_sha256']
    for r in primary_rows:
        if r['kind'] == 'teacher_forcing':
            losses[r['method'],r['fixture']].append(r)
    secondary = json.loads((root/'results/model_quality_secondary.json').read_text())
    secondary_raw = (root/'results/model_quality_secondary.jsonl').read_bytes()
    assert sha(secondary_raw) == secondary['raw_results_sha256']
    for line in secondary_raw.splitlines():
        r = json.loads(line)
        losses[r['method'],r['fixture']].append(r)
    methods = sorted({key[0] for key in losses})
    recomputed = {}
    for (method,fixture), values in losses.items():
        assert sorted(r['target_position'] for r in values) == list(range(289,384)), (method, fixture)
        assert all(math.isfinite(r['nll']) for r in values)
    base = math.fsum(r['nll'] for (method,_), values in losses.items() if method == 'fp16' for r in values)/380
    max_nll_difference = 0.0
    for method in methods:
        values = [r['nll'] for (m,_), rs in losses.items() if m == method for r in rs]
        assert len(values) == 380
        mean = math.fsum(values)/len(values)
        expected = secondary['aggregate']['mean_nll'] if method == 'u4_asymmetric_KV' else primary['aggregate'][method]['mean_nll']
        max_nll_difference = max(max_nll_difference,abs(mean-expected))
        assert abs(mean-expected) < 1e-12
        recomputed[method] = {'targets':len(values),'mean_nll':mean,'ppl_ratio':math.exp(mean-base)}
    # Derive sizes from shapes and format, independently of the stored totals.
    prefix_scalars = 30*3*288*64
    tail = 30*2*3*95*64*2
    sizes = {'fp16':30*2*3*383*64*2,
             'u4_V':prefix_scalars*(16+4.5)/8+tail,
             'astc_token6_V':prefix_scalars*(16+25/6)/8+tail,
             'u4_asymmetric_KV':prefix_scalars*(4+32/96+4.5)/8+tail}
    result = {'schema':'astckv.independent_audit.v1','scope':'Recalculation of archived results; no fresh model execution',
              'archive_sha256':sha(args.archive.read_bytes()),'archive_manifest_files_verified':len(manifest['files']),
              'local_result_files_verified_against_archive':local_verified,
              'snapshot_rows':len(rows),'snapshot_aggregates':len(summary),'max_snapshot_difference':max_snapshot_difference,
              'model_loss_rows':sum(map(len,losses.values())),'max_nll_difference':max_nll_difference,
              'selected_snapshot':selected,'model_recomputed':recomputed,'memory_bytes':sizes,
              'astc_V_saving_vs_u4_V_fraction':1-sizes['astc_token6_V']/sizes['u4_V'],
              'source_sha256':sha(Path(__file__).read_bytes()),'passed':True}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
