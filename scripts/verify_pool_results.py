#!/usr/bin/env python3
"""Independent post-run audit of runtime input hashes, pool cycles and metrics."""
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'results/kernel_v2/pool_v5'


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    source = ROOT/'results/kernel_v2/fixtures_page/t8192'
    manifest = json.loads((source/'manifest.json').read_text())
    expected = {'k_half_sha256':sha_bytes((source/manifest['k_file']).read_bytes()),
                'v_offset_sha256':sha_bytes((source/manifest['v_offset_file']).read_bytes()),
                'v_scale_sha256':sha_bytes((source/manifest['v_scale_file']).read_bytes()),
                'v_astc_payload_sha256':sha_bytes(b''.join((source/f).read_bytes()[16:] for f in manifest['v_files']))}
    # The independent hardware readback predates this pool experiment.
    readback = ROOT/'results/kernel_v2/llama_comparison_common_q/matched_inputs/t8192/v_hardware.f32'
    expected['v_half_sha256'] = sha_bytes(np.fromfile(readback,'<f4').astype('<f2').tobytes())
    qhash = sha_bytes((source/manifest['query_file']).read_bytes())
    registration = json.loads((FOLDER/'registration.json').read_text())
    for path, digest in registration['hashes_before'].items():
        assert sha_bytes((ROOT/path).read_bytes()) == digest, path
    summary = json.loads((FOLDER/'summary.json').read_text())
    max_error = 0.0
    slots_checked = 0
    for path in sorted(FOLDER.glob('s*_pool*.json')):
        data = json.loads(path.read_text())
        n = data['pool']['size']
        assert data['input_manifest'] == manifest
        assert data['pool']['query_sha256'] == qhash
        assert data['pool']['source_astc_payload_sha256'] == expected['v_astc_payload_sha256']
        assert len(data['pool_slot_hashes']) == n
        for i, slot in enumerate(data['pool_slot_hashes']):
            assert slot['slot']==i and slot['passed']
            assert all(slot[key]==value for key,value in expected.items())
            slots_checked += 1
        for case in data['pool_correctness']:
            assert case['passed_rtol_l2_1e_4'] and case['errors']['finite']
            max_error = max(max_error,case['errors']['relative_l2'])
        for variant in ['astc_factored_v','fp16_buffer']:
            cases = [c for c in data['pool_correctness'] if c['variant']==variant and c['phase']=='every_slot']
            assert [c['slot'] for c in cases] == list(range(n))
            samples = [s for s in data['raw_samples'] if s['variant']==variant]
            assert len(samples)==30
            for i,sample in enumerate(samples):
                assert sample['iterations']==n and sample['first_slot']==0 and sample['last_slot']==n-1
                assert sample['pool_cursor_before']==(i+1)*n
            median = statistics.median(s['gpu_batch_ms']/n*1000 for s in samples)
            row = next(r for r in summary['rows'] if r['raw']==path.name and r['variant']==variant)
            assert row['gpu_median_us']==median
        allocated = sum(s['total_allocated_bytes'] for s in data['pool']['slots'])
        assert allocated==data['memory']['pool_total_allocated_bytes']<=1<<30
    result = {'status':'PASS','processes':9,'slot_hash_sets_verified':slots_checked,
              'input_hashes_identical_across_all_sessions':True,'query_sha256':qhash,
              'kv_and_metadata_hashes':expected,'maximum_pool_reference_relative_l2':max_error,
              'outcome':summary['preregistered_outcome'],
              'scope':'Post-run independent audit of hashes actually recorded by GPU host, source fixtures, separate hardware readback, every-slot coverage, cycle schedule and raw medians. Does not establish cache miss rates.'}
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
