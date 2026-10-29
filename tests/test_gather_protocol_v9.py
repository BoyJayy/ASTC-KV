"""Reject malformed timing evidence instead of silently accepting its labels."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_gather_exchange_v9 import PAIRS, VARIANTS, extract_blocks, validate


def fixture():
    checks = {name: {'passed_rtol_l2_1e_4': True, 'vs_same_values_cpu_fp64_accumulation':
                     {'finite': True, 'relative_l2': 1e-6}} for name in VARIANTS}
    checks.update({k: {'passed_bit_exact_float32': True} for k in ('compact_decode_vs_original_normalized', 'gather_vs_direct')})
    samples = []
    for pair, (a, b) in PAIRS.items():
        for slot, name in enumerate((a, b, b, a)):
            samples.append({'pair_name': pair, 'round': 0, 'slot_in_block': slot,
                            'a_variant': a, 'b_variant': b, 'variant': name, 'block_order': 'ABBA',
                            'iterations': 20, 'gpu_batch_ms': 2 if name == a else 1})
    return {'status': 'executed_on_metal', 'correctness': checks, 'raw_samples': samples,
            'shape': {'tokens': 8192, 'tile_tokens': 256, 'head_dim': 64, 'kv_heads': 3, 'query_heads': 9}}


class ProtocolTests(unittest.TestCase):
    def parse(self, data):
        validate(data, 8192, 256)
        return extract_blocks(data, 0, 8192, 256, 20, 1, 'synthetic')

    def test_valid_ratio(self):
        self.assertEqual([r['gpu_ratio_b_over_a'] for r in self.parse(fixture())], [.5] * 4)

    def test_wrong_labels_and_missing_pair(self):
        for field in ('pair_name', 'a_variant', 'b_variant', 'variant', 'block_order', 'round', 'slot_in_block'):
            data = fixture()
            data['raw_samples'][0][field] = 'incorrect'
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.parse(data)

    def test_invalid_times_iterations_and_extra_samples(self):
        for value in (0, -1, float('nan'), float('inf')):
            data = fixture(); data['raw_samples'][0]['gpu_batch_ms'] = value
            with self.subTest(time=value), self.assertRaises(AssertionError):
                self.parse(data)
        data = fixture(); data['raw_samples'].append(copy.deepcopy(data['raw_samples'][0]))
        with self.assertRaises(AssertionError): self.parse(data)
        data = fixture(); data['raw_samples'][0]['iterations'] = 1
        with self.assertRaises(AssertionError): self.parse(data)

    def test_false_gate_renamed_variant_and_large_error(self):
        data = fixture(); data['correctness']['gather_vs_direct']['passed_bit_exact_float32'] = False
        with self.assertRaises(AssertionError): self.parse(data)
        data = fixture(); data['correctness']['wrong'] = data['correctness'].pop('compact_paired_gather_v')
        with self.assertRaises(AssertionError): self.parse(data)
        for error in (1.01e-4, float('nan'), float('inf'), -1):
            data = fixture()
            data['correctness']['compact_paired_gather_v']['vs_same_values_cpu_fp64_accumulation']['relative_l2'] = error
            with self.subTest(error=error), self.assertRaises(AssertionError): self.parse(data)
        data = fixture(); data['shape']['tokens'] = 384
        with self.assertRaises(AssertionError): self.parse(data)


if __name__ == '__main__':
    unittest.main()
