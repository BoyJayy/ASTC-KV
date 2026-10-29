"""Independent algebra checks for the per-token V affine transformation."""
import unittest
import numpy as np


class FactoredValueTests(unittest.TestCase):
    def check_case(self, scores, u, scale, offset):
        weights = np.exp(scores - np.max(scores))
        expected = weights @ (u * scale[:, None] + offset[:, None]) / weights.sum()
        # Independently compose split states in shifted exponent coordinates.
        states = []
        for start in range(0, len(scores), 7):
            end = min(start + 7, len(scores))
            local = scores[start:end]
            m = local.max()
            w = np.exp(local - m)
            numerator = (w * scale[start:end]) @ u[start:end] + w @ offset[start:end]
            states.append((m, w.sum(), numerator))
        maximum = max(x[0] for x in states)
        result = sum(np.exp(m-maximum)*v for m, z, v in states)
        result /= sum(np.exp(m-maximum)*z for m, z, v in states)
        np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)

    def test_constant_channels_and_zero_scale(self):
        scores = np.linspace(-12, 12, 23)
        self.check_case(scores, np.ones((23, 64)), np.zeros(23), np.linspace(-2, 3, 23))

    def test_extreme_scores_signed_offsets_and_partial_split(self):
        rng = np.random.default_rng(42)
        self.check_case(np.linspace(-1000, 1000, 31), rng.random((31, 64)),
                        rng.random(31)*10, rng.normal(size=31)*5)

    def test_nonuniform_scale_and_offsets(self):
        rng = np.random.default_rng(20261028)
        self.check_case(rng.normal(size=67)*3, rng.random((67, 64)),
                        np.exp(rng.normal(size=67)*2), rng.normal(size=67)*10)


if __name__ == '__main__':
    unittest.main()
