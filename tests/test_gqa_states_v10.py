"""Independent dense-oracle checks for three separate softmax states per KV head."""
import unittest
import numpy as np


class SharedGQAStates(unittest.TestCase):
    def test_separate_heads_partial_splits_and_extreme_scores(self):
        rng = np.random.default_rng(20261102)
        for tokens, tile in ((1, 1), (7, 1), (31, 7), (31, 13), (391, 13), (391, 256)):
            for magnitude in (1, 1000):
                q = rng.normal(size=(9, 64)) * magnitude
                k = rng.normal(size=(3, tokens, 64))
                u = rng.random((3, tokens, 64))
                scale = np.exp(rng.normal(size=(3, tokens)))
                offset = rng.normal(size=(3, tokens)) * 5
                v = scale[:, :, None] * u + offset[:, :, None]
                for restored_half in (False, True):
                    values = v.astype(np.float16).astype(np.float64) if restored_half else v
                    # Dense independent single-head reference.
                    expected = []
                    for h in range(9):
                        logits = (k[h // 3] @ q[h]) / 8
                        weights = np.exp(logits - logits.max())
                        expected.append(weights @ values[h // 3] / weights.sum())
                    # A shared K/V tile produces three independent softmax states.
                    actual = np.zeros((9, 64))
                    for kh in range(3):
                        qq = q[kh * 3:kh * 3 + 3]
                        states = []
                        for start in range(0, tokens, tile):
                            end = min(start + tile, tokens)
                            scores = qq @ k[kh, start:end].T / 8
                            maxima = scores.max(axis=1)
                            weights = np.exp(scores - maxima[:, None])
                            if restored_half:
                                numerator = weights @ values[kh, start:end]
                            else:
                                numerator = (weights * scale[kh, start:end]) @ u[kh, start:end]
                                numerator += (weights @ offset[kh, start:end])[:, None]
                            states.append((maxima, weights.sum(axis=1), numerator))
                        maxima = np.maximum.reduce([s[0] for s in states])
                        denominator = np.zeros(3)
                        numerator = np.zeros((3, 64))
                        for m, z, n in states:
                            weight = np.exp(m - maxima)
                            denominator += weight * z
                            numerator += weight[:, None] * n
                        actual[kh * 3:kh * 3 + 3] = numerator / denominator[:, None]
                    np.testing.assert_allclose(actual, np.array(expected), atol=2e-11, rtol=2e-11)


if __name__ == '__main__':
    unittest.main()
