"""Independent address/seam and weighted-pair checks for the gather experiment."""
import unittest
import numpy as np


def xy(head, token, channel, height):
    block = (head * (height // 6) + token // 6) * 11 + channel // 6
    return (block % 32) * 6 + channel % 6, (block // 32) * 6 + token % 6


class GatherPairTests(unittest.TestCase):
    def test_even_pairs_never_cross_astc_or_atlas_seams(self):
        for tokens in (1, 2, 5, 6, 7, 31, 96, 127, 384, 391, 2048, 8192):
            height = ((tokens + 95) // 96) * 96
            for head in range(3):
                for token in range(0, tokens - 1, 2):
                    for channel in range(0, 64, 2):
                        x, y = xy(head, token, channel, height)
                        self.assertEqual(xy(head, token, channel + 1, height), (x + 1, y))
                        self.assertEqual(xy(head, token + 1, channel, height), (x, y + 1))
                        self.assertEqual(xy(head, token + 1, channel + 1, height), (x + 1, y + 1))

    def test_odd_block_boundary_requires_fallback(self):
        # An odd starting token can cross an ASTC row/atlas seam.
        self.assertNotEqual(xy(0, 6, 12, 96), tuple(np.array(xy(0, 5, 12, 96)) + (0, 1)))

    def test_split_pair_coverage_and_mask(self):
        for tokens in (1, 2, 5, 6, 7, 31, 384, 391, 2048, 8192):
            for tile in (1, 2, 7, 8, 13, 32, 64, 255, 256):
                visited = []
                for start in range(0, tokens, tile):
                    end = min(start + tile, tokens)
                    for base in range(start, end, 8):
                        for lane in (0, 2, 4, 6):
                            t = base + lane
                            if t >= end:
                                continue
                            visited.append(t)
                            if t + 1 < end:
                                visited.append(t + 1)
                            gather = t % 2 == 0 and t + 1 < end
                            if gather:
                                x, y = xy(2, t, 62, ((tokens + 95) // 96) * 96)
                                self.assertEqual(xy(2, t + 1, 63, ((tokens + 95) // 96) * 96), (x + 1, y + 1))
                            else:
                                self.assertTrue(t % 2 == 1 or t + 1 >= end)
                self.assertEqual(visited, list(range(tokens)))

    def test_pair_algebra_with_softmax_extremes(self):
        rng = np.random.default_rng(20261031)
        for tokens in (1, 2, 7, 31, 391):
            for tile in (1, 7, 13, 64, 256):
                for score_range in (1, 20, 1000):
                    score = rng.normal(size=tokens) * score_range
                    u = rng.random((tokens, 64))
                    scale = np.exp(rng.normal(size=tokens) * 2)
                    offset = rng.normal(size=tokens) * 10
                    w = np.exp(score - score.max())
                    expected = w @ (u * scale[:, None] + offset[:, None]) / w.sum()
                    states = []
                    for start in range(0, tokens, tile):
                        end = min(start + tile, tokens)
                        m = score[start:end].max()
                        p = np.exp(score[start:end] - m)
                        numerator = np.zeros(64)
                        for t in range(start, end, 2):
                            i = t - start
                            numerator += p[i] * scale[t] * u[t] + p[i] * offset[t]
                            if t + 1 < end:
                                numerator += p[i + 1] * scale[t + 1] * u[t + 1] + p[i + 1] * offset[t + 1]
                        states.append((m, p.sum(), numerator))
                    maximum = max(s[0] for s in states)
                    actual = sum(np.exp(m - maximum) * v for m, z, v in states)
                    actual /= sum(np.exp(m - maximum) * z for m, z, v in states)
                    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)


if __name__ == '__main__':
    unittest.main()
