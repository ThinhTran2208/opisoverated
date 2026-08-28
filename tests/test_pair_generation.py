"""Tests for valid unordered pair generation and padding exclusion."""

import unittest

from src.scorer.dataset import build_pair_index_tuples, build_pair_mask, torch


class PairIndexTests(unittest.TestCase):
    def test_pair_counts(self):
        self.assertEqual(build_pair_index_tuples(3), [(0, 1), (0, 2), (1, 2)])
        self.assertEqual(len(build_pair_index_tuples(8)), 28)

    def test_pairs_are_upper_triangular(self):
        for i, j in build_pair_index_tuples(8):
            self.assertLess(i, j)


@unittest.skipUnless(torch is not None, "PyTorch is not installed in portability CI")
class PairMaskTorchTests(unittest.TestCase):
    def test_padding_does_not_create_pairs(self):
        item_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, False],
            ]
        )
        pair_mask = build_pair_mask(item_mask)
        self.assertEqual(pair_mask.sum(dim=(1, 2)).tolist(), [3, 6])
        self.assertFalse(pair_mask[:, 3:, 4:].any())

    def test_pair_mask_has_no_diagonal_or_reverse_pairs(self):
        item_mask = torch.ones((1, 4), dtype=torch.bool)
        pair_mask = build_pair_mask(item_mask)[0]
        self.assertFalse(torch.diagonal(pair_mask).any())
        self.assertFalse(torch.tril(pair_mask, diagonal=0).any())


if __name__ == "__main__":
    unittest.main()
