"""Regression tests for the min-items-2 scorer / min-items-3 LOO experiment."""

import unittest

from src.diagnosis.loo import (
    MIN_LOO_ORIGINAL_ITEMS,
    MIN_SCORER_ITEMS,
    build_loo_subsets,
    validate_loo_original_item_count,
)


class MinItems2LooBoundaryTests(unittest.TestCase):
    def test_contract_constants(self):
        self.assertEqual(MIN_SCORER_ITEMS, 2)
        self.assertEqual(MIN_LOO_ORIGINAL_ITEMS, 3)

    def test_three_item_outfit_produces_two_item_residuals(self):
        subsets = build_loo_subsets(["top", "bottom", "shoes"])

        self.assertEqual(len(subsets), 3)
        self.assertEqual([index for index, _ in subsets], [0, 1, 2])
        self.assertTrue(all(len(residual) == 2 for _, residual in subsets))
        self.assertEqual(subsets[0][1], ["bottom", "shoes"])
        self.assertEqual(subsets[2][1], ["top", "bottom"])

    def test_two_item_outfit_is_not_eligible_for_loo_localization(self):
        with self.assertRaisesRegex(ValueError, "at least 3 original items"):
            validate_loo_original_item_count(2)
        with self.assertRaises(ValueError):
            build_loo_subsets(["top", "shoes"])


if __name__ == "__main__":
    unittest.main()
