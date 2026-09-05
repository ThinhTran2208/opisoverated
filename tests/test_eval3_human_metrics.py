# -*- coding: utf-8 -*-
"""Unit tests for EVALUATION3 human-alignment metrics."""

import math
import unittest

from src.evaluation.eval3_human_metrics import (
    bootstrap_both,
    kendall_value,
    linear_quantile,
    ordering_probability,
    spearman_value,
)


class Eval3HumanMetricTests(unittest.TestCase):

    def test_perfect_positive_ordering(self):
        logits = [-2.0, -1.0, 0.5]
        quality = [1, 2, 3]
        self.assertAlmostEqual(kendall_value(logits, quality), 1.0, places=12)
        self.assertAlmostEqual(spearman_value(logits, quality), 1.0, places=12)

    def test_perfect_negative_ordering(self):
        logits = [2.0, 1.0, 0.0]
        quality = [1, 2, 3]
        self.assertAlmostEqual(kendall_value(logits, quality), -1.0, places=12)
        self.assertAlmostEqual(spearman_value(logits, quality), -1.0, places=12)

    def test_kendall_tau_b_with_human_ties(self):
        # Hand-check:
        # x strictly increases, y=[1,2,2,3,3]
        # 8 concordant, 0 discordant, 2 ties in y only.
        # tau_b = 8 / sqrt(10 * 8)
        logits = [-1.2, -0.3, 0.2, 1.4, 2.0]
        quality = [1, 2, 2, 3, 3]
        expected = 8.0 / math.sqrt(80.0)
        self.assertAlmostEqual(
            kendall_value(logits, quality),
            expected,
            places=12,
        )

    def test_spearman_average_rank_ties(self):
        # x ranks = [1,2,3,4,5]
        # y average ranks = [1,2.5,2.5,4.5,4.5]
        logits = [-0.8, 0.1, 0.4, 0.7, 1.2]
        quality = [1, 2, 2, 3, 3]
        expected = 9.0 / math.sqrt(10.0 * 9.0)
        self.assertAlmostEqual(
            spearman_value(logits, quality),
            expected,
            places=12,
        )

    def test_linear_quantile_matches_frozen_linear_definition(self):
        values = [0.0, 10.0, 20.0, 30.0]
        self.assertAlmostEqual(linear_quantile(values, 0.25), 7.5)
        self.assertAlmostEqual(linear_quantile(values, 0.50), 15.0)
        self.assertAlmostEqual(linear_quantile(values, 0.75), 22.5)

    def test_pairwise_ordering_all_wins(self):
        result = ordering_probability([0.8, 1.0, 1.2], [-1.0, -0.2])
        self.assertEqual(result["wins"], 6)
        self.assertEqual(result["ties"], 0)
        self.assertEqual(result["losses"], 0)
        self.assertEqual(result["cross_class_pairs"], 6)
        self.assertAlmostEqual(result["probability"], 1.0, places=12)

    def test_pairwise_tie_counts_half(self):
        result = ordering_probability([1.0, 2.0], [1.0, 0.0])
        self.assertEqual(result["wins"], 3)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["losses"], 0)
        self.assertEqual(result["cross_class_pairs"], 4)
        self.assertAlmostEqual(result["probability"], 0.875, places=12)

    def test_pairwise_can_be_below_chance(self):
        result = ordering_probability([-1.0, -0.5], [0.0, 0.5])
        self.assertEqual(result["wins"], 0)
        self.assertEqual(result["ties"], 0)
        self.assertEqual(result["losses"], 4)
        self.assertAlmostEqual(result["probability"], 0.0, places=12)

    def test_empty_pairwise_class_returns_nan(self):
        result = ordering_probability([], [0.1, 0.2])
        self.assertTrue(math.isnan(result["probability"]))
        self.assertEqual(result["cross_class_pairs"], 0)

    def test_bootstrap_seed_42_is_reproducible(self):
        logits = [-1.7, -1.0, -0.4, 0.0, 0.2, 0.8, 1.1, 1.8]
        quality = [1, 1, 2, 2, 2, 3, 3, 3]

        kwargs = {
            "seed": 42,
            "resamples": 200,
            "ci_level": 0.95,
        }

        k1, s1 = bootstrap_both(
            logits, quality, backend="reference", **kwargs
        )
        k2, s2 = bootstrap_both(
            logits, quality, backend="reference", **kwargs
        )

        self.assertEqual(k1, k2)
        self.assertEqual(s1, s2)
        self.assertGreaterEqual(k1.valid_resamples, 198)
        self.assertLessEqual(k1.ci_low, k1.estimate)
        self.assertLessEqual(k1.estimate, k1.ci_high)
        self.assertLessEqual(s1.ci_low, s1.estimate)
        self.assertLessEqual(s1.estimate, s1.ci_high)


if __name__ == "__main__":
    unittest.main()
