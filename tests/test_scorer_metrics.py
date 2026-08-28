"""Tests for canonical Scorer V1 ROC-AUC, FITB, and paired margins."""

import unittest

from src.scorer.metrics import compute_scorer_metrics, paired_logit_margins, roc_auc


class ScorerMetricTests(unittest.TestCase):
    def test_perfect_roc_auc(self):
        self.assertEqual(roc_auc([1, 0, 1, 0], [2.0, -1.0, 1.0, 0.0]), 1.0)

    def test_auc_handles_tied_scores(self):
        self.assertAlmostEqual(roc_auc([1, 0], [0.5, 0.5]), 0.5)

    def test_pair_metrics_use_ids_not_row_order(self):
        sample_ids = ["b_neg", "a_pos", "b_pos", "a_neg"]
        pair_ids = ["b_pos", None, None, "a_pos"]
        labels = [0, 1, 1, 0]
        logits = [0.2, 2.0, 0.1, 1.0]

        margins = paired_logit_margins(sample_ids, pair_ids, labels, logits)
        self.assertCountEqual(margins, [-0.1, 1.0])

        metrics = compute_scorer_metrics(sample_ids, pair_ids, labels, logits)
        self.assertEqual(metrics["paired_family_count"], 2)
        self.assertEqual(metrics["sample_count"], 4)
        self.assertEqual(metrics["fitb_2way"], 0.5)
        self.assertAlmostEqual(metrics["mean_logit_margin"], 0.45)
        self.assertAlmostEqual(metrics["median_logit_margin"], 0.45)

    def test_fitb_tie_is_incorrect(self):
        metrics = compute_scorer_metrics(
            ["p", "n"],
            [None, "p"],
            [1, 0],
            [0.0, 0.0],
        )
        self.assertEqual(metrics["fitb_2way"], 0.0)
        self.assertEqual(metrics["mean_logit_margin"], 0.0)

    def test_missing_pair_hard_fails(self):
        with self.assertRaises(ValueError):
            compute_scorer_metrics(
                ["p"],
                [None],
                [1],
                [1.0],
            )


if __name__ == "__main__":
    unittest.main()
