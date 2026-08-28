"""Regression tests for Scorer V1 S3 training/evaluation precision behavior."""

import unittest
from unittest.mock import patch

from src.scorer.model import torch

if torch is not None:
    from src.scorer.train import evaluate_epoch
else:  # Keep lightweight portability CI free of NumPy/PyTorch training deps.
    evaluate_epoch = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed in portability CI")
class ScorerTrainingRegressionTests(unittest.TestCase):
    def test_validation_never_enters_amp_autocast(self):
        class DummyScorer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.tensor(0.0))

            def forward(
                self,
                *,
                item_embeddings,
                coarse_category_ids,
                item_mask,
                pair_mask,
            ):
                del coarse_category_ids, item_mask, pair_mask
                logits = item_embeddings[:, 0, 0].float() + self.anchor * 0.0
                return {"compatibility_logit": logits}

        batch = {
            "item_embeddings": torch.zeros(2, 3, 512, dtype=torch.float32),
            "coarse_category_ids": torch.ones(2, 3, dtype=torch.long),
            "item_mask": torch.ones(2, 3, dtype=torch.bool),
            "pair_mask": torch.zeros(2, 3, 3, dtype=torch.bool),
            "labels": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "sample_ids": ["p1", "n1"],
            "paired_positive_sample_ids": [None, "p1"],
        }
        batch["item_embeddings"][0, 0, 0] = 1.0
        batch["item_embeddings"][1, 0, 0] = -1.0

        model = DummyScorer()
        criterion = torch.nn.BCEWithLogitsLoss()

        with patch(
            "src.scorer.train._autocast_context",
            side_effect=AssertionError("validation must not use autocast"),
        ):
            metrics = evaluate_epoch(
                model,
                [batch],
                criterion=criterion,
                device=torch.device("cpu"),
                mixed_precision=True,
            )

        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["fitb_2way"], 1.0)
        self.assertGreater(metrics["mean_logit_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
