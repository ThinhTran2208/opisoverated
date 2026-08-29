"""Tests for the eval-only Leave-One-Out diagnosis path."""

import unittest

from src.diagnosis.loo import (
    build_loo_variant_batch,
    diagnose_outfit,
    evaluate_loo_localization,
)
from src.scorer.model import TypeAwarePairwiseScorer, torch


@unittest.skipUnless(torch is not None, "PyTorch is not installed in portability CI")
class LooDiagnosisTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = TypeAwarePairwiseScorer(
            embedding_dim=4,
            category_embedding_dim=2,
            item_projection_dim=5,
            item_hidden_dim=3,
            pair_hidden_dim=4,
            output_hidden_dim=2,
            dropout=0.0,
            min_items=3,
            max_items=8,
        )

    def _three_item_outfit(self):
        return (
            torch.randn(3, 4),
            torch.tensor([1, 2, 5], dtype=torch.long),
        )

    def test_variant_batch_contains_full_then_each_removal(self):
        embeddings, categories = self._three_item_outfit()
        batch = build_loo_variant_batch(embeddings, categories, max_items=8)

        self.assertEqual(tuple(batch["item_embeddings"].shape), (4, 8, 4))
        self.assertEqual(batch["item_mask"].sum(dim=1).tolist(), [3, 2, 2, 2])
        self.assertEqual(batch["pair_mask"].sum(dim=(1, 2)).tolist(), [3, 1, 1, 1])
        self.assertEqual(batch["removed_item_indices"], [None, 0, 1, 2])
        self.assertEqual(batch["coarse_category_ids"][1, :2].tolist(), [2, 5])
        self.assertTrue(batch["uses_two_item_subsets"])

    def test_canonical_forward_still_rejects_two_items(self):
        self.model.eval()
        embeddings = torch.zeros(1, 8, 4)
        categories = torch.zeros(1, 8, dtype=torch.long)
        item_mask = torch.zeros(1, 8, dtype=torch.bool)
        embeddings[0, :2] = torch.randn(2, 4)
        categories[0, :2] = torch.tensor([1, 5])
        item_mask[0, :2] = True

        with self.assertRaisesRegex(ValueError, r"\[3, 8\]"):
            self.model(embeddings, categories, item_mask)

        output = self.model(
            embeddings,
            categories,
            item_mask,
            diagnostic_min_items=2,
        )
        self.assertEqual(tuple(output["compatibility_logit"].shape), (1,))

    def test_two_item_escape_hatch_is_rejected_during_training(self):
        self.model.train()
        embeddings = torch.zeros(1, 8, 4)
        categories = torch.zeros(1, 8, dtype=torch.long)
        item_mask = torch.zeros(1, 8, dtype=torch.bool)
        embeddings[0, :2] = torch.randn(2, 4)
        categories[0, :2] = torch.tensor([1, 5])
        item_mask[0, :2] = True

        with self.assertRaisesRegex(ValueError, "inference-only"):
            self.model(
                embeddings,
                categories,
                item_mask,
                diagnostic_min_items=2,
            )

    def test_single_outfit_diagnosis_marks_extrapolation(self):
        self.model.eval()
        embeddings, categories = self._three_item_outfit()
        result = diagnose_outfit(
            self.model,
            embeddings,
            categories,
            item_ids=["top", "bottom", "shoe"],
        )

        self.assertEqual(result["original_item_count"], 3)
        self.assertEqual(len(result["without_item_logits"]), 3)
        self.assertEqual(len(result["ranked_item_indices"]), 3)
        self.assertTrue(result["uses_two_item_extrapolation"])
        self.assertIn(result["problematic_item_id"], {"top", "bottom", "shoe"})

    def test_localization_metrics_are_split_by_original_size(self):
        class SumScorer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.max_items = 8
                self.anchor = torch.nn.Parameter(
                    torch.tensor(0.0), requires_grad=False
                )

            def forward(
                self,
                item_embeddings,
                coarse_category_ids,
                item_mask,
                pair_mask=None,
                *,
                diagnostic_min_items=None,
            ):
                del coarse_category_ids, pair_mask
                minimum = 2 if diagnostic_min_items == 2 else 3
                if self.training:
                    raise ValueError("eval only")
                if torch.any(item_mask.sum(dim=1) < minimum):
                    raise ValueError("too few items")
                logits = (
                    item_embeddings[..., 0] * item_mask.to(item_embeddings.dtype)
                ).sum(dim=1)
                return {"compatibility_logit": logits + self.anchor * 0.0}

        def sample(sample_id, values, target_index, label=0.0):
            item_count = len(values)
            embeddings = torch.zeros(item_count, 4)
            embeddings[:, 0] = torch.tensor(values, dtype=torch.float32)
            return {
                "sample_id": sample_id,
                "source_kit_id": sample_id,
                "item_ids": [f"{sample_id}_{index}" for index in range(item_count)],
                "item_embeddings": embeddings,
                "coarse_category_ids": torch.arange(1, item_count + 1).long(),
                "label": label,
                "negative_metadata": (
                    {"swapped_item_index": target_index} if label == 0.0 else None
                ),
            }

        class ListDataset:
            def __init__(self, rows):
                self.rows = rows

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                return self.rows[index]

        dataset = ListDataset(
            [
                sample("three", [2.0, -5.0, 1.0], 1),
                sample("four", [3.0, 2.0, -4.0, 1.0], 2),
                sample("positive_is_skipped", [1.0, 1.0, 1.0], 0, label=1.0),
            ]
        )
        scorer = SumScorer()
        scorer.eval()

        report = evaluate_loo_localization(
            scorer,
            dataset,
            outfit_batch_size=2,
        )

        self.assertEqual(report["overall"]["sample_count"], 2)
        self.assertEqual(report["overall"]["top1_localization_accuracy"], 1.0)
        self.assertEqual(report["overall"]["hit_at_2"], 1.0)
        self.assertEqual(report["overall"]["two_item_extrapolation_count"], 1)
        self.assertEqual(set(report["by_original_item_count"]), {"3", "4"})
        self.assertEqual(
            report["records"][0]["predicted_problematic_item_index"], 1
        )
        self.assertEqual(
            report["records"][1]["predicted_problematic_item_index"], 2
        )


if __name__ == "__main__":
    unittest.main()
