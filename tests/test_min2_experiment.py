import tempfile
import unittest
from pathlib import Path

from src.data.min2_experiment import (
    EXPERIMENT_DATASET_VERSION,
    LOO_MIN_ORIGINAL_ITEMS,
    MIN_SCORER_ITEMS,
    scorer_ready_path,
)
from src.diagnosis.loo import LOOInputError, build_leave_one_out_outfits

try:
    import torch
except ModuleNotFoundError:
    torch = None


class Min2ContractTests(unittest.TestCase):
    def test_min2_contract_constants_are_split_between_scorer_and_loo(self):
        self.assertEqual(MIN_SCORER_ITEMS, 2)
        self.assertEqual(LOO_MIN_ORIGINAL_ITEMS, 3)
        self.assertEqual(
            EXPERIMENT_DATASET_VERSION,
            "polyvore1000-core7-compat-min2-exp-v1",
        )

    def test_min2_scorer_artifact_name_is_isolated_from_frozen_v2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = scorer_ready_path(Path(temp_dir), "train")
            self.assertEqual(path.name, "scorer_ready_min2_exp_v1_train.jsonl")
            self.assertNotEqual(path.name, "scorer_ready_v2_train.jsonl")

    def test_loo_three_items_produces_two_item_residuals(self):
        residuals = build_leave_one_out_outfits(["top", "bottom", "shoes"])
        self.assertEqual(len(residuals), 3)
        self.assertTrue(all(len(residual) == 2 for residual in residuals))

    def test_loo_two_items_hard_fails_before_scorer(self):
        with self.assertRaises(LOOInputError):
            build_leave_one_out_outfits(["top", "shoes"])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class Min2ScorerTests(unittest.TestCase):
    def test_min2_collator_and_model_accept_two_real_items(self):
        from src.scorer.min2_experiment import collate_min2_scorer_batch
        from src.scorer.model import TypeAwarePairwiseScorer

        sample = {
            "sample_id": "kit_pos",
            "source_kit_id": "kit",
            "paired_positive_sample_id": None,
            "item_ids": ["kit_1", "kit_2"],
            "item_embeddings": torch.randn(2, 512),
            "coarse_category_ids": torch.tensor([1, 5], dtype=torch.long),
            "label": 1.0,
            "negative_metadata": None,
        }
        batch = collate_min2_scorer_batch([sample])
        self.assertEqual(int(batch["item_mask"].sum().item()), 2)
        self.assertEqual(int(batch["pair_mask"].sum().item()), 1)

        model = TypeAwarePairwiseScorer(min_items=2, max_items=8, dropout=0.0)
        model.eval()
        output = model(
            item_embeddings=batch["item_embeddings"],
            coarse_category_ids=batch["coarse_category_ids"],
            item_mask=batch["item_mask"],
            pair_mask=batch["pair_mask"],
        )
        logits = output["compatibility_logit"]
        self.assertEqual(tuple(logits.shape), (1,))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_min2_model_rejects_one_real_item(self):
        from src.scorer.model import TypeAwarePairwiseScorer

        model = TypeAwarePairwiseScorer(min_items=2, max_items=8, dropout=0.0)
        embeddings = torch.zeros((1, 8, 512), dtype=torch.float32)
        categories = torch.zeros((1, 8), dtype=torch.long)
        mask = torch.zeros((1, 8), dtype=torch.bool)
        embeddings[0, 0, 0] = 1.0
        categories[0, 0] = 1
        mask[0, 0] = True

        with self.assertRaisesRegex(ValueError, "Every outfit must contain"):
            model(
                item_embeddings=embeddings,
                coarse_category_ids=categories,
                item_mask=mask,
            )


if __name__ == "__main__":
    unittest.main()
