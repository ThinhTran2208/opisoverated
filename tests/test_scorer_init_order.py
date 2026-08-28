"""Regression test for category rescaling without downstream RNG drift."""

import unittest

from src.scorer.model import TypeAwarePairwiseScorer, torch


@unittest.skipUnless(torch is not None, "PyTorch is not installed in portability CI")
class ScorerInitializationOrderTests(unittest.TestCase):
    def test_category_rescale_preserves_legacy_mlp_initialization_order(self):
        seed = 1234

        # Reference the initialization order used by the strongest FP32
        # diagnostic: construct embedding + all MLPs first, then rescale only
        # the category table.
        torch.manual_seed(seed)
        ref_category = torch.nn.Embedding(8, 32, padding_idx=0)
        ref_item = torch.nn.Sequential(
            torch.nn.Linear(544, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
        )
        ref_pair = torch.nn.Sequential(
            torch.nn.Linear(576, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1),
        )
        ref_output = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
        torch.nn.init.normal_(ref_category.weight, mean=0.0, std=32 ** -0.5)
        with torch.no_grad():
            ref_category.weight[0].zero_()

        torch.manual_seed(seed)
        model = TypeAwarePairwiseScorer()

        self.assertEqual(
            model.category_embedding_init_policy,
            "post_mlp_scale_preserving",
        )
        self.assertTrue(torch.equal(model.category_embedding.weight, ref_category.weight))
        self.assertTrue(torch.equal(model.item_mlp[0].weight, ref_item[0].weight))
        self.assertTrue(torch.equal(model.item_mlp[3].weight, ref_item[3].weight))
        self.assertTrue(torch.equal(model.pair_mlp[0].weight, ref_pair[0].weight))
        self.assertTrue(torch.equal(model.pair_mlp[3].weight, ref_pair[3].weight))
        self.assertTrue(torch.equal(model.output_mlp[0].weight, ref_output[0].weight))
        self.assertTrue(torch.equal(model.output_mlp[2].weight, ref_output[2].weight))


if __name__ == "__main__":
    unittest.main()
