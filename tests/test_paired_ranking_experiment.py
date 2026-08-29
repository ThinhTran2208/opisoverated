"""Tests for the controlled V5 paired-ranking recheck helpers."""

import unittest

from src.scorer.paired_ranking_experiment import (
    PairedFamilyBatchSampler,
    evaluate_pure_loo_4plus,
    paired_logit_margins,
    torch,
)


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class PairedRankingExperimentTests(unittest.TestCase):
    def test_paired_sampler_keeps_complete_families(self):
        generator = torch.Generator().manual_seed(42)
        families = [(0, 1), (2, 3), (4, 5), (6, 7)]
        sampler = PairedFamilyBatchSampler(
            families,
            sample_batch_size=4,
            generator=generator,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 2)
        known = {frozenset(family) for family in families}
        for batch in batches:
            self.assertEqual(len(batch), 4)
            batch_pairs = {
                frozenset(batch[offset : offset + 2])
                for offset in range(0, len(batch), 2)
            }
            self.assertTrue(batch_pairs.issubset(known))

    def test_paired_margins_use_ids_not_adjacency(self):
        logits = torch.tensor([-1.0, 3.0, 2.0, 0.5])
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        sample_ids = ["neg_b", "pos_a", "pos_b", "neg_a"]
        pair_ids = ["pos_b", None, None, "pos_a"]
        margins = paired_logit_margins(logits, labels, sample_ids, pair_ids)
        self.assertEqual(sorted(round(float(x), 4) for x in margins), [2.5, 3.0])

    def test_pure_loo_4plus_localizes_bad_item(self):
        class SumScorer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.max_items = 8
                self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

            def forward(
                self,
                item_embeddings,
                coarse_category_ids,
                item_mask,
                pair_mask=None,
            ):
                del coarse_category_ids, pair_mask
                if torch.any(item_mask.sum(dim=1) < 3):
                    raise ValueError("canonical min is 3")
                logits = (
                    item_embeddings[..., 0] * item_mask.to(item_embeddings.dtype)
                ).sum(dim=1)
                return {"compatibility_logit": logits + self.anchor * 0.0}

        embeddings = torch.zeros(4, 512)
        embeddings[:, 0] = torch.tensor([2.0, 1.0, -5.0, 3.0])
        sample = {
            "sample_id": "negative_four",
            "item_embeddings": embeddings,
            "coarse_category_ids": torch.tensor([1, 2, 5, 6], dtype=torch.long),
            "label": 0.0,
            "negative_metadata": {"swapped_item_index": 2},
        }

        class Dataset:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                self.assert_index(index)
                return sample

            @staticmethod
            def assert_index(index):
                if index != 0:
                    raise IndexError(index)

        model = SumScorer().eval()
        report = evaluate_pure_loo_4plus(model, Dataset(), device="cpu")
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["top1_localization_accuracy"], 1.0)
        self.assertEqual(report["hit_at_2"], 1.0)


if __name__ == "__main__":
    unittest.main()
