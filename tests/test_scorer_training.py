"""Tests for canonical Scorer V1 training and S2.5 helpers."""

import unittest

from src.scorer.dataset import build_pair_mask
from src.scorer.train import (
    build_full_training_loaders,
    build_optimizer,
    build_tiny_overfit_loader,
    run_tiny_overfit,
    set_reproducible_seed,
    torch,
    train_one_epoch,
)


@unittest.skipUnless(torch is not None, "PyTorch is not installed in portability CI")
class ScorerTrainingTests(unittest.TestCase):
    class ToyPairModel(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.0))
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(
            self,
            item_embeddings,
            coarse_category_ids,
            item_mask,
            pair_mask=None,
        ):
            del coarse_category_ids, item_mask, pair_mask
            feature = item_embeddings[:, 0, 0]
            return {"compatibility_logit": self.scale * feature + self.bias}

    class ToyFamilyDataset:
        def __init__(self, family_count=6):
            self.pair_families = [
                (2 * index, 2 * index + 1) for index in range(family_count)
            ]
            self.samples = []
            for index in range(family_count):
                positive_id = f"p{index}"
                self.samples.extend(
                    [
                        self._sample(positive_id, 1, None, 1.0),
                        self._sample(f"n{index}", 0, positive_id, -1.0),
                    ]
                )

        @staticmethod
        def _sample(sample_id, label, pair_id, feature):
            embeddings = torch.zeros(3, 512)
            embeddings[0, 0] = feature
            return {
                "sample_id": sample_id,
                "source_kit_id": sample_id,
                "paired_positive_sample_id": pair_id,
                "item_ids": [f"{sample_id}_{i}" for i in range(3)],
                "item_embeddings": embeddings,
                "coarse_category_ids": torch.tensor([1, 2, 5]),
                "label": float(label),
                "negative_metadata": None,
            }

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    def _batch(self, family_count=4):
        batch_size = family_count * 2
        embeddings = torch.zeros(batch_size, 8, 512)
        categories = torch.zeros(batch_size, 8, dtype=torch.long)
        categories[:, :3] = torch.tensor([1, 2, 5])
        item_mask = categories != 0
        labels = torch.tensor(
            [value for _ in range(family_count) for value in (1.0, 0.0)]
        )
        embeddings[:, 0, 0] = torch.where(labels == 1.0, 1.0, -1.0)

        sample_ids = []
        pair_ids = []
        for index in range(family_count):
            positive_id = f"p{index}"
            sample_ids.extend([positive_id, f"n{index}"])
            pair_ids.extend([None, positive_id])
        return {
            "item_embeddings": embeddings,
            "coarse_category_ids": categories,
            "item_mask": item_mask,
            "pair_mask": build_pair_mask(item_mask),
            "labels": labels,
            "sample_ids": sample_ids,
            "source_kit_ids": sample_ids,
            "paired_positive_sample_ids": pair_ids,
            "item_ids": [[f"i{i}"] * 3 for i in range(batch_size)],
            "negative_metadata": [None] * batch_size,
        }

    def test_seed_controls_torch_randomness(self):
        set_reproducible_seed(42)
        first = torch.randn(5)
        set_reproducible_seed(42)
        second = torch.randn(5)
        self.assertTrue(torch.equal(first, second))

    def test_tiny_loader_keeps_complete_families(self):
        dataset = self.ToyFamilyDataset(family_count=6)
        loader, selection = build_tiny_overfit_loader(
            dataset,
            family_count=3,
            batch_size=6,
            seed=42,
        )
        batch = next(iter(loader))
        self.assertEqual(selection["family_count"], 3)
        self.assertEqual(selection["sample_count"], 6)
        self.assertEqual(len(set(selection["sample_indices"])), 6)
        positive_ids = {
            sample_id
            for sample_id, label in zip(batch["sample_ids"], batch["labels"])
            if float(label) == 1.0
        }
        negative_pair_ids = {
            pair_id
            for pair_id, label in zip(
                batch["paired_positive_sample_ids"], batch["labels"]
            )
            if float(label) == 0.0
        }
        self.assertEqual(positive_ids, negative_pair_ids)

    def test_full_loaders_shuffle_train_only(self):
        train_dataset = self.ToyFamilyDataset(family_count=4)
        valid_dataset = self.ToyFamilyDataset(family_count=2)
        train_loader, valid_loader = build_full_training_loaders(
            train_dataset,
            valid_dataset,
            {
                "data": {"max_items": 8},
                "training": {"batch_size": 4, "seed": 42},
            },
        )
        self.assertEqual(type(train_loader.sampler).__name__, "RandomSampler")
        self.assertEqual(type(valid_loader.sampler).__name__, "SequentialSampler")
        self.assertEqual(len(train_loader.dataset), 8)
        self.assertEqual(len(valid_loader.dataset), 4)
        self.assertEqual(len(next(iter(train_loader))["sample_ids"]), 4)

    def test_full_loaders_reject_non_integer_seed(self):
        dataset = self.ToyFamilyDataset(family_count=2)
        for invalid_seed in (True, 42.5):
            with self.subTest(seed=invalid_seed), self.assertRaises(ValueError):
                build_full_training_loaders(
                    dataset,
                    dataset,
                    {
                        "data": {"max_items": 8},
                        "training": {"batch_size": 4, "seed": invalid_seed},
                    },
                )

    def test_train_one_epoch_updates_parameters(self):
        model = self.ToyPairModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        before = float(model.scale.detach())
        result = train_one_epoch(model, [self._batch()], optimizer, device="cpu")
        after = float(model.scale.detach())
        self.assertNotEqual(before, after)
        self.assertEqual(result["sample_count"], 8)
        self.assertEqual(result["global_step"], 1)
        self.assertGreater(result["loss"], 0.0)

    def test_tiny_overfit_reaches_perfect_pair_metrics(self):
        model = self.ToyPairModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
        report = run_tiny_overfit(
            model,
            [self._batch()],
            {"training": {"optimizer": "adamw"}},
            optimizer=optimizer,
            device="cpu",
            max_epochs=50,
            expected_family_count=4,
            target_roc_auc=0.99,
            target_fitb=0.99,
            max_loss_ratio=0.5,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["final"]["roc_auc"], 1.0)
        self.assertEqual(report["final"]["fitb_2way"], 1.0)
        self.assertLessEqual(report["final"]["loss_ratio"], 0.5)

    def test_optimizer_contract_rejects_non_adamw(self):
        model = self.ToyPairModel()
        with self.assertRaises(ValueError):
            build_optimizer(model, {"training": {"optimizer": "sgd"}})


if __name__ == "__main__":
    unittest.main()
