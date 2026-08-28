"""Tests for canonical Scorer V1 training and S2.5 helpers."""

import unittest

from src.scorer.dataset import build_pair_mask
from src.scorer.train import (
    _capture_rng_state,
    _restore_rng_state,
    build_full_training_loaders,
    build_optimizer,
    build_paired_training_loaders,
    build_tiny_overfit_loader,
    paired_batch_logit_margins,
    paired_logistic_ranking_loss,
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

    def test_paired_loader_keeps_every_family_in_one_batch(self):
        config = {
            "data": {"max_items": 8},
            "training": {
                "batch_size": 4,
                "seed": 42,
                "objective": "bce_plus_paired_logistic",
                "paired_batching": True,
                "paired_ranking_weight": 0.5,
            },
        }
        train_dataset = self.ToyFamilyDataset(family_count=5)
        valid_dataset = self.ToyFamilyDataset(family_count=2)
        train_loader, valid_loader = build_paired_training_loaders(
            train_dataset,
            valid_dataset,
            config,
        )
        self.assertIs(train_loader.generator, train_loader.batch_sampler.generator)

        seen_ids = []
        for batch in train_loader:
            self.assertEqual(len(batch["sample_ids"]) % 2, 0)
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
            seen_ids.extend(batch["sample_ids"])

        self.assertEqual(len(seen_ids), len(train_dataset))
        self.assertEqual(len(seen_ids), len(set(seen_ids)))
        self.assertEqual(type(valid_loader.sampler).__name__, "SequentialSampler")

    def test_paired_loader_order_is_reproducible(self):
        config = {
            "data": {"max_items": 8},
            "training": {
                "batch_size": 4,
                "seed": 42,
                "objective": "bce_plus_paired_logistic",
                "paired_batching": True,
                "paired_ranking_weight": 0.5,
            },
        }
        dataset = self.ToyFamilyDataset(family_count=6)
        first, _ = build_paired_training_loaders(dataset, dataset, config)
        second, _ = build_paired_training_loaders(dataset, dataset, config)
        first_order = [value for batch in first for value in batch["sample_ids"]]
        second_order = [value for batch in second for value in batch["sample_ids"]]
        self.assertEqual(first_order, second_order)

    def test_paired_loader_order_resumes_at_next_epoch(self):
        config = {
            "data": {"max_items": 8},
            "training": {
                "batch_size": 4,
                "seed": 42,
                "objective": "bce_plus_paired_logistic",
                "paired_batching": True,
                "paired_ranking_weight": 0.5,
            },
        }
        dataset = self.ToyFamilyDataset(family_count=6)
        uninterrupted, _ = build_paired_training_loaders(dataset, dataset, config)
        list(uninterrupted)
        saved_state = _capture_rng_state(uninterrupted)
        expected_next_epoch = [
            value for batch in uninterrupted for value in batch["sample_ids"]
        ]

        resumed, _ = build_paired_training_loaders(dataset, dataset, config)
        _restore_rng_state(saved_state, resumed)
        actual_next_epoch = [
            value for batch in resumed for value in batch["sample_ids"]
        ]
        self.assertEqual(actual_next_epoch, expected_next_epoch)

    def test_paired_loader_rejects_odd_sample_batch_size(self):
        dataset = self.ToyFamilyDataset(family_count=2)
        with self.assertRaisesRegex(ValueError, "even"):
            build_paired_training_loaders(
                dataset,
                dataset,
                {
                    "data": {"max_items": 8},
                    "training": {
                        "batch_size": 3,
                        "seed": 42,
                        "objective": "bce_plus_paired_logistic",
                        "paired_batching": True,
                        "paired_ranking_weight": 0.5,
                    },
                },
            )

    def test_paired_loader_rejects_incomplete_index_coverage(self):
        dataset = self.ToyFamilyDataset(family_count=2)
        dataset.pair_families = [(0, 1), (2, 99)]
        with self.assertRaisesRegex(ValueError, "cover every train row"):
            build_paired_training_loaders(
                dataset,
                dataset,
                {
                    "data": {"max_items": 8},
                    "training": {
                        "batch_size": 4,
                        "seed": 42,
                        "objective": "bce_plus_paired_logistic",
                        "paired_batching": True,
                        "paired_ranking_weight": 0.5,
                    },
                },
            )

    def test_paired_margin_lookup_does_not_depend_on_row_order(self):
        logits = torch.tensor([-0.5, 1.5, 0.25, 0.5], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        margins = paired_batch_logit_margins(
            logits,
            labels,
            ["n1", "p2", "n2", "p1"],
            ["p1", None, "p2", None],
        )
        self.assertTrue(torch.equal(margins, torch.tensor([1.25, 1.0])))

        loss = paired_logistic_ranking_loss(
            logits,
            labels,
            ["n1", "p2", "n2", "p1"],
            ["p1", None, "p2", None],
        )
        expected = torch.nn.functional.softplus(-margins).mean()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_paired_margin_rejects_incomplete_batch(self):
        with self.assertRaisesRegex(ValueError, "incomplete families"):
            paired_batch_logit_margins(
                torch.tensor([0.5, -0.5]),
                torch.tensor([1.0, 0.0]),
                ["p1", "n2"],
                [None, "p2"],
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

    def test_train_one_epoch_reports_paired_objective_components(self):
        model = self.ToyPairModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        result = train_one_epoch(
            model,
            [self._batch()],
            optimizer,
            device="cpu",
            paired_ranking_weight=0.5,
        )
        self.assertEqual(result["paired_family_count"], 4)
        self.assertGreater(result["bce_loss"], 0.0)
        self.assertGreater(result["paired_ranking_loss"], 0.0)
        self.assertAlmostEqual(
            result["loss"],
            result["bce_loss"] + 0.5 * result["paired_ranking_loss"],
            places=6,
        )

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
