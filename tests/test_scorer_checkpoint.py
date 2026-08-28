"""Tests for the locked Scorer V1 checkpoint schema and restoration."""

import tempfile
import unittest
from pathlib import Path

from src.scorer.checkpoint import (
    build_checkpoint_payload,
    canonical_provenance,
    restore_checkpoint,
    save_epoch_checkpoints,
    validate_checkpoint_payload,
)
from src.scorer.model import torch


@unittest.skipUnless(torch is not None, "PyTorch is not installed in portability CI")
class ScorerCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        self.provenance = canonical_provenance(
            dataset_manifest_sha256="a" * 64,
            embedding_manifest_sha256="b" * 64,
        )
        self.config = {
            "model": {"name": "type_aware_pairwise_v1"},
            "training": {"optimizer": "adamw"},
        }
        self.metrics = {
            "roc_auc": 0.75,
            "fitb_2way": 0.70,
            "mean_logit_margin": 0.2,
            "median_logit_margin": 0.1,
            "sample_count": 20,
            "paired_family_count": 10,
        }

    def _payload(self):
        return build_checkpoint_payload(
            model=self.model,
            optimizer=self.optimizer,
            epoch=3,
            global_step=12,
            config=self.config,
            provenance=self.provenance,
            git_commit="c" * 40,
            git_tree_clean=True,
            seed=42,
            best_valid_roc_auc=0.75,
            validation_metrics=self.metrics,
        )

    def test_payload_contains_locked_provenance(self):
        payload = self._payload()
        self.assertEqual(
            payload["dataset_version"], "polyvore1000-core7-compat-v2"
        )
        self.assertEqual(payload["category_mapping_version"], "core7-v2")
        self.assertEqual(payload["negative_protocol_version"], "negative-v1")
        self.assertEqual(payload["embedding_version"], "fashionclip-512-l2-v1")
        validate_checkpoint_payload(payload)

    def test_save_and_restore_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_weight = self.model.weight.detach().clone()
            paths = save_epoch_checkpoints(tmp, self._payload(), is_best=True)
            self.assertTrue(paths["last"].is_file())
            self.assertTrue(paths["best"].is_file())

            with torch.no_grad():
                self.model.weight.add_(10.0)
            restore_checkpoint(
                paths["last"],
                model=self.model,
                optimizer=self.optimizer,
                expected=self.provenance,
            )
            self.assertTrue(torch.equal(self.model.weight, original_weight))

    def test_provenance_mismatch_hard_fails_before_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "last.pt"
            save_epoch_checkpoints(tmp, self._payload(), is_best=False)
            before = self.model.weight.detach().clone()
            with self.assertRaises(ValueError):
                restore_checkpoint(
                    checkpoint_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    expected={"dataset_manifest_sha256": "d" * 64},
                )
            self.assertTrue(torch.equal(self.model.weight, before))

    def test_missing_schema_key_is_rejected(self):
        payload = self._payload()
        del payload["git_commit"]
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(payload)


if __name__ == "__main__":
    unittest.main()
