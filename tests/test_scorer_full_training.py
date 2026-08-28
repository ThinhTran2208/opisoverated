"""Tests for S3 full-training orchestration and resume behavior."""

import json
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.scorer.checkpoint import canonical_provenance, load_checkpoint
from src.scorer.train import (
    _capture_rng_state,
    _restore_rng_state,
    run_full_training,
    torch,
)


@unittest.skipUnless(torch is not None, "PyTorch is not installed in portability CI")
class ScorerFullTrainingTests(unittest.TestCase):
    class ToyModel(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(
            self,
            item_embeddings,
            coarse_category_ids,
            item_mask,
            pair_mask=None,
        ):
            del coarse_category_ids, item_mask, pair_mask
            return {
                "compatibility_logit": self.weight
                + item_embeddings[:, 0, 0]
            }

    def setUp(self):
        self.config = {
            "model": {"name": "type_aware_pairwise_v1"},
            "data": {"min_items": 3, "max_items": 8},
            "training": {
                "optimizer": "adamw",
                "learning_rate": 0.0003,
                "weight_decay": 0.0001,
                "batch_size": 4,
                "max_epochs": 10,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.0,
                "lr_scheduler": "none",
                "gradient_clipping": "none",
                "mixed_precision": False,
                "seed": 42,
            },
            "selection": {
                "primary_metric": "roc_auc",
                "guardrail_metric": "fitb_2way",
            },
        }
        self.provenance = canonical_provenance(
            dataset_manifest_sha256="a" * 64,
            embedding_manifest_sha256="b" * 64,
        )
        self.git_state = {"git_commit": "c" * 40, "git_tree_clean": True}
        self.loader = [object()]

    @staticmethod
    def _validation(auc: float) -> dict[str, float | int]:
        return {
            "loss": 1.0 - auc / 2.0,
            "roc_auc": auc,
            "fitb_2way": auc,
            "mean_logit_margin": auc - 0.5,
            "median_logit_margin": auc - 0.5,
            "sample_count": 4,
            "paired_family_count": 2,
        }

    @staticmethod
    def _train_side_effect():
        calls = {"count": 0}

        def fake_train(*args, global_step=0, **kwargs):
            del args, kwargs
            calls["count"] += 1
            return {
                "loss": 1.0 / calls["count"],
                "sample_count": 4,
                "batch_count": 1,
                "global_step": global_step + 1,
            }

        return fake_train

    def test_best_last_history_and_early_stopping(self):
        validation_sequence = [
            self._validation(0.60),
            self._validation(0.70),
            self._validation(0.69),
            self._validation(0.68),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.scorer.train.train_one_epoch",
            side_effect=self._train_side_effect(),
        ), patch(
            "src.scorer.train._evaluation_snapshot",
            side_effect=validation_sequence,
        ):
            summary = run_full_training(
                self.ToyModel(),
                self.loader,
                self.loader,
                self.config,
                output_dir=tmp,
                provenance=self.provenance,
                git_state=self.git_state,
                device="cpu",
            )

            self.assertEqual(summary["status"], "EARLY_STOPPED")
            self.assertEqual(summary["epochs_completed"], 4)
            self.assertEqual(summary["best_epoch"], 2)
            self.assertEqual(summary["best_valid_roc_auc"], 0.70)

            best = load_checkpoint(Path(tmp) / "best.pt")
            last = load_checkpoint(Path(tmp) / "last.pt")
            self.assertEqual(best["epoch"], 2)
            self.assertEqual(last["epoch"], 4)
            self.assertEqual(last["epochs_without_improvement"], 2)

            history = json.loads(
                (Path(tmp) / "training_history.json").read_text(
                    encoding="utf-8"
                )
            )["epochs"]
            self.assertEqual(len(history), 4)
            self.assertEqual([row["improved"] for row in history], [1, 1, 0, 0])
            self.assertTrue((Path(tmp) / "run_config.json").is_file())
            self.assertTrue((Path(tmp) / "validation_metrics.json").is_file())
            self.assertTrue((Path(tmp) / "run_summary.json").is_file())

    def test_paired_experiment_passes_locked_weight_and_records_components(self):
        config = json.loads(json.dumps(self.config))
        config["experiment"] = {
            "stage": "S3.1",
            "name": "s3_1_paired_ranking_v1",
        }
        config["training"].update(
            {
                "max_epochs": 1,
                "objective": "bce_plus_paired_logistic",
                "paired_batching": True,
                "paired_ranking_weight": 0.5,
            }
        )
        seen = {}

        def fake_train(*args, global_step=0, paired_ranking_weight=0.0, **kwargs):
            del args, kwargs
            seen["weight"] = paired_ranking_weight
            return {
                "loss": 0.9,
                "bce_loss": 0.6,
                "paired_ranking_loss": 0.6,
                "mean_paired_margin": 0.2,
                "paired_family_count": 2,
                "sample_count": 4,
                "batch_count": 1,
                "global_step": global_step + 1,
            }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.scorer.train.train_one_epoch",
            side_effect=fake_train,
        ), patch(
            "src.scorer.train._evaluation_snapshot",
            return_value=self._validation(0.65),
        ):
            summary = run_full_training(
                self.ToyModel(),
                self.loader,
                self.loader,
                config,
                output_dir=tmp,
                provenance=self.provenance,
                git_state=self.git_state,
                device="cpu",
            )

        self.assertEqual(seen["weight"], 0.5)
        self.assertEqual(summary["stage"], "S3.1")
        self.assertEqual(summary["experiment_name"], "s3_1_paired_ranking_v1")
        self.assertEqual(summary["objective"], "bce_plus_paired_logistic")
        self.assertEqual(summary["history"][0]["train_bce_loss"], 0.6)
        self.assertEqual(
            summary["history"][0]["train_paired_ranking_loss"], 0.6
        )

    def test_paired_objective_rejects_missing_pair_batching(self):
        config = json.loads(json.dumps(self.config))
        config["training"].update(
            {
                "objective": "bce_plus_paired_logistic",
                "paired_batching": False,
                "paired_ranking_weight": 0.5,
            }
        )
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "paired_batching=true"
        ):
            run_full_training(
                self.ToyModel(),
                self.loader,
                self.loader,
                config,
                output_dir=tmp,
                provenance=self.provenance,
                git_state=self.git_state,
                device="cpu",
            )

    def test_resume_continues_after_last_completed_epoch(self):
        config = json.loads(json.dumps(self.config))
        config["training"]["max_epochs"] = 4
        config["training"]["early_stopping_patience"] = 10

        first_train = self._train_side_effect()
        calls = {"count": 0}

        def interrupt_on_epoch_three(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 3:
                raise RuntimeError("simulated Colab disconnect")
            return first_train(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "src.scorer.train.train_one_epoch",
                side_effect=interrupt_on_epoch_three,
            ), patch(
                "src.scorer.train._evaluation_snapshot",
                side_effect=[self._validation(0.60), self._validation(0.65)],
            ):
                with self.assertRaisesRegex(RuntimeError, "disconnect"):
                    run_full_training(
                        self.ToyModel(),
                        self.loader,
                        self.loader,
                        config,
                        output_dir=tmp,
                        provenance=self.provenance,
                        git_state=self.git_state,
                        device="cpu",
                    )

            last_before_resume = load_checkpoint(Path(tmp) / "last.pt")
            self.assertEqual(last_before_resume["epoch"], 2)

            with patch(
                "src.scorer.train.train_one_epoch",
                side_effect=self._train_side_effect(),
            ), patch(
                "src.scorer.train._evaluation_snapshot",
                side_effect=[self._validation(0.66), self._validation(0.67)],
            ):
                summary = run_full_training(
                    self.ToyModel(),
                    self.loader,
                    self.loader,
                    config,
                    output_dir=tmp,
                    provenance=self.provenance,
                    git_state=self.git_state,
                    device="cpu",
                    resume=True,
                )

            self.assertEqual(summary["status"], "COMPLETED_MAX_EPOCHS")
            self.assertEqual(summary["epochs_completed"], 4)
            self.assertEqual(
                [row["epoch"] for row in summary["history"]], [1, 2, 3, 4]
            )
            self.assertEqual(load_checkpoint(Path(tmp) / "last.pt")["epoch"], 4)

    def test_rng_restore_uses_cpu_byte_tensors(self):
        class Loader:
            generator = torch.Generator().manual_seed(123)

        loader = Loader()
        saved = _capture_rng_state(loader)
        expected_torch_state = saved["torch_cpu_rng_state"].clone()
        expected_loader_state = saved["dataloader_generator_state"].clone()

        torch.manual_seed(999)
        loader.generator.manual_seed(999)
        _restore_rng_state(saved, loader)

        self.assertEqual(torch.get_rng_state().device.type, "cpu")
        self.assertEqual(torch.get_rng_state().dtype, torch.uint8)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch_state))
        self.assertTrue(
            torch.equal(loader.generator.get_state(), expected_loader_state)
        )

    def test_rng_restore_rejects_non_byte_cpu_state(self):
        class Loader:
            generator = torch.Generator().manual_seed(123)

        loader = Loader()
        saved = _capture_rng_state(loader)
        saved["torch_cpu_rng_state"] = saved["torch_cpu_rng_state"].to(
            torch.int64
        )

        with self.assertRaisesRegex(ValueError, "rank-1 torch.uint8"):
            _restore_rng_state(saved, loader)

    def test_dirty_git_tree_is_blocked_before_training(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            RuntimeError, "clean Git tree"
        ):
            run_full_training(
                self.ToyModel(),
                self.loader,
                self.loader,
                self.config,
                output_dir=tmp,
                provenance=self.provenance,
                git_state={"git_commit": "c" * 40, "git_tree_clean": False},
                device="cpu",
            )

    def test_existing_run_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "last.pt").write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                run_full_training(
                    self.ToyModel(),
                    self.loader,
                    self.loader,
                    self.config,
                    output_dir=tmp,
                    provenance=self.provenance,
                    git_state=self.git_state,
                    device="cpu",
                )

    def test_full_runner_has_no_test_loader_parameter(self):
        parameters = inspect.signature(run_full_training).parameters
        self.assertEqual(
            list(parameters)[:4],
            ["model", "train_dataloader", "valid_dataloader", "config"],
        )
        self.assertNotIn("test_dataloader", parameters)

    def test_resume_does_not_continue_an_already_early_stopped_run(self):
        config = json.loads(json.dumps(self.config))
        config["training"]["early_stopping_patience"] = 1

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "src.scorer.train.train_one_epoch",
                side_effect=self._train_side_effect(),
            ), patch(
                "src.scorer.train._evaluation_snapshot",
                side_effect=[self._validation(0.60), self._validation(0.59)],
            ):
                first = run_full_training(
                    self.ToyModel(),
                    self.loader,
                    self.loader,
                    config,
                    output_dir=tmp,
                    provenance=self.provenance,
                    git_state=self.git_state,
                    device="cpu",
                )
            self.assertEqual(first["status"], "EARLY_STOPPED")

            with patch("src.scorer.train.train_one_epoch") as train_mock:
                resumed = run_full_training(
                    self.ToyModel(),
                    self.loader,
                    self.loader,
                    config,
                    output_dir=tmp,
                    provenance=self.provenance,
                    git_state=self.git_state,
                    device="cpu",
                    resume=True,
                )
            train_mock.assert_not_called()
            self.assertEqual(resumed["status"], "EARLY_STOPPED")
            self.assertEqual(resumed["epochs_completed"], 2)

    def test_resume_repairs_best_checkpoint_from_complete_last_state(self):
        config = json.loads(json.dumps(self.config))
        config["training"]["max_epochs"] = 1

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "src.scorer.train.train_one_epoch",
                side_effect=self._train_side_effect(),
            ), patch(
                "src.scorer.train._evaluation_snapshot",
                return_value=self._validation(0.75),
            ):
                run_full_training(
                    self.ToyModel(),
                    self.loader,
                    self.loader,
                    config,
                    output_dir=tmp,
                    provenance=self.provenance,
                    git_state=self.git_state,
                    device="cpu",
                )

            best_path = Path(tmp) / "best.pt"
            best_path.unlink()
            resumed = run_full_training(
                self.ToyModel(),
                self.loader,
                self.loader,
                config,
                output_dir=tmp,
                provenance=self.provenance,
                git_state=self.git_state,
                device="cpu",
                resume=True,
            )
            self.assertTrue(best_path.is_file())
            self.assertEqual(load_checkpoint(best_path)["epoch"], 1)
            self.assertEqual(resumed["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
