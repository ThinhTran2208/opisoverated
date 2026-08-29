"""Prevent Colab-only paths from returning to the data notebooks."""

import json
import unittest
from pathlib import Path


class NotebookPortabilityTests(unittest.TestCase):
    def test_nb2_to_nb4_have_no_required_colab_or_my_drive_code(self):
        repo_root = Path(__file__).resolve().parents[1]
        notebook_dir = repo_root / "notebooks" / "experiments"
        names = (
            "NB2_core7_drop_and_clean_positives.ipynb",
            "NB3_core7_embedding_validation.ipynb",
            "NB4_build_core7_scorer_dataset_v1.ipynb",
        )
        forbidden = (
            "from google.colab import drive",
            "drive.mount(",
            "/content/drive/MyDrive",
            "Path('/content/opisoverated')",
            'Path("/content/opisoverated")',
        )

        for name in names:
            notebook = json.loads((notebook_dir / name).read_text(encoding="utf-8"))
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell.get("cell_type") == "code"
            )
            for fragment in forbidden:
                self.assertNotIn(fragment, code, msg=f"{name}: {fragment}")
            self.assertIn("load_runtime_paths", code, msg=name)

    def test_gitignore_only_ignores_root_data_directory(self):
        repo_root = Path(__file__).resolve().parents[1]
        lines = {
            line.strip()
            for line in (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
        }
        self.assertIn("/data/", lines)
        self.assertNotIn("data/", lines)

    def test_nb4_requires_live_clean_git_provenance_for_freeze(self):
        repo_root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (
                repo_root
                / "notebooks/experiments/NB4_build_core7_scorer_dataset_v1.ipynb"
            ).read_text(encoding="utf-8")
        )
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn("inspect_git_provenance(REPO_ROOT)", code)
        self.assertIn("GIT_TREE_CLEAN", code)
        self.assertIn("repo_root=REPO_ROOT", code)
        self.assertIn("BLOCKED", code)

    def test_nb5_exposes_separate_amp_control_and_paired_experiment(self):
        repo_root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (
                repo_root
                / "notebooks/experiments/NB5_type_aware_pairwise_v1.ipynb"
            ).read_text(encoding="utf-8")
        )
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn(
            "scorer_type_aware_pairwise_v1_amp_off_control.yaml", code
        )
        self.assertIn(
            "scorer_type_aware_pairwise_v1_paired_ranking_v1.yaml", code
        )
        self.assertIn("build_paired_training_loaders", code)
        self.assertIn("RUN_S3_AMP_OFF_CONTROL = False", code)
        self.assertIn("RUN_S3_1_PAIRED = False", code)
        self.assertIn('metrics["mean_logit_margin"]', code)
        self.assertIn('metrics["median_logit_margin"]', code)

    def test_nb5_exposes_balanced_60_epoch_paired_control(self):
        repo_root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (
                repo_root
                / "notebooks/experiments/NB5_type_aware_pairwise_v1.ipynb"
            ).read_text(encoding="utf-8")
        )
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn(
            "scorer_type_aware_pairwise_v1_paired_ranking_balanced_60ep.yaml",
            code,
        )
        self.assertIn("RUN_S3_1_PAIRED_BALANCED_60 = False", code)
        self.assertIn("RESUME_S3_1_PAIRED_BALANCED_60 = False", code)
        self.assertIn("1.0 / math.sqrt(32.0)", code)
        self.assertIn('"s3_1_paired_ranking_balanced_init_60ep"', code)


if __name__ == "__main__":
    unittest.main()
