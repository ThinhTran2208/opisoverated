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


if __name__ == "__main__":
    unittest.main()
