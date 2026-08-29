"""Static safety checks for the canonical VLM Colab notebook."""

import json
import unittest
from pathlib import Path


class VlmNotebookTests(unittest.TestCase):
    def test_nb8_locks_model_scope_and_validation_only_demo(self):
        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "notebooks/experiments/NB8_vlm_explanation_v1.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )

        self.assertIn('BRANCH = "feat/vlm-explanation-v1"', code)
        self.assertIn("Qwen/Qwen3-VL-4B-Instruct", code + markdown)
        self.assertIn("Qwen3VLBackend.from_config", code)
        self.assertIn("vlm-visual-analysis-v1", code)
        self.assertIn("real_qwen_smoke_report.json", code)
        self.assertIn("[MERGE GATE] REAL QWEN SMOKE: PASS", code)
        self.assertIn('"test_split_loaded": False', code)
        self.assertIn('build_dataset_from_runtime(paths, "valid"', code)
        self.assertNotIn('build_dataset_from_runtime(paths, "test"', code)
        self.assertNotIn('load_dataset("codewaly/polyvore1000", "items", split="test"', code)
        self.assertIn("recommendation is not implemented", markdown.lower())
        self.assertIn("không có field free-text", markdown.lower())
        self.assertNotIn("print(os.environ", code)
        self.assertNotIn("HF_TOKEN =", code)


if __name__ == "__main__":
    unittest.main()
