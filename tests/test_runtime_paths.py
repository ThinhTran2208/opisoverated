"""Tests for platform-independent artifact path resolution."""

import json
import tempfile
import unittest
from pathlib import Path

from src.data.runtime_paths import load_runtime_paths


class RuntimePathTests(unittest.TestCase):
    def _create_repo(self, root: Path) -> None:
        (root / "configs").mkdir(parents=True)
        (root / "src" / "data").mkdir(parents=True)
        (root / "configs" / "category_mapping_core7_v1.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (root / "src" / "data" / "prepare_core7_dataset.py").write_text(
            "# marker\n", encoding="utf-8"
        )
        (root / "configs" / "data_paths.example.json").write_text(
            json.dumps(
                {
                    "artifact_root": "./data",
                    "embedding_cache": "cache/embeddings.pt",
                    "embedding_manifest": "cache/embedding_manifest_v1.json",
                    "core7_dir": "core7",
                    "scorer_ready_dir": "scorer",
                }
            ),
            encoding="utf-8",
        )

    def test_example_config_defaults_to_repo_local_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_repo(root)

            paths = load_runtime_paths(repo_root=root, env={})

            self.assertEqual(paths.artifact_root, (root / "data").resolve())
            self.assertEqual(paths.embedding_cache, (root / "data/cache/embeddings.pt").resolve())
            self.assertEqual(
                paths.embedding_manifest,
                (root / "data/cache/embedding_manifest_v1.json").resolve(),
            )
            self.assertEqual(paths.core7_dir, (root / "data/core7").resolve())
            self.assertEqual(paths.scorer_ready_dir, (root / "data/scorer").resolve())

    def test_environment_overrides_work_on_any_platform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            external = Path(directory) / "external"
            self._create_repo(root)

            paths = load_runtime_paths(
                repo_root=root,
                env={
                    "FASHION_ARTIFACT_ROOT": str(external),
                    "FASHION_EMBEDDING_CACHE": "shared/fashionclip.pt",
                    "FASHION_EMBEDDING_MANIFEST": "shared/manifest.json",
                },
            )

            self.assertEqual(paths.artifact_root, external.resolve())
            self.assertEqual(
                paths.embedding_cache,
                (external / "shared/fashionclip.pt").resolve(),
            )
            self.assertEqual(
                paths.embedding_manifest,
                (external / "shared/manifest.json").resolve(),
            )

    def test_local_config_takes_precedence_over_example(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_repo(root)
            (root / "configs" / "data_paths.local.json").write_text(
                json.dumps({"artifact_root": "./private_artifacts"}),
                encoding="utf-8",
            )

            paths = load_runtime_paths(repo_root=root, env={})

            self.assertEqual(
                paths.artifact_root,
                (root / "private_artifacts").resolve(),
            )
            self.assertEqual(paths.config_path.name, "data_paths.local.json")


if __name__ == "__main__":
    unittest.main()
