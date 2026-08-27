"""Keep dataset and component version namespaces explicit and consistent."""

import json
import unittest
from pathlib import Path

from src.data.build_core7_scorer_dataset import (
    CATEGORY_MAPPING_VERSION,
    DATASET_VERSION_V1,
    DATASET_VERSION_V2,
    NEGATIVE_VERSION,
)


class VersionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.v1 = json.loads(
            (repo_root / "artifacts/data_v1_reference.json").read_text(
                encoding="utf-8"
            )
        )
        cls.v2 = json.loads(
            (repo_root / "artifacts/data_v2_reference.json").read_text(
                encoding="utf-8"
            )
        )

    def test_dataset_versions_use_canonical_benchmark_namespace(self):
        self.assertEqual(self.v1["dataset_version"], DATASET_VERSION_V1)
        self.assertEqual(self.v2["dataset_version"], DATASET_VERSION_V2)
        self.assertNotEqual(
            self.v2["dataset_version"], self.v2["category_mapping_version"]
        )

    def test_v2_component_versions_are_separate_and_canonical(self):
        self.assertEqual(
            self.v2["category_mapping_version"], CATEGORY_MAPPING_VERSION
        )
        self.assertEqual(
            self.v2["negative_protocol_version"], NEGATIVE_VERSION
        )
        self.assertEqual(
            self.v2["embedding_version"], "fashionclip-512-l2-v1"
        )

    def test_v1_reference_names_its_components_separately(self):
        self.assertEqual(self.v1["category_mapping_version"], "core7-v1")
        self.assertEqual(self.v1["negative_protocol_version"], "negative-v1")
        self.assertEqual(
            self.v1["embedding_version"], "fashionclip-512-l2-v1"
        )


if __name__ == "__main__":
    unittest.main()
