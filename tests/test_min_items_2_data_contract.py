"""Pure-Python contract tests for min-items-2 data processing wrappers."""

import unittest
from pathlib import Path

from src.data.prepare_core7_dataset import DEFAULT_MIN_OUTFIT_ITEMS
from src.data.build_core7_scorer_dataset_min2 import (
    EXPERIMENT_ID,
    MIN_LOO_ORIGINAL_ITEMS,
    MIN_SCORER_ITEMS,
    _require_isolated_output_dir,
)
from src.data.validate_core7_embeddings_min2 import repair_split_min2


class MinItems2DataContractTests(unittest.TestCase):
    def test_core7_cleaning_default_is_two(self):
        self.assertEqual(DEFAULT_MIN_OUTFIT_ITEMS, 2)
        self.assertEqual(MIN_SCORER_ITEMS, 2)
        self.assertEqual(MIN_LOO_ORIGINAL_ITEMS, 3)
        self.assertEqual(EXPERIMENT_ID, "min-items-2-loo-min3-v1")

    def test_experiment_rejects_canonical_scorer_ready_directory(self):
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            _require_isolated_output_dir(Path("/tmp/scorer_ready_v2"))

    def test_experiment_accepts_isolated_output_directory(self):
        _require_isolated_output_dir(Path("/tmp/scorer_ready_v2_min2_exp"))

    def test_embedding_repair_keeps_two_valid_items(self):
        positives = [
            {
                "sample_id": "kit_a_pos",
                "source_kit_id": "kit_a",
                "paired_positive_sample_id": None,
                "items": ["a", "b", "c"],
                "label": 1,
                "negative_metadata": None,
            }
        ]
        metadata = [
            {
                "item_metadata_version": "core7-item-metadata-v1",
                "category_mapping_version": "core7-v2",
                "split": "train",
                "item_id": item_id,
                "source_kit_id": "kit_a",
                "master_category": "T-Shirts",
                "coarse_category": "TOP",
            }
            for item_id in ("a", "b", "c")
        ]

        repaired, repaired_metadata, report = repair_split_min2(
            positives,
            metadata,
            {"a", "b"},
        )

        self.assertEqual(repaired[0]["items"], ["a", "b"])
        self.assertEqual({row["item_id"] for row in repaired_metadata}, {"a", "b"})
        self.assertEqual(report["dropped_outfit_count"], 0)


if __name__ == "__main__":
    unittest.main()
