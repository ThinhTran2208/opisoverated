"""Unit tests for the Core-7 category-drop stage."""

import json
import tempfile
import unittest
from pathlib import Path

from src.data.prepare_core7_dataset import (
    build_clean_positive_samples,
    filter_items_by_core_category,
    load_category_mapping,
    validate_clean_positive_samples,
    validate_mapping_coverage,
)
from src.data.prepare_core7_dataset_v2 import load_category_mapping_v2


class Core7DropTests(unittest.TestCase):
    def setUp(self):
        self.raw_items = {
            "kit_a": [
                {"item_id": "kit_a_1", "kit_id": "kit_a", "category": "T-Shirts"},
                {"item_id": "kit_a_2", "kit_id": "kit_a", "category": "Jeans"},
                {"item_id": "kit_a_3", "kit_id": "kit_a", "category": "Sneakers"},
                {"item_id": "kit_a_4", "kit_id": "kit_a", "category": "Lipstick"},
            ],
            "kit_b": [
                {"item_id": "kit_b_1", "kit_id": "kit_b", "category": "T-Shirts"},
                {"item_id": "kit_b_2", "kit_id": "kit_b", "category": "Jeans"},
                {"item_id": "kit_b_3", "kit_id": "kit_b", "category": "Lipstick"},
            ],
        }
        self.mapping = {
            "T-Shirts": "TOP",
            "Jeans": "BOTTOM",
            "Sneakers": "SHOES",
            "Lipstick": "DROP",
        }

    def test_load_versioned_mapping(self):
        payload = {
            "mapping_version": "unit-test-v1",
            "status": "draft",
            "mapping": self.mapping,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            metadata, mapping = load_category_mapping(path)

        self.assertEqual(metadata["mapping_version"], "unit-test-v1")
        self.assertEqual(mapping, self.mapping)

    def test_declared_decision_counts_must_match_mapping(self):
        payload = {
            "mapping_version": "unit-test-v1",
            "status": "frozen",
            "decision_counts": {"TOP": 99},
            "mapping": self.mapping,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "decision_counts"):
                load_category_mapping(path)

    def test_frozen_v1_is_preserved_and_v2_contains_policy_changes(self):
        root = Path(__file__).resolve().parents[1]
        v1_metadata, v1 = load_category_mapping(
            root / "configs" / "category_mapping_core7_v1.json"
        )
        v2_metadata, v2 = load_category_mapping_v2(
            root / "configs" / "category_mapping_core7_v2.json"
        )

        self.assertEqual(v1_metadata["mapping_version"], "core7-v1")
        self.assertEqual(v1_metadata["status"], "frozen")
        self.assertEqual(v2_metadata["mapping_version"], "core7-v2")
        self.assertEqual(v2_metadata["status"], "frozen")
        self.assertEqual(v2_metadata["base_mapping_version"], "core7-v1")
        self.assertEqual(len(v1), len(v2))

        changed_to_drop = {
            "Activewear Jackets": "OUTERWEAR",
            "Activewear Pants": "BOTTOM",
            "Activewear Shorts": "BOTTOM",
            "Activewear Skirts": "BOTTOM",
            "Activewear Tank Tops": "TOP",
            "Activewear Tops": "TOP",
            "Bags & Cases": "BAG",
            "Camisoles": "TOP",
            "Men's Activewear Jackets": "OUTERWEAR",
            "Men's Activewear Pants": "BOTTOM",
            "Men's Activewear Shorts": "BOTTOM",
            "Men's Activewear Tops": "TOP",
            "Men's Bags & Wallets": "BAG",
        }
        for category, v1_decision in changed_to_drop.items():
            self.assertEqual(v1[category], v1_decision)
            self.assertEqual(v2[category], "DROP")

        self.assertEqual(v2["Athletic Shoes"], "SHOES")
        self.assertEqual(v2["Men's Athletic Shoes"], "SHOES")
        self.assertEqual(v2["Men's Bags"], "BAG")
        self.assertEqual(v2["Slippers"], "SHOES")
        self.assertEqual(v2["Men's Slippers"], "SHOES")

    def test_mapping_must_cover_every_observed_category(self):
        with self.assertRaisesRegex(ValueError, "Missing master categories"):
            validate_mapping_coverage(
                ["T-Shirts", "Jeans", "Unknown Category"],
                self.mapping,
            )

    def test_filter_drops_items_and_preserves_order(self):
        filtered, report = filter_items_by_core_category(self.raw_items, self.mapping)

        self.assertEqual(
            [item["item_id"] for item in filtered["kit_a"]],
            ["kit_a_1", "kit_a_2", "kit_a_3"],
        )
        self.assertEqual(
            [item["coarse_category"] for item in filtered["kit_a"]],
            ["TOP", "BOTTOM", "SHOES"],
        )
        self.assertEqual(report["raw_item_count"], 7)
        self.assertEqual(report["kept_item_count"], 5)
        self.assertEqual(report["dropped_item_count"], 2)

    def test_clean_positives_keep_only_outfits_with_three_items(self):
        filtered, _ = filter_items_by_core_category(self.raw_items, self.mapping)
        kits = [{"kit_id": "kit_a"}, {"kit_id": "kit_b"}]

        samples, report = build_clean_positive_samples(
            kits,
            self.raw_items,
            filtered,
            min_items=3,
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(
            samples[0],
            {
                "sample_id": "kit_a_pos",
                "source_kit_id": "kit_a",
                "paired_positive_sample_id": None,
                "items": ["kit_a_1", "kit_a_2", "kit_a_3"],
                "label": 1,
                "negative_metadata": None,
            },
        )
        self.assertEqual(report["outfits_kept"], 1)
        self.assertEqual(report["outfits_dropped_below_min_items"], 1)

        item_to_coarse = {
            item["item_id"]: item["coarse_category"]
            for items in filtered.values()
            for item in items
        }
        validation = validate_clean_positive_samples(
            samples,
            item_to_coarse,
            min_items=3,
        )
        self.assertTrue(validation["pass"])

    def test_positive_pair_reference_must_be_null(self):
        invalid_sample = {
            "sample_id": "kit_a_pos",
            "source_kit_id": "kit_a",
            "paired_positive_sample_id": "kit_a_pos",
            "items": ["kit_a_1", "kit_a_2", "kit_a_3"],
            "label": 1,
            "negative_metadata": None,
        }
        item_to_coarse = {
            "kit_a_1": "TOP",
            "kit_a_2": "BOTTOM",
            "kit_a_3": "SHOES",
        }

        with self.assertRaisesRegex(ValueError, "validation failed"):
            validate_clean_positive_samples(
                [invalid_sample],
                item_to_coarse,
                min_items=3,
            )


if __name__ == "__main__":
    unittest.main()
