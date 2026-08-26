"""Unit tests for Core-7 FashionCLIP embedding validation."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.data.validate_core7_embeddings import (
    inspect_embedding_cache,
    repair_split,
    validate_core7_embedding_coverage,
    validate_split,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record))
            stream.write("\n")


class Core7EmbeddingValidationTests(unittest.TestCase):
    def setUp(self):
        self.positives = [
            {
                "sample_id": "kit_a_pos",
                "source_kit_id": "kit_a",
                "items": ["a", "b", "c"],
                "label": 1,
                "negative_metadata": None,
            }
        ]
        self.metadata = [
            {"item_id": item_id, "split": "train", "coarse_category": "TOP"}
            for item_id in ("a", "b", "c")
        ]

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_valid_cache_and_split_pass(self):
        cache = {
            "model_id": "patrickjohncyh/fashion-clip",
            "item_ids": ["a", "b", "c"],
            "embeddings": torch.eye(3, 512),
            "normalized": True,
        }
        cache_report, usable_ids = inspect_embedding_cache(cache)
        split_report = validate_split(
            self.positives,
            self.metadata,
            usable_ids,
            split="train",
        )

        self.assertTrue(cache_report["pass"])
        self.assertTrue(split_report["pass"])
        self.assertEqual(split_report["embedding_coverage"], 1.0)

    def test_missing_embedding_fails_split(self):
        report = validate_split(
            self.positives,
            self.metadata,
            {"a", "b"},
            split="train",
        )

        self.assertFalse(report["pass"])
        self.assertEqual(report["missing_or_invalid_embedding_count"], 1)
        self.assertEqual(report["missing_or_invalid_embedding_examples"], ["c"])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_nonfinite_and_bad_norm_rows_fail_cache(self):
        embeddings = torch.eye(3, 512)
        embeddings[0, 0] = float("nan")
        embeddings[1] *= 2.0
        cache = {
            "model_id": "patrickjohncyh/fashion-clip",
            "item_ids": ["a", "b", "c"],
            "embeddings": embeddings,
            "normalized": True,
        }

        report, usable_ids = inspect_embedding_cache(cache)

        self.assertFalse(report["pass"])
        self.assertEqual(report["nonfinite_row_count"], 1)
        self.assertEqual(report["bad_norm_row_count"], 1)
        self.assertEqual(usable_ids, {"c"})

    def test_repair_recounts_items_and_drops_short_outfit(self):
        repaired, metadata, report = repair_split(
            self.positives,
            self.metadata,
            {"a", "b"},
            min_items=3,
        )

        self.assertEqual(repaired, [])
        self.assertEqual(metadata, [])
        self.assertEqual(report["dropped_outfit_count"], 1)
        self.assertEqual(report["removed_item_reference_count"], 1)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_end_to_end_report_marks_existing_positives_as_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "cache.pt"
            positives_path = root / "positives.jsonl"
            metadata_path = root / "metadata.jsonl"
            report_path = root / "report.json"

            torch.save(
                {
                    "model_id": "patrickjohncyh/fashion-clip",
                    "item_ids": ["a", "b", "c"],
                    "embeddings": torch.eye(3, 512).half(),
                    "normalized": True,
                },
                cache_path,
            )
            write_jsonl(positives_path, self.positives)
            write_jsonl(metadata_path, self.metadata)

            report = validate_core7_embedding_coverage(
                cache_path=cache_path,
                positives_by_split={"train": positives_path},
                metadata_by_split={"train": metadata_path},
                report_path=report_path,
            )

            self.assertTrue(report["pass"])
            self.assertTrue(report["reuse_category_clean_as_final"])
            self.assertTrue(report["ready_for_negative_sampling"])
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
