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
    fingerprint_category_mapping,
    inspect_embedding_cache,
    inspect_embedding_manifest,
    repair_split,
    sha256_file,
    validate_core7_embedding_coverage,
    validate_split,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record))
            stream.write("\n")


def write_mapping_files(root: Path) -> Path:
    v1_path = root / "category_mapping_core7_v1.json"
    v2_path = root / "category_mapping_core7_v2.json"
    v1_path.write_text(
        json.dumps(
            {
                "mapping_version": "core7-v1",
                "status": "frozen",
                "mapping": {"T-Shirts": "TOP", "Jeans": "BOTTOM"},
            }
        ),
        encoding="utf-8",
    )
    v2_path.write_text(
        json.dumps(
            {
                "mapping_version": "core7-v2",
                "status": "frozen",
                "base_mapping": v1_path.name,
                "overrides": {"T-Shirts": "DROP"},
            }
        ),
        encoding="utf-8",
    )
    return v2_path


class Core7EmbeddingValidationTests(unittest.TestCase):
    def setUp(self):
        self.positives = [
            {
                "sample_id": "kit_a_pos",
                "source_kit_id": "kit_a",
                "paired_positive_sample_id": None,
                "items": ["a", "b", "c"],
                "label": 1,
                "negative_metadata": None,
            }
        ]
        self.metadata = [
            {
                "item_metadata_version": "core7-item-metadata-v1",
                "category_mapping_version": "core7-v2",
                "item_id": item_id,
                "source_kit_id": "kit_a",
                "split": "train",
                "master_category": "T-Shirts",
                "coarse_category": "TOP",
            }
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

    def test_wrong_mapping_version_fails_split(self):
        metadata = [dict(row) for row in self.metadata]
        metadata[0]["category_mapping_version"] = "core7-v1"
        report = validate_split(
            self.positives,
            metadata,
            {"a", "b", "c"},
            split="train",
        )
        self.assertFalse(report["pass"])
        self.assertEqual(report["wrong_mapping_version_count"], 1)

    def test_mapping_fingerprint_binds_base_override_and_resolved_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping_path = write_mapping_files(root)

            first = fingerprint_category_mapping(mapping_path)
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
            payload["overrides"]["Jeans"] = "DROP"
            mapping_path.write_text(json.dumps(payload), encoding="utf-8")
            second = fingerprint_category_mapping(mapping_path)

            self.assertNotEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["base_sha256"], second["base_sha256"])
            self.assertNotEqual(
                first["resolved_mapping_sha256"],
                second["resolved_mapping_sha256"],
            )

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

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_manifest_must_match_cache_identity(self):
        cache = {
            "model_id": "patrickjohncyh/fashion-clip",
            "item_ids": ["a", "b", "c"],
            "embeddings": torch.eye(3, 512).half(),
            "normalized": True,
        }
        cache_report, _ = inspect_embedding_cache(cache)
        manifest = {
            "embedding_version": "fashionclip-512-l2-v1",
            "model_name_or_version": "patrickjohncyh/fashion-clip",
            "preprocessing_version": "fashionclip-preprocess-v1",
            "embedding_dimension": 512,
            "normalization": "l2",
            "dtype": "float16",
            "item_count": 3,
            "cache_sha256": "wrong-hash",
        }
        report = inspect_embedding_manifest(
            manifest,
            cache_report=cache_report,
            cache_sha256="actual-hash",
        )
        self.assertFalse(report["pass"])
        self.assertFalse(report["cache_sha256_matches"])

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
    def test_end_to_end_report_contains_exact_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "cache.pt"
            manifest_path = root / "embedding_manifest_v1.json"
            positives_path = root / "positives.jsonl"
            metadata_path = root / "metadata.jsonl"
            report_path = root / "report.json"
            mapping_path = write_mapping_files(root)

            torch.save(
                {
                    "model_id": "patrickjohncyh/fashion-clip",
                    "item_ids": ["a", "b", "c"],
                    "embeddings": torch.eye(3, 512).half(),
                    "normalized": True,
                },
                cache_path,
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "embedding_version": "fashionclip-512-l2-v1",
                        "model_name_or_version": "patrickjohncyh/fashion-clip",
                        "preprocessing_version": "fashionclip-preprocess-v1",
                        "embedding_dimension": 512,
                        "normalization": "l2",
                        "dtype": "float16",
                        "item_count": 3,
                        "cache_sha256": sha256_file(cache_path),
                    }
                ),
                encoding="utf-8",
            )
            write_jsonl(positives_path, self.positives)
            write_jsonl(metadata_path, self.metadata)

            report = validate_core7_embedding_coverage(
                mapping_path=mapping_path,
                cache_path=cache_path,
                manifest_path=manifest_path,
                positives_by_split={"train": positives_path},
                metadata_by_split={"train": metadata_path},
                report_path=report_path,
            )

            self.assertTrue(report["pass"])
            self.assertTrue(report["manifest"]["pass"])
            self.assertTrue(report["reuse_category_clean_as_final"])
            self.assertTrue(report["ready_for_negative_sampling"])
            self.assertEqual(
                report["inputs"]["category_mapping"]["sha256"],
                sha256_file(mapping_path),
            )
            self.assertEqual(
                report["mapping_sha256"],
                sha256_file(mapping_path),
            )
            self.assertEqual(
                report["inputs"]["embedding_cache"]["sha256"],
                sha256_file(cache_path),
            )
            self.assertEqual(
                report["inputs"]["splits"]["train"]["positive_sha256"],
                sha256_file(positives_path),
            )
            self.assertEqual(
                report["inputs"]["splits"]["train"]["metadata_sha256"],
                sha256_file(metadata_path),
            )
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
