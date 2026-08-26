"""Tests for Core-7 negative sampling, merge, and final validation."""

import json
import tempfile
import unittest
from pathlib import Path

from src.data.build_core7_scorer_dataset import (
    NEGATIVE_TYPE,
    build_scorer_dataset_v1,
    generate_negative_records,
    merge_positive_negative_families,
    validate_all_splits,
    validate_scorer_split,
)


def positive(kit_id: str, items: list[str]) -> dict:
    return {
        "sample_id": f"{kit_id}_pos",
        "source_kit_id": kit_id,
        "paired_positive_sample_id": None,
        "items": items,
        "label": 1,
        "negative_metadata": None,
    }


def metadata_row(
    item_id: str,
    kit_id: str,
    split: str,
    master_category: str,
    coarse_category: str,
) -> dict:
    return {
        "item_metadata_version": "core7-item-metadata-v1",
        "category_mapping_version": "core7-v1",
        "split": split,
        "item_id": item_id,
        "source_kit_id": kit_id,
        "slot_index": int(item_id.rsplit("_", 1)[1]),
        "master_category": master_category,
        "coarse_category": coarse_category,
    }


def make_split(split: str, prefix: str) -> tuple[list[dict], list[dict]]:
    positives = [
        positive(f"{prefix}1", [f"{prefix}1_1", f"{prefix}1_2", f"{prefix}1_3"]),
        positive(f"{prefix}2", [f"{prefix}2_1", f"{prefix}2_2", f"{prefix}2_3"]),
    ]
    metadata = []
    specs = [
        (1, "T-Shirts", "TOP"),
        (2, "Jeans", "BOTTOM"),
        (3, "Sneakers", "SHOES"),
    ]
    for kit_number in (1, 2):
        kit_id = f"{prefix}{kit_number}"
        for slot, master, coarse in specs:
            metadata.append(
                metadata_row(
                    f"{kit_id}_{slot}",
                    kit_id,
                    split,
                    master,
                    coarse,
                )
            )
    return positives, metadata


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record))
            stream.write("\n")


class Core7ScorerDatasetTests(unittest.TestCase):
    def test_negative_is_one_same_category_different_kit_swap(self):
        positives, metadata = make_split("train", "tr")
        negatives, report = generate_negative_records(
            positives,
            metadata,
            split="train",
            seed=42,
        )

        self.assertTrue(report["pass"])
        self.assertEqual(len(negatives), 2)
        for source, negative in zip(positives, negatives):
            block = negative["negative_metadata"]
            differences = [
                index
                for index, pair in enumerate(zip(source["items"], negative["items"]))
                if pair[0] != pair[1]
            ]
            self.assertEqual(differences, [block["swapped_item_index"]])
            self.assertEqual(block["negative_type"], NEGATIVE_TYPE)
            self.assertNotEqual(
                block["replacement_kit_id"], source["source_kit_id"]
            )
            self.assertNotIn(block["replacement_item_id"], source["items"])

    def test_sampling_is_reproducible_for_same_seed(self):
        positives, metadata = make_split("train", "tr")
        first, _ = generate_negative_records(
            positives, metadata, split="train", seed=123
        )
        second, _ = generate_negative_records(
            positives, metadata, split="train", seed=123
        )
        self.assertEqual(first, second)

    def test_missing_candidate_is_reported_and_merge_is_blocked(self):
        positives = [positive("solo", ["solo_1", "solo_2", "solo_3"])]
        metadata = [
            metadata_row("solo_1", "solo", "train", "T-Shirts", "TOP"),
            metadata_row("solo_2", "solo", "train", "Jeans", "BOTTOM"),
            metadata_row("solo_3", "solo", "train", "Sneakers", "SHOES"),
        ]
        negatives, sampling = generate_negative_records(
            positives, metadata, split="train"
        )
        scorer, merge = merge_positive_negative_families(positives, negatives)

        self.assertFalse(sampling["pass"])
        self.assertEqual(sampling["failed_positive_count"], 1)
        self.assertEqual(scorer, [])
        self.assertFalse(merge["pass"])

    def test_valid_scorer_split_passes_independent_pair_check(self):
        positives, metadata = make_split("train", "tr")
        negatives, _ = generate_negative_records(
            positives, metadata, split="train", seed=42
        )
        scorer, _ = merge_positive_negative_families(positives, negatives)

        report = validate_scorer_split(scorer, metadata, split="train")

        self.assertTrue(report["pass"])
        self.assertEqual(report["positive_count"], 2)
        self.assertEqual(report["negative_count"], 2)
        self.assertEqual(report["issue_count"], 0)

    def test_validator_catches_wrong_swap_category(self):
        positives, metadata = make_split("train", "tr")
        negatives, _ = generate_negative_records(
            positives, metadata, split="train", seed=42
        )
        negatives[0]["negative_metadata"]["swap_category"] = "Wrong Category"
        scorer, _ = merge_positive_negative_families(positives, negatives)

        report = validate_scorer_split(scorer, metadata, split="train")

        self.assertFalse(report["pass"])
        self.assertEqual(
            report["issue_counts"]["swap_category_metadata_mismatch"], 1
        )

    def test_all_split_validation_passes_without_leakage(self):
        records_by_split = {}
        metadata_by_split = {}
        sampling_reports = {}
        embedding_splits = {}
        for index, split in enumerate(("train", "valid", "test")):
            positives, metadata = make_split(split, f"s{index}")
            negatives, sampling = generate_negative_records(
                positives, metadata, split=split, seed=42 + index
            )
            scorer, _ = merge_positive_negative_families(positives, negatives)
            records_by_split[split] = scorer
            metadata_by_split[split] = metadata
            sampling_reports[split] = sampling
            embedding_splits[split] = {
                "pass": True,
                "embedding_coverage": 1.0,
            }

        report = validate_all_splits(
            records_by_split,
            metadata_by_split,
            embedding_report={"pass": True, "splits": embedding_splits},
            sampling_reports=sampling_reports,
        )

        self.assertTrue(report["pass"])
        self.assertEqual(report["status"], "READY_TO_TRAIN")

    def test_end_to_end_builder_writes_ready_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "input"
            output_dir = root / "output"
            data_dir.mkdir()
            embedding_splits = {}

            for index, split in enumerate(("train", "valid", "test")):
                positives, metadata = make_split(split, f"s{index}")
                write_jsonl(data_dir / f"category_clean_{split}.jsonl", positives)
                write_jsonl(
                    data_dir / f"core7_item_metadata_v1_{split}.jsonl",
                    metadata,
                )
                embedding_splits[split] = {
                    "pass": True,
                    "embedding_coverage": 1.0,
                }

            embedding_report_path = data_dir / "embedding_report.json"
            embedding_report_path.write_text(
                json.dumps(
                    {
                        "pass": True,
                        "cache": {
                            "model_id": "patrickjohncyh/fashion-clip",
                            "embedding_dim": 512,
                        },
                        "splits": embedding_splits,
                    }
                ),
                encoding="utf-8",
            )

            result = build_scorer_dataset_v1(
                data_dir=data_dir,
                output_dir=output_dir,
                embedding_report_path=embedding_report_path,
                git_commit="unit-test",
            )

            self.assertEqual(result["status"], "READY_TO_TRAIN")
            self.assertTrue((output_dir / "dataset_manifest_v1.json").exists())
            self.assertTrue((output_dir / "split_manifest_v1.json").exists())
            manifest = json.loads(
                (output_dir / "dataset_manifest_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["status"], "READY_TO_TRAIN")
            self.assertEqual(manifest["negative_seed"], 42)


if __name__ == "__main__":
    unittest.main()
