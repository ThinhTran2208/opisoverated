"""Tests for Scorer V1 dataset lookup, padding, and frozen pair families."""

import json
import tempfile
import unittest
from pathlib import Path

from src.scorer.dataset import (
    CATEGORY_TO_ID,
    ScorerDataset,
    build_metadata_index,
    collate_scorer_batch,
    flatten_family_indices,
    metadata_split_path,
    paired_family_indices,
    scorer_split_path,
    torch,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row))
            stream.write("\n")


def metadata(item_id: str, kit_id: str, coarse: str, master: str) -> dict:
    return {
        "item_metadata_version": "core7-item-metadata-v1",
        "category_mapping_version": "core7-v2",
        "split": "train",
        "item_id": item_id,
        "source_kit_id": kit_id,
        "slot_index": 0,
        "master_category": master,
        "coarse_category": coarse,
    }


class ScorerDatasetSchemaTests(unittest.TestCase):
    def test_category_ids_are_locked(self):
        self.assertEqual(CATEGORY_TO_ID["TOP"], 1)
        self.assertEqual(CATEGORY_TO_ID["HAT"], 7)
        self.assertEqual(len(CATEGORY_TO_ID), 7)

    def test_canonical_split_paths(self):
        self.assertEqual(
            scorer_split_path("/tmp/scorer", "valid").name,
            "scorer_ready_v2_valid.jsonl",
        )
        self.assertEqual(
            metadata_split_path("/tmp/core7", "test").name,
            "core7_item_metadata_v1_test.jsonl",
        )

    def test_metadata_rejects_unknown_coarse_category(self):
        with self.assertRaises(ValueError):
            build_metadata_index([metadata("x", "k", "UNKNOWN", "Thing")])

    def test_pair_family_lookup_does_not_depend_on_adjacency(self):
        rows = [
            {
                "sample_id": "a_pos",
                "paired_positive_sample_id": None,
                "label": 1,
            },
            {
                "sample_id": "b_pos",
                "paired_positive_sample_id": None,
                "label": 1,
            },
            {
                "sample_id": "b_neg_1",
                "paired_positive_sample_id": "b_pos",
                "label": 0,
            },
            {
                "sample_id": "a_neg_1",
                "paired_positive_sample_id": "a_pos",
                "label": 0,
            },
        ]
        families = paired_family_indices(rows)
        self.assertEqual(families, [(0, 3), (1, 2)])
        self.assertEqual(flatten_family_indices(families[:1]), [0, 3])


@unittest.skipUnless(torch is not None, "PyTorch is not installed in portability CI")
class ScorerDatasetTorchTests(unittest.TestCase):
    def test_dataset_lookup_and_fixed_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_path = root / "scorer_ready_v2_train.jsonl"
            metadata_path = root / "core7_item_metadata_v1_train.jsonl"
            cache_path = root / "fashionclip_item_embeddings.pt"

            rows = [
                {
                    "sample_id": "k1_pos",
                    "source_kit_id": "k1",
                    "paired_positive_sample_id": None,
                    "items": ["top1", "bottom1", "shoe1"],
                    "label": 1,
                    "negative_metadata": None,
                },
                {
                    "sample_id": "k1_neg_1",
                    "source_kit_id": "k1",
                    "paired_positive_sample_id": "k1_pos",
                    "items": ["top1", "bottom1", "shoe2"],
                    "label": 0,
                    "negative_metadata": {"swapped_item_index": 2},
                },
            ]
            metadata_rows = [
                metadata("top1", "k1", "TOP", "T-Shirts"),
                metadata("bottom1", "k1", "BOTTOM", "Jeans"),
                metadata("shoe1", "k1", "SHOES", "Sneakers"),
                metadata("shoe2", "k2", "SHOES", "Sneakers"),
            ]
            write_jsonl(samples_path, rows)
            write_jsonl(metadata_path, metadata_rows)
            torch.save(
                {
                    "item_ids": ["top1", "bottom1", "shoe1", "shoe2"],
                    "embeddings": torch.randn(4, 512, dtype=torch.float16),
                },
                cache_path,
            )

            dataset = ScorerDataset(samples_path, metadata_path, cache_path)
            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertEqual(tuple(sample["item_embeddings"].shape), (3, 512))
            self.assertEqual(sample["coarse_category_ids"].tolist(), [1, 2, 5])

            batch = collate_scorer_batch([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["item_embeddings"].shape), (2, 8, 512))
            self.assertEqual(tuple(batch["coarse_category_ids"].shape), (2, 8))
            self.assertEqual(tuple(batch["item_mask"].shape), (2, 8))
            self.assertEqual(tuple(batch["pair_mask"].shape), (2, 8, 8))
            self.assertEqual(batch["item_mask"].sum(dim=1).tolist(), [3, 3])
            self.assertEqual(batch["pair_mask"].sum(dim=(1, 2)).tolist(), [3, 3])
            self.assertTrue((batch["item_embeddings"][:, 3:] == 0).all())
            self.assertTrue((batch["coarse_category_ids"][:, 3:] == 0).all())


if __name__ == "__main__":
    unittest.main()
