"""Two-item scorer dataset/collator coverage for the min2 experiment."""

import json
import tempfile
import unittest
from pathlib import Path

from src.scorer.dataset import ScorerDataset, collate_scorer_batch, torch


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row))
            stream.write("\n")


@unittest.skipUnless(torch is not None, "PyTorch is not installed in portability CI")
class ScorerDatasetMin2Tests(unittest.TestCase):
    def test_two_item_family_collates_to_exactly_one_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples_path = root / "scorer_ready_v2_train.jsonl"
            metadata_path = root / "core7_item_metadata_v1_train.jsonl"
            cache_path = root / "fashionclip_item_embeddings.pt"

            rows = [
                {
                    "sample_id": "k1_pos",
                    "source_kit_id": "k1",
                    "paired_positive_sample_id": None,
                    "items": ["top1", "shoe1"],
                    "label": 1,
                    "negative_metadata": None,
                },
                {
                    "sample_id": "k1_neg_1",
                    "source_kit_id": "k1",
                    "paired_positive_sample_id": "k1_pos",
                    "items": ["top1", "shoe2"],
                    "label": 0,
                    "negative_metadata": {"swapped_item_index": 1},
                },
            ]
            metadata_rows = [
                {
                    "item_metadata_version": "core7-item-metadata-v1",
                    "category_mapping_version": "core7-v2",
                    "split": "train",
                    "item_id": "top1",
                    "source_kit_id": "k1",
                    "slot_index": 1,
                    "master_category": "T-Shirts",
                    "coarse_category": "TOP",
                },
                {
                    "item_metadata_version": "core7-item-metadata-v1",
                    "category_mapping_version": "core7-v2",
                    "split": "train",
                    "item_id": "shoe1",
                    "source_kit_id": "k1",
                    "slot_index": 2,
                    "master_category": "Sneakers",
                    "coarse_category": "SHOES",
                },
                {
                    "item_metadata_version": "core7-item-metadata-v1",
                    "category_mapping_version": "core7-v2",
                    "split": "train",
                    "item_id": "shoe2",
                    "source_kit_id": "k2",
                    "slot_index": 2,
                    "master_category": "Sneakers",
                    "coarse_category": "SHOES",
                },
            ]

            write_jsonl(samples_path, rows)
            write_jsonl(metadata_path, metadata_rows)
            torch.save(
                {
                    "item_ids": ["top1", "shoe1", "shoe2"],
                    "embeddings": torch.randn(3, 512, dtype=torch.float16),
                },
                cache_path,
            )

            dataset = ScorerDataset(samples_path, metadata_path, cache_path)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.min_items, 2)

            batch = collate_scorer_batch([dataset[0], dataset[1]])
            self.assertEqual(batch["item_mask"].sum(dim=1).tolist(), [2, 2])
            self.assertEqual(batch["pair_mask"].sum(dim=(1, 2)).tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
