"""Tests for the EVALUATION3/Polyvore overlap audit."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.evaluation.evaluation3_overlap import (
    BKTree,
    discover_evaluation3_outfits,
    hamming_distance,
    load_evaluation3_annotations,
    merge_evaluation3_annotations,
    run_overlap_audit,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row))
            stream.write("\n")


def make_image(path: Path, color: tuple[int, int, int], *, size=(40, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    # Add asymmetric detail so the perceptual fingerprint is meaningful.
    for x in range(5, min(17, size[0])):
        for y in range(3, min(13, size[1])):
            image.putpixel((x, y), tuple(255 - channel for channel in color))
    image.save(path)


class Evaluation3OverlapTests(unittest.TestCase):
    def test_nb10_bootstraps_the_feature_branch_on_colab(self):
        repo_root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (
                repo_root
                / "notebooks/experiments/NB10_evaluation3_overlap_audit.ipynb"
            ).read_text(encoding="utf-8")
        )
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn("feat/evaluation3-overlap-audit", code)
        self.assertIn("'clone', '--branch', BRANCH, '--single-branch'", code)
        self.assertIn("os.environ['FASHION_PROJECT_ROOT']", code)
        self.assertIn("drive.mount('/content/drive'", code)
        self.assertIn("DRIVE_ROOT / 'EVALUATION3'", code)
        self.assertIn("DRIVE_ROOT / 'scorer_ready_v2'", code)
        self.assertIn("INPUTS_READY = not missing", code)
        self.assertNotIn("raise FileNotFoundError('Thiếu input", code)

    def test_annotation_tables_join_by_item(self):
        merged = merge_evaluation3_annotations(
            {"436320": {"cmt_raw": "3", "reason_raw": "1", "group": ""}},
            {"436320": {"cmt_raw": None, "reason_raw": None, "group": "A-Test2000"}},
        )
        self.assertEqual(
            merged["436320"],
            {"cmt_raw": "3", "reason_raw": "1", "group": "A-Test2000"},
        )

    def test_bk_tree_returns_only_values_inside_hamming_radius(self):
        tree = BKTree()
        values = (0b0000, 0b0011, 0b1111)
        for value in values:
            tree.add(value)

        self.assertEqual(hamming_distance(0b0000, 0b0011), 2)
        self.assertEqual(tree.query(0b0001, 1), [(1, 0b0000), (1, 0b0011)])

    def test_annotation_csv_and_group_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations_path = root / "annotations.csv"
            with annotations_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["ITEM", "Cmt", "Reason", "Group"])
                writer.writerow([101, 1, "", "A-Test2000"])
                writer.writerow([102, 3, 1, "A-Train29479"])

            for outfit_id in ("101", "102"):
                for slot, color in zip("UBSG", ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12))):
                    make_image(root / "outfit" / outfit_id / f"{slot}.jpg", color)

            annotations = load_evaluation3_annotations(annotations_path)
            outfits = discover_evaluation3_outfits(
                root / "outfit",
                annotations=annotations,
                selected_groups=["A-Test2000"],
            )

            self.assertEqual(len(outfits), 1)
            self.assertEqual(outfits[0]["e3_outfit_id"], "101")
            self.assertEqual(outfits[0]["cmt_raw"], "1")
            self.assertEqual(outfits[0]["missing_slots"], [])

    def test_outputs_full_model_clean_and_strict_clean_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scorer = root / "scorer"
            polyvore = root / "polyvore_images"
            evaluation3 = root / "evaluation3" / "outfit"
            output = root / "audit"
            scorer.mkdir()

            write_jsonl(
                scorer / "train.jsonl",
                [
                    {
                        "sample_id": "train_kit_pos",
                        "source_kit_id": "train_kit",
                        "items": ["train_kit_1", "train_kit_2"],
                        "label": 1,
                    }
                ],
            )
            write_jsonl(
                scorer / "valid.jsonl",
                [
                    {
                        "sample_id": "valid_kit_pos",
                        "source_kit_id": "valid_kit",
                        "items": ["valid_kit_1"],
                        "label": 1,
                    }
                ],
            )
            write_jsonl(
                scorer / "test.jsonl",
                [
                    {
                        "sample_id": "test_kit_pos",
                        "source_kit_id": "test_kit",
                        "items": ["test_kit_1"],
                        "label": 1,
                    }
                ],
            )

            colors = {
                "train_kit_1": (220, 20, 20),
                "train_kit_2": (20, 220, 20),
                "valid_kit_1": (20, 20, 220),
                "test_kit_1": (180, 90, 10),
            }
            for item_id, color in colors.items():
                make_image(polyvore / f"{item_id}.png", color)

            # Exact train-image overlap.
            make_image(evaluation3 / "e3_train_overlap" / "U.png", colors["train_kit_1"])
            # Test-only overlap: model-clean but not strict-clean.
            make_image(evaluation3 / "e3_test_overlap" / "U.png", colors["test_kit_1"])
            # No ID or image overlap.
            make_image(evaluation3 / "e3_clean" / "U.jpg", (90, 40, 170))
            # Same ID as a scorer source kit is conservatively a candidate.
            make_image(evaluation3 / "train_kit" / "U.jpg", (30, 130, 190))

            # Complete every E3 outfit so the official readiness gate passes.
            for index, outfit_dir in enumerate(sorted(evaluation3.iterdir())):
                for slot_offset, slot in enumerate("BSG", start=1):
                    make_image(
                        outfit_dir / f"{slot}.jpg",
                        (40 + index * 20, 30 + slot_offset * 17, 70 + index * 13),
                    )

            summary, paths = run_overlap_audit(
                evaluation3_root=evaluation3,
                development_split_paths={
                    "train": scorer / "train.jsonl",
                    "valid": scorer / "valid.jsonl",
                    "test": scorer / "test.jsonl",
                },
                output_dir=output,
                polyvore_image_root=polyvore,
                model_development_splits={"train", "valid"},
                near_hamming_threshold=0,
            )

            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["manifests"], {"full": 4, "model_clean": 2, "strict_clean": 1})
            self.assertEqual(summary["overlap"]["outfit_counts_by_method"]["id_candidate_splits"], 1)
            self.assertEqual(summary["overlap"]["outfit_counts_by_method"]["exact_pixel_splits"], 2)

            full = [json.loads(line) for line in paths["full"].read_text().splitlines()]
            model_clean = [
                json.loads(line) for line in paths["model_clean"].read_text().splitlines()
            ]
            strict_clean = [
                json.loads(line) for line in paths["strict_clean"].read_text().splitlines()
            ]
            self.assertEqual(len(full), 4)
            self.assertEqual(
                {row["e3_outfit_id"] for row in model_clean},
                {"e3_clean", "e3_test_overlap"},
            )
            self.assertEqual(
                {row["e3_outfit_id"] for row in strict_clean},
                {"e3_clean"},
            )
            self.assertTrue(paths["evidence"].is_file())

    def test_missing_polyvore_images_block_official_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scorer = root / "train.jsonl"
            write_jsonl(
                scorer,
                [
                    {
                        "sample_id": "kit_pos",
                        "source_kit_id": "kit",
                        "items": ["missing_item"],
                        "label": 1,
                    }
                ],
            )
            (root / "polyvore").mkdir()
            for slot in "UBSG":
                make_image(root / "evaluation3" / "1" / f"{slot}.jpg", (1, 2, 3))

            with self.assertRaisesRegex(ValueError, "image index is incomplete"):
                run_overlap_audit(
                    evaluation3_root=root / "evaluation3",
                    development_split_paths={"train": scorer},
                    output_dir=root / "output",
                    polyvore_image_root=root / "polyvore",
                    model_development_splits={"train"},
                )


if __name__ == "__main__":
    unittest.main()
