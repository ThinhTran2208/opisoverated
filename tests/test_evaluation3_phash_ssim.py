"""Tests for calibrated EVALUATION3 pHash + SSIM overlap handling."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.evaluation.evaluation3_phash_ssim import (
    DUPLICATE,
    MANUAL_REVIEW,
    NON_DUPLICATE,
    classify_pair,
    finalize_overlap_audit,
    fingerprint_image,
    hamming_distance,
    image_to_ssim_array,
    ssim_score,
)


class Evaluation3PhashSsimTests(unittest.TestCase):
    def test_three_state_frozen_rule(self):
        self.assertEqual(
            classify_pair(phash_distance=4, ssim=0.92),
            (DUPLICATE, "phash_ssim_auto"),
        )
        self.assertEqual(
            classify_pair(phash_distance=4, ssim=0.90),
            (MANUAL_REVIEW, "phash_ssim_manual"),
        )
        self.assertEqual(
            classify_pair(phash_distance=4, ssim=0.8999),
            (NON_DUPLICATE, "phash_ssim_non_duplicate"),
        )
        self.assertEqual(
            classify_pair(phash_distance=5, ssim=None),
            (NON_DUPLICATE, "phash_outside_radius"),
        )

    def test_exact_pixel_is_optional_shortcut_only(self):
        self.assertEqual(
            classify_pair(
                phash_distance=0,
                ssim=1.0,
                exact_pixel_match=True,
                use_exact_pixel=False,
            ),
            (DUPLICATE, "phash_ssim_auto"),
        )
        self.assertEqual(
            classify_pair(
                phash_distance=0,
                ssim=1.0,
                exact_pixel_match=True,
                use_exact_pixel=True,
            ),
            (DUPLICATE, "exact_pixel"),
        )

    def test_identical_images_have_identical_phash_and_ssim_one(self):
        image = Image.new("RGB", (90, 70), (40, 110, 190))
        for x in range(8, 30):
            for y in range(5, 23):
                image.putpixel((x, y), (230, 25, 70))
        copied = image.copy()
        left = fingerprint_image(image)
        right = fingerprint_image(copied)
        self.assertEqual(hamming_distance(left.phash64, right.phash64), 0)
        self.assertEqual(left.pixel_sha256, right.pixel_sha256)
        self.assertAlmostEqual(
            ssim_score(image_to_ssim_array(image), image_to_ssim_array(copied)),
            1.0,
            places=7,
        )

    def test_two_notebook_variants_are_isolated_and_use_same_thresholds(self):
        repo_root = Path(__file__).resolve().parents[1]
        notebook_dir = repo_root / "notebooks/experiments/evaluation3_overlap_v2"
        expected = {
            "NB10A_evaluation3_phash_ssim_only.ipynb": "USE_EXACT_PIXEL = False",
            "NB10B_evaluation3_phash_ssim_exact_pixel.ipynb": "USE_EXACT_PIXEL = True",
        }
        for name, flag in expected.items():
            notebook = json.loads((notebook_dir / name).read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell.get("cell_type") == "code"
            )
            self.assertIn("feat/evaluation3-phash-ssim-overlap-v2", source)
            self.assertIn(flag, source)
            self.assertIn("phash_threshold=4", source)
            self.assertIn("ssim_auto_threshold=0.92", source)
            self.assertIn("ssim_manual_lower_bound=0.90", source)
            self.assertIn("evaluation3_overlap_phash_ssim_v2", source)

    def test_finalize_binary_manual_labels_preserves_model_vs_strict_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pre_rows = [
                {
                    "e3_outfit_id": "manual_test_only",
                    "missing_slots": [],
                    "overlap": {
                        "decision_pre_review": MANUAL_REVIEW,
                        "auto_duplicate_splits": [],
                        "manual_candidate_splits": ["test"],
                        "manual_review_pair_ids": ["PAIR000001"],
                        "image_results": [],
                    },
                },
                {
                    "e3_outfit_id": "manual_train",
                    "missing_slots": [],
                    "overlap": {
                        "decision_pre_review": MANUAL_REVIEW,
                        "auto_duplicate_splits": [],
                        "manual_candidate_splits": ["train"],
                        "manual_review_pair_ids": ["PAIR000002"],
                        "image_results": [],
                    },
                },
                {
                    "e3_outfit_id": "clean",
                    "missing_slots": [],
                    "overlap": {
                        "decision_pre_review": NON_DUPLICATE,
                        "auto_duplicate_splits": [],
                        "manual_candidate_splits": [],
                        "manual_review_pair_ids": [],
                        "image_results": [],
                    },
                },
            ]
            with (root / "evaluation3_overlap_audit_pre_review.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                for row in pre_rows:
                    stream.write(json.dumps(row) + "\n")
            with (root / "evaluation3_manual_review_BLIND.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(["pair_id", "preview_file", "human_label"])
                writer.writerow(["PAIR000001", "a.jpg", "DUPLICATE"])
                writer.writerow(["PAIR000002", "b.jpg", "DUPLICATE"])
            with (root / "evaluation3_manual_review_KEY.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(["pair_id", "polyvore_splits"])
                writer.writerow(["PAIR000001", "test"])
                writer.writerow(["PAIR000002", "train"])
            (root / "evaluation3_overlap_summary_pre_review.json").write_text(
                json.dumps(
                    {
                        "evaluation3": {"outfits_missing_required_images": 0},
                        "development_data": {"unresolved_item_count": 0},
                    }
                ),
                encoding="utf-8",
            )

            summary, paths = finalize_overlap_audit(
                output_dir=root,
                manual_labels_path=root / "evaluation3_manual_review_BLIND.csv",
            )
            self.assertEqual(summary["status"], "PASS")
            model_clean = [
                json.loads(line)
                for line in paths["model_clean"].read_text().splitlines()
            ]
            strict_clean = [
                json.loads(line)
                for line in paths["strict_clean"].read_text().splitlines()
            ]
            self.assertEqual(
                {row["e3_outfit_id"] for row in model_clean},
                {"manual_test_only", "clean"},
            )
            self.assertEqual(
                {row["e3_outfit_id"] for row in strict_clean},
                {"clean"},
            )

    def test_finalize_rejects_non_binary_manual_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation3_overlap_audit_pre_review.jsonl").write_text(
                json.dumps(
                    {
                        "e3_outfit_id": "x",
                        "missing_slots": [],
                        "overlap": {
                            "decision_pre_review": MANUAL_REVIEW,
                            "auto_duplicate_splits": [],
                            "manual_candidate_splits": ["train"],
                            "manual_review_pair_ids": ["PAIR000001"],
                            "image_results": [],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "evaluation3_manual_review_BLIND.csv").write_text(
                "pair_id,preview_file,human_label\nPAIR000001,a.jpg,UNCERTAIN\n",
                encoding="utf-8",
            )
            (root / "evaluation3_manual_review_KEY.csv").write_text(
                "pair_id,polyvore_splits\nPAIR000001,train\n",
                encoding="utf-8",
            )
            (root / "evaluation3_overlap_summary_pre_review.json").write_text(
                json.dumps(
                    {
                        "evaluation3": {"outfits_missing_required_images": 0},
                        "development_data": {"unresolved_item_count": 0},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "only DUPLICATE or NON_DUPLICATE"
            ):
                finalize_overlap_audit(
                    output_dir=root,
                    manual_labels_path=root / "evaluation3_manual_review_BLIND.csv",
                )


if __name__ == "__main__":
    unittest.main()
