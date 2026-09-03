"""Tests for NB11 EVALUATION3 active-learning helpers."""

import unittest

import numpy as np
import pandas as pd

from src.evaluation.evaluation3_preview_active_learning import (
    SAME_PRODUCT_DIFFERENT_IMAGE,
    attach_labels,
    fit_models,
    select_resolution_batch,
)
from src.evaluation.evaluation3_active_learning import (
    DUPLICATE,
    NON_DUPLICATE,
    apply_triage,
    choose_triage_thresholds,
    diverse_batch,
    prepare_feature_frame,
)


class Evaluation3ActiveLearningTests(unittest.TestCase):
    @staticmethod
    def _rows(count=12):
        rows = []
        for index in range(count):
            rows.append(
                {
                    "pair_id": f"PAIR{index:06d}",
                    "preview_file": f"manual_review_previews/PAIR{index:06d}.jpg",
                    "e3_outfit_id": f"OUTFIT{index:03d}",
                    "e3_slot": "B",
                    "phash_distance": index % 5,
                    "alignment_success": index % 3 != 0,
                    "ecc_correlation": 0.80 + 0.01 * index,
                    "rotation_degrees": (-1) ** index * 0.2 * index,
                    "translation_x": (-1) ** index * index,
                    "translation_y": index / 2,
                    "rgb_ssim": 0.70 + 0.02 * index,
                    "gray_ssim": 0.75 + 0.015 * index,
                    "foreground_iou": 0.72 + 0.015 * index,
                    "edge_ssim": 0.68 + 0.02 * index,
                    "mean_lab_delta": 20 - index,
                    "interior_mae": 0.15 - 0.007 * index,
                    "patch_mae_max": 0.25 - 0.01 * index,
                    "patch_mae_p90": 0.20 - 0.008 * index,
                    "patch_count": 10 + index,
                }
            )
        return pd.DataFrame(rows)

    def test_prepare_feature_frame_is_numeric_and_stable(self):
        frame = prepare_feature_frame(self._rows(4))
        self.assertEqual(len(frame), 4)
        self.assertTrue(np.isfinite(frame.to_numpy()).all())
        self.assertIn("abs_rotation_degrees", frame.columns)
        self.assertGreaterEqual(float(frame["abs_translation_x"].min()), 0.0)

    def test_diverse_batch_respects_requested_size(self):
        selected = diverse_batch(self._rows(12), batch_size=5, random_state=7)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected["e3_outfit_id"].nunique(), 5)

    def test_two_threshold_triage_keeps_middle_manual(self):
        # Only the two extreme low/high regions are perfectly reliable; the
        # intentionally mixed middle must remain MANUAL_REVIEW.
        y = np.array([0, 0, 1, 0, 1, 0, 1, 1], dtype=int)
        p = np.array([0.01, 0.05, 0.20, 0.40, 0.60, 0.80, 0.95, 0.99])
        thresholds = choose_triage_thresholds(
            y,
            p,
            target_auto_duplicate_precision=1.0,
            target_auto_non_npv=1.0,
            minimum_auto_examples=2,
        )
        triage = apply_triage(p, thresholds)
        self.assertTrue((triage[:2] == NON_DUPLICATE).all())
        self.assertTrue((triage[-2:] == DUPLICATE).all())
        self.assertIn("MANUAL_REVIEW", set(triage))

    def test_overlap_of_safe_regions_collapses_to_manual(self):
        y = np.array([0, 1, 0, 1], dtype=int)
        p = np.array([0.40, 0.45, 0.55, 0.60])
        thresholds = choose_triage_thresholds(
            y,
            p,
            target_auto_duplicate_precision=0.50,
            target_auto_non_npv=0.50,
            minimum_auto_examples=1,
        )
        self.assertLess(
            thresholds.auto_non_max_probability,
            thresholds.auto_duplicate_min_probability,
        )

    def test_preview_models_accept_valid_row_weights(self):
        rows = self._rows(20)
        # Preview models use a smaller feature contract than NB10D models.
        preview = rows.assign(
            foreground_mae=np.linspace(0.05, 0.25, len(rows)),
            histogram_intersection=np.linspace(0.95, 0.55, len(rows)),
            foreground_area_ratio_delta=np.linspace(0.01, 0.20, len(rows)),
            bbox_aspect_ratio_delta=np.linspace(0.01, 0.30, len(rows)),
            target=np.arange(len(rows)) % 2,
        )
        train = preview.iloc[:16].copy()
        validation = preview.iloc[16:].copy()
        fitted, report = fit_models(
            train,
            validation,
            random_state=3,
            sample_weight=np.linspace(0.25, 2.0, len(train)),
        )
        self.assertEqual(set(fitted), {"logistic_regression", "svm_rbf", "random_forest"})
        self.assertEqual(len(report), 3)

    def test_preview_models_reject_misaligned_weights(self):
        rows = self._rows(20).assign(
            foreground_mae=0.1,
            histogram_intersection=0.8,
            foreground_area_ratio_delta=0.1,
            bbox_aspect_ratio_delta=0.1,
            target=np.arange(20) % 2,
        )
        with self.assertRaisesRegex(ValueError, "one value per train row"):
            fit_models(
                rows.iloc[:16],
                rows.iloc[16:],
                sample_weight=np.ones(3),
            )

    def test_same_product_different_image_maps_to_duplicate(self):
        pool = pd.DataFrame({"pair_id": ["PAIR1"]})
        labels = pd.DataFrame(
            {
                "pair_id": ["PAIR1"],
                "human_label": [SAME_PRODUCT_DIFFERENT_IMAGE],
            }
        )
        path = self._write_temp_csv(labels)
        try:
            attached = attach_labels(pool, [path])
            self.assertEqual(int(attached.loc[0, "target"]), 1)
        finally:
            path.unlink(missing_ok=True)

    def test_resolution_batch_is_one_per_unresolved_group(self):
        class ProbabilityFromRgb:
            classes_ = np.array([0, 1])

            @staticmethod
            def predict_proba(features):
                p = features["rgb_ssim"].to_numpy(dtype=float)
                return np.column_stack([1 - p, p])

        pool = self._rows(6).assign(
            foreground_mae=0.1,
            histogram_intersection=0.8,
            foreground_area_ratio_delta=0.1,
            bbox_aspect_ratio_delta=0.1,
            e3_outfit_id=["A", "A", "B", "B", "C", "C"],
        )
        labeled = pool.iloc[[2]].assign(
            human_label=DUPLICATE,
            target=1,
        )
        selected = select_resolution_batch(
            ProbabilityFromRgb(),
            pool,
            batch_size=10,
            labeled_rows=labeled,
        )
        self.assertNotIn("B", set(selected["e3_outfit_id"]))
        self.assertEqual(selected["e3_outfit_id"].nunique(), len(selected))
        self.assertEqual(set(selected["e3_outfit_id"]), {"A", "C"})

    @staticmethod
    def _write_temp_csv(frame):
        import tempfile
        from pathlib import Path

        handle = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        handle.close()
        path = Path(handle.name)
        frame.to_csv(path, index=False)
        return path


if __name__ == "__main__":
    unittest.main()
