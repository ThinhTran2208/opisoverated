"""Tests for NB11 EVALUATION3 active-learning helpers."""

import unittest

import numpy as np
import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
