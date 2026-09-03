# -*- coding: utf-8 -*-

import math
import tempfile
import unittest
from pathlib import Path

from src.calibration import (
    CALIBRATION_VERSION,
    PlattCalibrator,
    calibration_metrics,
    fit_platt_calibrator,
    load_calibrator,
    save_calibrator,
)


class CalibrationV1Tests(unittest.TestCase):
    def test_monotonic_product_score_and_bounds(self):
        calibrator = PlattCalibrator(
            scale=0.5,
            bias=-0.2,
            scorer_version="type_aware_pairwise_v1",
        )
        logits = [-100.0, -2.0, 0.0, 2.0, 100.0]
        probabilities = calibrator.transform_many(logits)
        self.assertEqual(probabilities, sorted(probabilities))
        scores = [calibrator.compatibility_score(value) for value in logits]
        self.assertEqual(scores, sorted(scores))
        self.assertTrue(all(0 <= value <= 100 for value in scores))

    def test_artifact_round_trip(self):
        calibrator = PlattCalibrator(
            scale=0.47217959118640485,
            bias=-0.17733224823027438,
            scorer_version="type_aware_pairwise_v1",
            metadata={"fit_split": "valid"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            save_calibrator(calibrator, path)
            restored = load_calibrator(path)
        self.assertEqual(restored.calibration_version, CALIBRATION_VERSION)
        self.assertAlmostEqual(restored.scale, calibrator.scale)
        self.assertAlmostEqual(restored.bias, calibrator.bias)
        self.assertEqual(restored.metadata["fit_split"], "valid")

    def test_fitter_improves_synthetic_nll(self):
        logits = [-5.0, -3.0, -1.5, -0.5, 0.5, 1.5, 3.0, 5.0]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        raw = calibration_metrics(logits, labels)
        calibrator = fit_platt_calibrator(
            logits,
            labels,
            scorer_version="type_aware_pairwise_v1",
            max_steps=2000,
        )
        fitted = calibration_metrics(
            logits,
            labels,
            scale=calibrator.scale,
            bias=calibrator.bias,
        )
        self.assertGreater(calibrator.scale, 0.0)
        self.assertLess(fitted["nll"], raw["nll"])
        self.assertTrue(math.isfinite(fitted["brier"]))

    def test_committed_v1_artifact_contract(self):
        repo_root = Path(__file__).resolve().parents[1]
        calibrator = load_calibrator(
            repo_root
            / "artifacts"
            / "calibration"
            / "type_aware_pairwise_v1"
            / "platt_logistic_v1.json"
        )
        self.assertEqual(calibrator.scorer_version, "type_aware_pairwise_v1")
        self.assertGreater(calibrator.scale, 0.0)
        self.assertEqual(calibrator.metadata["fit_split"], "valid")
        self.assertFalse(calibrator.metadata["test_split_loaded"])


if __name__ == "__main__":
    unittest.main()
