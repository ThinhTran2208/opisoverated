"""Lightweight contract tests for RF-DETR + FashionCLIP Core-7 detection."""

import json
import tempfile
import unittest
from pathlib import Path

from src.detection.config import (
    CORE7_CATEGORY_TO_ID,
    EXPECTED_EMBEDDING_DIM,
    load_detection_config,
)
from src.detection.fashionclip import select_core7_prediction
from src.detection.pipeline import build_scorer_batch_lists, expand_and_clamp_xyxy
from src.detection.rfdetr import DEFAULT_CORE7_DETECTOR_LABELS, build_detection_candidates
from src.detection.schema import CategoryPrediction, DetectedGarment, DetectionCandidate, DetectionResult


class DetectionCore7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.config_path = cls.repo_root / "configs/detection_rfdetr_fashionclip_core7_v1.json"
        cls.config = load_detection_config(cls.config_path)

    def test_config_matches_canonical_core7_vocab(self):
        self.assertEqual(
            CORE7_CATEGORY_TO_ID,
            {
                "TOP": 1,
                "BOTTOM": 2,
                "DRESS": 3,
                "OUTERWEAR": 4,
                "SHOES": 5,
                "BAG": 6,
                "HAT": 7,
            },
        )
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["scorer_handoff"]["category_ids"], CORE7_CATEGORY_TO_ID)
        self.assertEqual(self.config.embedding_dim, EXPECTED_EMBEDDING_DIM)
        self.assertEqual(self.config.category_mapping_version, "core7-v2")

    def test_detector_filter_removes_parts_without_mapping_label_to_core7(self):
        accepted, rejected = build_detection_candidates(
            boxes=[[0, 0, 50, 80], [10, 10, 20, 20], [30, 20, 60, 45]],
            confidences=[0.9, 0.8, 0.7],
            class_ids=[0, 31, 24],
            class_names=["shirt, blouse", "sleeve", "bag, wallet"],
            supported_labels=DEFAULT_CORE7_DETECTOR_LABELS,
        )
        self.assertEqual([row.detector_label for row in accepted], ["shirt, blouse", "bag, wallet"])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["detector_label"], "sleeve")
        self.assertNotIn("coarse_category", rejected[0])

    def test_zero_shot_selection_reports_top1_and_margin(self):
        scores = {
            "TOP": 0.45,
            "BOTTOM": 0.20,
            "DRESS": 0.10,
            "OUTERWEAR": 0.32,
            "SHOES": 0.05,
            "BAG": -0.01,
            "HAT": 0.02,
        }
        decision = select_core7_prediction(scores)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.coarse_category, "TOP")
        self.assertEqual(decision.coarse_category_id, 1)
        self.assertAlmostEqual(decision.similarity, 0.45)
        self.assertAlmostEqual(decision.margin, 0.13)

    def test_zero_shot_threshold_can_reject_ambiguous_prediction(self):
        scores = {
            "TOP": 0.31,
            "BOTTOM": 0.30,
            "DRESS": 0.10,
            "OUTERWEAR": 0.20,
            "SHOES": 0.05,
            "BAG": 0.04,
            "HAT": 0.02,
        }
        decision = select_core7_prediction(scores, min_margin=0.02)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.rejection_reason, "category_margin_below_threshold")

    def test_crop_padding_is_clamped_to_image(self):
        crop = expand_and_clamp_xyxy(
            [-5.0, 4.0, 98.0, 110.0],
            image_width=100,
            image_height=100,
            padding_ratio=0.03,
        )
        self.assertEqual(crop[0], 0)
        self.assertEqual(crop[1], 0)
        self.assertEqual(crop[2], 100)
        self.assertEqual(crop[3], 100)

    def _fake_result(self, count=3):
        garments = []
        categories = list(CORE7_CATEGORY_TO_ID.items())
        for index in range(count):
            category, category_id = categories[index % len(categories)]
            candidate = DetectionCandidate(
                detection_index=index,
                box_xyxy=(0, 0, 20, 30),
                detector_label="shirt, blouse",
                detector_confidence=0.9,
                detector_class_id=0,
            )
            prediction = CategoryPrediction(
                coarse_category=category,
                coarse_category_id=category_id,
                similarity=0.4,
                margin=0.1,
                similarities={name: 0.1 for name in CORE7_CATEGORY_TO_ID},
                source="fashionclip-zero-shot-core7-v1",
            )
            garments.append(
                DetectedGarment(
                    candidate=candidate,
                    crop_box_xyxy=(0, 0, 20, 30),
                    category=prediction,
                    embedding=[0.0] * EXPECTED_EMBEDDING_DIM,
                )
            )
        return DetectionResult(
            detection_version="rfdetr-fashionclip-core7-v1",
            detector_repo_id="resoa/garment-detector-seg",
            fashionclip_model_id="patrickjohncyh/fashion-clip",
            category_classifier_version="fashionclip-zero-shot-core7-v1",
            category_mapping_version="core7-v2",
            image_width=200,
            image_height=300,
            garments=garments,
            rejected_detections=[],
        )

    def test_scorer_handoff_contains_only_embedding_coarse_ids_and_mask(self):
        result = self._fake_result(3)
        batch = build_scorer_batch_lists(result, min_items=3, max_items=8)
        self.assertEqual(set(batch), {"item_embeddings", "coarse_category_ids", "item_mask"})
        self.assertEqual(batch["coarse_category_ids"], [1, 2, 3])
        self.assertEqual(batch["item_mask"], [True, True, True])
        metadata = result.metadata_dict()
        self.assertIsNone(metadata["taxonomy"]["master_category"])
        for garment in metadata["garments"]:
            self.assertNotIn("master_category", garment)

    def test_scorer_handoff_does_not_silently_truncate(self):
        with self.assertRaisesRegex(ValueError, "supports <= 8"):
            build_scorer_batch_lists(self._fake_result(9), min_items=3, max_items=8)

    def test_detection_notebook_is_portable_and_uses_new_pipeline(self):
        notebook_path = (
            self.repo_root
            / "notebooks/experiments/NB9_detection_rfdetr_fashionclip_core7_v1.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertNotIn("from google.colab import drive", code)
        self.assertNotIn("/content/drive/MyDrive", code)
        self.assertNotIn("GroundingDino", code)
        self.assertIn("DetectionPipeline", code)
        self.assertIn("load_detection_config", code)

    def test_config_rejects_missing_category_prompt(self):
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        del payload["coarse_category_classifier"]["prompts"]["HAT"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly the seven"):
                load_detection_config(path)


if __name__ == "__main__":
    unittest.main()
