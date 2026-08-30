# -*- coding: utf-8 -*-

import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.inference import InferenceContext, ProductionInferencePipeline


@unittest.skipIf(torch is None, "PyTorch is not installed in lightweight portability CI")
class ProductionInferenceV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.pipeline = ProductionInferencePipeline.load_from_manifest(
            cls.repo_root / "configs" / "production_inference_v1.json",
            repo_root=cls.repo_root,
            device="cpu",
        )

    @staticmethod
    def _item(index, category_id):
        vector = [0.0] * 512
        vector[index] = 1.0
        return {
            "item_id": f"garment-{index}",
            "embedding": vector,
            "coarse_category_id": category_id,
            "coarse_category": {1: "TOP", 2: "BOTTOM", 5: "SHOES"}.get(
                category_id, "TOP"
            ),
            "bbox": [index, index, index + 10, index + 20],
            "detection_confidence": 0.9,
        }

    def test_real_checkpoint_calibration_and_loo_end_to_end(self):
        items = [self._item(0, 1), self._item(1, 2), self._item(2, 5)]
        result = self.pipeline.analyze_precomputed(items)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["item_count"], 3)
        self.assertTrue(0 <= result["compatibility"]["compatibility_score"] <= 100)
        self.assertEqual(
            result["compatibility"]["calibration_version"], "platt-logistic-v1"
        )
        self.assertIn(result["diagnosis"]["problematic_item_index"], (0, 1, 2))
        self.assertEqual(len(result["diagnosis"]["deltas_without_minus_full"]), 3)
        self.assertTrue(result["diagnosis"]["uses_two_item_extrapolation"])
        self.assertNotIn("embedding", result["items"][0])

    def test_insufficient_garments_is_structured_error(self):
        result = self.pipeline.analyze_precomputed_safe(
            [self._item(0, 1), self._item(1, 2)]
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "insufficient_garments")
        self.assertEqual(result["error"]["details"]["minimum_required"], 3)

    def test_more_than_eight_is_not_silently_truncated(self):
        items = []
        for index in range(9):
            vector = [0.0] * 512
            vector[index] = 1.0
            items.append(
                {
                    "item_id": f"garment-{index}",
                    "embedding": vector,
                    "coarse_category_id": (index % 7) + 1,
                }
            )
        result = self.pipeline.analyze_precomputed_safe(items)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "too_many_garments")
        self.assertEqual(result["error"]["details"]["maximum_supported"], 8)

    def test_non_normalized_embedding_is_rejected(self):
        items = [self._item(0, 1), self._item(1, 2), self._item(2, 5)]
        items[1]["embedding"][1] = 2.0
        result = self.pipeline.analyze_precomputed_safe(items)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "embedding_not_l2_normalized")

    def test_image_boundary_fails_cleanly_until_detection_adapter_exists(self):
        result = self.pipeline.analyze_image_safe(object())
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "image_preprocessor_unavailable")

    def test_image_flow_passes_raw_loo_and_crop_refs_to_vlm_adapter(self):
        closed = {"value": False}
        captured = {}

        class FakeDetectionAdapter:
            def prepare(self, image):
                garments = []
                embeddings = []
                categories = [1, 2, 5]
                names = ["TOP", "BOTTOM", "SHOES"]
                for index, (category_id, name) in enumerate(zip(categories, names)):
                    vector = torch.zeros(512, dtype=torch.float32)
                    vector[index] = 1.0
                    embeddings.append(vector)
                    garments.append(
                        {
                            "item_id": f"garment-{index}",
                            "coarse_category_id": category_id,
                            "coarse_category": name,
                            "bbox": [index, index, index + 10, index + 20],
                        }
                    )
                return InferenceContext(
                    garments=garments,
                    embeddings=torch.stack(embeddings),
                    categories=torch.tensor(categories, dtype=torch.long),
                    crop_image_refs=[Path(f"crop-{index}.png") for index in range(3)],
                    original_image=image,
                    request_id="request-123",
                    metadata={"detection": {"detection_version": "fake-v1"}},
                    cleanup=lambda: closed.__setitem__("value", True),
                )

        class FakeVLMAdapter:
            def explain(self, loo_result, garments, crop_image_refs, *, sample_id):
                captured["loo_result"] = loo_result
                captured["garments"] = list(garments)
                captured["crop_image_refs"] = list(crop_image_refs)
                captured["sample_id"] = sample_id
                return {"headline": "fake explanation"}

        pipeline = ProductionInferencePipeline(
            scorer=self.pipeline.scorer,
            calibrator=self.pipeline.calibrator,
            detection_adapter=FakeDetectionAdapter(),
            vlm_adapter=FakeVLMAdapter(),
        )
        result = pipeline.analyze_image(object())

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["request_id"], "request-123")
        self.assertEqual(result["explanation"]["headline"], "fake explanation")
        self.assertEqual(captured["sample_id"], "request-123")
        self.assertEqual(len(captured["crop_image_refs"]), 3)
        self.assertEqual(len(captured["garments"]), 3)
        self.assertIn("full_logit", captured["loo_result"])
        self.assertIn("without_item_logits", captured["loo_result"])
        self.assertIn("deltas_without_minus_full", captured["loo_result"])
        self.assertTrue(closed["value"])


if __name__ == "__main__":
    unittest.main()
