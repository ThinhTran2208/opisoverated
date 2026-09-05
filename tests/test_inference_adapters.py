# -*- coding: utf-8 -*-

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.inference import DetectionAdapter, VLMAdapter


class _FakeCrop:
    def save(self, path, format=None):
        Path(path).write_bytes(b"fake-image")


class _FakeImage:
    def save(self, path, format=None):
        Path(path).write_bytes(b"fake-original-image")

    def crop(self, box):
        return _FakeCrop()


class _FakeDetectionResult:
    def __init__(self, garments):
        self.garments = garments

    def metadata_dict(self):
        return {"detection_version": "fake-detection-v1", "garment_count": len(self.garments)}


@unittest.skipIf(torch is None, "PyTorch is not installed in lightweight portability CI")
class InferenceAdapterTests(unittest.TestCase):
    def test_detection_adapter_builds_and_cleans_context(self):
        garments = []
        names = ["TOP", "BOTTOM", "SHOES"]
        category_ids = [1, 2, 5]
        for index, (name, category_id) in enumerate(zip(names, category_ids)):
            embedding = torch.zeros(512, dtype=torch.float32)
            embedding[index] = 1.0
            garments.append(
                SimpleNamespace(
                    candidate=SimpleNamespace(
                        detector_label=name.lower(),
                        detector_confidence=0.9,
                        box_xyxy=(0.0, 0.0, 10.0, 20.0),
                    ),
                    crop_box_xyxy=(0, 0, 10, 20),
                    category=SimpleNamespace(
                        coarse_category=name,
                        coarse_category_id=category_id,
                        similarity=0.8,
                        margin=0.2,
                    ),
                    embedding=embedding,
                )
            )

        class FakeDetectionPipeline:
            def run(self, image):
                return _FakeDetectionResult(garments), _FakeImage()

        context = DetectionAdapter(FakeDetectionPipeline()).prepare(object())
        crop_refs = [Path(value) for value in context.crop_image_refs]
        self.assertEqual(context.embeddings.shape, (3, 512))
        self.assertEqual(context.categories.tolist(), category_ids)
        self.assertEqual(context.coarse_categories, names)
        self.assertTrue(all(path.is_file() for path in crop_refs))
        original_ref = Path(context.original_image_ref)
        self.assertTrue(original_ref.is_file())
        context.close()
        self.assertTrue(all(not path.exists() for path in crop_refs))
        self.assertFalse(original_ref.exists())

    def test_vlm_adapter_receives_raw_loo_and_builds_vlm_evidence(self):
        captured = {}

        class FakeExplanationPipeline:
            def explain(self, evidence, image_refs):
                captured["evidence"] = evidence
                captured["image_refs"] = list(image_refs)
                return {"schema_version": "fake-vlm-output"}

        loo_result = {
            "protocol_version": "loo-diagnostic-v1",
            "original_item_count": 3,
            "full_logit": 0.0,
            "without_item_logits": [0.1, 0.2, 0.0],
            "deltas_without_minus_full": [0.1, 0.2, 0.0],
            "ranked_item_indices": [1, 0, 2],
            "problematic_item_index": 1,
            "problematic_item_id": "garment-1",
            "uses_two_item_extrapolation": True,
        }
        garments = [
            {"item_id": "garment-0", "coarse_category": "TOP"},
            {"item_id": "garment-1", "coarse_category": "BOTTOM"},
            {"item_id": "garment-2", "coarse_category": "SHOES"},
        ]
        crop_refs = [Path("crop-0.png"), Path("crop-1.png"), Path("crop-2.png")]

        result = VLMAdapter(FakeExplanationPipeline()).explain(
            loo_result,
            garments,
            crop_refs,
            sample_id="request-123",
        )
        self.assertEqual(result["schema_version"], "fake-vlm-output")
        self.assertEqual(captured["evidence"]["sample_id"], "request-123")
        self.assertEqual(
            captured["evidence"]["diagnosis"]["problematic_item_index"], 1
        )
        self.assertEqual(captured["image_refs"], crop_refs)


if __name__ == "__main__":
    unittest.main()
