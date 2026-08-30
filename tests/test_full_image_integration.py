# -*- coding: utf-8 -*-

import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.inference import DetectionAdapter, ProductionInferencePipeline, VLMAdapter


class _FakeCrop:
    def save(self, path, format=None):
        Path(path).write_bytes(b"fake-crop")


class _FakeImage:
    def crop(self, box):
        return _FakeCrop()


class _FakeDetectionResult:
    def __init__(self, garments):
        self.garments = garments

    def metadata_dict(self):
        return {
            "detection_version": "contract-e2e-detection-v1",
            "garment_count": len(self.garments),
            "detector": {"runtime_ms": 1.0},
        }


@unittest.skipIf(torch is None, "PyTorch is not installed in lightweight portability CI")
class FullImageIntegrationTests(unittest.TestCase):
    def test_image_detection_scorer_calibration_loo_vlm_contract(self):
        repo_root = Path(__file__).resolve().parents[1]
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

        captured = {}

        class FakeExplanationPipeline:
            def explain(self, evidence, image_refs):
                captured["evidence"] = evidence
                captured["image_refs"] = list(image_refs)
                captured["refs_exist_during_vlm"] = [
                    Path(value).is_file() for value in image_refs
                ]
                return {
                    "schema_version": "contract-e2e-vlm-v1",
                    "problematic_item_index": evidence["diagnosis"][
                        "problematic_item_index"
                    ],
                }

        pipeline = ProductionInferencePipeline.load_from_manifest(
            repo_root / "configs" / "production_inference_v1.json",
            repo_root=repo_root,
            device="cpu",
            detection_adapter=DetectionAdapter(FakeDetectionPipeline()),
            vlm_adapter=VLMAdapter(FakeExplanationPipeline()),
        )
        result = pipeline.analyze_image(object())

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["item_count"], 3)
        self.assertTrue(0 <= result["compatibility"]["compatibility_score"] <= 100)
        self.assertIn(result["diagnosis"]["problematic_item_index"], [0, 1, 2])
        self.assertEqual(
            result["explanation"]["problematic_item_index"],
            result["diagnosis"]["problematic_item_index"],
        )
        self.assertEqual(captured["refs_exist_during_vlm"], [True, True, True])
        self.assertEqual(
            captured["evidence"]["diagnosis"]["problematic_item_index"],
            result["diagnosis"]["problematic_item_index"],
        )
        self.assertEqual(
            captured["evidence"]["scorer"]["compatibility_logit"],
            result["compatibility"]["compatibility_logit"],
        )


if __name__ == "__main__":
    unittest.main()
