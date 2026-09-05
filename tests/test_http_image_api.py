# -*- coding: utf-8 -*-

import importlib.util
import unittest
from pathlib import Path


HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(HAS_FASTAPI, "FastAPI runtime dependencies are not installed")
class ImageHTTPAPITests(unittest.TestCase):
    def test_analyze_outfit_accepts_image_upload(self):
        from fastapi.testclient import TestClient
        from src.inference.http_api import create_app

        captured = {}

        class FakePipeline:
            pipeline_version = "outfit-production-inference-v1"
            versions = {"pipeline_version": pipeline_version}
            detection_adapter = object()
            vlm_adapter = object()

            def analyze_image_safe(self, image):
                captured["size"] = image.size
                return {
                    "status": "ok",
                    "request_id": "request-http-1",
                    "item_count": 3,
                    "items": [],
                    "compatibility": {"compatibility_score": 75},
                    "diagnosis": {"problematic_item_index": 1},
                    "explanation": {"headline": "ok"},
                    "versions": self.versions,
                }

            def analyze_precomputed_safe(self, items):
                return {"status": "ok", "items": items, "versions": self.versions}

        client = TestClient(create_app(FakePipeline()))
        image_path = Path(__file__).resolve().parent / "animage.jpg"
        response = client.post(
            "/v1/analyze-outfit",
            files={"image": ("outfit.jpg", image_path.read_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["explanation"]["headline"], "ok")
        self.assertGreater(captured["size"][0], 0)
        self.assertGreater(captured["size"][1], 0)

    def test_analyze_outfit_rejects_non_image_upload(self):
        from fastapi.testclient import TestClient
        from src.inference.http_api import create_app

        class FakePipeline:
            pipeline_version = "outfit-production-inference-v1"
            versions = {"pipeline_version": pipeline_version}
            detection_adapter = object()
            vlm_adapter = object()

            def analyze_image_safe(self, image):
                raise AssertionError("non-image upload must not reach inference")

            def analyze_precomputed_safe(self, items):
                return {"status": "ok", "items": items, "versions": self.versions}

        client = TestClient(create_app(FakePipeline()))
        response = client.post(
            "/v1/analyze-outfit",
            files={"image": ("not-image.txt", b"hello", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["error"]["code"], "unsupported_media_type")

    def test_analyze_outfit_keeps_unexpected_failures_json_shaped(self):
        from fastapi.testclient import TestClient
        from src.inference.http_api import create_app

        class FakePipeline:
            pipeline_version = "outfit-production-inference-v1"
            versions = {"pipeline_version": pipeline_version}
            detection_adapter = object()
            vlm_adapter = object()

            def analyze_image_safe(self, image):
                raise KeyError("simulated failure")

            def analyze_precomputed_safe(self, items):
                return {"status": "ok", "items": items, "versions": self.versions}

        client = TestClient(create_app(FakePipeline()))
        image_path = Path(__file__).resolve().parent / "animage.jpg"
        response = client.post(
            "/v1/analyze-outfit",
            files={"image": ("outfit.jpg", image_path.read_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "internal_server_error")


if __name__ == "__main__":
    unittest.main()
