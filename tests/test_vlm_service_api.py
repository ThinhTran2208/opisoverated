# -*- coding: utf-8 -*-

import base64
import importlib.util
import unittest
from pathlib import Path


HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(HAS_FASTAPI, "FastAPI runtime dependencies are not installed")
class VLMServiceAPITests(unittest.TestCase):
    def test_service_decodes_crops_and_calls_adapter(self):
        from fastapi.testclient import TestClient
        from src.inference.vlm_http_api import create_app

        captured = {}

        class FakeAdapter:
            def explain(self, loo_result, garments, crop_image_refs, *, sample_id):
                captured["loo_result"] = loo_result
                captured["garments"] = list(garments)
                captured["sample_id"] = sample_id
                captured["crop_exists"] = [Path(value).is_file() for value in crop_image_refs]
                return {"schema_version": "fake-vlm", "headline": "explanation"}

        class FakeRuntime:
            config_path = Path("fake-vlm-config.json")
            loaded = False

            def get_adapter(self):
                self.loaded = True
                return FakeAdapter()

        runtime = FakeRuntime()
        client = TestClient(create_app(runtime))
        encoded = base64.b64encode(b"fake-image-bytes").decode("ascii")
        response = client.post(
            "/v1/explain",
            json={
                "sample_id": "sample-1",
                "loo_result": {"protocol_version": "loo-diagnostic-v1"},
                "garments": [
                    {"item_id": "garment-0", "coarse_category": "TOP"},
                    {"item_id": "garment-1", "coarse_category": "BOTTOM"},
                    {"item_id": "garment-2", "coarse_category": "SHOES"},
                ],
                "crop_images": [
                    {"filename": f"crop-{index}.png", "base64": encoded}
                    for index in range(3)
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["explanation"]["headline"], "explanation")
        self.assertEqual(captured["sample_id"], "sample-1")
        self.assertEqual(captured["crop_exists"], [True, True, True])

    def test_service_rejects_crop_count_mismatch(self):
        from fastapi.testclient import TestClient
        from src.inference.vlm_http_api import create_app

        class FakeRuntime:
            config_path = Path("fake-vlm-config.json")
            loaded = False

            def get_adapter(self):
                raise AssertionError("invalid request must not load the model")

        client = TestClient(create_app(FakeRuntime()))
        response = client.post(
            "/v1/explain",
            json={
                "sample_id": "sample-1",
                "loo_result": {},
                "garments": [{"item_id": "garment-0"}],
                "crop_images": [],
            },
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
