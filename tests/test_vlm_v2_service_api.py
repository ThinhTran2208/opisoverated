# -*- coding: utf-8 -*-

import base64
import importlib.util
import unittest
from pathlib import Path


HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(HAS_FASTAPI, "FastAPI runtime dependencies are not installed")
class VLMV2ServiceAPITests(unittest.TestCase):
    def test_v2_reason_service_passes_full_and_target_images(self):
        from fastapi.testclient import TestClient
        from src.inference.vlm_http_api import create_app

        captured = {}

        class FakeAdapter:
            def explain_reason(
                self,
                *,
                target_item,
                original_image_ref,
                target_image_ref,
                must_exist,
            ):
                captured["target_item"] = target_item
                captured["original_exists"] = Path(original_image_ref).is_file()
                captured["target_exists"] = Path(target_image_ref).is_file()
                captured["must_exist"] = must_exist
                return "Chiếc áo hiện tại lệch tông với tổng thể outfit."

        class FakeRuntime:
            config_path = Path("fake-vlm-v2-config.json")
            loaded = False

            def get_adapter(self):
                return FakeAdapter()

        client = TestClient(create_app(FakeRuntime(), FakeRuntime()))
        encoded = base64.b64encode(b"fake-image-bytes").decode("ascii")
        response = client.post(
            "/v2/reason",
            json={
                "sample_id": "sample-v2-reason",
                "target_item": {
                    "item_index": 2,
                    "item_id": "garment-2",
                    "coarse_category": "OUTERWEAR",
                },
                "original_image": {"filename": "original.png", "base64": encoded},
                "target_image": {"filename": "target.png", "base64": encoded},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["reason"],
            "Chiếc áo hiện tại lệch tông với tổng thể outfit.",
        )
        self.assertEqual(captured["target_item"]["item_index"], 2)
        self.assertTrue(captured["original_exists"])
        self.assertTrue(captured["target_exists"])
        self.assertTrue(captured["must_exist"])

    def test_v2_service_rejects_non_top3_candidate_image_map(self):
        from fastapi.testclient import TestClient
        from src.inference.vlm_http_api import create_app

        class FakeRuntime:
            config_path = Path("fake-vlm-v2-config.json")
            loaded = False

            def get_adapter(self):
                raise AssertionError("invalid request must not load the model")

        client = TestClient(create_app(FakeRuntime(), FakeRuntime()))
        response = client.post(
            "/v2/explain",
            json={
                "sample_id": "sample-v2-1",
                "evidence": {"schema_version": "vlm-evidence-v2"},
                "outfit_images": [{"filename": "garment.png", "base64": "aA=="}],
                "recommendation_images": {
                    "item-a": {"filename": "item-a.jpg", "base64": "aA=="},
                },
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_v2_service_decodes_outfit_and_candidate_images(self):
        from fastapi.testclient import TestClient
        from src.inference.vlm_http_api import create_app

        captured = {}

        class FakeAdapter:
            def explain(
                self,
                evidence,
                outfit_image_refs,
                recommendation_image_refs,
                *,
                must_exist,
                original_image_ref=None,
            ):
                captured["evidence"] = evidence
                captured["outfit_exists"] = [
                    Path(value).is_file() for value in outfit_image_refs
                ]
                captured["recommendation_exists"] = {
                    key: Path(value).is_file()
                    for key, value in recommendation_image_refs.items()
                }
                captured["must_exist"] = must_exist
                captured["original_exists"] = (
                    original_image_ref is not None
                    and Path(original_image_ref).is_file()
                )
                return {"user_facing": {"schema_version": "vlm-user-facing-v2"}}

        class FakeRuntime:
            config_path = Path("fake-vlm-v2-config.json")
            loaded = False

            def get_adapter(self):
                self.loaded = True
                return FakeAdapter()

        client = TestClient(create_app(FakeRuntime(), FakeRuntime()))
        encoded = base64.b64encode(b"fake-image-bytes").decode("ascii")
        response = client.post(
            "/v2/explain",
            json={
                "sample_id": "sample-v2-1",
                "evidence": {"schema_version": "vlm-evidence-v2"},
                "original_image": {
                    "filename": "original-outfit.png",
                    "base64": encoded,
                },
                "outfit_images": [
                    {"filename": "garment-0.png", "base64": encoded},
                    {"filename": "garment-1.png", "base64": encoded},
                ],
                "recommendation_images": {
                    "item-a": {"filename": "item-a.jpg", "base64": encoded},
                    "item-b": {"filename": "item-b.jpg", "base64": encoded},
                    "item-c": {"filename": "item-c.jpg", "base64": encoded},
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["explanation"]["schema_version"],
            "vlm-user-facing-v2",
        )
        self.assertEqual(captured["outfit_exists"], [True, True])
        self.assertEqual(
            captured["recommendation_exists"],
            {"item-a": True, "item-b": True, "item-c": True},
        )
        self.assertTrue(captured["must_exist"])
        self.assertTrue(captured["original_exists"])

    def test_v2_service_keeps_unexpected_failures_json_shaped(self):
        from fastapi.testclient import TestClient
        from src.inference.vlm_http_api import create_app

        class FakeAdapter:
            def explain(self, *args, **kwargs):
                raise KeyError("simulated failure")

        class FakeRuntime:
            config_path = Path("fake-vlm-v2-config.json")
            loaded = False

            def get_adapter(self):
                return FakeAdapter()

        client = TestClient(create_app(FakeRuntime(), FakeRuntime()))
        encoded = base64.b64encode(b"fake-image-bytes").decode("ascii")
        response = client.post(
            "/v2/explain",
            json={
                "sample_id": "sample-v2-1",
                "evidence": {"schema_version": "vlm-evidence-v2"},
                "outfit_images": [{"filename": "garment.png", "base64": encoded}],
                "recommendation_images": {
                    "item-a": {"filename": "item-a.jpg", "base64": encoded},
                    "item-b": {"filename": "item-b.jpg", "base64": encoded},
                    "item-c": {"filename": "item-c.jpg", "base64": encoded},
                },
            },
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "internal_server_error")


if __name__ == "__main__":
    unittest.main()
