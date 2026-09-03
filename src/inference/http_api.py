# -*- coding: utf-8 -*-
"""Deploy-facing inference-core HTTP API.

The core process owns Detection V1, FashionCLIP, the frozen scorer, Calibration V1,
and LOO.  Visual explanation is reached through ``RemoteVLMAdapter`` so the
RF-DETR/Transformers-v5 runtime does not share a Python environment with the
Qwen/Transformers-v4 runtime.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse
except ModuleNotFoundError as error:  # pragma: no cover - runtime dependency.
    raise RuntimeError(
        "FastAPI runtime dependencies are missing; install requirements-runtime.txt"
    ) from error

from .adapters import DetectionAdapter, RemoteVLMAdapter, VLMServiceError
from .pipeline import InferenceInputError, ProductionInferencePipeline


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(
    os.environ.get(
        "FASHION_INFERENCE_MANIFEST",
        str(REPO_ROOT / "configs" / "production_inference_v1.json"),
    )
)
DETECTION_CONFIG_PATH = Path(
    os.environ.get(
        "FASHION_DETECTION_CONFIG",
        str(REPO_ROOT / "configs" / "detection_rfdetr_fashionclip_core7_v1.json"),
    )
)
DEVICE = os.environ.get("FASHION_INFERENCE_DEVICE", "cpu")
DETECTION_DEVICE = os.environ.get("FASHION_DETECTION_DEVICE", DEVICE)
VLM_SERVICE_URL = os.environ.get("FASHION_VLM_SERVICE_URL", "").strip()
REQUIRE_VLM = os.environ.get("FASHION_REQUIRE_VLM", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
MAX_UPLOAD_BYTES = int(os.environ.get("FASHION_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
VLM_TIMEOUT_SECONDS = float(os.environ.get("FASHION_VLM_TIMEOUT_SECONDS", "180"))


def build_default_pipeline() -> ProductionInferencePipeline:
    detection_adapter = DetectionAdapter.from_config(
        DETECTION_CONFIG_PATH,
        device=DETECTION_DEVICE,
    )
    vlm_adapter = None
    if VLM_SERVICE_URL:
        vlm_adapter = RemoteVLMAdapter(
            VLM_SERVICE_URL,
            timeout_seconds=VLM_TIMEOUT_SECONDS,
        )
    elif REQUIRE_VLM:
        raise RuntimeError(
            "FASHION_REQUIRE_VLM=true but FASHION_VLM_SERVICE_URL is not configured"
        )

    return ProductionInferencePipeline.load_from_manifest(
        MANIFEST_PATH,
        repo_root=REPO_ROOT,
        device=DEVICE,
        detection_adapter=detection_adapter,
        vlm_adapter=vlm_adapter,
    )


def create_app(pipeline: ProductionInferencePipeline) -> FastAPI:
    application = FastAPI(
        title="Outfit Production Inference V1",
        version=pipeline.pipeline_version,
    )

    @application.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "versions": pipeline.versions,
            "image_endpoint": True,
            "detection_adapter": pipeline.detection_adapter is not None,
            "vlm_adapter": pipeline.vlm_adapter is not None,
        }

    @application.post("/v1/analyze-precomputed")
    def analyze_precomputed(payload: dict):
        items = payload.get("items")
        result = pipeline.analyze_precomputed_safe(items)
        if result["status"] == "error":
            return JSONResponse(status_code=422, content=result)
        return result

    @application.post("/v1/analyze-outfit")
    async def analyze_outfit(image: UploadFile = File(...)):
        content_type = (image.content_type or "").lower()
        if content_type and not content_type.startswith("image/"):
            return JSONResponse(
                status_code=415,
                content={
                    "status": "error",
                    "error": {
                        "code": "unsupported_media_type",
                        "message": "Upload must be an image",
                        "details": {"content_type": content_type},
                    },
                    "versions": pipeline.versions,
                },
            )

        raw = await image.read(MAX_UPLOAD_BYTES + 1)
        if not raw:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "error": {
                        "code": "empty_upload",
                        "message": "Uploaded image is empty",
                        "details": {},
                    },
                    "versions": pipeline.versions,
                },
            )
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "status": "error",
                    "error": {
                        "code": "image_too_large",
                        "message": "Uploaded image exceeds the configured size limit",
                        "details": {"maximum_bytes": MAX_UPLOAD_BYTES},
                    },
                    "versions": pipeline.versions,
                },
            )

        try:
            from PIL import Image, UnidentifiedImageError

            pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
        except (ModuleNotFoundError, UnidentifiedImageError, OSError, ValueError) as error:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "error": {
                        "code": "invalid_image",
                        "message": f"Could not decode uploaded image: {error}",
                        "details": {},
                    },
                    "versions": pipeline.versions,
                },
            )

        try:
            result = pipeline.analyze_image_safe(pil_image)
        except VLMServiceError as error:
            return JSONResponse(
                status_code=502,
                content={
                    "status": "error",
                    "error": {
                        "code": "vlm_service_error",
                        "message": str(error),
                        "details": {},
                    },
                    "versions": pipeline.versions,
                },
            )
        except RuntimeError as error:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "error": {
                        "code": "inference_runtime_error",
                        "message": str(error),
                        "details": {},
                    },
                    "versions": pipeline.versions,
                },
            )
        except InferenceInputError as error:  # defensive: safe path normally catches this.
            result = {
                "status": "error",
                "error": error.to_dict(),
                "versions": pipeline.versions,
            }

        if result["status"] == "error":
            return JSONResponse(status_code=422, content=result)
        if REQUIRE_VLM and "explanation" not in result:
            return JSONResponse(
                status_code=502,
                content={
                    "status": "error",
                    "error": {
                        "code": "vlm_explanation_missing",
                        "message": "VLM explanation is required but was not returned",
                        "details": {},
                    },
                    "versions": pipeline.versions,
                },
            )
        return result

    return application


pipeline = build_default_pipeline()
app = create_app(pipeline)
