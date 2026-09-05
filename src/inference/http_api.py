# -*- coding: utf-8 -*-
"""Deploy-facing inference-core HTTP API.

The core process owns Detection V1, FashionCLIP, the frozen scorer, Calibration V1,
LOO, and optional Recommendation V2. Visual explanation is reached through a
remote adapter so the RF-DETR/Transformers-v5 runtime does not share a Python
environment with the Qwen/Transformers-v4 runtime.
"""

from __future__ import annotations

import io
import os
import asyncio
from pathlib import Path

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
except ModuleNotFoundError as error:  # pragma: no cover - runtime dependency.
    raise RuntimeError(
        "FastAPI runtime dependencies are missing; install requirements-runtime.txt"
    ) from error

from .adapters import (
    DetectionAdapter,
    RemoteVLMAdapter,
    RemoteVLMAdapterV2,
    VLMServiceError,
)
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
VLM_PROTOCOL = os.environ.get("FASHION_VLM_PROTOCOL", "v1").strip().lower()
RECOMMENDATION_CONFIG_PATH = Path(
    os.environ.get(
        "FASHION_RECOMMENDATION_CONFIG",
        str(REPO_ROOT / "configs" / "recommendation_category_aware_v2.json"),
    )
)
RECOMMENDATION_ARTIFACT_ROOT = os.environ.get(
    "FASHION_RECOMMENDATION_ARTIFACT_ROOT", ""
).strip()
RECOMMENDATION_IMAGE_ROOT = os.environ.get(
    "FASHION_RECOMMENDATION_IMAGE_ROOT", ""
).strip()
MAX_PENDING_REQUESTS = int(os.environ.get("FASHION_MAX_PENDING_REQUESTS", "3"))
if MAX_PENDING_REQUESTS < 1:
    raise ValueError("FASHION_MAX_PENDING_REQUESTS must be >= 1")
CORS_ORIGINS = tuple(
    origin.strip()
    for origin in os.environ.get("FASHION_CORS_ORIGINS", "").split(",")
    if origin.strip()
)


def build_default_pipeline() -> ProductionInferencePipeline:
    detection_adapter = DetectionAdapter.from_config(
        DETECTION_CONFIG_PATH,
        device=DETECTION_DEVICE,
    )
    recommendation_pipeline = None
    if RECOMMENDATION_ARTIFACT_ROOT or RECOMMENDATION_IMAGE_ROOT:
        if not RECOMMENDATION_ARTIFACT_ROOT or not RECOMMENDATION_IMAGE_ROOT:
            raise RuntimeError(
                "Recommendation V2 requires both FASHION_RECOMMENDATION_ARTIFACT_ROOT "
                "and FASHION_RECOMMENDATION_IMAGE_ROOT"
            )
        from src.recommendation import RecommendationPipeline

        recommendation_pipeline = RecommendationPipeline.load_from_directories(
            RECOMMENDATION_CONFIG_PATH,
            artifact_root=RECOMMENDATION_ARTIFACT_ROOT,
            image_root=RECOMMENDATION_IMAGE_ROOT,
            device=DEVICE,
        )

    vlm_adapter = None
    if VLM_SERVICE_URL:
        if VLM_PROTOCOL == "v2":
            if recommendation_pipeline is None:
                raise RuntimeError(
                    "FASHION_VLM_PROTOCOL=v2 requires Recommendation V2 artifacts"
                )
            vlm_adapter = RemoteVLMAdapterV2(
                VLM_SERVICE_URL,
                image_resolver=recommendation_pipeline.image_resolver,
                timeout_seconds=VLM_TIMEOUT_SECONDS,
            )
        elif VLM_PROTOCOL == "v1":
            vlm_adapter = RemoteVLMAdapter(
                VLM_SERVICE_URL,
                timeout_seconds=VLM_TIMEOUT_SECONDS,
            )
        else:
            raise RuntimeError("FASHION_VLM_PROTOCOL must be 'v1' or 'v2'")
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
        recommendation_provider=recommendation_pipeline,
    )


def create_app(pipeline: ProductionInferencePipeline) -> FastAPI:
    application = FastAPI(
        title="Outfit Production Inference",
        version=pipeline.pipeline_version,
    )
    if CORS_ORIGINS:
        allow_all_origins = CORS_ORIGINS == ("*",)
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(CORS_ORIGINS),
            allow_credentials=not allow_all_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    try:
        from src.recommendation.http_api import create_image_router
    except ModuleNotFoundError:  # pragma: no cover - lightweight runtime.
        create_image_router = None
    recommendation_provider = getattr(pipeline, "recommendation_provider", None)
    recommendation_resolver = getattr(recommendation_provider, "image_resolver", None)
    if create_image_router is not None and recommendation_resolver is not None:
        application.include_router(
            create_image_router(recommendation_resolver)
        )

    inference_semaphore = asyncio.Semaphore(1)
    pending_requests = 0
    frontend_path = REPO_ROOT / "frontend" / "index.html"

    @application.get("/", include_in_schema=False)
    def frontend():
        if not frontend_path.is_file():
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Frontend is not installed"},
            )
        return FileResponse(frontend_path, media_type="text/html")

    @application.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "versions": pipeline.versions,
            "image_endpoint": recommendation_resolver is not None,
            "detection_adapter": pipeline.detection_adapter is not None,
            "vlm_adapter": pipeline.vlm_adapter is not None,
            "recommendation_v2": recommendation_provider is not None,
            "max_pending_requests": MAX_PENDING_REQUESTS,
        }

    @application.post("/v1/analyze-precomputed")
    def analyze_precomputed(payload: dict):
        items = payload.get("items")
        result = pipeline.analyze_precomputed_safe(items)
        if result["status"] == "error":
            return JSONResponse(status_code=422, content=result)
        return result

    @application.post("/v1/analyze-outfit")
    @application.post("/v2/analyze-outfit")
    async def analyze_outfit(image: UploadFile = File(...)):
        nonlocal pending_requests
        if pending_requests >= MAX_PENDING_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "error": {
                        "code": "server_busy",
                        "message": "The inference service is busy; please retry shortly",
                        "details": {"maximum_pending_requests": MAX_PENDING_REQUESTS},
                    },
                    "versions": pipeline.versions,
                },
            )
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

        pending_requests += 1
        try:
            from starlette.concurrency import run_in_threadpool

            async with inference_semaphore:
                result = await run_in_threadpool(
                    pipeline.analyze_image_safe,
                    pil_image,
                )
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
        finally:
            pending_requests -= 1

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
