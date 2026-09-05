# -*- coding: utf-8 -*-
"""Dedicated HTTP service for Qwen VLM Explanation V1/V2.

This service intentionally lives in a separate Python environment from RF-DETR.
The model is loaded lazily on the first explanation request so health checks and
container startup do not require an immediate model download.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Mapping

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
except ModuleNotFoundError as error:  # pragma: no cover - runtime dependency.
    raise RuntimeError(
        "FastAPI runtime dependencies are missing; install requirements-runtime.txt"
    ) from error

from .adapters import VLMAdapter


LOGGER = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
VLM_CONFIG_PATH = Path(
    os.environ.get(
        "FASHION_VLM_CONFIG",
        str(REPO_ROOT / "configs" / "vlm_qwen3_vl_4b_instruct_v1.json"),
    )
)
VLM_V2_CONFIG_PATH = Path(
    os.environ.get(
        "FASHION_VLM_V2_CONFIG",
        str(REPO_ROOT / "configs" / "vlm_qwen3_vl_4b_instruct_v2.json"),
    )
)
MAX_CROP_BYTES = int(os.environ.get("FASHION_VLM_MAX_CROP_BYTES", str(5 * 1024 * 1024)))


class LazyVLMRuntime:
    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self._adapter = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._adapter is not None

    def get_adapter(self):
        if self._adapter is None:
            with self._lock:
                if self._adapter is None:
                    self._adapter = VLMAdapter.from_config(self.config_path)
        return self._adapter


class LazyVLMRuntimeV2:
    """Lazy loader for the canonical VLM V2 pipeline."""

    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self._adapter = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._adapter is not None

    def get_adapter(self):
        if self._adapter is None:
            with self._lock:
                if self._adapter is None:
                    from src.vlm import (
                        Qwen3VLBackendV2,
                        VLMExplanationPipelineV2,
                        load_vlm_config_v2,
                    )

                    config = load_vlm_config_v2(self.config_path)
                    backend = Qwen3VLBackendV2.from_config(config)
                    self._adapter = VLMExplanationPipelineV2(backend, config)
        return self._adapter


runtime = LazyVLMRuntime(VLM_CONFIG_PATH)
runtime_v2 = LazyVLMRuntimeV2(VLM_V2_CONFIG_PATH)


def _decode_crop(value: object, *, index: int, directory: Path) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"crop_images[{index}] must be an object")
    encoded = value.get("base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"crop_images[{index}].base64 must be non-empty")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"crop_images[{index}] is not valid base64") from error
    if not raw:
        raise ValueError(f"crop_images[{index}] decoded to an empty file")
    if len(raw) > MAX_CROP_BYTES:
        raise ValueError(
            f"crop_images[{index}] exceeds maximum size {MAX_CROP_BYTES} bytes"
        )

    filename = value.get("filename")
    suffix = Path(str(filename)).suffix.lower() if filename else ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    path = directory / f"crop-{index:02d}{suffix}"
    path.write_bytes(raw)
    return path


def create_app(
    vlm_runtime: LazyVLMRuntime,
    vlm_runtime_v2: LazyVLMRuntimeV2 | None = None,
) -> FastAPI:
    application = FastAPI(title="Outfit VLM Explanation", version="vlm-explanation-v2")

    @application.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "service": "vlm-explanation",
            "model_loaded": vlm_runtime.loaded,
            "config_path": str(vlm_runtime.config_path),
            "v2_available": vlm_runtime_v2 is not None,
            "v2_config_path": (
                None if vlm_runtime_v2 is None else str(vlm_runtime_v2.config_path)
            ),
            "v2_model_loaded": (
                False if vlm_runtime_v2 is None else vlm_runtime_v2.loaded
            ),
        }

    @application.post("/v1/explain")
    def explain(payload: dict):
        sample_id = payload.get("sample_id")
        loo_result = payload.get("loo_result")
        garments = payload.get("garments")
        crop_images = payload.get("crop_images")

        if not isinstance(sample_id, str) or not sample_id.strip():
            return JSONResponse(
                status_code=422,
                content={"status": "error", "error": "sample_id must be non-empty"},
            )
        if not isinstance(loo_result, Mapping):
            return JSONResponse(
                status_code=422,
                content={"status": "error", "error": "loo_result must be an object"},
            )
        if not isinstance(garments, list):
            return JSONResponse(
                status_code=422,
                content={"status": "error", "error": "garments must be a list"},
            )
        if not isinstance(crop_images, list) or len(crop_images) != len(garments):
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "error": "crop_images must contain exactly one image per garment",
                },
            )

        try:
            with tempfile.TemporaryDirectory(prefix="vlm-crops-") as directory:
                root = Path(directory)
                crop_refs = [
                    _decode_crop(value, index=index, directory=root)
                    for index, value in enumerate(crop_images)
                ]
                adapter = vlm_runtime.get_adapter()
                explanation = adapter.explain(
                    loo_result,
                    garments,
                    crop_refs,
                    sample_id=sample_id,
                )
        except (TypeError, ValueError, FileNotFoundError) as error:
            return JSONResponse(
                status_code=422,
                content={"status": "error", "error": str(error)},
            )
        except (ImportError, OSError, RuntimeError) as error:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "error": str(error)},
            )
        except Exception as error:  # defensive: never return plain-text 500s.
            LOGGER.exception("Unhandled VLM request failure")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error": {
                        "code": "internal_server_error",
                        "message": "VLM request failed; check the VLM service log",
                        "details": {"exception": type(error).__name__},
                    },
                },
            )

        return {"status": "ok", "explanation": explanation}

    @application.post("/v2/explain")
    def explain_v2(payload: dict):
        if vlm_runtime_v2 is None:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "error": "VLM V2 is not configured"},
            )

        sample_id = payload.get("sample_id")
        evidence = payload.get("evidence")
        outfit_images = payload.get("outfit_images")
        recommendation_images = payload.get("recommendation_images")
        if not isinstance(sample_id, str) or not sample_id.strip():
            return JSONResponse(
                status_code=422,
                content={"status": "error", "error": "sample_id must be non-empty"},
            )
        if not isinstance(evidence, Mapping):
            return JSONResponse(
                status_code=422,
                content={"status": "error", "error": "evidence must be an object"},
            )
        evidence_sample_id = evidence.get("sample_id")
        if evidence_sample_id is not None and str(evidence_sample_id) != sample_id:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "error": "evidence.sample_id must match sample_id",
                },
            )
        if not isinstance(outfit_images, list) or not outfit_images:
            return JSONResponse(
                status_code=422,
                content={"status": "error", "error": "outfit_images must be a non-empty list"},
            )
        if not isinstance(recommendation_images, Mapping):
            return JSONResponse(
                status_code=422,
                content={"status": "error", "error": "recommendation_images must be an object"},
            )
        if len(recommendation_images) != 3:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "error": "recommendation_images must contain exactly three items",
                },
            )

        try:
            with tempfile.TemporaryDirectory(prefix="vlm-v2-images-") as directory:
                root = Path(directory)
                outfit_refs = [
                    _decode_crop(value, index=index, directory=root)
                    for index, value in enumerate(outfit_images)
                ]
                recommendation_refs: dict[str, Path] = {}
                for index, (item_id, value) in enumerate(recommendation_images.items()):
                    if not isinstance(item_id, str) or not item_id.strip():
                        raise ValueError("recommendation image keys must be non-empty strings")
                    recommendation_refs[item_id] = _decode_crop(
                        value, index=len(outfit_refs) + index, directory=root
                    )

                run = vlm_runtime_v2.get_adapter().explain(
                    evidence,
                    outfit_refs,
                    recommendation_refs,
                    must_exist=True,
                )
        except (TypeError, ValueError, FileNotFoundError) as error:
            return JSONResponse(
                status_code=422,
                content={"status": "error", "error": str(error)},
            )
        except RuntimeError as error:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "error": str(error)},
            )
        except Exception as error:  # defensive: never return plain-text 500s.
            LOGGER.exception("Unhandled VLM V2 request failure")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error": {
                        "code": "internal_server_error",
                        "message": "VLM V2 request failed; check the VLM service log",
                        "details": {"exception": type(error).__name__},
                    },
                },
            )

        if not isinstance(run, Mapping) or "user_facing" not in run:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "error": "VLM V2 returned an invalid run"},
            )
        return {
            "status": "ok",
            "protocol_version": "vlm-explanation-v2",
            "explanation": run["user_facing"],
        }

    return application


app = create_app(runtime, runtime_v2)
