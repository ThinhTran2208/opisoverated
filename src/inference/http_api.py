# -*- coding: utf-8 -*-
"""Minimal deploy-facing HTTP wrapper for Production Inference V1.

This endpoint accepts precomputed garment embeddings while detection/FashionCLIP
image preprocessing is still being integrated.  The stable ML response schema
is already the same scorer -> calibration -> diagnosis contract that the final
image endpoint will use.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
except ModuleNotFoundError as error:  # pragma: no cover - runtime dependency.
    raise RuntimeError(
        "FastAPI runtime dependencies are missing; install requirements-runtime.txt"
    ) from error

from .pipeline import ProductionInferencePipeline


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(
    os.environ.get(
        "FASHION_INFERENCE_MANIFEST",
        str(REPO_ROOT / "configs" / "production_inference_v1.json"),
    )
)
DEVICE = os.environ.get("FASHION_INFERENCE_DEVICE", "cpu")

pipeline = ProductionInferencePipeline.load_from_manifest(
    MANIFEST_PATH,
    repo_root=REPO_ROOT,
    device=DEVICE,
)

app = FastAPI(title="Outfit Production Inference V1", version=pipeline.pipeline_version)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "versions": pipeline.versions}


@app.post("/v1/analyze-precomputed")
def analyze_precomputed(payload: dict):
    items = payload.get("items")
    result = pipeline.analyze_precomputed_safe(items)
    if result["status"] == "error":
        return JSONResponse(status_code=422, content=result)
    return result
