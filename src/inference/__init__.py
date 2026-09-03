# -*- coding: utf-8 -*-
"""Production inference boundary for deploy-facing outfit analysis."""

from .adapters import (
    DetectionAdapter,
    ExplanationProvider,
    GarmentPreprocessor,
    RemoteVLMAdapter,
    VLMAdapter,
    VLMServiceError,
)
from .context import InferenceContext
from .pipeline import (
    PIPELINE_VERSION,
    InferenceInputError,
    ProductionInferencePipeline,
)

__all__ = [
    "PIPELINE_VERSION",
    "DetectionAdapter",
    "ExplanationProvider",
    "GarmentPreprocessor",
    "InferenceContext",
    "InferenceInputError",
    "ProductionInferencePipeline",
    "RemoteVLMAdapter",
    "VLMAdapter",
    "VLMServiceError",
]
