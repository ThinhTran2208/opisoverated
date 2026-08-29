# -*- coding: utf-8 -*-
"""Production inference boundary for deploy-facing outfit analysis."""

from .pipeline import (
    PIPELINE_VERSION,
    InferenceInputError,
    ProductionInferencePipeline,
)

__all__ = [
    "PIPELINE_VERSION",
    "InferenceInputError",
    "ProductionInferencePipeline",
]
