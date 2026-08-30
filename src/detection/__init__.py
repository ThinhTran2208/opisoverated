"""Garment detection and Core-7 inference taxonomy utilities."""

from .config import (
    CATEGORY_CLASSIFIER_VERSION,
    CORE7_CATEGORIES,
    CORE7_CATEGORY_TO_ID,
    DETECTION_VERSION,
    DetectionConfig,
    load_detection_config,
)
from .pipeline import DetectionPipeline, build_scorer_batch_lists

__all__ = [
    "CATEGORY_CLASSIFIER_VERSION",
    "CORE7_CATEGORIES",
    "CORE7_CATEGORY_TO_ID",
    "DETECTION_VERSION",
    "DetectionConfig",
    "DetectionPipeline",
    "build_scorer_batch_lists",
    "load_detection_config",
]
