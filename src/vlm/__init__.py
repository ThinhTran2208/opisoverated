"""Grounded VLM explanation layer for frozen scorer + LOO evidence."""

from .config import CANONICAL_MODEL_ID, VLM_PROTOCOL_VERSION, load_vlm_config
from .pipeline import VLMExplanationPipeline, validate_explanation
from .schema import (
    EVIDENCE_SCHEMA_VERSION,
    build_vlm_evidence,
    validate_vlm_evidence,
)

__all__ = [
    "CANONICAL_MODEL_ID",
    "EVIDENCE_SCHEMA_VERSION",
    "VLMExplanationPipeline",
    "VLM_PROTOCOL_VERSION",
    "build_vlm_evidence",
    "load_vlm_config",
    "validate_explanation",
    "validate_vlm_evidence",
]
