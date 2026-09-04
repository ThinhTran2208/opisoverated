"""Grounded VLM explanation layer for frozen scorer + LOO evidence."""

from .config import CANONICAL_MODEL_ID, VLM_PROTOCOL_VERSION, load_vlm_config
from .pipeline import (
    VLMExplanationPipeline,
    render_explanation_vi,
    validate_visual_analysis,
)
from .schema import (
    EVIDENCE_SCHEMA_VERSION,
    build_vlm_evidence,
    validate_vlm_evidence,
)
from .schema_v2 import (
    CANONICAL_RECOMMENDATION_VERSION,
    EVIDENCE_SCHEMA_VERSION_V2,
    GROUNDING_RULES_V2,
    build_recommendation_evidence,
    build_vlm_evidence_v2,
    canonical_evidence_json_v2,
    validate_recommendation_evidence,
    validate_vlm_evidence_v2,
)

__all__ = [
    "CANONICAL_MODEL_ID",
    "CANONICAL_RECOMMENDATION_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "EVIDENCE_SCHEMA_VERSION_V2",
    "GROUNDING_RULES_V2",
    "VLMExplanationPipeline",
    "VLM_PROTOCOL_VERSION",
    "build_recommendation_evidence",
    "build_vlm_evidence",
    "build_vlm_evidence_v2",
    "canonical_evidence_json_v2",
    "load_vlm_config",
    "render_explanation_vi",
    "validate_recommendation_evidence",
    "validate_vlm_evidence",
    "validate_vlm_evidence_v2",
    "validate_visual_analysis",
]
