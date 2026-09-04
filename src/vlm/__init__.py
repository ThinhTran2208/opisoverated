"""Grounded VLM explanation layer for frozen scorer + LOO evidence."""

from .config import CANONICAL_MODEL_ID, VLM_PROTOCOL_VERSION, load_vlm_config
from .config_v2 import (
    VLM_PROTOCOL_VERSION_V2,
    load_vlm_config_v2,
    validate_vlm_config_v2,
)
from .pipeline import (
    VLMExplanationPipeline,
    render_explanation_vi,
    validate_visual_analysis,
)
from .pipeline_v2 import (
    EXPLANATION_SCHEMA_VERSION_V2,
    HANDOFF_SCHEMA_VERSION_V2,
    RUN_SCHEMA_VERSION_V2,
    VLMExplanationPipelineV2,
    build_handoff_result_v2,
    render_explanation_vi_v2,
    validate_visual_analysis_v2,
)
from .prompt_v2 import (
    VISUAL_ANALYSIS_SCHEMA_VERSION_V2,
    append_repair_request_v2,
    build_qwen_messages_v2,
    expected_output_shape_v2,
    required_limitations_v2,
)
from .qwen_backend_v2 import Qwen3VLBackendV2
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
from .user_renderer_v2 import (
    USER_FACING_SCHEMA_VERSION_V2,
    render_user_facing_vi_v2,
)

__all__ = [
    "CANONICAL_MODEL_ID",
    "CANONICAL_RECOMMENDATION_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "EVIDENCE_SCHEMA_VERSION_V2",
    "EXPLANATION_SCHEMA_VERSION_V2",
    "GROUNDING_RULES_V2",
    "HANDOFF_SCHEMA_VERSION_V2",
    "Qwen3VLBackendV2",
    "RUN_SCHEMA_VERSION_V2",
    "USER_FACING_SCHEMA_VERSION_V2",
    "VISUAL_ANALYSIS_SCHEMA_VERSION_V2",
    "VLMExplanationPipeline",
    "VLMExplanationPipelineV2",
    "VLM_PROTOCOL_VERSION",
    "VLM_PROTOCOL_VERSION_V2",
    "append_repair_request_v2",
    "build_handoff_result_v2",
    "build_qwen_messages_v2",
    "build_recommendation_evidence",
    "build_vlm_evidence",
    "build_vlm_evidence_v2",
    "canonical_evidence_json_v2",
    "expected_output_shape_v2",
    "load_vlm_config",
    "load_vlm_config_v2",
    "render_explanation_vi",
    "render_explanation_vi_v2",
    "render_user_facing_vi_v2",
    "required_limitations_v2",
    "validate_recommendation_evidence",
    "validate_vlm_config_v2",
    "validate_vlm_evidence",
    "validate_vlm_evidence_v2",
    "validate_visual_analysis",
    "validate_visual_analysis_v2",
]
