# -*- coding: utf-8 -*-
"""VLM V2 validation, internal rendering, and safe handoff pipeline.

Qwen is restricted to closed-taxonomy visual evidence. Frozen scorer + LOO +
Recommendation V2 remain authoritative for every numerical decision, candidate
identity, and candidate rank.

Boundary:
- ``run['evidence']``, ``run['visual_analysis']`` and ``run['explanation']`` are
  internal/audit artifacts and may contain raw numerical model outputs.
- ``run['handoff']`` is a score-free integration payload.
- ``run['user_facing']`` is the final Vietnamese UI payload and is safe to render
  directly to an end user.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .config_v2 import VLM_PROTOCOL_VERSION_V2, validate_vlm_config_v2
from .pipeline import (
    CATEGORY_LABELS_VI,
    CONFIDENCE_LABELS_VI,
    DIMENSION_LABELS_VI,
    extract_json_object,
)
from .prompt_v2 import (
    DIAGNOSIS_EFFECTS_V2,
    DIAGNOSIS_OVERALL_SUPPORT_V2,
    RECOMMENDATION_EFFECTS_V2,
    RECOMMENDATION_OVERALL_SUPPORT_V2,
    VISUAL_ANALYSIS_SCHEMA_VERSION_V2,
    VISUAL_CONFIDENCE_LEVELS_V2,
    VISUAL_DIMENSIONS_V2,
    append_repair_request_v2,
    build_qwen_messages_v2,
    build_qwen_reason_messages_v2,
    required_limitations_v2,
)
from .schema_v2 import canonical_evidence_json_v2, validate_vlm_evidence_v2


EXPLANATION_SCHEMA_VERSION_V2 = "vlm-explanation-v2"
RUN_SCHEMA_VERSION_V2 = "vlm-run-v2"
HANDOFF_SCHEMA_VERSION_V2 = "vlm-handoff-v2"

_USER_REASON_MAX_CHARS = 300
_FORBIDDEN_USER_REASON_TERMS = (
    "score",
    "điểm",
    "logit",
    "probability",
    "confidence",
    "phần trăm",
    "xếp hạng",
    "rank",
    "qwen",
    "mô hình",
)


def _validate_user_reason(
    value: object,
    *,
    overall_visual_support: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError("diagnosis.user_reason must be a string")
    reason = value.strip()
    if len(reason) > _USER_REASON_MAX_CHARS:
        raise ValueError(
            f"diagnosis.user_reason must be at most {_USER_REASON_MAX_CHARS} characters"
        )
    if "\n" in reason or "\r" in reason:
        raise ValueError("diagnosis.user_reason must be a single-line string")
    lowered = reason.casefold()
    if any(term in lowered for term in _FORBIDDEN_USER_REASON_TERMS):
        raise ValueError("diagnosis.user_reason contains an internal scoring term")
    if overall_visual_support == "supports_loo" and not reason:
        raise ValueError(
            "diagnosis.user_reason is required when diagnosis supports_loo"
        )
    if overall_visual_support != "supports_loo" and reason:
        raise ValueError(
            "diagnosis.user_reason must be empty unless diagnosis supports_loo"
        )
    return reason


def _sanitize_generated_reason(value: object) -> str:
    """Keep only a short, safe sentence from the dedicated reason call."""

    if not isinstance(value, str):
        return ""
    reason = value.strip()
    if reason.startswith("```") and reason.endswith("```"):
        reason = reason[3:-3].strip()
    if reason.casefold().startswith("lý do:"):
        reason = reason.split(":", 1)[1].strip()
    if reason.startswith(('"', "'")) and reason.endswith(('"', "'")):
        reason = reason[1:-1].strip()
    if not reason or "{" in reason or "}" in reason:
        return ""
    try:
        return _validate_user_reason(reason, overall_visual_support="supports_loo")
    except ValueError:
        return ""

DIAGNOSIS_EFFECT_LABELS_VI = {
    "supports_loo": "ủng hộ chẩn đoán LOO",
    "ambiguous": "chưa cho tín hiệu thị giác rõ ràng",
    "contradicts_loo": "không ủng hộ chẩn đoán LOO",
}
DIAGNOSIS_OVERALL_LABELS_VI = {
    "supports_loo": "Các quan sát thị giác có xu hướng ủng hộ chẩn đoán LOO.",
    "ambiguous": "Các quan sát thị giác chưa đủ rõ để ủng hộ hoặc bác bỏ chẩn đoán LOO.",
    "contradicts_loo": (
        "Các quan sát thị giác không ủng hộ chẩn đoán LOO; quyết định problematic item "
        "vẫn do frozen scorer và LOO quyết định."
    ),
}
RECOMMENDATION_EFFECT_LABELS_VI = {
    "supports_recommendation": "ủng hộ recommendation",
    "ambiguous": "chưa cho tín hiệu thị giác rõ ràng",
    "contradicts_recommendation": "không ủng hộ recommendation",
}
RECOMMENDATION_OVERALL_LABELS_VI = {
    "supports_recommendation": "Các quan sát thị giác có xu hướng ủng hộ candidate này.",
    "ambiguous": "Các quan sát thị giác chưa đủ rõ để ủng hộ hoặc bác bỏ candidate này.",
    "contradicts_recommendation": (
        "Các quan sát thị giác không ủng hộ candidate này; rank vẫn được giữ nguyên "
        "vì frozen Recommendation V2 là nguồn quyết định authoritative."
    ),
}


class VLMBackendV2(Protocol):
    model_id: str

    def generate(
        self,
        messages: Sequence[Mapping[str, object]],
        generation: Mapping[str, object],
    ) -> str:
        """Return raw model text for one multimodal conversation."""


def _enum(value: object, name: str, allowed: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}")
    return value


def _validate_overall_support(
    *,
    overall: str,
    observations: Sequence[Mapping[str, object]],
    support_token: str,
    contradict_token: str,
    name: str,
) -> None:
    """Require non-ambiguous overall labels to have matching visual evidence."""

    if not observations:
        if overall != "ambiguous":
            raise ValueError(f"{name} must be ambiguous when observations are empty")
        return

    effects = {str(row["effect"]) for row in observations}
    if overall == support_token and support_token not in effects:
        raise ValueError(f"{name}={support_token} requires a supporting observation")
    if overall == contradict_token and contradict_token not in effects:
        raise ValueError(
            f"{name}={contradict_token} requires a contradicting observation"
        )


def _validate_diagnosis_observations(
    value: object,
    *,
    problem_index: int,
    item_count: int,
) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) > item_count:
        raise ValueError("diagnosis.visual_observations must be a bounded list")

    normalized: list[dict[str, object]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != {
            "item_indices",
            "dimension",
            "effect",
            "confidence",
        }:
            raise ValueError(
                f"diagnosis.visual_observations[{row_index}] has invalid schema"
            )

        indices = row.get("item_indices")
        if (
            not isinstance(indices, list)
            or len(indices) < 2
            or len(indices) > item_count
        ):
            raise ValueError(
                "diagnosis visual observation item_indices must contain the problematic "
                "item plus at least one other original outfit item"
            )

        normalized_indices: list[int] = []
        for value_index in indices:
            if isinstance(value_index, bool) or not isinstance(value_index, int):
                raise ValueError("diagnosis visual item indices must be integers")
            if not 0 <= value_index < item_count:
                raise ValueError("diagnosis visual observation references an unknown item")
            if value_index in normalized_indices:
                raise ValueError("diagnosis visual item indices must be unique")
            normalized_indices.append(value_index)

        if problem_index not in normalized_indices:
            raise ValueError(
                "every diagnosis visual observation must reference the problematic item"
            )

        normalized.append(
            {
                "item_indices": normalized_indices,
                "dimension": _enum(
                    row.get("dimension"),
                    f"diagnosis.visual_observations[{row_index}].dimension",
                    VISUAL_DIMENSIONS_V2,
                ),
                "effect": _enum(
                    row.get("effect"),
                    f"diagnosis.visual_observations[{row_index}].effect",
                    DIAGNOSIS_EFFECTS_V2,
                ),
                "confidence": _enum(
                    row.get("confidence"),
                    f"diagnosis.visual_observations[{row_index}].confidence",
                    VISUAL_CONFIDENCE_LEVELS_V2,
                ),
            }
        )
    return normalized


def _validate_recommendation_observations(
    value: object,
    *,
    allowed_context_indices: set[int],
    recommendation_position: int,
) -> list[dict[str, object]]:
    max_rows = len(allowed_context_indices)
    if not isinstance(value, list) or len(value) > max_rows:
        raise ValueError(
            f"recommendations[{recommendation_position}].visual_observations "
            "must be a bounded list"
        )

    normalized: list[dict[str, object]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != {
            "context_item_indices",
            "dimension",
            "effect",
            "confidence",
        }:
            raise ValueError(
                f"recommendations[{recommendation_position}].visual_observations"
                f"[{row_index}] has invalid schema"
            )

        indices = row.get("context_item_indices")
        if (
            not isinstance(indices, list)
            or not indices
            or len(indices) > len(allowed_context_indices)
        ):
            raise ValueError(
                "recommendation context_item_indices must be a non-empty bounded list"
            )

        normalized_indices: list[int] = []
        for value_index in indices:
            if isinstance(value_index, bool) or not isinstance(value_index, int):
                raise ValueError("recommendation context item indices must be integers")
            if value_index not in allowed_context_indices:
                raise ValueError(
                    "recommendation visual observation may reference only remaining "
                    "original outfit context items"
                )
            if value_index in normalized_indices:
                raise ValueError("recommendation context item indices must be unique")
            normalized_indices.append(value_index)

        normalized.append(
            {
                "context_item_indices": normalized_indices,
                "dimension": _enum(
                    row.get("dimension"),
                    (
                        f"recommendations[{recommendation_position}].visual_observations"
                        f"[{row_index}].dimension"
                    ),
                    VISUAL_DIMENSIONS_V2,
                ),
                "effect": _enum(
                    row.get("effect"),
                    (
                        f"recommendations[{recommendation_position}].visual_observations"
                        f"[{row_index}].effect"
                    ),
                    RECOMMENDATION_EFFECTS_V2,
                ),
                "confidence": _enum(
                    row.get("confidence"),
                    (
                        f"recommendations[{recommendation_position}].visual_observations"
                        f"[{row_index}].confidence"
                    ),
                    VISUAL_CONFIDENCE_LEVELS_V2,
                ),
            }
        )
    return normalized


def _recommendation_visual_signature(row: Mapping[str, object]) -> tuple:
    observations = row.get("visual_observations")
    if not isinstance(observations, list):
        return ()
    observation_signature = tuple(
        (
            tuple(int(index) for index in observation["context_item_indices"]),
            str(observation["dimension"]),
            str(observation["effect"]),
            str(observation["confidence"]),
        )
        for observation in observations
    )
    return (str(row.get("overall_visual_support")), observation_signature)


def _validate_recommendation_independence(
    recommendations: Sequence[Mapping[str, object]],
) -> None:
    """Reject the exact overconfident clone pattern seen in real-Qwen testing."""

    if len(recommendations) != 3:
        return
    signatures = [_recommendation_visual_signature(row) for row in recommendations]
    if not signatures[0] or len(set(signatures)) != 1:
        return

    overall = str(recommendations[0].get("overall_visual_support"))
    observations = recommendations[0].get("visual_observations")
    if overall == "ambiguous" or not isinstance(observations, list) or not observations:
        return
    if all(str(row.get("confidence")) == "high" for row in observations):
        raise ValueError(
            "Top-3 recommendation visual analyses are an exact cloned high-confidence "
            "pattern; re-inspect each candidate independently"
        )


def validate_visual_analysis_v2(
    analysis: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict:
    """Hard-fail any VLM attempt to alter diagnosis or Recommendation V2."""

    normalized_evidence = validate_vlm_evidence_v2(evidence)
    if not isinstance(analysis, Mapping):
        raise TypeError("visual analysis must be a mapping")

    required_top_keys = {
        "schema_version",
        "problematic_item_index",
        "problematic_item_id",
        "diagnosis",
        "recommendations",
        "limitations",
    }
    if set(analysis) != required_top_keys:
        raise ValueError(
            f"V2 visual analysis keys must be exactly {sorted(required_top_keys)}"
        )
    if analysis.get("schema_version") != VISUAL_ANALYSIS_SCHEMA_VERSION_V2:
        raise ValueError(
            f"schema_version must be {VISUAL_ANALYSIS_SCHEMA_VERSION_V2!r}"
        )

    diagnosis_evidence = normalized_evidence["diagnosis"]
    problem_index = int(diagnosis_evidence["problematic_item_index"])
    problem_id = str(diagnosis_evidence["problematic_item_id"])

    output_problem_index = analysis.get("problematic_item_index")
    if (
        isinstance(output_problem_index, bool)
        or not isinstance(output_problem_index, int)
        or output_problem_index != problem_index
    ):
        raise ValueError("VLM attempted to change problematic_item_index")
    if analysis.get("problematic_item_id") != problem_id:
        raise ValueError("VLM attempted to change problematic_item_id")

    diagnosis = analysis.get("diagnosis")
    diagnosis_keys = set(diagnosis) if isinstance(diagnosis, Mapping) else set()
    legacy_diagnosis_keys = {
        "overall_visual_support",
        "visual_observations",
    }
    diagnosis_keys_with_reason = legacy_diagnosis_keys | {"user_reason"}
    if diagnosis_keys != legacy_diagnosis_keys and diagnosis_keys != diagnosis_keys_with_reason:
        raise ValueError("diagnosis visual analysis has invalid schema")

    diagnosis_overall = _enum(
        diagnosis.get("overall_visual_support"),
        "diagnosis.overall_visual_support",
        DIAGNOSIS_OVERALL_SUPPORT_V2,
    )
    item_count = len(normalized_evidence["items"])
    diagnosis_observations = _validate_diagnosis_observations(
        diagnosis.get("visual_observations"),
        problem_index=problem_index,
        item_count=item_count,
    )
    _validate_overall_support(
        overall=diagnosis_overall,
        observations=diagnosis_observations,
        support_token="supports_loo",
        contradict_token="contradicts_loo",
        name="diagnosis.overall_visual_support",
    )
    user_reason = ""
    if "user_reason" in diagnosis:
        user_reason = _validate_user_reason(
            diagnosis.get("user_reason"),
            overall_visual_support=diagnosis_overall,
        )

    recommendation_rows = analysis.get("recommendations")
    authoritative_rows = normalized_evidence["recommendation"]["items"]
    if not isinstance(recommendation_rows, list) or len(recommendation_rows) != len(
        authoritative_rows
    ):
        raise ValueError("recommendations must contain exactly authoritative Top-3 rows")

    allowed_context_indices = {
        int(row["item_index"])
        for row in normalized_evidence["items"]
        if int(row["item_index"]) != problem_index
    }
    normalized_recommendations: list[dict[str, object]] = []
    for position, (row, authoritative) in enumerate(
        zip(recommendation_rows, authoritative_rows)
    ):
        if not isinstance(row, Mapping) or set(row) != {
            "rank",
            "item_id",
            "overall_visual_support",
            "visual_observations",
        }:
            raise ValueError(f"recommendations[{position}] has invalid schema")

        expected_rank = int(authoritative["rank"])
        rank = row.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank != expected_rank:
            raise ValueError("VLM attempted to change recommendation rank/order")
        expected_item_id = str(authoritative["item_id"])
        if row.get("item_id") != expected_item_id:
            raise ValueError("VLM attempted to change recommendation candidate identity")

        overall = _enum(
            row.get("overall_visual_support"),
            f"recommendations[{position}].overall_visual_support",
            RECOMMENDATION_OVERALL_SUPPORT_V2,
        )
        observations = _validate_recommendation_observations(
            row.get("visual_observations"),
            allowed_context_indices=allowed_context_indices,
            recommendation_position=position,
        )
        _validate_overall_support(
            overall=overall,
            observations=observations,
            support_token="supports_recommendation",
            contradict_token="contradicts_recommendation",
            name=f"recommendations[{position}].overall_visual_support",
        )
        normalized_recommendations.append(
            {
                "rank": expected_rank,
                "item_id": expected_item_id,
                "overall_visual_support": overall,
                "visual_observations": observations,
            }
        )

    _validate_recommendation_independence(normalized_recommendations)

    limitations = analysis.get("limitations")
    expected_limitations = list(required_limitations_v2(normalized_evidence))
    if limitations != expected_limitations:
        raise ValueError(
            "limitations must exactly match required VLM V2 disclosures: "
            f"{expected_limitations}"
        )

    return {
        "schema_version": VISUAL_ANALYSIS_SCHEMA_VERSION_V2,
        "problematic_item_index": problem_index,
        "problematic_item_id": problem_id,
        "diagnosis": {
            "overall_visual_support": diagnosis_overall,
            "visual_observations": diagnosis_observations,
            "user_reason": user_reason,
        },
        "recommendations": normalized_recommendations,
        "limitations": expected_limitations,
    }


def render_explanation_vi_v2(
    analysis: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict:
    """Render an internal/audit explanation; raw scores are allowed here."""

    normalized_evidence = validate_vlm_evidence_v2(evidence)
    normalized_analysis = validate_visual_analysis_v2(analysis, normalized_evidence)

    diagnosis = normalized_evidence["diagnosis"]
    scorer = normalized_evidence["scorer"]
    problem_index = int(diagnosis["problematic_item_index"])
    problem_category = CATEGORY_LABELS_VI[str(diagnosis["problematic_category"])]
    top_row = diagnosis["ranked_items"][0]

    diagnosis_observations = []
    for row in normalized_analysis["diagnosis"]["visual_observations"]:
        indices = ", ".join(str(index) for index in row["item_indices"])
        diagnosis_observations.append(
            {
                "item_indices": list(row["item_indices"]),
                "dimension": row["dimension"],
                "effect": row["effect"],
                "confidence": row["confidence"],
                "text": (
                    f"Với item {indices}, {DIMENSION_LABELS_VI[row['dimension']]} "
                    f"được phân loại là {DIAGNOSIS_EFFECT_LABELS_VI[row['effect']]} "
                    f"với độ tin cậy thị giác {CONFIDENCE_LABELS_VI[row['confidence']]}."
                ),
            }
        )

    diagnosis_section = {
        "headline": (
            f"LOO xếp item {problem_index} ({problem_category}) là item problematic nhất."
        ),
        "evidence_summary": [
            (
                f"Khi bỏ item {problem_index}, compatibility logit đổi từ "
                f"{float(scorer['compatibility_logit']):.4f} thành "
                f"{float(top_row['without_item_logit']):.4f}, tương ứng LOO delta "
                f"{float(top_row['loo_delta']):+.4f}."
            ),
            (
                f"Khoảng cách LOO delta giữa Top-1 và Top-2 là "
                f"{float(diagnosis['top1_top2_delta_gap']):.4f}; đây không phải xác suất "
                "hay độ chắc chắn đã hiệu chỉnh."
            ),
        ],
        "visual_summary": DIAGNOSIS_OVERALL_LABELS_VI[
            normalized_analysis["diagnosis"]["overall_visual_support"]
        ],
        "visual_observations": diagnosis_observations,
    }

    analysis_by_id = {
        str(row["item_id"]): row for row in normalized_analysis["recommendations"]
    }
    recommendation_sections = []
    for candidate in normalized_evidence["recommendation"]["items"]:
        item_id = str(candidate["item_id"])
        visual = analysis_by_id[item_id]
        rendered_observations = []
        for row in visual["visual_observations"]:
            context = ", ".join(str(index) for index in row["context_item_indices"])
            rendered_observations.append(
                {
                    "context_item_indices": list(row["context_item_indices"]),
                    "dimension": row["dimension"],
                    "effect": row["effect"],
                    "confidence": row["confidence"],
                    "text": (
                        f"So với các item context {context}, "
                        f"{DIMENSION_LABELS_VI[row['dimension']]} được phân loại là "
                        f"{RECOMMENDATION_EFFECT_LABELS_VI[row['effect']]} với độ tin cậy "
                        f"thị giác {CONFIDENCE_LABELS_VI[row['confidence']]}."
                    ),
                }
            )

        recommendation_sections.append(
            {
                "rank": int(candidate["rank"]),
                "item_id": item_id,
                "master_category": str(candidate["master_category"]),
                "coarse_category": str(candidate["coarse_category"]),
                "compatibility_logit": float(candidate["compatibility_logit"]),
                "improvement_logit": float(candidate["improvement_logit"]),
                "headline": f"Recommendation #{int(candidate['rank'])}: item {item_id}.",
                "score_summary": (
                    f"Frozen scorer cho compatibility logit "
                    f"{float(candidate['compatibility_logit']):.4f}, cải thiện "
                    f"{float(candidate['improvement_logit']):+.4f} so với outfit gốc. "
                    "Các logit này không phải xác suất."
                ),
                "visual_summary": RECOMMENDATION_OVERALL_LABELS_VI[
                    visual["overall_visual_support"]
                ],
                "visual_observations": rendered_observations,
            }
        )

    uncertainty_parts = [
        "Scorer/LOO/recommendation logits là đầu ra chưa hiệu chỉnh, không phải xác suất.",
        "Các nhãn thị giác của Qwen là suy luận từ ảnh theo taxonomy đóng.",
        "Qwen không được thay đổi problematic item, candidate identity hoặc Top-3 rank.",
    ]
    if diagnosis["uses_two_item_extrapolation"]:
        uncertainty_parts.append(
            "Outfit gốc có 3 item nên LOO phải chấm subset còn 2 item; đây là two-item extrapolation."
        )

    return {
        "schema_version": EXPLANATION_SCHEMA_VERSION_V2,
        "problematic_item_index": problem_index,
        "problematic_item_id": str(diagnosis["problematic_item_id"]),
        "headline": diagnosis_section["headline"],
        "diagnosis": diagnosis_section,
        "recommendations": recommendation_sections,
        "explanation": (
            "Frozen scorer và LOO quyết định problematic item; Recommendation V2 quyết định "
            "Top-3 candidate và thứ tự. Qwen chỉ bổ sung các quan sát thị giác đã qua validator."
        ),
        "uncertainty_note": " ".join(uncertainty_parts),
        "limitations": list(normalized_analysis["limitations"]),
    }


def build_handoff_result_v2(
    explanation: Mapping[str, object],
    *,
    model_id: str,
    generation_attempts: int,
) -> dict:
    """Build a score-free integration handoff from an internal explanation.

    Raw compatibility/improvement logits, LOO deltas, score summaries, model
    names, and raw model text intentionally remain outside this payload. The
    frontend may use IDs/ranks for image binding and use ``run['user_facing']``
    for end-user prose.
    """

    if not isinstance(explanation, Mapping):
        raise TypeError("explanation must be a mapping")
    required = {
        "schema_version",
        "problematic_item_index",
        "problematic_item_id",
        "headline",
        "diagnosis",
        "recommendations",
        "explanation",
        "uncertainty_note",
        "limitations",
    }
    if (
        set(explanation) != required
        or explanation.get("schema_version") != EXPLANATION_SCHEMA_VERSION_V2
    ):
        raise ValueError("explanation is not a canonical VLM explanation V2 payload")
    if (
        isinstance(generation_attempts, bool)
        or not isinstance(generation_attempts, int)
        or generation_attempts < 1
    ):
        raise ValueError("generation_attempts must be an integer >= 1")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")

    diagnosis = explanation["diagnosis"]
    recommendations = explanation["recommendations"]
    if not isinstance(diagnosis, Mapping):
        raise ValueError("handoff diagnosis must be an object")
    if not isinstance(recommendations, list) or len(recommendations) != 3:
        raise ValueError("handoff requires exactly three recommendation rows")

    safe_recommendations: list[dict[str, object]] = []
    for row in recommendations:
        if not isinstance(row, Mapping):
            raise ValueError("handoff recommendation row must be an object")
        safe_recommendations.append(
            {
                "rank": int(row["rank"]),
                "item_id": str(row["item_id"]),
                "master_category": str(row["master_category"]),
                "coarse_category": str(row["coarse_category"]),
                "visual_summary": str(row["visual_summary"]),
                "visual_observations": list(row["visual_observations"]),
            }
        )

    return {
        "schema_version": HANDOFF_SCHEMA_VERSION_V2,
        "protocol_version": VLM_PROTOCOL_VERSION_V2,
        "problematic_item_index": int(explanation["problematic_item_index"]),
        "problematic_item_id": str(explanation["problematic_item_id"]),
        "diagnosis": {
            "visual_summary": str(diagnosis["visual_summary"]),
            "visual_observations": list(diagnosis["visual_observations"]),
        },
        "recommendations": safe_recommendations,
        "limitations": list(explanation["limitations"]),
    }


class VLMExplanationPipelineV2:
    """End-to-end VLM V2 wrapper with one deterministic schema-repair retry."""

    def __init__(self, backend: VLMBackendV2, config: Mapping[str, object]) -> None:
        self.backend = backend
        self.config = validate_vlm_config_v2(config)
        if backend.model_id != self.config["model"]["id"]:
            raise ValueError("Backend model_id does not match the frozen VLM V2 config")

    def explain(
        self,
        evidence: Mapping[str, object],
        outfit_image_refs: Sequence[str | Path],
        recommendation_image_refs: Mapping[str, str | Path],
        *,
        must_exist: bool = True,
        original_image_ref: str | Path | None = None,
    ) -> dict:
        """Run Qwen -> validator -> internal + score-free public renderers."""

        normalized_evidence = validate_vlm_evidence_v2(evidence)
        vision = self.config["vision"]
        messages = build_qwen_messages_v2(
            normalized_evidence,
            outfit_image_refs,
            recommendation_image_refs,
            min_pixels=int(vision["min_pixels"]),
            max_pixels=int(vision["max_pixels"]),
            must_exist=must_exist,
            original_image_ref=original_image_ref,
        )
        generation = dict(self.config["generation"])
        max_retries = int(generation.pop("max_validation_retries"))

        raw_response = ""
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            raw_response = self.backend.generate(messages, generation)
            try:
                parsed = extract_json_object(raw_response)
                visual_analysis = validate_visual_analysis_v2(parsed, normalized_evidence)
                break
            except (TypeError, ValueError) as error:
                last_error = error
                if attempt >= max_retries:
                    raise ValueError(
                        "VLM V2 output failed validation after "
                        f"{attempt + 1} attempt(s): {error}"
                    ) from error
                messages = append_repair_request_v2(
                    messages,
                    raw_response=raw_response,
                    validation_error=str(error),
                )
        else:  # pragma: no cover
            raise RuntimeError(f"Unreachable VLM V2 validation state: {last_error}")

        rendered_explanation = render_explanation_vi_v2(
            visual_analysis, normalized_evidence
        )
        evidence_json = canonical_evidence_json_v2(normalized_evidence)
        generation_attempts = attempt + 1
        handoff = build_handoff_result_v2(
            rendered_explanation,
            model_id=self.backend.model_id,
            generation_attempts=generation_attempts,
        )

        # Local import avoids a module-level cycle because the user renderer
        # itself imports this module's validator.
        from .user_renderer_v2 import render_user_facing_vi_v2

        user_facing = render_user_facing_vi_v2(
            visual_analysis,
            normalized_evidence,
        )

        run = {
            "schema_version": RUN_SCHEMA_VERSION_V2,
            "protocol_version": VLM_PROTOCOL_VERSION_V2,
            "model_id": self.backend.model_id,
            "generation_attempts": generation_attempts,
            "evidence_sha256": hashlib.sha256(evidence_json.encode("utf-8")).hexdigest(),
            "evidence": normalized_evidence,
            "visual_analysis": visual_analysis,
            "explanation": rendered_explanation,
            "handoff": handoff,
            "user_facing": user_facing,
        }
        if self.config["output"]["include_raw_response"]:
            run["raw_response"] = raw_response
        return run

    def explain_reason(
        self,
        *,
        target_item: Mapping[str, object],
        original_image_ref: str | Path,
        target_image_ref: str | Path,
        must_exist: bool = True,
    ) -> str:
        """Run a dedicated short natural-language visual reason request."""

        vision = self.config["vision"]
        messages = build_qwen_reason_messages_v2(
            target_item,
            original_image_ref,
            target_image_ref,
            min_pixels=int(vision["min_pixels"]),
            max_pixels=int(vision["max_pixels"]),
            must_exist=must_exist,
        )
        generation = dict(self.config["generation"])
        generation.pop("max_validation_retries", None)
        generation["max_new_tokens"] = 96
        raw_reason = self.backend.generate(messages, generation)
        return _sanitize_generated_reason(raw_reason)
