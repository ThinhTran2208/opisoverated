# -*- coding: utf-8 -*-
"""Validate constrained VLM analysis and render safe Vietnamese explanations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .config import validate_vlm_config
from .prompt import (
    EXPLANATION_SCHEMA_VERSION,
    OVERALL_VISUAL_SUPPORT_LEVELS,
    VISUAL_ANALYSIS_SCHEMA_VERSION,
    VISUAL_CONFIDENCE_LEVELS,
    VISUAL_DIMENSIONS,
    VISUAL_EFFECTS,
    append_repair_request,
    build_qwen_messages,
    required_limitations,
)
from .schema import canonical_evidence_json, validate_vlm_evidence


RUN_SCHEMA_VERSION = "vlm-run-v1"

CATEGORY_LABELS_VI = {
    "TOP": "áo trên",
    "BOTTOM": "quần hoặc váy dưới",
    "DRESS": "váy liền",
    "OUTERWEAR": "áo khoác",
    "SHOES": "giày",
    "BAG": "túi",
    "HAT": "mũ",
}
DIMENSION_LABELS_VI = {
    "color_harmony": "hòa hợp màu sắc",
    "pattern_coherence": "tính nhất quán họa tiết",
    "silhouette_balance": "cân bằng phom dáng",
    "formality_alignment": "mức độ đồng nhất về tính trang trọng",
    "style_coherence": "tính nhất quán phong cách",
}
EFFECT_LABELS_VI = {
    "supports_loo": "ủng hộ chẩn đoán LOO",
    "ambiguous": "chưa cho tín hiệu thị giác rõ ràng",
    "contradicts_loo": "không ủng hộ chẩn đoán LOO",
}
CONFIDENCE_LABELS_VI = {
    "low": "thấp",
    "medium": "trung bình",
    "high": "cao",
}
OVERALL_SUPPORT_LABELS_VI = {
    "supports_loo": "Các nhãn thị giác có xu hướng ủng hộ chẩn đoán LOO.",
    "ambiguous": "Các nhãn thị giác chưa đủ rõ để ủng hộ hoặc bác bỏ chẩn đoán LOO.",
    "contradicts_loo": (
        "Các nhãn thị giác không ủng hộ chẩn đoán LOO; kết luận định lượng "
        "vẫn do scorer và LOO quyết định."
    ),
}


class VLMBackend(Protocol):
    model_id: str

    def generate(
        self,
        messages: Sequence[Mapping[str, object]],
        generation: Mapping[str, object],
    ) -> str:
        """Return raw model text for one multimodal conversation."""


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"VLM response contains duplicate JSON key: {key}")
        value[key] = child
    return value


def extract_json_object(raw_text: str) -> dict:
    """Extract exactly one JSON object, allowing only a surrounding code fence."""

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("VLM returned an empty response")
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if not text.startswith("{"):
        raise ValueError("VLM response contains text before the JSON object")
    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_json_keys)
    try:
        value, end = decoder.raw_decode(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"VLM response is not valid JSON: {error}") from error
    trailing = text[end:].strip()
    if trailing:
        raise ValueError("VLM response contains text after the JSON object")
    if not isinstance(value, dict):
        raise ValueError("VLM response JSON must be an object")
    return value


def _enum(value: object, name: str, allowed: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}")
    return value


def validate_visual_analysis(
    analysis: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict:
    """Hard-fail any free text, unknown token, or diagnosis override.

    The VLM has no free-text output field. Consequently a replacement suggestion
    cannot pass by hiding inside ``headline`` or ``explanation``; those fields are
    not part of this schema at all.
    """

    normalized_evidence = validate_vlm_evidence(evidence)
    if not isinstance(analysis, Mapping):
        raise TypeError("visual analysis must be a mapping")
    required_keys = {
        "schema_version",
        "problematic_item_index",
        "problematic_item_id",
        "overall_visual_support",
        "visual_observations",
        "limitations",
    }
    if set(analysis) != required_keys:
        raise ValueError(
            f"Visual analysis keys must be exactly {sorted(required_keys)}"
        )
    if analysis.get("schema_version") != VISUAL_ANALYSIS_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {VISUAL_ANALYSIS_SCHEMA_VERSION!r}"
        )

    diagnosis = normalized_evidence["diagnosis"]
    problem_index = int(diagnosis["problematic_item_index"])
    output_problem_index = analysis.get("problematic_item_index")
    if (
        isinstance(output_problem_index, bool)
        or not isinstance(output_problem_index, int)
        or output_problem_index != problem_index
    ):
        raise ValueError("VLM attempted to change problematic_item_index")
    if analysis.get("problematic_item_id") != diagnosis["problematic_item_id"]:
        raise ValueError("VLM attempted to change problematic_item_id")

    normalized: dict[str, object] = {
        "schema_version": VISUAL_ANALYSIS_SCHEMA_VERSION,
        "problematic_item_index": problem_index,
        "problematic_item_id": diagnosis["problematic_item_id"],
        "overall_visual_support": _enum(
            analysis.get("overall_visual_support"),
            "overall_visual_support",
            OVERALL_VISUAL_SUPPORT_LEVELS,
        ),
    }

    observations = analysis.get("visual_observations")
    item_count = len(normalized_evidence["items"])
    if not isinstance(observations, list) or len(observations) > item_count:
        raise ValueError("visual_observations must be a bounded list")
    normalized_observations: list[dict] = []
    for row_index, row in enumerate(observations):
        if not isinstance(row, Mapping) or set(row) != {
            "item_indices",
            "dimension",
            "effect",
            "confidence",
        }:
            raise ValueError(f"visual_observations[{row_index}] has invalid schema")
        indices = row.get("item_indices")
        if not isinstance(indices, list) or not indices or len(indices) > item_count:
            raise ValueError("visual observation item_indices must be a bounded list")
        normalized_indices: list[int] = []
        for value in indices:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("visual observation item indices must be integers")
            if not 0 <= value < item_count:
                raise ValueError("visual observation references an unknown item")
            if value in normalized_indices:
                raise ValueError("visual observation item indices must be unique")
            normalized_indices.append(value)
        if problem_index not in normalized_indices:
            raise ValueError(
                "every visual observation must reference the problematic item"
            )
        normalized_observations.append(
            {
                "item_indices": normalized_indices,
                "dimension": _enum(
                    row.get("dimension"), "dimension", VISUAL_DIMENSIONS
                ),
                "effect": _enum(row.get("effect"), "effect", VISUAL_EFFECTS),
                "confidence": _enum(
                    row.get("confidence"),
                    "confidence",
                    VISUAL_CONFIDENCE_LEVELS,
                ),
            }
        )
    normalized["visual_observations"] = normalized_observations

    overall_support = normalized["overall_visual_support"]
    effects = {row["effect"] for row in normalized_observations}
    if not normalized_observations and overall_support != "ambiguous":
        raise ValueError(
            "overall_visual_support must be ambiguous when observations are empty"
        )
    if overall_support == "supports_loo" and "supports_loo" not in effects:
        raise ValueError("overall supports_loo requires a supporting observation")
    if overall_support == "contradicts_loo" and "contradicts_loo" not in effects:
        raise ValueError(
            "overall contradicts_loo requires a contradicting observation"
        )

    limitations = analysis.get("limitations")
    expected_limitations = list(required_limitations(normalized_evidence))
    if limitations != expected_limitations:
        raise ValueError(
            "limitations must exactly match the required machine-readable "
            f"disclosures: {expected_limitations}"
        )
    normalized["limitations"] = expected_limitations
    return normalized


def render_explanation_vi(
    analysis: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict:
    """Render all human-facing text from reviewed templates, never VLM prose."""

    normalized_evidence = validate_vlm_evidence(evidence)
    normalized_analysis = validate_visual_analysis(analysis, normalized_evidence)
    diagnosis = normalized_evidence["diagnosis"]
    scorer = normalized_evidence["scorer"]
    problem_index = int(diagnosis["problematic_item_index"])
    problem_category = CATEGORY_LABELS_VI[diagnosis["problematic_category"]]
    top_row = diagnosis["ranked_items"][0]

    rendered_observations = []
    for row in normalized_analysis["visual_observations"]:
        indices = ", ".join(str(index) for index in row["item_indices"])
        rendered_observations.append(
            {
                "item_indices": list(row["item_indices"]),
                "observation": (
                    f"Với item {indices}, chiều {DIMENSION_LABELS_VI[row['dimension']]} "
                    f"được phân loại là {EFFECT_LABELS_VI[row['effect']]} "
                    f"với độ tin cậy thị giác {CONFIDENCE_LABELS_VI[row['confidence']]}."
                ),
            }
        )

    evidence_summary = [
        (
            f"Khi bỏ item {problem_index}, compatibility logit đổi từ "
            f"{float(scorer['compatibility_logit']):.4f} thành "
            f"{float(top_row['without_item_logit']):.4f}, tương ứng LOO delta "
            f"{float(top_row['loo_delta']):+.4f}."
        ),
        (
            f"Khoảng cách delta giữa Top-1 và Top-2 là "
            f"{float(diagnosis['top1_top2_delta_gap']):.4f}; giá trị này không "
            "phải xác suất hoặc độ chắc chắn đã hiệu chỉnh."
        ),
    ]

    uncertainty_parts = [
        "Độ chắc chắn của LOO chưa được hiệu chỉnh và các nhãn thị giác chỉ là suy luận từ ảnh."
    ]
    if diagnosis["uses_two_item_extrapolation"]:
        uncertainty_parts.append(
            "Outfit gốc có 3 item nên LOO đã chấm các subset còn 2 item, nằm ngoài "
            "phân phối huấn luyện 3–8 item của scorer; kết quả này là extrapolation."
        )

    return {
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "problematic_item_index": problem_index,
        "problematic_item_id": diagnosis["problematic_item_id"],
        "headline": (
            f"LOO xếp item {problem_index} ({problem_category}) đứng đầu trong "
            "phép chẩn đoán loại-từng-item."
        ),
        "evidence_summary": evidence_summary,
        "visual_observations": rendered_observations,
        "explanation": (
            "Item được nêu trên do frozen scorer và LOO quyết định. "
            f"{OVERALL_SUPPORT_LABELS_VI[normalized_analysis['overall_visual_support']]} "
            "Qwen chỉ phân loại quan hệ nhìn thấy theo taxonomy đóng và không được "
            "thay đổi kết luận định lượng."
        ),
        "uncertainty_note": " ".join(uncertainty_parts),
        "limitations": list(normalized_analysis["limitations"]),
    }


class VLMExplanationPipeline:
    """One-case inference wrapper with one deterministic schema-repair retry."""

    def __init__(self, backend: VLMBackend, config: Mapping[str, object]) -> None:
        self.backend = backend
        self.config = validate_vlm_config(config)
        if backend.model_id != self.config["model"]["id"]:
            raise ValueError("Backend model_id does not match the frozen VLM config")

    def explain(
        self,
        evidence: Mapping[str, object],
        image_refs: Sequence[str | Path],
        *,
        must_exist: bool = True,
    ) -> dict:
        normalized_evidence = validate_vlm_evidence(evidence)
        vision = self.config["vision"]
        messages = build_qwen_messages(
            normalized_evidence,
            image_refs,
            min_pixels=int(vision["min_pixels"]),
            max_pixels=int(vision["max_pixels"]),
            must_exist=must_exist,
        )
        generation = dict(self.config["generation"])
        max_retries = int(generation.pop("max_validation_retries"))

        raw_response = ""
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            raw_response = self.backend.generate(messages, generation)
            try:
                parsed = extract_json_object(raw_response)
                visual_analysis = validate_visual_analysis(parsed, normalized_evidence)
                break
            except (TypeError, ValueError) as error:
                last_error = error
                if attempt >= max_retries:
                    raise ValueError(
                        "VLM output failed validation after "
                        f"{attempt + 1} attempt(s): {error}"
                    ) from error
                messages = append_repair_request(
                    messages,
                    raw_response=raw_response,
                    validation_error=str(error),
                )
        else:  # pragma: no cover - defensive; loop always breaks or raises.
            raise RuntimeError(f"Unreachable VLM validation state: {last_error}")

        rendered_explanation = render_explanation_vi(
            visual_analysis, normalized_evidence
        )
        evidence_json = canonical_evidence_json(normalized_evidence)
        run = {
            "schema_version": RUN_SCHEMA_VERSION,
            "protocol_version": self.config["protocol_version"],
            "model_id": self.backend.model_id,
            "generation_attempts": attempt + 1,
            "evidence_sha256": hashlib.sha256(evidence_json.encode("utf-8")).hexdigest(),
            "evidence": normalized_evidence,
            "visual_analysis": visual_analysis,
            "explanation": rendered_explanation,
        }
        if self.config["output"]["include_raw_response"]:
            run["raw_response"] = raw_response
        return run
