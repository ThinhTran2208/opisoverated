# -*- coding: utf-8 -*-
"""Deterministic validation for VLM visual analysis V2.

This module is intentionally model-independent. Qwen may only emit a closed
machine-readable visual analysis; every authoritative decision continues to come
from frozen scorer + LOO + Recommendation V2 evidence.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .prompt_v2 import (
    DIAGNOSIS_EFFECTS_V2,
    DIAGNOSIS_OVERALL_SUPPORT_V2,
    RECOMMENDATION_EFFECTS_V2,
    RECOMMENDATION_OVERALL_SUPPORT_V2,
    VISUAL_ANALYSIS_SCHEMA_VERSION_V2,
    VISUAL_CONFIDENCE_LEVELS_V2,
    VISUAL_DIMENSIONS_V2,
    required_limitations_v2,
)
from .schema_v2 import validate_vlm_evidence_v2


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
        if not isinstance(indices, list) or not indices or len(indices) > item_count:
            raise ValueError(
                "diagnosis visual observation item_indices must be a bounded list"
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


def validate_visual_analysis_v2(
    analysis: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict:
    """Hard-fail any VLM attempt to alter diagnosis or Recommendation V2.

    The output schema contains no free-text field. Candidate identity and rank
    must exactly match authoritative Recommendation V2 evidence. Recommendation
    observations may reference only the outfit items that remain after removing
    the LOO-selected problematic item.
    """

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
    if not isinstance(diagnosis, Mapping) or set(diagnosis) != {
        "overall_visual_support",
        "visual_observations",
    }:
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
        },
        "recommendations": normalized_recommendations,
        "limitations": expected_limitations,
    }
