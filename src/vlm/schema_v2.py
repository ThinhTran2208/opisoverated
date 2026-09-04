# -*- coding: utf-8 -*-
"""Structured evidence V2 joining frozen LOO with frozen Recommendation V2.

V1 remains unchanged and explicitly forbids recommendation payloads.  This
module adds a new evidence schema whose recommendation section is copied from
an authoritative ``RecommendationResult`` and cross-checked against the same
frozen scorer/LOO evidence before it can reach a VLM.
"""

from __future__ import annotations

import json
import math
from typing import Mapping, Sequence

from .schema import (
    ALLOWED_CATEGORIES,
    CANONICAL_CHECKPOINT,
    GROUNDING_RULES as V1_GROUNDING_RULES,
    LOO_PROTOCOL_VERSION,
    SCORER_VERSION,
    build_vlm_evidence,
    validate_vlm_evidence,
)


EVIDENCE_SCHEMA_VERSION_V2 = "vlm-evidence-v2"
CANONICAL_RECOMMENDATION_VERSION = "category-aware-hybrid-v2"
RECOMMENDATION_STATUS = "available"
RECOMMENDATION_COUNT = 3
RECOMMENDATION_SCORE_SEMANTICS = "uncalibrated_logits_not_probabilities"
RECOMMENDATION_RANKING_SEMANTICS = "frozen_scorer_descending_compatibility_logit"

GROUNDING_RULES_V2 = (
    "scorer_logit_is_not_probability",
    "loo_diagnosis_controls_problematic_item",
    "recommendation_v2_controls_candidate_identity_and_rank",
    "recommendation_scores_are_not_probabilities",
    "visual_claims_must_be_visible",
    "vlm_may_not_invent_or_rerank_recommendations",
)

# Synthetic benchmark/evaluation state must never become explanation evidence.
# ``rank`` itself is intentionally allowed because recommendation rank 1..3 is
# runtime output, not an evaluation target.
FORBIDDEN_V2_LEAKAGE_KEYS = frozenset(
    {
        "label",
        "negative_metadata",
        "swapped_item_index",
        "target_swapped_item_index",
        "original_item_id",
        "ground_truth",
        "ground_truth_item_id",
        "replacement_item_id",
        "top1_correct",
        "hit_at_1",
        "hit_at_2",
        "hit_at_3",
        "mrr",
        "source_split",
        "ground_truth_rank",
        "pre_rerank_ground_truth_rank",
        "post_rerank_ground_truth_rank",
    }
)


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _find_forbidden_keys(value: object, *, path: str = "root") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in FORBIDDEN_V2_LEAKAGE_KEYS:
                found.append(child_path)
            found.extend(_find_forbidden_keys(child, path=child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, path=f"{path}[{index}]"))
    return found


def _recommendation_result_parts(
    recommendation_result: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return public recommendation output plus private runtime metadata.

    Canonical runtime passes ``RecommendationResult``.  A plain mapping is also
    accepted for adapters/tests when it contains the public keys plus
    ``internal_metadata``.  Evaluation-only/public-only records are rejected
    because they cannot prove that item ranks and scorer logits refer to the
    same reranking call.
    """

    if isinstance(recommendation_result, Mapping):
        if "internal_metadata" not in recommendation_result:
            raise ValueError(
                "recommendation_result mapping must include internal_metadata"
            )
        public = {
            "status": recommendation_result.get("status"),
            "recommendation_version": recommendation_result.get(
                "recommendation_version"
            ),
            "items": recommendation_result.get("items"),
        }
        internal = recommendation_result.get("internal_metadata")
    else:
        to_public_dict = getattr(recommendation_result, "to_public_dict", None)
        internal = getattr(recommendation_result, "internal_metadata", None)
        if not callable(to_public_dict):
            raise TypeError(
                "recommendation_result must be RecommendationResult-like or a mapping"
            )
        public = to_public_dict()

    if not isinstance(public, Mapping):
        raise ValueError("recommendation_result public output must be an object")
    if not isinstance(internal, Mapping):
        raise ValueError("recommendation_result internal_metadata must be an object")

    combined = {"public": dict(public), "internal_metadata": dict(internal)}
    leaked = _find_forbidden_keys(combined, path="recommendation_result")
    if leaked:
        raise ValueError(
            f"Recommendation result contains forbidden evaluation leakage: {leaked}"
        )
    return dict(public), dict(internal)


def build_recommendation_evidence(
    recommendation_result: object,
    *,
    base_evidence: Mapping[str, object],
) -> dict:
    """Build the authoritative recommendation section for ``vlm-evidence-v2``.

    Candidate identity/rank/category comes from the public Top-3 result.  Scores
    come from the corresponding first three rows of ``reranked_candidates``.
    The builder verifies both views agree and that every improvement logit is
    consistent with the same baseline compatibility logit used by LOO.
    """

    base = validate_vlm_evidence(base_evidence)
    public, internal = _recommendation_result_parts(recommendation_result)

    if public.get("status") != "ok":
        raise ValueError("recommendation_result.status must be 'ok'")
    if public.get("recommendation_version") != CANONICAL_RECOMMENDATION_VERSION:
        raise ValueError(
            "recommendation_result must come from frozen category-aware-hybrid-v2"
        )

    diagnosis = base["diagnosis"]
    problem_index = int(diagnosis["problematic_item_index"])
    problem_id = str(diagnosis["problematic_item_id"])
    problem_category = str(diagnosis["problematic_category"])

    internal_problem_index = internal.get("problematic_item_index")
    if (
        isinstance(internal_problem_index, bool)
        or not isinstance(internal_problem_index, int)
        or internal_problem_index != problem_index
    ):
        raise ValueError(
            "Recommendation problematic_item_index does not match LOO diagnosis"
        )
    internal_loo_version = internal.get("loo_protocol_version")
    if internal_loo_version not in (None, LOO_PROTOCOL_VERSION):
        raise ValueError("Recommendation metadata targets a different LOO protocol")

    public_items = public.get("items")
    if not isinstance(public_items, (list, tuple)) or len(public_items) != RECOMMENDATION_COUNT:
        raise ValueError(
            f"Recommendation V2 evidence requires exactly Top-{RECOMMENDATION_COUNT} items"
        )

    reranked = internal.get("reranked_candidates")
    if not isinstance(reranked, list) or len(reranked) < RECOMMENDATION_COUNT:
        raise ValueError(
            "Recommendation internal_metadata must contain at least three reranked candidates"
        )

    outfit_item_ids = {str(row["item_id"]) for row in base["items"]}
    baseline = float(base["scorer"]["compatibility_logit"])
    normalized_items: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for position in range(RECOMMENDATION_COUNT):
        public_row = public_items[position]
        score_row = reranked[position]
        if not isinstance(public_row, Mapping):
            raise ValueError(f"recommendation items[{position}] must be an object")
        if not isinstance(score_row, Mapping):
            raise ValueError(
                f"reranked_candidates[{position}] must be an object"
            )

        expected_rank = position + 1
        rank = public_row.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank != expected_rank:
            raise ValueError("Recommendation public ranks must be exactly 1, 2, 3")

        item_id = _nonempty_string(
            public_row.get("item_id"), f"recommendation.items[{position}].item_id"
        )
        if item_id in seen_ids:
            raise ValueError("Recommendation Top-3 item IDs must be unique")
        if item_id in outfit_item_ids:
            raise ValueError("Recommendation candidate may not already be in the outfit")
        seen_ids.add(item_id)

        score_item_id = _nonempty_string(
            score_row.get("item_id"),
            f"reranked_candidates[{position}].item_id",
        )
        if score_item_id != item_id:
            raise ValueError(
                "Recommendation public Top-3 does not match frozen scorer reranking order"
            )

        master_category = _nonempty_string(
            public_row.get("master_category"),
            f"recommendation.items[{position}].master_category",
        )
        coarse_category = _nonempty_string(
            public_row.get("coarse_category"),
            f"recommendation.items[{position}].coarse_category",
        ).upper()
        if coarse_category not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"Unknown recommendation coarse category: {coarse_category}"
            )
        if coarse_category != problem_category:
            raise ValueError(
                "Recommendation candidate coarse category must match the LOO problematic category"
            )

        compatibility_logit = _finite_float(
            score_row.get("compatibility_logit"),
            f"reranked_candidates[{position}].compatibility_logit",
        )
        improvement_logit = _finite_float(
            score_row.get("improvement_logit"),
            f"reranked_candidates[{position}].improvement_logit",
        )
        if not math.isclose(
            compatibility_logit - baseline,
            improvement_logit,
            abs_tol=1e-5,
        ):
            raise ValueError(
                "Recommendation improvement_logit is inconsistent with the LOO/scorer baseline"
            )

        normalized_items.append(
            {
                "rank": expected_rank,
                "item_id": item_id,
                "master_category": master_category,
                "coarse_category": coarse_category,
                "compatibility_logit": compatibility_logit,
                "improvement_logit": improvement_logit,
            }
        )

    expected_order = sorted(
        normalized_items,
        key=lambda row: (-float(row["compatibility_logit"]), str(row["item_id"])),
    )
    if [row["item_id"] for row in normalized_items] != [
        row["item_id"] for row in expected_order
    ]:
        raise ValueError(
            "Recommendation Top-3 ranks are inconsistent with frozen scorer logits"
        )

    recommendation = {
        "status": RECOMMENDATION_STATUS,
        "version": CANONICAL_RECOMMENDATION_VERSION,
        "problematic_item_index": problem_index,
        "problematic_item_id": problem_id,
        "ranking_semantics": RECOMMENDATION_RANKING_SEMANTICS,
        "score_semantics": RECOMMENDATION_SCORE_SEMANTICS,
        "items": normalized_items,
    }
    return validate_recommendation_evidence(recommendation, base_evidence=base)


def validate_recommendation_evidence(
    recommendation: Mapping[str, object],
    *,
    base_evidence: Mapping[str, object],
) -> dict:
    """Validate a serialized recommendation section without model-side trust."""

    base = validate_vlm_evidence(base_evidence)
    if not isinstance(recommendation, Mapping):
        raise TypeError("recommendation must be an object")
    leaked = _find_forbidden_keys(recommendation, path="recommendation")
    if leaked:
        raise ValueError(f"Recommendation evidence contains forbidden leakage: {leaked}")

    required = {
        "status",
        "version",
        "problematic_item_index",
        "problematic_item_id",
        "ranking_semantics",
        "score_semantics",
        "items",
    }
    if set(recommendation) != required:
        raise ValueError("recommendation evidence has unexpected fields")
    if recommendation.get("status") != RECOMMENDATION_STATUS:
        raise ValueError(f"recommendation.status must be {RECOMMENDATION_STATUS!r}")
    if recommendation.get("version") != CANONICAL_RECOMMENDATION_VERSION:
        raise ValueError("recommendation.version is not canonical Recommendation V2")
    if recommendation.get("ranking_semantics") != RECOMMENDATION_RANKING_SEMANTICS:
        raise ValueError("recommendation.ranking_semantics is invalid")
    if recommendation.get("score_semantics") != RECOMMENDATION_SCORE_SEMANTICS:
        raise ValueError("recommendation.score_semantics is invalid")

    diagnosis = base["diagnosis"]
    problem_index = recommendation.get("problematic_item_index")
    if (
        isinstance(problem_index, bool)
        or not isinstance(problem_index, int)
        or problem_index != diagnosis["problematic_item_index"]
    ):
        raise ValueError("recommendation problematic index must match LOO")
    if recommendation.get("problematic_item_id") != diagnosis["problematic_item_id"]:
        raise ValueError("recommendation problematic item ID must match LOO")

    rows = recommendation.get("items")
    if not isinstance(rows, list) or len(rows) != RECOMMENDATION_COUNT:
        raise ValueError(
            f"recommendation.items must contain exactly {RECOMMENDATION_COUNT} rows"
        )

    outfit_item_ids = {str(row["item_id"]) for row in base["items"]}
    problem_category = str(diagnosis["problematic_category"])
    baseline = float(base["scorer"]["compatibility_logit"])
    seen_ids: set[str] = set()
    normalized_rows: list[dict[str, object]] = []
    row_keys = {
        "rank",
        "item_id",
        "master_category",
        "coarse_category",
        "compatibility_logit",
        "improvement_logit",
    }

    for position, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != row_keys:
            raise ValueError(f"recommendation.items[{position}] has invalid schema")
        expected_rank = position + 1
        rank = row.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank != expected_rank:
            raise ValueError("recommendation ranks must be contiguous 1, 2, 3")
        item_id = _nonempty_string(row.get("item_id"), "recommendation item_id")
        if item_id in seen_ids:
            raise ValueError("recommendation item IDs must be unique")
        if item_id in outfit_item_ids:
            raise ValueError("recommendation item may not already be in the outfit")
        seen_ids.add(item_id)
        master_category = _nonempty_string(
            row.get("master_category"), "recommendation master_category"
        )
        coarse_category = _nonempty_string(
            row.get("coarse_category"), "recommendation coarse_category"
        ).upper()
        if coarse_category not in ALLOWED_CATEGORIES or coarse_category != problem_category:
            raise ValueError(
                "recommendation coarse category must match the problematic category"
            )
        compatibility_logit = _finite_float(
            row.get("compatibility_logit"), "recommendation compatibility_logit"
        )
        improvement_logit = _finite_float(
            row.get("improvement_logit"), "recommendation improvement_logit"
        )
        if not math.isclose(
            compatibility_logit - baseline,
            improvement_logit,
            abs_tol=1e-5,
        ):
            raise ValueError(
                "recommendation improvement_logit is inconsistent with scorer baseline"
            )
        normalized_rows.append(
            {
                "rank": expected_rank,
                "item_id": item_id,
                "master_category": master_category,
                "coarse_category": coarse_category,
                "compatibility_logit": compatibility_logit,
                "improvement_logit": improvement_logit,
            }
        )

    expected_order = sorted(
        normalized_rows,
        key=lambda row: (-float(row["compatibility_logit"]), str(row["item_id"])),
    )
    if [row["item_id"] for row in normalized_rows] != [
        row["item_id"] for row in expected_order
    ]:
        raise ValueError("recommendation rank order is inconsistent with scorer logits")

    return {
        "status": RECOMMENDATION_STATUS,
        "version": CANONICAL_RECOMMENDATION_VERSION,
        "problematic_item_index": int(problem_index),
        "problematic_item_id": str(recommendation["problematic_item_id"]),
        "ranking_semantics": RECOMMENDATION_RANKING_SEMANTICS,
        "score_semantics": RECOMMENDATION_SCORE_SEMANTICS,
        "items": normalized_rows,
    }


def build_vlm_evidence_v2(
    loo_result: Mapping[str, object],
    recommendation_result: object,
    *,
    sample_id: str,
    item_ids: Sequence[str],
    coarse_categories: Sequence[str],
    scorer_version: str = SCORER_VERSION,
    checkpoint: str = CANONICAL_CHECKPOINT,
) -> dict:
    """Build VLM evidence containing frozen LOO plus authoritative Top-3."""

    base_v1 = build_vlm_evidence(
        loo_result,
        sample_id=sample_id,
        item_ids=item_ids,
        coarse_categories=coarse_categories,
        scorer_version=scorer_version,
        checkpoint=checkpoint,
    )
    recommendation = build_recommendation_evidence(
        recommendation_result,
        base_evidence=base_v1,
    )
    evidence = json.loads(json.dumps(base_v1))
    evidence["schema_version"] = EVIDENCE_SCHEMA_VERSION_V2
    evidence["recommendation"] = recommendation
    evidence["grounding_rules"] = list(GROUNDING_RULES_V2)
    return validate_vlm_evidence_v2(evidence)


def validate_vlm_evidence_v2(evidence: Mapping[str, object]) -> dict:
    """Hard-fail malformed, leaked, or internally inconsistent V2 evidence."""

    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    leaked = _find_forbidden_keys(evidence)
    if leaked:
        raise ValueError(f"V2 evidence contains forbidden target leakage: {leaked}")

    required_top = {
        "schema_version",
        "sample_id",
        "output_language",
        "scorer",
        "items",
        "diagnosis",
        "recommendation",
        "grounding_rules",
    }
    if set(evidence) != required_top:
        raise ValueError("V2 evidence top-level schema is invalid")
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION_V2:
        raise ValueError(
            f"schema_version must be {EVIDENCE_SCHEMA_VERSION_V2!r}"
        )
    if evidence.get("grounding_rules") != list(GROUNDING_RULES_V2):
        raise ValueError("grounding_rules must match the frozen V2 evidence contract")

    # Reuse the fully-audited V1 scorer/items/LOO validator without relaxing V1.
    v1_view = json.loads(json.dumps(evidence))
    v1_view["schema_version"] = "vlm-evidence-v1"
    v1_view["recommendation"] = {"status": "not_implemented", "items": []}
    v1_view["grounding_rules"] = list(V1_GROUNDING_RULES)
    base_v1 = validate_vlm_evidence(v1_view)

    recommendation = validate_recommendation_evidence(
        evidence.get("recommendation"),
        base_evidence=base_v1,
    )
    normalized = json.loads(json.dumps(evidence))
    normalized["recommendation"] = recommendation
    return normalized


def canonical_evidence_json_v2(evidence: Mapping[str, object]) -> str:
    normalized = validate_vlm_evidence_v2(evidence)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
