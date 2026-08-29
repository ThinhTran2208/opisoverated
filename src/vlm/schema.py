# -*- coding: utf-8 -*-
"""Strict structured-evidence contract between frozen LOO and the VLM."""

from __future__ import annotations

import json
import math
from typing import Mapping, Sequence


EVIDENCE_SCHEMA_VERSION = "vlm-evidence-v1"
SCORER_VERSION = "type_aware_pairwise_v1"
CANONICAL_CHECKPOINT = "final_val_auc_v5_seed42/best.pt"
LOO_PROTOCOL_VERSION = "loo-diagnostic-v1"
MIN_ITEMS = 3
MAX_ITEMS = 8
ALLOWED_CATEGORIES = frozenset(
    {"TOP", "BOTTOM", "DRESS", "OUTERWEAR", "SHOES", "BAG", "HAT"}
)
GROUNDING_RULES = (
    "scorer_logit_is_not_probability",
    "loo_diagnosis_controls_problematic_item",
    "visual_claims_must_be_visible",
    "recommendation_not_implemented",
)
FORBIDDEN_LEAKAGE_KEYS = frozenset(
    {
        "label",
        "negative_metadata",
        "swapped_item_index",
        "target_swapped_item_index",
        "top1_correct",
        "hit_at_2",
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


def _index(value: object, name: str, item_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= value < item_count:
        raise ValueError(f"{name} is outside [0, {item_count})")
    return value


def _find_forbidden_keys(value: object, *, path: str = "root") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in FORBIDDEN_LEAKAGE_KEYS:
                found.append(child_path)
            found.extend(_find_forbidden_keys(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, path=f"{path}[{index}]"))
    return found


def build_vlm_evidence(
    loo_result: Mapping[str, object],
    *,
    sample_id: str,
    item_ids: Sequence[str],
    coarse_categories: Sequence[str],
    scorer_version: str = SCORER_VERSION,
    checkpoint: str = CANONICAL_CHECKPOINT,
) -> dict:
    """Create evidence without exposing synthetic labels to the VLM.

    ``loo_result`` should be the output of ``diagnose_outfit``. Ground-truth
    swap metadata is deliberately neither accepted nor serialized.
    """

    if not isinstance(loo_result, Mapping):
        raise TypeError("loo_result must be a mapping")
    leaked = _find_forbidden_keys(loo_result, path="loo_result")
    if leaked:
        raise ValueError(f"LOO result contains forbidden target leakage: {leaked}")
    if loo_result.get("protocol_version") != LOO_PROTOCOL_VERSION:
        raise ValueError(f"LOO protocol_version must be {LOO_PROTOCOL_VERSION!r}")

    normalized_sample_id = _nonempty_string(sample_id, "sample_id")
    normalized_item_ids = [
        _nonempty_string(value, f"item_ids[{index}]")
        for index, value in enumerate(item_ids)
    ]
    if not MIN_ITEMS <= len(normalized_item_ids) <= MAX_ITEMS:
        raise ValueError(
            f"VLM evidence requires [{MIN_ITEMS}, {MAX_ITEMS}] original items"
        )
    if len(normalized_item_ids) != len(set(normalized_item_ids)):
        raise ValueError("item_ids must be unique")

    normalized_categories = [
        _nonempty_string(value, f"coarse_categories[{index}]").upper()
        for index, value in enumerate(coarse_categories)
    ]
    if len(normalized_categories) != len(normalized_item_ids):
        raise ValueError("coarse_categories length must match item_ids")
    invalid_categories = sorted(set(normalized_categories) - ALLOWED_CATEGORIES)
    if invalid_categories:
        raise ValueError(f"Unknown coarse categories: {invalid_categories}")

    item_count = len(normalized_item_ids)
    declared_count = loo_result.get("original_item_count")
    if declared_count != item_count:
        raise ValueError("LOO original_item_count does not match item_ids")

    full_logit = _finite_float(loo_result.get("full_logit"), "full_logit")
    raw_without_logits = loo_result.get("without_item_logits")
    raw_deltas = loo_result.get("deltas_without_minus_full")
    raw_ranking = loo_result.get("ranked_item_indices")
    if not isinstance(raw_without_logits, list) or len(raw_without_logits) != item_count:
        raise ValueError("without_item_logits must contain one value per item")
    if not isinstance(raw_deltas, list) or len(raw_deltas) != item_count:
        raise ValueError("deltas_without_minus_full must contain one value per item")
    if not isinstance(raw_ranking, list) or len(raw_ranking) != item_count:
        raise ValueError("ranked_item_indices must contain one index per item")

    without_logits = [
        _finite_float(value, f"without_item_logits[{index}]")
        for index, value in enumerate(raw_without_logits)
    ]
    deltas = [
        _finite_float(value, f"deltas_without_minus_full[{index}]")
        for index, value in enumerate(raw_deltas)
    ]
    ranking = [
        _index(value, f"ranked_item_indices[{rank}]", item_count)
        for rank, value in enumerate(raw_ranking)
    ]
    if sorted(ranking) != list(range(item_count)):
        raise ValueError("ranked_item_indices must be a permutation of item indices")
    expected_ranking = sorted(range(item_count), key=lambda i: (-deltas[i], i))
    if ranking != expected_ranking:
        raise ValueError("ranked_item_indices is inconsistent with LOO deltas")

    for index, (without_logit, delta) in enumerate(zip(without_logits, deltas)):
        if not math.isclose(without_logit - full_logit, delta, abs_tol=1e-5):
            raise ValueError(
                f"LOO delta at item {index} is inconsistent with scorer logits"
            )

    problematic_index = _index(
        loo_result.get("problematic_item_index"),
        "problematic_item_index",
        item_count,
    )
    if problematic_index != ranking[0]:
        raise ValueError("problematic_item_index must equal the top-ranked LOO item")
    result_problematic_id = loo_result.get("problematic_item_id")
    if result_problematic_id not in (None, normalized_item_ids[problematic_index]):
        raise ValueError("problematic_item_id does not match item_ids")

    ranked_items = []
    for rank, item_index in enumerate(ranking, start=1):
        ranked_items.append(
            {
                "rank": rank,
                "item_index": item_index,
                "item_id": normalized_item_ids[item_index],
                "coarse_category": normalized_categories[item_index],
                "without_item_logit": without_logits[item_index],
                "loo_delta": deltas[item_index],
            }
        )

    top2_delta = deltas[ranking[1]] if item_count > 1 else deltas[ranking[0]]
    uses_two_item_extrapolation = loo_result.get("uses_two_item_extrapolation", False)
    if not isinstance(uses_two_item_extrapolation, bool):
        raise ValueError("uses_two_item_extrapolation must be boolean")
    if scorer_version != SCORER_VERSION:
        raise ValueError(f"scorer_version must be {SCORER_VERSION!r}")
    if checkpoint != CANONICAL_CHECKPOINT:
        raise ValueError(f"checkpoint must be {CANONICAL_CHECKPOINT!r}")

    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "sample_id": normalized_sample_id,
        "output_language": "vi",
        "scorer": {
            "version": scorer_version,
            "checkpoint": checkpoint,
            "compatibility_logit": full_logit,
            "semantics": "uncalibrated_logit_not_probability",
        },
        "items": [
            {
                "item_index": index,
                "item_id": item_id,
                "coarse_category": normalized_categories[index],
            }
            for index, item_id in enumerate(normalized_item_ids)
        ],
        "diagnosis": {
            "protocol_version": LOO_PROTOCOL_VERSION,
            "problematic_item_index": problematic_index,
            "problematic_item_id": normalized_item_ids[problematic_index],
            "problematic_category": normalized_categories[problematic_index],
            "ranked_items": ranked_items,
            "top1_top2_delta_gap": deltas[ranking[0]] - top2_delta,
            "certainty": "not_calibrated",
            "uses_two_item_extrapolation": uses_two_item_extrapolation,
        },
        "recommendation": {"status": "not_implemented", "items": []},
        "grounding_rules": list(GROUNDING_RULES),
    }
    return validate_vlm_evidence(evidence)


def validate_vlm_evidence(evidence: Mapping[str, object]) -> dict:
    """Hard-fail malformed, label-leaking, or internally inconsistent evidence."""

    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    leaked = _find_forbidden_keys(evidence)
    if leaked:
        raise ValueError(f"Evidence contains forbidden target leakage: {leaked}")

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
        raise ValueError(
            f"Evidence top-level keys must be exactly {sorted(required_top)}"
        )
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {EVIDENCE_SCHEMA_VERSION!r}")
    _nonempty_string(evidence.get("sample_id"), "sample_id")
    if evidence.get("output_language") != "vi":
        raise ValueError("output_language must be 'vi'")

    scorer = evidence.get("scorer")
    if not isinstance(scorer, Mapping):
        raise ValueError("scorer must be an object")
    if set(scorer) != {"version", "checkpoint", "compatibility_logit", "semantics"}:
        raise ValueError("scorer has unexpected fields")
    if scorer.get("version") != SCORER_VERSION:
        raise ValueError(f"scorer.version must be {SCORER_VERSION!r}")
    if scorer.get("checkpoint") != CANONICAL_CHECKPOINT:
        raise ValueError(f"scorer.checkpoint must be {CANONICAL_CHECKPOINT!r}")
    _finite_float(scorer.get("compatibility_logit"), "scorer.compatibility_logit")
    if scorer.get("semantics") != "uncalibrated_logit_not_probability":
        raise ValueError("scorer.semantics is invalid")

    items = evidence.get("items")
    if not isinstance(items, list) or not MIN_ITEMS <= len(items) <= MAX_ITEMS:
        raise ValueError(f"items must contain [{MIN_ITEMS}, {MAX_ITEMS}] entries")
    item_ids: list[str] = []
    for expected_index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"items[{expected_index}] must be an object")
        if set(item) != {"item_index", "item_id", "coarse_category"}:
            raise ValueError(f"items[{expected_index}] has unexpected fields")
        _index(item.get("item_index"), "item_index", len(items))
        if item.get("item_index") != expected_index:
            raise ValueError("items must use contiguous ordered item_index values")
        item_ids.append(_nonempty_string(item.get("item_id"), "item_id"))
        category = _nonempty_string(item.get("coarse_category"), "coarse_category")
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"Unknown coarse category: {category}")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("items contain duplicate item IDs")

    diagnosis = evidence.get("diagnosis")
    if not isinstance(diagnosis, Mapping):
        raise ValueError("diagnosis must be an object")
    expected_diagnosis_keys = {
        "protocol_version",
        "problematic_item_index",
        "problematic_item_id",
        "problematic_category",
        "ranked_items",
        "top1_top2_delta_gap",
        "certainty",
        "uses_two_item_extrapolation",
    }
    if set(diagnosis) != expected_diagnosis_keys:
        raise ValueError("diagnosis has unexpected fields")
    if diagnosis.get("protocol_version") != LOO_PROTOCOL_VERSION:
        raise ValueError(
            f"diagnosis.protocol_version must be {LOO_PROTOCOL_VERSION!r}"
        )
    problem_index = _index(
        diagnosis.get("problematic_item_index"),
        "diagnosis.problematic_item_index",
        len(items),
    )
    if diagnosis.get("problematic_item_id") != item_ids[problem_index]:
        raise ValueError("diagnosis problematic item ID/index mismatch")
    if diagnosis.get("problematic_category") != items[problem_index]["coarse_category"]:
        raise ValueError("diagnosis problematic category/index mismatch")
    if diagnosis.get("certainty") != "not_calibrated":
        raise ValueError("diagnosis.certainty must remain 'not_calibrated'")
    if not isinstance(diagnosis.get("uses_two_item_extrapolation"), bool):
        raise ValueError("uses_two_item_extrapolation must be boolean")

    ranked_items = diagnosis.get("ranked_items")
    if not isinstance(ranked_items, list) or len(ranked_items) != len(items):
        raise ValueError("diagnosis.ranked_items must cover every item")
    ranking: list[int] = []
    deltas: dict[int, float] = {}
    for expected_rank, row in enumerate(ranked_items, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("ranked_items rows must be objects")
        required = {
            "rank",
            "item_index",
            "item_id",
            "coarse_category",
            "without_item_logit",
            "loo_delta",
        }
        rank = row.get("rank")
        if (
            set(row) != required
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank != expected_rank
        ):
            raise ValueError("ranked_items rank/schema is invalid")
        index = _index(row.get("item_index"), "ranked item_index", len(items))
        if row.get("item_id") != item_ids[index]:
            raise ValueError("ranked item ID/index mismatch")
        if row.get("coarse_category") != items[index]["coarse_category"]:
            raise ValueError("ranked item category/index mismatch")
        without_item_logit = _finite_float(
            row.get("without_item_logit"), "without_item_logit"
        )
        deltas[index] = _finite_float(row.get("loo_delta"), "loo_delta")
        if not math.isclose(
            without_item_logit - float(scorer["compatibility_logit"]),
            deltas[index],
            abs_tol=1e-5,
        ):
            raise ValueError("ranked item LOO delta is inconsistent with scorer logits")
        ranking.append(index)
    if sorted(ranking) != list(range(len(items))):
        raise ValueError("ranked_items must be a permutation")
    expected_ranking = sorted(range(len(items)), key=lambda i: (-deltas[i], i))
    if ranking != expected_ranking or problem_index != ranking[0]:
        raise ValueError("diagnosis ranking is inconsistent with LOO deltas")
    expected_gap = deltas[ranking[0]] - deltas[ranking[1]]
    gap = _finite_float(
        diagnosis.get("top1_top2_delta_gap"), "diagnosis.top1_top2_delta_gap"
    )
    if not math.isclose(gap, expected_gap, abs_tol=1e-7):
        raise ValueError("top1_top2_delta_gap is inconsistent with ranked_items")

    recommendation = evidence.get("recommendation")
    if recommendation != {"status": "not_implemented", "items": []}:
        raise ValueError("recommendation must remain explicitly not implemented")
    if evidence.get("grounding_rules") != list(GROUNDING_RULES):
        raise ValueError("grounding_rules must match the frozen evidence contract")

    return json.loads(json.dumps(evidence))


def canonical_evidence_json(evidence: Mapping[str, object]) -> str:
    normalized = validate_vlm_evidence(evidence)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
