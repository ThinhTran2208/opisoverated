# -*- coding: utf-8 -*-
"""Constrained Qwen3-VL prompt for diagnosis + Recommendation V2 explanation.

Baseline V2 visual input intentionally contains only:
- one crop image for each original outfit item; and
- exactly one image for each authoritative Top-3 recommendation candidate.

The optional full original outfit image is deliberately outside this contract so
it can be evaluated later as a separate ablation without changing the baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

from .schema_v2 import EVIDENCE_SCHEMA_VERSION_V2, validate_vlm_evidence_v2


VISUAL_ANALYSIS_SCHEMA_VERSION_V2 = "vlm-visual-analysis-v2"

VISUAL_DIMENSIONS_V2 = (
    "color_harmony",
    "pattern_coherence",
    "silhouette_balance",
    "formality_alignment",
    "style_coherence",
)
VISUAL_CONFIDENCE_LEVELS_V2 = ("low", "medium", "high")

DIAGNOSIS_EFFECTS_V2 = (
    "supports_loo",
    "ambiguous",
    "contradicts_loo",
)
DIAGNOSIS_OVERALL_SUPPORT_V2 = DIAGNOSIS_EFFECTS_V2

RECOMMENDATION_EFFECTS_V2 = (
    "supports_recommendation",
    "ambiguous",
    "contradicts_recommendation",
)
RECOMMENDATION_OVERALL_SUPPORT_V2 = RECOMMENDATION_EFFECTS_V2

BASE_REQUIRED_LIMITATIONS_V2 = (
    "compatibility_logit_is_not_probability",
    "recommendation_scores_are_not_probabilities",
    "recommendation_identity_and_rank_are_authoritative",
    "vlm_visual_observations_are_inferences",
)
TWO_ITEM_EXTRAPOLATION_LIMITATION = "loo_uses_two_item_extrapolation"


SYSTEM_PROMPT_V2 = """You are the constrained visual-analysis layer of a
fashion-compatibility system.

The frozen scorer, Leave-One-Out (LOO) diagnosis, and Recommendation V2 module
have already made all numerical decisions. Candidate identity and rank are
authoritative. You may only classify visible relations using the closed taxonomy
in the requested JSON. Application code, not you, will render final Vietnamese
prose.

Hard rules:
1. Copy diagnosis.problematic_item_index and problematic_item_id exactly. Never
   choose a different problematic item.
2. Copy all three recommendation ranks and item_ids exactly and in the same
   order. Never invent, remove, replace, or rerank recommendation candidates.
3. Return no natural-language prose. Every generated string other than copied
   item_ids must be one of the explicitly allowed enum or schema tokens.
4. Use only visible color, pattern, silhouette, formality, and style relations.
   Do not infer brand, material, price, occasion, user intent, demographics, or
   any property not directly supported by the supplied images.
5. Recommendation observations must compare each candidate with the remaining
   original outfit context. Do not use the problematic original item as a
   context item because the candidate is intended to replace it.
6. Treat scorer logits, LOO deltas, and recommendation improvement logits as
   uncalibrated model outputs, never as probabilities, percentages, or objective
   fashion truth.
7. Visual observations may support, contradict, or remain ambiguous relative to
   the authoritative numerical decisions. Do not force agreement with LOO or
   Recommendation V2.
8. Copy the exact required limitations list, including the conditional
   loo_uses_two_item_extrapolation token when requested.
9. Return exactly one JSON object with no Markdown fence and no extra keys.
"""


def required_limitations_v2(evidence: Mapping[str, object]) -> tuple[str, ...]:
    """Return exact machine-readable disclosures for one V2 case."""

    normalized = validate_vlm_evidence_v2(evidence)
    limitations = list(BASE_REQUIRED_LIMITATIONS_V2)
    if normalized["diagnosis"]["uses_two_item_extrapolation"]:
        limitations.append(TWO_ITEM_EXTRAPOLATION_LIMITATION)
    return tuple(limitations)


def _normalize_image_ref_v2(value: object, *, must_exist: bool) -> str:
    if not isinstance(value, (str, Path)):
        raise TypeError("Every image reference must be a string or Path")
    text = str(value).strip()
    if not text:
        raise ValueError("Image references may not be empty")
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https", "data", "file"}:
        return text
    path = Path(text).expanduser().resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"Missing VLM V2 image: {path}")
    return path.as_uri()


def _validate_pixel_budget(min_pixels: object, max_pixels: object) -> tuple[int, int]:
    if isinstance(min_pixels, bool) or not isinstance(min_pixels, int):
        raise ValueError("min_pixels must be an integer")
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int):
        raise ValueError("max_pixels must be an integer")
    if min_pixels < 1 or min_pixels > max_pixels:
        raise ValueError("Require 1 <= min_pixels <= max_pixels")
    return min_pixels, max_pixels


def _normalize_recommendation_image_refs(
    evidence: Mapping[str, object],
    recommendation_image_refs: Mapping[str, str | Path],
    *,
    must_exist: bool,
) -> dict[str, str]:
    if not isinstance(recommendation_image_refs, Mapping):
        raise TypeError(
            "recommendation_image_refs must be a mapping keyed by candidate item_id"
        )
    candidate_ids = [str(row["item_id"]) for row in evidence["recommendation"]["items"]]
    supplied_keys = {str(key) for key in recommendation_image_refs}
    expected_keys = set(candidate_ids)
    if supplied_keys != expected_keys:
        missing = sorted(expected_keys - supplied_keys)
        extra = sorted(supplied_keys - expected_keys)
        raise ValueError(
            "recommendation_image_refs keys must exactly match authoritative Top-3 "
            f"candidate IDs; missing={missing}, extra={extra}"
        )

    normalized: dict[str, str] = {}
    for item_id in candidate_ids:
        normalized[item_id] = _normalize_image_ref_v2(
            recommendation_image_refs[item_id], must_exist=must_exist
        )
    return normalized


def expected_output_shape_v2(evidence: Mapping[str, object]) -> dict:
    """Build one closed-taxonomy example preserving all authoritative identities."""

    normalized = validate_vlm_evidence_v2(evidence)
    diagnosis = normalized["diagnosis"]
    problem_index = int(diagnosis["problematic_item_index"])
    context_indices = [
        int(row["item_index"])
        for row in normalized["items"]
        if int(row["item_index"]) != problem_index
    ]
    example_context_index = context_indices[0]

    recommendations = []
    for candidate in normalized["recommendation"]["items"]:
        recommendations.append(
            {
                "rank": int(candidate["rank"]),
                "item_id": candidate["item_id"],
                "overall_visual_support": "ambiguous",
                "visual_observations": [
                    {
                        "context_item_indices": [example_context_index],
                        "dimension": "style_coherence",
                        "effect": "ambiguous",
                        "confidence": "low",
                    }
                ],
            }
        )

    return {
        "schema_version": VISUAL_ANALYSIS_SCHEMA_VERSION_V2,
        "problematic_item_index": problem_index,
        "problematic_item_id": diagnosis["problematic_item_id"],
        "diagnosis": {
            "overall_visual_support": "ambiguous",
            "visual_observations": [
                {
                    "item_indices": [problem_index, example_context_index],
                    "dimension": "style_coherence",
                    "effect": "ambiguous",
                    "confidence": "low",
                }
            ],
        },
        "recommendations": recommendations,
        "limitations": list(required_limitations_v2(normalized)),
    }


def build_qwen_messages_v2(
    evidence: Mapping[str, object],
    outfit_image_refs: Sequence[str | Path],
    recommendation_image_refs: Mapping[str, str | Path],
    *,
    min_pixels: int,
    max_pixels: int,
    must_exist: bool = True,
) -> list[dict]:
    """Bind original garment crops + authoritative Top-3 images to V2 evidence.

    Outfit crops remain positional because item_index is canonical and already
    frozen by V1. Recommendation images are keyed by item_id rather than passed
    positionally; this prevents an accidental rank/image mismatch at the caller
    boundary.
    """

    normalized = validate_vlm_evidence_v2(evidence)
    items = normalized["items"]
    if isinstance(outfit_image_refs, (str, bytes)) or not isinstance(
        outfit_image_refs, Sequence
    ):
        raise TypeError("outfit_image_refs must be a sequence")
    if len(outfit_image_refs) != len(items):
        raise ValueError(
            "outfit_image_refs must contain exactly one crop image per original item"
        )
    min_pixels, max_pixels = _validate_pixel_budget(min_pixels, max_pixels)
    candidate_images = _normalize_recommendation_image_refs(
        normalized,
        recommendation_image_refs,
        must_exist=must_exist,
    )

    problem_index = int(normalized["diagnosis"]["problematic_item_index"])
    content: list[dict] = []

    content.append(
        {
            "type": "text",
            "text": (
                "VISUAL INPUT GROUP: ORIGINAL OUTFIT ITEM CROPS. These images are "
                "the original outfit before replacement."
            ),
        }
    )
    for item, image_ref in zip(items, outfit_image_refs):
        item_index = int(item["item_index"])
        content.append(
            {
                "type": "text",
                "text": (
                    "ORIGINAL OUTFIT ITEM: "
                    f"item_index={item_index}, "
                    f"item_id={item['item_id']}, "
                    f"coarse_category={item['coarse_category']}, "
                    f"problematic_item={'true' if item_index == problem_index else 'false'}."
                ),
            }
        )
        content.append(
            {
                "type": "image",
                "image": _normalize_image_ref_v2(image_ref, must_exist=must_exist),
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )

    content.append(
        {
            "type": "text",
            "text": (
                "VISUAL INPUT GROUP: AUTHORITATIVE RECOMMENDATION CANDIDATES. "
                "Each candidate replaces the problematic original item; identity "
                "and rank are fixed by Recommendation V2."
            ),
        }
    )
    for candidate in normalized["recommendation"]["items"]:
        item_id = str(candidate["item_id"])
        content.append(
            {
                "type": "text",
                "text": (
                    "AUTHORITATIVE RECOMMENDATION CANDIDATE: "
                    f"rank={candidate['rank']}, "
                    f"item_id={item_id}, "
                    f"master_category={candidate['master_category']}, "
                    f"coarse_category={candidate['coarse_category']}. "
                    "Do not change this candidate identity or rank."
                ),
            }
        )
        content.append(
            {
                "type": "image",
                "image": candidate_images[item_id],
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )

    prompt_payload = json.dumps(normalized, indent=2, ensure_ascii=False)
    output_shape = json.dumps(
        expected_output_shape_v2(normalized), indent=2, ensure_ascii=False
    )
    context_indices = [
        int(row["item_index"])
        for row in normalized["items"]
        if int(row["item_index"]) != problem_index
    ]

    content.append(
        {
            "type": "text",
            "text": (
                f"The following {EVIDENCE_SCHEMA_VERSION_V2} JSON is authoritative.\n"
                f"EVIDENCE:\n{prompt_payload}\n\n"
                "Return JSON with exactly the requested shape and no free-text fields. "
                "Diagnosis visual_observations may be repeated or empty; every diagnosis "
                "observation must include the authoritative problematic item. Recommendation "
                "visual_observations may be repeated or empty and may reference only these "
                f"remaining original outfit context indices: {context_indices}. "
                "Recommendation rows must remain exactly rank 1, 2, 3 with the authoritative "
                "item_ids shown in the requested shape. If an observation list is empty, its "
                "overall_visual_support must be ambiguous.\n"
                f"Allowed dimension tokens: {list(VISUAL_DIMENSIONS_V2)}.\n"
                f"Allowed diagnosis effect/overall tokens: {list(DIAGNOSIS_EFFECTS_V2)}.\n"
                "Allowed recommendation effect/overall tokens: "
                f"{list(RECOMMENDATION_EFFECTS_V2)}.\n"
                f"Allowed confidence tokens: {list(VISUAL_CONFIDENCE_LEVELS_V2)}.\n"
                f"REQUESTED SHAPE:\n{output_shape}"
            ),
        }
    )

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT_V2}],
        },
        {"role": "user", "content": content},
    ]


def append_repair_request_v2(
    messages: Sequence[Mapping[str, object]],
    *,
    raw_response: str,
    validation_error: str,
) -> list[dict]:
    """Request one schema repair without allowing identity/rank changes."""

    repaired = json.loads(json.dumps(messages))
    repaired.append({"role": "assistant", "content": raw_response})
    repaired.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Your previous response failed deterministic VLM V2 validation: "
                        f"{validation_error}. Return corrected JSON only. Do not emit "
                        "natural-language strings. Do not change the problematic item, "
                        "recommendation candidate identities, or recommendation ranks. "
                        "Use only enum tokens from the requested shape."
                    ),
                }
            ],
        }
    )
    return repaired
