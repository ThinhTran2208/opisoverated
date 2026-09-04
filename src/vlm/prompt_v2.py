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


SYSTEM_PROMPT_V2 = """You are the constrained visual-evidence extraction layer of a
fashion-compatibility system. You are NOT a second decision-maker and you are NOT
a stylist who is allowed to replace the upstream decisions.

The frozen scorer, Leave-One-Out (LOO) diagnosis, and Recommendation V2 module
have already decided which original item is problematic and which three
replacement candidates are ranked 1, 2, and 3. Those identities and ranks are
authoritative. Application code will always present those decisions to the end
user. Your job is narrower: inspect the supplied images and extract only useful,
image-grounded visual evidence that can help explain the fixed decisions.

Think of your output as internal evidence for a deterministic renderer, not as a
vote on whether the upstream system was right. Do not write user-facing prose.

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
5. Every diagnosis observation is relational: it must include the problematic
   item and at least one other original outfit item. Prefer one strongest useful
   relation; add a second only when it contributes a genuinely different visual
   dimension.
6. For diagnosis, first look for visible evidence that helps explain why the
   already-fixed problematic item fits the outfit less well. Use supports_loo
   only when such evidence is actually visible. If no convincing supporting
   relation is visible, use ambiguous rather than inventing a reason.
   contradicts_loo is reserved for clear internal-QA evidence that the fixed item
   visually aligns with the outfit; do not search for contradiction as the main
   task and do not manufacture it merely because the item looks acceptable in
   isolation.
7. Each recommendation candidate has already been selected as a replacement.
   Evaluate each candidate independently against the remaining original outfit
   context, excluding the problematic original item. First look for one concrete
   positive visible relation that can explain why the candidate works with the
   remaining outfit. Use supports_recommendation only when grounded in the
   images.
8. If a recommendation candidate has no clear positive visual reason, use
   ambiguous with an empty visual_observations list instead of producing filler,
   weak negative commentary, or a made-up justification. Absence of a strong
   visual explanation does NOT mean the candidate should be removed or reranked.
9. Use contradicts_recommendation only for a clear visible clash with the
   remaining outfit. This token is internal quality-control evidence only; it
   never authorizes you to remove, replace, or rerank the candidate.
10. Do not compare recommendation candidates against each other to decide their
    rank. Their rank is already frozen. Your task is to explain each candidate's
    relation to the remaining outfit, not to recreate the ranking.
11. Treat scorer logits, LOO deltas, and recommendation improvement logits as
    uncalibrated model outputs. Never turn them into probabilities, percentages,
    confidence scores, or visual evidence. Never infer a visual label from a
    numerical score.
12. Copy the exact required limitations list, including the conditional
    loo_uses_two_item_extrapolation token when requested.
13. Return exactly one JSON object with no Markdown fence and no extra keys.
14. There is deliberately no populated visual answer in the prompt. Determine
    every dimension, effect, confidence, and overall label from the supplied
    images. Do not use a default label merely because it is listed as allowed.
15. Inspect the diagnosis and each recommendation independently. Do not
    mechanically reuse the same context indices, dimension, effect, confidence,
    or observation list across candidates. In particular, do not clone the exact
    same non-ambiguous high-confidence analysis across all Top-3 candidates. If
    candidates are visually too similar to distinguish confidently, use
    ambiguous or lower confidence rather than asserting copied evidence.
16. Confidence describes only how clearly the claimed visual relation is visible
    in the supplied images. It is not confidence in the scorer, LOO decision, or
    recommendation rank.
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
    if any(not isinstance(key, str) for key in recommendation_image_refs):
        raise TypeError("recommendation_image_refs keys must be candidate item_id strings")
    candidate_ids = [str(row["item_id"]) for row in evidence["recommendation"]["items"]]
    supplied_keys = set(recommendation_image_refs)
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
    """Build a validator-safe fixture preserving authoritative identities.

    This helper is intentionally retained for deterministic unit tests and fake
    backends. It MUST NOT be embedded in the real Qwen prompt because populated
    semantic values can anchor the model toward the fixture instead of the
    supplied images.
    """

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


def _output_contract_text_v2(evidence: Mapping[str, object]) -> str:
    """Describe the exact output structure without seeding semantic answers."""

    normalized = validate_vlm_evidence_v2(evidence)
    diagnosis = normalized["diagnosis"]
    problem_index = int(diagnosis["problematic_item_index"])
    problem_id = str(diagnosis["problematic_item_id"])
    context_indices = [
        int(row["item_index"])
        for row in normalized["items"]
        if int(row["item_index"]) != problem_index
    ]
    limitations = list(required_limitations_v2(normalized))
    recommendation_identity_lines = "\n".join(
        (
            f"  - row {position}: rank={int(candidate['rank'])}, "
            f"item_id={candidate['item_id']}"
        )
        for position, candidate in enumerate(
            normalized["recommendation"]["items"], start=1
        )
    )

    return (
        "OUTPUT JSON CONTRACT (schema instructions, NOT an example analysis):\n"
        "Top-level keys must be exactly: schema_version, problematic_item_index, "
        "problematic_item_id, diagnosis, recommendations, limitations; no free-text fields "
        "are allowed.\n"
        f"- schema_version must be exactly {VISUAL_ANALYSIS_SCHEMA_VERSION_V2}.\n"
        f"- problematic_item_index must be exactly {problem_index}.\n"
        f"- problematic_item_id must be exactly {problem_id}.\n"
        "- diagnosis must contain exactly overall_visual_support and visual_observations.\n"
        f"  * overall_visual_support: choose one of {list(DIAGNOSIS_OVERALL_SUPPORT_V2)} "
        "from the outfit images. This field is internal visual evidence; it does not change "
        "the authoritative problematic item.\n"
        "  * First seek a concrete visible relation that supports why the fixed problematic "
        "item fits less well. If no grounded supporting relation is visible, use ambiguous "
        "rather than inventing one. Use contradicts_loo only for clear internal-QA evidence.\n"
        "  * visual_observations: a JSON list of zero or more objects. Prefer one strongest "
        "useful visible relation; add a second only when it contributes a different dimension. "
        "Do not add filler observations.\n"
        "  * each diagnosis observation must contain exactly item_indices, dimension, "
        "effect, confidence. item_indices must contain at least two indices: the problematic "
        "index plus at least one other original outfit item. Only original outfit items may "
        "be referenced.\n"
        f"  * dimension: choose one of {list(VISUAL_DIMENSIONS_V2)}.\n"
        f"  * effect: choose one of {list(DIAGNOSIS_EFFECTS_V2)}.\n"
        f"  * confidence: choose one of {list(VISUAL_CONFIDENCE_LEVELS_V2)}.\n"
        "- recommendations must be a JSON list of exactly three objects in this exact "
        "identity order:\n"
        f"{recommendation_identity_lines}\n"
        "  * each recommendation object must contain exactly rank, item_id, "
        "overall_visual_support, visual_observations.\n"
        f"  * overall_visual_support: choose one of {list(RECOMMENDATION_OVERALL_SUPPORT_V2)} "
        "from that candidate image versus the remaining outfit context. This field does not "
        "change candidate identity or rank.\n"
        "  * For each candidate, first seek one concrete positive visible relation that can "
        "help explain why it works with the remaining outfit. If none is clearly visible, "
        "use ambiguous with an empty visual_observations list. Do not create negative filler "
        "merely because no strong visual reason is available.\n"
        "  * visual_observations: a JSON list of zero or more objects. Prefer one strongest "
        "useful relation; add a second only when it contributes a genuinely different "
        "dimension. Evaluate each candidate independently.\n"
        "  * each recommendation observation must contain exactly context_item_indices, "
        "dimension, effect, confidence. It may reference only the remaining original outfit "
        f"context indices. The remaining original outfit context indices: {context_indices}. "
        "Use every context index materially involved in the claimed visible relation; do "
        "not default to the first index.\n"
        "  * Do not use the problematic original item as recommendation context. Do not "
        "compare candidates against each other to recreate the ranking.\n"
        "  * Do not clone the exact same non-ambiguous high-confidence observation pattern "
        "across all three candidates. If they are visually too similar to distinguish, use "
        "lower confidence or ambiguous rather than identical high-confidence claims.\n"
        f"  * dimension: choose one of {list(VISUAL_DIMENSIONS_V2)}.\n"
        f"  * effect: choose one of {list(RECOMMENDATION_EFFECTS_V2)}.\n"
        f"  * confidence: choose one of {list(VISUAL_CONFIDENCE_LEVELS_V2)}.\n"
        "- If any visual_observations list is empty, its overall_visual_support must be "
        "ambiguous. Otherwise the overall label must summarize the visible observations. "
        "Never force support from scorer/LOO/recommendation numbers.\n"
        f"- limitations must be exactly this JSON array: {json.dumps(limitations, ensure_ascii=False)}.\n"
        "Before producing JSON, inspect all supplied images. Your goal is useful grounded "
        "evidence for an already-fixed decision, not a second fashion verdict. Do not infer "
        "visual support from numerical scores and do not copy a semantic default from these "
        "schema instructions."
    )


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
    output_contract = _output_contract_text_v2(normalized)

    content.append(
        {
            "type": "text",
            "text": (
                f"The following {EVIDENCE_SCHEMA_VERSION_V2} JSON is authoritative.\n"
                f"EVIDENCE:\n{prompt_payload}\n\n"
                "Inspect the supplied images first. Remember: the problematic item and Top-3 "
                "candidate identities/ranks are already fixed; your task is to extract useful "
                "grounded visual evidence for those decisions, not to make new decisions. "
                "Then return exactly one JSON object that follows the contract below. There "
                "is intentionally no populated visual-analysis example to copy.\n\n"
                f"{output_contract}"
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
                        "Diagnosis observations must include the problematic item plus at "
                        "least one other original outfit item. For each recommendation, seek "
                        "a grounded positive visual relation to the remaining outfit; if no "
                        "clear positive reason is visible, use ambiguous with an empty "
                        "visual_observations list rather than negative filler or invented "
                        "support. Do not clone the same non-ambiguous high-confidence "
                        "recommendation analysis across all three candidates. Use only enum "
                        "tokens from the output contract. Repair schema or identity mistakes "
                        "without replacing image-grounded labels with mechanical defaults."
                    ),
                }
            ],
        }
    )
    return repaired
