# -*- coding: utf-8 -*-
"""Constrained Qwen3-VL messages for grounded visual analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

from .schema import EVIDENCE_SCHEMA_VERSION, validate_vlm_evidence


VISUAL_ANALYSIS_SCHEMA_VERSION = "vlm-visual-analysis-v1"
EXPLANATION_SCHEMA_VERSION = "vlm-explanation-v1"

BASE_REQUIRED_LIMITATIONS = (
    "recommendation_not_implemented",
    "compatibility_logit_is_not_probability",
    "vlm_visual_observations_are_inferences",
)
TWO_ITEM_EXTRAPOLATION_LIMITATION = "loo_uses_two_item_extrapolation"

VISUAL_DIMENSIONS = (
    "color_harmony",
    "pattern_coherence",
    "silhouette_balance",
    "formality_alignment",
    "style_coherence",
)
VISUAL_EFFECTS = (
    "supports_loo",
    "ambiguous",
    "contradicts_loo",
)
VISUAL_CONFIDENCE_LEVELS = ("low", "medium", "high")
OVERALL_VISUAL_SUPPORT_LEVELS = (
    "supports_loo",
    "ambiguous",
    "contradicts_loo",
)

SYSTEM_PROMPT = """You are the constrained visual-analysis layer of a
fashion-compatibility system.

The frozen scorer and Leave-One-Out (LOO) module have already made the numerical
decision. You may only classify visible relations using the closed taxonomy in
the requested JSON. Application code, not you, will render the final Vietnamese
explanation.

Hard rules:
1. Copy diagnosis.problematic_item_index and problematic_item_id exactly. Never
   choose a different problematic item.
2. Return no natural-language prose. Every string other than the copied item_id
   must be one of the explicitly allowed enum tokens or schema tokens.
3. Do not suggest, describe, rank, or fabricate any replacement item.
4. Use only visible color, pattern, silhouette, formality, and style relations.
   Do not infer brand, material, price, occasion, user intent, or demographics.
5. Treat scorer logits and LOO deltas as uncalibrated model outputs, never as
   probabilities, percentages, or objective fashion truth.
6. Copy the exact required limitations list, including the conditional
   loo_uses_two_item_extrapolation token when it is present in the requested shape.
7. Return exactly one JSON object with no Markdown fence and no extra keys.
"""


def required_limitations(evidence: Mapping[str, object]) -> tuple[str, ...]:
    """Return the exact machine-readable disclosures required for one case."""

    normalized = validate_vlm_evidence(evidence)
    limitations = list(BASE_REQUIRED_LIMITATIONS)
    if normalized["diagnosis"]["uses_two_item_extrapolation"]:
        limitations.append(TWO_ITEM_EXTRAPOLATION_LIMITATION)
    return tuple(limitations)


def _normalize_image_ref(value: object, *, must_exist: bool) -> str:
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
        raise FileNotFoundError(f"Missing item image: {path}")
    return path.as_uri()


def expected_output_shape(evidence: Mapping[str, object]) -> dict:
    """Build an example whose values all belong to the closed output taxonomy."""

    normalized = validate_vlm_evidence(evidence)
    diagnosis = normalized["diagnosis"]
    problem_index = int(diagnosis["problematic_item_index"])
    comparison_index = next(
        index for index in range(len(normalized["items"])) if index != problem_index
    )
    return {
        "schema_version": VISUAL_ANALYSIS_SCHEMA_VERSION,
        "problematic_item_index": problem_index,
        "problematic_item_id": diagnosis["problematic_item_id"],
        "overall_visual_support": "ambiguous",
        "visual_observations": [
            {
                "item_indices": [problem_index, comparison_index],
                "dimension": "style_coherence",
                "effect": "ambiguous",
                "confidence": "low",
            }
        ],
        "limitations": list(required_limitations(normalized)),
    }


def build_qwen_messages(
    evidence: Mapping[str, object],
    image_refs: Sequence[str | Path],
    *,
    min_pixels: int,
    max_pixels: int,
    must_exist: bool = True,
) -> list[dict]:
    """Build ordered image/evidence messages for constrained Qwen3-VL output."""

    normalized = validate_vlm_evidence(evidence)
    items = normalized["items"]
    if len(image_refs) != len(items):
        raise ValueError("image_refs must contain exactly one image per item")
    if isinstance(min_pixels, bool) or not isinstance(min_pixels, int):
        raise ValueError("min_pixels must be an integer")
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int):
        raise ValueError("max_pixels must be an integer")
    if min_pixels < 1 or min_pixels > max_pixels:
        raise ValueError("Require 1 <= min_pixels <= max_pixels")

    content: list[dict] = []
    for item, image_ref in zip(items, image_refs):
        content.append(
            {
                "type": "text",
                "text": (
                    "Visual input mapping: "
                    f"item_index={item['item_index']}, "
                    f"item_id={item['item_id']}, "
                    f"coarse_category={item['coarse_category']}."
                ),
            }
        )
        content.append(
            {
                "type": "image",
                "image": _normalize_image_ref(image_ref, must_exist=must_exist),
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )

    prompt_payload = json.dumps(normalized, indent=2, ensure_ascii=False)
    output_shape = json.dumps(
        expected_output_shape(normalized), indent=2, ensure_ascii=False
    )
    content.append(
        {
            "type": "text",
            "text": (
                f"The following {EVIDENCE_SCHEMA_VERSION} JSON is authoritative.\n"
                f"EVIDENCE:\n{prompt_payload}\n\n"
                "Return JSON with exactly the requested shape and no free-text "
                "fields. You may repeat visual_observations rows or return an "
                "empty list. Every observation must include the authoritative "
                "problematic item. Allowed dimension tokens: "
                f"{list(VISUAL_DIMENSIONS)}. Allowed effect tokens: "
                f"{list(VISUAL_EFFECTS)}. Allowed confidence tokens: "
                f"{list(VISUAL_CONFIDENCE_LEVELS)}. Allowed overall tokens: "
                f"{list(OVERALL_VISUAL_SUPPORT_LEVELS)}.\n"
                f"REQUESTED SHAPE:\n{output_shape}"
            ),
        }
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def append_repair_request(
    messages: Sequence[Mapping[str, object]],
    *,
    raw_response: str,
    validation_error: str,
) -> list[dict]:
    repaired = json.loads(json.dumps(messages))
    repaired.append({"role": "assistant", "content": raw_response})
    repaired.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Your previous response failed deterministic validation: "
                        f"{validation_error}. Return corrected JSON only. Do not "
                        "emit natural-language strings or change the authoritative "
                        "problematic item. Use only the enum tokens in the requested shape."
                    ),
                }
            ],
        }
    )
    return repaired
