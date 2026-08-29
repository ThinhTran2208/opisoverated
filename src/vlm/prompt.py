# -*- coding: utf-8 -*-
"""Grounded Qwen3-VL messages for compatibility explanations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

from .schema import EVIDENCE_SCHEMA_VERSION, validate_vlm_evidence


EXPLANATION_SCHEMA_VERSION = "vlm-explanation-v1"
REQUIRED_LIMITATIONS = (
    "recommendation_not_implemented",
    "compatibility_logit_is_not_probability",
    "vlm_visual_observations_are_inferences",
)

SYSTEM_PROMPT = """You are the grounded explanation layer of a fashion-compatibility system.

The frozen scorer and Leave-One-Out (LOO) module have already made the numerical
decision. You must explain that evidence; you must not replace, override, or
recalculate it.

Hard rules:
1. The evidence field diagnosis.problematic_item_index is authoritative. Copy
   that index and its item_id exactly into your output even if your visual
   impression differs.
2. Treat compatibility_logit and LOO deltas as model outputs, not probabilities,
   percentages, beauty scores, or objective fashion truth.
3. Put image-derived statements only in visual_observations. Mention only
   directly visible color, silhouette, pattern, formality, or style relations.
   Do not invent brand, material, price, occasion, user intent, or demographics.
4. Recommendation is not implemented. Do not suggest, rank, or fabricate any
   replacement item.
5. If uses_two_item_extrapolation is true, explicitly state that the diagnosis
   includes an out-of-training-distribution two-item LOO subset.
6. If visual evidence is weak or ambiguous, say so. LOO certainty is not calibrated.
7. Return one JSON object only: no Markdown fences and no text outside JSON.
8. Write every human-facing string in Vietnamese.
"""


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
    diagnosis = evidence["diagnosis"]
    return {
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "problematic_item_index": diagnosis["problematic_item_index"],
        "problematic_item_id": diagnosis["problematic_item_id"],
        "headline": "Một câu tóm tắt ngắn bằng tiếng Việt",
        "evidence_summary": [
            "Một đến bốn câu chỉ diễn giải scorer/LOO evidence"
        ],
        "visual_observations": [
            {
                "item_indices": [diagnosis["problematic_item_index"]],
                "observation": "Nhận xét chỉ dựa trên chi tiết nhìn thấy",
            }
        ],
        "explanation": "Đoạn giải thích ngắn gọn bằng tiếng Việt",
        "uncertainty_note": "Nêu rõ giới hạn và mức độ mơ hồ của evidence",
        "limitations": list(REQUIRED_LIMITATIONS),
    }


def build_qwen_messages(
    evidence: Mapping[str, object],
    image_refs: Sequence[str | Path],
    *,
    min_pixels: int,
    max_pixels: int,
    must_exist: bool = True,
) -> list[dict]:
    """Build ordered image/evidence messages for Qwen3-VL.

    Exactly one image is required for every evidence item. Image order is
    explicitly labeled so visual claims can be mapped back to canonical IDs.
    """

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
                "Return JSON with exactly this shape and no extra keys. Replace "
                f"the instructional strings with grounded Vietnamese text:\n{output_shape}"
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
                        f"{validation_error}. Return a corrected JSON object only. "
                        "Keep the authoritative problematic item unchanged."
                    ),
                }
            ],
        }
    )
    return repaired
