"""Deploy-facing Vietnamese renderer for VLM V2.

The internal VLM analysis may record support, ambiguity, or disagreement for
quality control. End users should not see that internal debate. The authoritative
problematic item and Top-3 recommendation ranks come from the frozen upstream
pipeline; Qwen contributes only optional image-grounded positive visual reasons.
"""

from __future__ import annotations

from typing import Mapping

from .pipeline import CATEGORY_LABELS_VI
from .pipeline_v2 import validate_visual_analysis_v2
from .schema_v2 import validate_vlm_evidence_v2


USER_FACING_SCHEMA_VERSION_V2 = "vlm-user-facing-v2"

_CURRENT_ITEM_LABEL_VI = {
    "TOP": "chiếc áo hiện tại",
    "BOTTOM": "món quần/váy hiện tại",
    "DRESS": "chiếc váy liền hiện tại",
    "OUTERWEAR": "chiếc áo khoác hiện tại",
    "SHOES": "đôi giày hiện tại",
    "BAG": "chiếc túi hiện tại",
    "HAT": "chiếc mũ hiện tại",
}

_RECOMMENDATION_NOUN_VI = {
    "TOP": "mẫu áo",
    "BOTTOM": "mẫu quần/váy",
    "DRESS": "mẫu váy liền",
    "OUTERWEAR": "mẫu áo khoác",
    "SHOES": "mẫu giày",
    "BAG": "mẫu túi",
    "HAT": "mẫu mũ",
}

_CONTEXT_NOUN_VI = {
    "TOP": "áo",
    "BOTTOM": "quần/váy",
    "DRESS": "váy liền",
    "OUTERWEAR": "áo khoác",
    "SHOES": "giày",
    "BAG": "túi",
    "HAT": "mũ",
}

_DIAGNOSIS_SUPPORT_PHRASES = {
    "color_harmony": "màu sắc chưa phối hợp tốt với {context}",
    "pattern_coherence": "họa tiết chưa ăn khớp tốt với {context}",
    "silhouette_balance": "phom dáng chưa cân bằng tốt với {context}",
    "formality_alignment": "mức độ trang trọng chưa đồng nhất với {context}",
    "style_coherence": "phong cách tổng thể chưa đồng nhất với {context}",
}

_RECOMMENDATION_SUPPORT_PHRASES = {
    "color_harmony": "màu sắc phối hợp tốt với các món còn lại",
    "pattern_coherence": "họa tiết phối hợp tốt với các món còn lại",
    "silhouette_balance": "phom dáng giúp tổng thể outfit cân đối hơn",
    "formality_alignment": "mức độ trang trọng phù hợp với các món còn lại",
    "style_coherence": "phong cách tổng thể đồng nhất hơn với outfit",
}


def _sentence_case(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def _join_vi(values: list[str]) -> str:
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    if not unique:
        return "các món còn lại"
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} và {unique[1]}"
    return ", ".join(unique[:-1]) + f" và {unique[-1]}"


def _context_label(
    indices: list[int],
    problem_index: int,
    category_by_index: Mapping[int, str],
) -> str:
    labels = [
        _CONTEXT_NOUN_VI[category_by_index[index]]
        for index in indices
        if index != problem_index and index in category_by_index
    ]
    return _join_vi(labels)


def _diagnosis_support_reason(
    analysis: Mapping[str, object],
    *,
    problem_index: int,
    category_by_index: Mapping[int, str],
) -> str | None:
    if str(analysis["overall_visual_support"]) != "supports_loo":
        return None

    phrases: list[str] = []
    for row in analysis["visual_observations"]:
        if row["effect"] != "supports_loo":
            continue
        dimension = str(row["dimension"])
        context = _context_label(
            list(row["item_indices"]),
            problem_index,
            category_by_index,
        )
        phrases.append(_DIAGNOSIS_SUPPORT_PHRASES[dimension].format(context=context))
        if len(phrases) == 2:
            break

    if not phrases:
        return None
    if len(phrases) == 1:
        return "Điều dễ nhận thấy nhất là " + phrases[0] + "."
    return "Các khác biệt dễ nhận thấy là " + "; ".join(phrases) + "."


def _recommendation_support_reason(visual: Mapping[str, object]) -> str | None:
    if str(visual["overall_visual_support"]) != "supports_recommendation":
        return None

    phrases: list[str] = []
    for row in visual["visual_observations"]:
        if row["effect"] != "supports_recommendation":
            continue
        phrases.append(_RECOMMENDATION_SUPPORT_PHRASES[str(row["dimension"])])
        if len(phrases) == 2:
            break

    if not phrases:
        return None
    return "; ".join(phrases)


def _rank_lead(display_name: str, rank: int) -> str:
    if rank == 1:
        return f"{display_name} là lựa chọn ưu tiên nhất"
    if rank == 2:
        return f"{display_name} là lựa chọn thứ hai"
    return f"{display_name} là lựa chọn thứ ba"


def render_user_facing_vi_v2(
    analysis: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict:
    """Render one decisive end-user explanation without exposing internal debate.

    Upstream diagnosis and recommendation ranks remain authoritative. Qwen's
    visual analysis is used only when it provides positive, image-grounded
    evidence that helps explain an already-fixed decision. Ambiguous or
    contradictory internal visual labels remain available in ``vlm-run-v2`` for
    QA but are deliberately not surfaced as user-facing counter-arguments.
    """

    normalized_evidence = validate_vlm_evidence_v2(evidence)
    normalized_analysis = validate_visual_analysis_v2(analysis, normalized_evidence)

    diagnosis = normalized_evidence["diagnosis"]
    problem_index = int(diagnosis["problematic_item_index"])
    problem_id = str(diagnosis["problematic_item_id"])
    problem_coarse = str(diagnosis["problematic_category"])
    problem_category = CATEGORY_LABELS_VI[problem_coarse]
    current_item_inline = _CURRENT_ITEM_LABEL_VI[problem_coarse]
    authoritative_candidates = normalized_evidence["recommendation"]["items"]
    all_improve = all(
        float(row["improvement_logit"]) > 0 for row in authoritative_candidates
    )
    replacement_noun = _RECOMMENDATION_NOUN_VI[problem_coarse]
    if all_improve:
        summary = (
            f"Ba {replacement_noun} bên dưới đều là những phương án phù hợp hơn "
            f"với outfit khi thay cho {current_item_inline}."
        )
    else:
        summary = (
            f"Ba {replacement_noun} bên dưới là những phương án thay thế được ưu tiên nhất "
            f"cho {current_item_inline}."
        )

    analysis_by_id = {
        str(row["item_id"]): row for row in normalized_analysis["recommendations"]
    }
    recommendations: list[dict[str, object]] = []
    recommendation_sentences: list[str] = []

    for candidate in authoritative_candidates:
        rank = int(candidate["rank"])
        item_id = str(candidate["item_id"])
        coarse_category = str(candidate["coarse_category"])
        display_name = f"{_RECOMMENDATION_NOUN_VI[coarse_category].capitalize()} {rank}"
        lead = _rank_lead(display_name, rank)
        improves_original = float(candidate["improvement_logit"]) > 0
        visual_reason = _recommendation_support_reason(analysis_by_id[item_id])

        if visual_reason:
            reason = f"{lead}, với {visual_reason}."
        elif improves_original:
            reason = (
                f"{lead} và cũng phù hợp hơn với outfit so với "
                f"{current_item_inline}."
            )
        else:
            reason = f"{lead} trong danh sách gợi ý."

        recommendation_sentences.append(reason)
        recommendations.append(
            {
                "rank": rank,
                "item_id": item_id,
                "master_category": str(candidate["master_category"]),
                "coarse_category": coarse_category,
                "display_name": display_name,
                "headline": display_name,
                "reason": reason,
            }
        )

    closing = (
        f"Bạn có thể tham khảo ba mẫu trên để thay cho {current_item_inline} và chọn phương án "
        "phù hợp nhất với sở thích của mình."
    )

    final_text = " ".join(
        [
            summary,
            *recommendation_sentences,
            closing,
        ]
    )

    return {
        "schema_version": USER_FACING_SCHEMA_VERSION_V2,
        "text": final_text,
        "problematic_item": {
            "item_index": problem_index,
            "item_id": problem_id,
            "category": problem_category,
        },
        "summary": summary,
        "recommendations": recommendations,
        "caution": closing,
    }
