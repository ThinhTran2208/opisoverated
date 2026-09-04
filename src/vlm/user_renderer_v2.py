"""Deploy-facing Vietnamese renderer for VLM V2.

The internal VLM analysis may record support, ambiguity, or disagreement for
quality control. End users should not see that internal debate. The authoritative
problematic item and Top-3 recommendation ranks come from the frozen upstream
pipeline; Qwen contributes only optional image-grounded visual reasons.
"""

from __future__ import annotations

from typing import Mapping

from .pipeline import CATEGORY_LABELS_VI
from .pipeline_v2 import validate_visual_analysis_v2
from .schema_v2 import validate_vlm_evidence_v2


USER_FACING_SCHEMA_VERSION_V2 = "vlm-user-facing-v2"

_CURRENT_ITEM_LABEL_VI = {
    "TOP": "Chiếc áo hiện tại",
    "BOTTOM": "Món quần/váy hiện tại",
    "DRESS": "Chiếc váy liền hiện tại",
    "OUTERWEAR": "Chiếc áo khoác hiện tại",
    "SHOES": "Đôi giày hiện tại",
    "BAG": "Chiếc túi hiện tại",
    "HAT": "Chiếc mũ hiện tại",
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
        return "Điểm chưa phù hợp dễ thấy nhất là " + phrases[0] + "."
    return "Các điểm chưa phù hợp dễ thấy là " + "; ".join(phrases) + "."


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
        return f"{display_name} là lựa chọn được xếp hạng đầu tiên"
    if rank == 2:
        return f"{display_name} là phương án thay thế thứ hai"
    return f"{display_name} là phương án thay thế thứ ba"


def render_user_facing_vi_v2(
    analysis: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict:
    """Render a decisive end-user explanation without exposing internal debate.

    Upstream diagnosis and recommendation ranks remain authoritative. Qwen's
    visual analysis is used only when it provides positive, image-grounded
    evidence that helps explain an already-fixed decision. Ambiguous or
    contradictory internal visual labels are retained in ``vlm-run-v2`` for QA
    but are deliberately not surfaced as user-facing counter-arguments.
    """

    normalized_evidence = validate_vlm_evidence_v2(evidence)
    normalized_analysis = validate_visual_analysis_v2(analysis, normalized_evidence)

    diagnosis = normalized_evidence["diagnosis"]
    problem_index = int(diagnosis["problematic_item_index"])
    problem_id = str(diagnosis["problematic_item_id"])
    problem_coarse = str(diagnosis["problematic_category"])
    problem_category = CATEGORY_LABELS_VI[problem_coarse]
    current_item_label = _CURRENT_ITEM_LABEL_VI[problem_coarse]
    current_item_inline = current_item_label[0].lower() + current_item_label[1:]

    category_by_index = {
        int(row["item_index"]): str(row["coarse_category"])
        for row in normalized_evidence["items"]
    }

    problematic_headline = (
        f"{current_item_label} là món được đánh giá kém phù hợp nhất và được ưu tiên thay trong outfit."
    )
    diagnosis_reason = _diagnosis_support_reason(
        normalized_analysis["diagnosis"],
        problem_index=problem_index,
        category_by_index=category_by_index,
    )
    problematic_reason = diagnosis_reason or (
        "Đây là món được ưu tiên thay để cải thiện độ phù hợp tổng thể của outfit."
    )

    authoritative_candidates = normalized_evidence["recommendation"]["items"]
    all_improve = all(float(row["improvement_logit"]) > 0 for row in authoritative_candidates)
    replacement_noun = _RECOMMENDATION_NOUN_VI[problem_coarse]
    if all_improve:
        summary = (
            f"Cả ba {replacement_noun} bên dưới đều được đánh giá phù hợp hơn khi thay cho "
            f"{current_item_inline}."
        )
    else:
        summary = (
            f"Ba {replacement_noun} bên dưới là các phương án thay thế được xếp hạng cao nhất "
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
            reason = f"{lead}; {visual_reason}."
        elif improves_original:
            reason = f"{lead} và được đánh giá phù hợp hơn {current_item_inline}."
        else:
            reason = f"{lead} trong ba lựa chọn được đề xuất."

        recommendation_sentences.append(reason)
        recommendations.append(
            {
                "rank": rank,
                "item_id": item_id,
                "master_category": str(candidate["master_category"]),
                "coarse_category": coarse_category,
                "display_name": display_name,
                "headline": f"{display_name}: phương án thay thế cho món hiện tại.",
                "reason": reason,
            }
        )

    caution = (
        f"Bạn có thể tham khảo ba phương án trên để thay cho {current_item_inline} và chọn mẫu "
        "phù hợp nhất với sở thích của mình."
    )

    final_text = " ".join(
        [
            problematic_headline,
            problematic_reason,
            summary,
            *recommendation_sentences,
            caution,
        ]
    )

    return {
        "schema_version": USER_FACING_SCHEMA_VERSION_V2,
        "text": final_text,
        "problematic_item": {
            "item_index": problem_index,
            "item_id": problem_id,
            "category": problem_category,
            "headline": problematic_headline,
            "reason": problematic_reason,
        },
        "summary": summary,
        "recommendations": recommendations,
        "caution": caution,
    }
