"""Deploy-facing Vietnamese renderer for VLM V2.

This module deliberately separates user-facing copy from internal scorer/LOO/Qwen
artifacts. The authoritative identities and ranks still come from validated
VLM V2 evidence/analysis, but the returned prose avoids implementation terms
such as LOO, Qwen, logits, validator names, and raw confidence taxonomy.
"""

from __future__ import annotations

from typing import Mapping

from .pipeline import CATEGORY_LABELS_VI
from .pipeline_v2 import validate_visual_analysis_v2
from .schema_v2 import validate_vlm_evidence_v2


USER_FACING_SCHEMA_VERSION_V2 = "vlm-user-facing-v2"

_DIMENSION_NOUN_VI = {
    "color_harmony": "màu sắc",
    "pattern_coherence": "họa tiết",
    "silhouette_balance": "phom dáng",
    "formality_alignment": "mức độ trang trọng",
    "style_coherence": "phong cách tổng thể",
}

# Natural nouns for end-user prose. These deliberately use the frozen coarse
# category rather than asking Qwen to invent a more specific garment subtype.
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

_DIAGNOSIS_SUPPORT_PHRASES = {
    "color_harmony": "màu sắc chưa hòa hợp tốt với {context}",
    "pattern_coherence": "họa tiết chưa ăn khớp tốt với {context}",
    "silhouette_balance": "phom dáng chưa cân bằng tốt với {context}",
    "formality_alignment": "mức độ trang trọng chưa đồng nhất với {context}",
    "style_coherence": "phong cách tổng thể chưa đồng nhất với {context}",
}

_DIAGNOSIS_CONTRADICT_PHRASES = {
    "color_harmony": "màu sắc vẫn khá hòa hợp với {context}",
    "pattern_coherence": "họa tiết vẫn khá ăn khớp với {context}",
    "silhouette_balance": "phom dáng vẫn khá cân bằng với {context}",
    "formality_alignment": "mức độ trang trọng vẫn khá đồng nhất với {context}",
    "style_coherence": "phong cách tổng thể vẫn khá đồng nhất với {context}",
}

_RECOMMENDATION_SUPPORT_PHRASES = {
    "color_harmony": "màu sắc hài hòa với các món còn lại",
    "pattern_coherence": "họa tiết phối hợp tốt với các món còn lại",
    "silhouette_balance": "phom dáng giúp outfit cân đối hơn",
    "formality_alignment": "mức độ trang trọng phù hợp với outfit",
    "style_coherence": "phong cách tổng thể đồng nhất với outfit",
}

_RECOMMENDATION_CONTRADICT_PHRASES = {
    "color_harmony": "màu sắc chưa thật sự hòa hợp với các món còn lại",
    "pattern_coherence": "họa tiết chưa thật sự ăn khớp với các món còn lại",
    "silhouette_balance": "phom dáng chưa tạo được sự cân bằng rõ ràng",
    "formality_alignment": "mức độ trang trọng chưa thật sự đồng nhất với outfit",
    "style_coherence": "phong cách tổng thể chưa thật sự đồng nhất với outfit",
}


def _context_label(indices: list[int], problem_index: int) -> str:
    context = [index for index in indices if index != problem_index]
    if not context:
        return "các món còn lại"
    if len(context) == 1:
        return f"món số {context[0] + 1}"
    return "các món số " + ", ".join(str(index + 1) for index in context)


def _diagnosis_observation_text(row: Mapping[str, object], problem_index: int) -> str:
    dimension = str(row["dimension"])
    effect = str(row["effect"])
    context = _context_label(list(row["item_indices"]), problem_index)

    if effect == "supports_loo":
        template = _DIAGNOSIS_SUPPORT_PHRASES[dimension]
        return template.format(context=context)
    if effect == "contradicts_loo":
        template = _DIAGNOSIS_CONTRADICT_PHRASES[dimension]
        return template.format(context=context)
    return f"{_DIMENSION_NOUN_VI[dimension]} chưa cho tín hiệu đủ rõ khi so với {context}"


def _diagnosis_reason(analysis: Mapping[str, object], problem_index: int) -> str:
    overall = str(analysis["overall_visual_support"])
    observations = list(analysis["visual_observations"])
    details = [_diagnosis_observation_text(row, problem_index) for row in observations[:2]]

    if overall == "supports_loo":
        prefix = "Hình ảnh cũng cho thấy món này có dấu hiệu kém phù hợp với outfit"
    elif overall == "contradicts_loo":
        prefix = (
            "Tuy nhiên, khi nhìn riêng về mặt thị giác, món này không có dấu hiệu lệch outfit quá rõ"
        )
    else:
        prefix = "Hình ảnh chưa cho thấy dấu hiệu lệch outfit đủ rõ"

    if details:
        return prefix + ": " + "; ".join(details) + "."
    return prefix + "."


def _recommendation_reason(
    visual: Mapping[str, object],
    *,
    display_name: str,
) -> str:
    overall = str(visual["overall_visual_support"])
    observations = list(visual["visual_observations"])

    if overall == "ambiguous":
        return (
            f"{display_name} chưa cho thấy ưu điểm thị giác đủ rõ để nổi bật hơn "
            "các lựa chọn còn lại."
        )

    phrases: list[str] = []
    for row in observations[:2]:
        dimension = str(row["dimension"])
        effect = str(row["effect"])
        if effect == "supports_recommendation":
            phrases.append(_RECOMMENDATION_SUPPORT_PHRASES[dimension])
        elif effect == "contradicts_recommendation":
            phrases.append(_RECOMMENDATION_CONTRADICT_PHRASES[dimension])

    if overall == "supports_recommendation":
        if phrases:
            return f"{display_name} phù hợp về mặt thị giác vì " + "; ".join(phrases) + "."
        return f"{display_name} có tín hiệu thị giác tích cực khi đặt cạnh các món còn lại."

    if phrases:
        return (
            f"{display_name} vẫn là một phương án thay thế, nhưng đánh giá hình ảnh chưa "
            "ủng hộ mạnh vì "
            + "; ".join(phrases)
            + "."
        )
    return (
        f"{display_name} vẫn là một phương án thay thế, nhưng đánh giá hình ảnh chưa ủng hộ mạnh."
    )


def render_user_facing_vi_v2(
    analysis: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict:
    """Return a concise deploy-facing Vietnamese payload for end users.

    Machine identities/ranks are preserved so the frontend can bind the correct
    images, while all prose is intentionally free of internal implementation
    vocabulary. Raw scorer/recommendation logits stay in the internal run only.
    """

    normalized_evidence = validate_vlm_evidence_v2(evidence)
    normalized_analysis = validate_visual_analysis_v2(analysis, normalized_evidence)

    diagnosis = normalized_evidence["diagnosis"]
    problem_index = int(diagnosis["problematic_item_index"])
    problem_id = str(diagnosis["problematic_item_id"])
    problem_coarse = str(diagnosis["problematic_category"])
    problem_category = CATEGORY_LABELS_VI[problem_coarse]
    diagnosis_visual = normalized_analysis["diagnosis"]

    current_item_label = _CURRENT_ITEM_LABEL_VI[problem_coarse]
    problematic_headline = (
        f"{current_item_label} được hệ thống đánh giá là món kém phù hợp nhất với outfit."
    )
    problematic_reason = _diagnosis_reason(diagnosis_visual, problem_index)

    analysis_by_id = {
        str(row["item_id"]): row for row in normalized_analysis["recommendations"]
    }
    recommendations: list[dict[str, object]] = []
    recommendation_sentences: list[str] = []
    for candidate in normalized_evidence["recommendation"]["items"]:
        rank = int(candidate["rank"])
        item_id = str(candidate["item_id"])
        coarse_category = str(candidate["coarse_category"])
        display_name = f"{_RECOMMENDATION_NOUN_VI[coarse_category].capitalize()} {rank}"
        visual = analysis_by_id[item_id]
        reason = _recommendation_reason(visual, display_name=display_name)
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

    replacement_noun = _RECOMMENDATION_NOUN_VI[problem_coarse]
    summary = (
        f"Bạn có thể thử thay món này bằng một trong ba {replacement_noun} bên dưới để "
        "outfit hài hòa hơn."
    )

    overall = str(diagnosis_visual["overall_visual_support"])
    if overall == "contradicts_loo":
        caution = (
            "Đánh giá hình ảnh và điểm tương thích của hệ thống chưa hoàn toàn đồng thuận, "
            "nên bạn có thể xem các phương án thay thế như những lựa chọn để thử thay vì "
            "một kết luận chắc chắn."
        )
    elif overall == "ambiguous":
        caution = (
            "Hình ảnh chưa cho tín hiệu đủ rõ về món hiện tại, nên bạn có thể thử các phương án "
            "thay thế và chọn lựa chọn hợp mắt nhất."
        )
    else:
        caution = (
            "Các phương án dưới đây được hệ thống ưu tiên; bạn vẫn có thể chọn lựa chọn phù hợp "
            "nhất với sở thích cá nhân."
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
