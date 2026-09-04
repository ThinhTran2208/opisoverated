"""Tests for the deploy-facing Vietnamese VLM V2 renderer."""

import json
import unittest

from src.vlm.schema_v2 import build_vlm_evidence_v2
from src.vlm.user_renderer_v2 import (
    USER_FACING_SCHEMA_VERSION_V2,
    render_user_facing_vi_v2,
)


def loo_fixture():
    return {
        "protocol_version": "loo-diagnostic-v1",
        "original_item_count": 4,
        "full_logit": -1.2,
        "without_item_logits": [-1.0, -1.1, -1.15, 0.2],
        "deltas_without_minus_full": [0.2, 0.1, 0.05, 1.4],
        "ranked_item_indices": [3, 0, 1, 2],
        "problematic_item_index": 3,
        "problematic_item_id": "bag-current",
        "uses_two_item_extrapolation": False,
    }


def recommendation_fixture():
    public_items = [
        {
            "rank": 1,
            "item_id": "bag-a",
            "image_url": "/recommendation/images/bag-a",
            "master_category": "Backpacks",
            "coarse_category": "BAG",
        },
        {
            "rank": 2,
            "item_id": "bag-b",
            "image_url": "/recommendation/images/bag-b",
            "master_category": "Backpacks",
            "coarse_category": "BAG",
        },
        {
            "rank": 3,
            "item_id": "bag-c",
            "image_url": "/recommendation/images/bag-c",
            "master_category": "Backpacks",
            "coarse_category": "BAG",
        },
    ]
    reranked = [
        {"item_id": "bag-a", "compatibility_logit": 0.9, "improvement_logit": 2.1},
        {"item_id": "bag-b", "compatibility_logit": 0.8, "improvement_logit": 2.0},
        {"item_id": "bag-c", "compatibility_logit": 0.7, "improvement_logit": 1.9},
    ]
    return {
        "status": "ok",
        "recommendation_version": "category-aware-hybrid-v2",
        "items": public_items,
        "internal_metadata": {
            "problematic_item_index": 3,
            "loo_protocol_version": "loo-diagnostic-v1",
            "reranked_candidates": reranked,
        },
    }


def evidence_fixture():
    return build_vlm_evidence_v2(
        loo_fixture(),
        recommendation_fixture(),
        sample_id="user-renderer-v2-demo",
        item_ids=["top", "bottom", "shoe", "bag-current"],
        coarse_categories=["TOP", "BOTTOM", "SHOES", "BAG"],
    )


def analysis_fixture():
    return {
        "schema_version": "vlm-visual-analysis-v2",
        "problematic_item_index": 3,
        "problematic_item_id": "bag-current",
        "diagnosis": {
            "overall_visual_support": "contradicts_loo",
            "visual_observations": [
                {
                    "item_indices": [3, 0],
                    "dimension": "color_harmony",
                    "effect": "contradicts_loo",
                    "confidence": "high",
                },
                {
                    "item_indices": [3, 1],
                    "dimension": "formality_alignment",
                    "effect": "contradicts_loo",
                    "confidence": "high",
                },
            ],
        },
        "recommendations": [
            {
                "rank": 1,
                "item_id": "bag-a",
                "overall_visual_support": "supports_recommendation",
                "visual_observations": [
                    {
                        "context_item_indices": [0, 1, 2],
                        "dimension": "color_harmony",
                        "effect": "supports_recommendation",
                        "confidence": "high",
                    }
                ],
            },
            {
                "rank": 2,
                "item_id": "bag-b",
                "overall_visual_support": "ambiguous",
                "visual_observations": [],
            },
            {
                "rank": 3,
                "item_id": "bag-c",
                "overall_visual_support": "supports_recommendation",
                "visual_observations": [
                    {
                        "context_item_indices": [0, 1, 2],
                        "dimension": "silhouette_balance",
                        "effect": "supports_recommendation",
                        "confidence": "medium",
                    }
                ],
            },
        ],
        "limitations": [
            "compatibility_logit_is_not_probability",
            "recommendation_scores_are_not_probabilities",
            "recommendation_identity_and_rank_are_authoritative",
            "vlm_visual_observations_are_inferences",
        ],
    }


class VlmUserRendererV2Tests(unittest.TestCase):
    def test_renders_concise_context_aware_copy(self):
        rendered = render_user_facing_vi_v2(analysis_fixture(), evidence_fixture())

        self.assertEqual(rendered["schema_version"], USER_FACING_SCHEMA_VERSION_V2)
        self.assertIn(
            "chiếc túi hiện tại là món được ưu tiên thay",
            rendered["problematic_item"]["headline"],
        )
        # Internal contradiction must not be turned into a fabricated user-facing reason.
        self.assertIsNone(rendered["problematic_item"]["reason"])

        self.assertIn("Ba mẫu túi", rendered["summary"])
        self.assertIn("phù hợp hơn", rendered["summary"])
        self.assertEqual(len(rendered["recommendations"]), 3)
        self.assertEqual(
            [row["display_name"] for row in rendered["recommendations"]],
            ["Mẫu túi 1", "Mẫu túi 2", "Mẫu túi 3"],
        )

        self.assertEqual(
            rendered["recommendations"][0]["reason"],
            "Màu sắc phối hợp tốt với áo, quần/váy và giày.",
        )
        self.assertIsNone(rendered["recommendations"][1]["reason"])
        self.assertEqual(
            rendered["recommendations"][2]["reason"],
            "Phom dáng tạo cảm giác cân bằng với áo, quần/váy và giày.",
        )

    def test_does_not_repeat_rank_boilerplate(self):
        rendered = render_user_facing_vi_v2(analysis_fixture(), evidence_fixture())
        prose = rendered["text"].lower()

        for boilerplate in (
            "lựa chọn ưu tiên nhất",
            "lựa chọn thứ hai",
            "lựa chọn thứ ba",
            "ảnh hưởng lớn nhất",
        ):
            self.assertNotIn(boilerplate, prose)

        self.assertIn("mẫu túi 1:", prose)
        self.assertIn("mẫu túi 3:", prose)

    def test_internal_visual_disagreement_is_not_shown_to_user(self):
        rendered = render_user_facing_vi_v2(analysis_fixture(), evidence_fixture())
        prose = rendered["text"].lower()

        for awkward_internal_phrase in (
            "tuy nhiên",
            "không ủng hộ",
            "contradicts",
            "ambiguous",
        ):
            self.assertNotIn(awkward_internal_phrase, prose)

    def test_user_prose_hides_internal_implementation_terms(self):
        rendered = render_user_facing_vi_v2(analysis_fixture(), evidence_fixture())
        values = [
            rendered["text"],
            rendered["problematic_item"]["headline"],
            rendered["summary"],
            rendered["caution"],
            *[row["headline"] for row in rendered["recommendations"]],
            *[
                row["reason"]
                for row in rendered["recommendations"]
                if row["reason"] is not None
            ],
        ]
        if rendered["problematic_item"]["reason"] is not None:
            values.append(rendered["problematic_item"]["reason"])

        prose = " ".join(values).lower()
        for forbidden in ("loo", "qwen", "logit", "validator", "probability"):
            self.assertNotIn(forbidden, prose)

    def test_user_payload_does_not_expose_raw_scores(self):
        rendered = render_user_facing_vi_v2(analysis_fixture(), evidence_fixture())
        serialized = json.dumps(rendered, ensure_ascii=False)
        self.assertNotIn("compatibility_logit", serialized)
        self.assertNotIn("improvement_logit", serialized)
        self.assertNotIn("score_summary", serialized)


if __name__ == "__main__":
    unittest.main()
