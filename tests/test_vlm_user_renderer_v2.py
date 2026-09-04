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
        {
            "item_id": "bag-a",
            "compatibility_logit": 0.9,
            "improvement_logit": 2.1,
        },
        {
            "item_id": "bag-b",
            "compatibility_logit": 0.8,
            "improvement_logit": 2.0,
        },
        {
            "item_id": "bag-c",
            "compatibility_logit": 0.7,
            "improvement_logit": 1.9,
        },
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
    def test_renders_plain_vietnamese_for_end_users(self):
        rendered = render_user_facing_vi_v2(analysis_fixture(), evidence_fixture())
        self.assertEqual(rendered["schema_version"], USER_FACING_SCHEMA_VERSION_V2)
        self.assertIn("Item 3", rendered["problematic_item"]["headline"])
        self.assertIn("món đồ có vấn đề nhất", rendered["problematic_item"]["headline"])
        self.assertIn("Bạn nên thử thay Item 3", rendered["summary"])
        self.assertEqual(len(rendered["recommendations"]), 3)
        self.assertIn("màu sắc hài hòa", rendered["recommendations"][0]["reason"])
        self.assertIn("chưa cho thấy ưu điểm thị giác đủ rõ", rendered["recommendations"][1]["reason"])
        self.assertIn("phom dáng giúp outfit cân đối hơn", rendered["recommendations"][2]["reason"])

    def test_user_prose_hides_internal_implementation_terms(self):
        rendered = render_user_facing_vi_v2(analysis_fixture(), evidence_fixture())
        prose = " ".join(
            [
                rendered["problematic_item"]["headline"],
                rendered["problematic_item"]["reason"],
                rendered["summary"],
                rendered["caution"],
                *[row["headline"] for row in rendered["recommendations"]],
                *[row["reason"] for row in rendered["recommendations"]],
            ]
        ).lower()
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
