"""Tests for deterministic VLM visual-analysis V2 validation."""

import copy
import unittest

from src.vlm.pipeline_v2 import validate_visual_analysis_v2
from src.vlm.prompt_v2 import expected_output_shape_v2
from src.vlm.schema_v2 import build_vlm_evidence_v2


def loo_fixture():
    return {
        "protocol_version": "loo-diagnostic-v1",
        "original_item_count": 4,
        "full_logit": -0.4,
        "without_item_logits": [-0.3, 0.2, -0.35, -0.5],
        "deltas_without_minus_full": [0.1, 0.6, 0.05, -0.1],
        "ranked_item_indices": [1, 0, 2, 3],
        "problematic_item_index": 1,
        "problematic_item_id": "bottom",
        "uses_two_item_extrapolation": False,
    }


def recommendation_fixture():
    public_items = [
        {
            "rank": 1,
            "item_id": "candidate-a",
            "image_url": "/recommendation/images/candidate-a",
            "master_category": "Skirts",
            "coarse_category": "BOTTOM",
        },
        {
            "rank": 2,
            "item_id": "candidate-b",
            "image_url": "/recommendation/images/candidate-b",
            "master_category": "Jeans",
            "coarse_category": "BOTTOM",
        },
        {
            "rank": 3,
            "item_id": "candidate-c",
            "image_url": "/recommendation/images/candidate-c",
            "master_category": "Trousers",
            "coarse_category": "BOTTOM",
        },
    ]
    reranked = [
        {
            "item_id": "candidate-a",
            "compatibility_logit": 0.8,
            "improvement_logit": 1.2,
            "category_id_used": 2,
            "used_category_fallback": False,
        },
        {
            "item_id": "candidate-b",
            "compatibility_logit": 0.5,
            "improvement_logit": 0.9,
            "category_id_used": 2,
            "used_category_fallback": False,
        },
        {
            "item_id": "candidate-c",
            "compatibility_logit": 0.1,
            "improvement_logit": 0.5,
            "category_id_used": 2,
            "used_category_fallback": False,
        },
    ]
    return {
        "status": "ok",
        "recommendation_version": "category-aware-hybrid-v2",
        "items": public_items,
        "internal_metadata": {
            "problematic_item_index": 1,
            "loo_protocol_version": "loo-diagnostic-v1",
            "reranked_candidates": reranked,
        },
    }


def evidence_fixture():
    return build_vlm_evidence_v2(
        loo_fixture(),
        recommendation_fixture(),
        sample_id="validator-v2-demo",
        item_ids=["top", "bottom", "shoe", "bag"],
        coarse_categories=["TOP", "BOTTOM", "SHOES", "BAG"],
    )


def valid_analysis():
    analysis = expected_output_shape_v2(evidence_fixture())
    analysis["diagnosis"] = {
        "overall_visual_support": "supports_loo",
        "visual_observations": [
            {
                "item_indices": [1, 0],
                "dimension": "style_coherence",
                "effect": "supports_loo",
                "confidence": "medium",
            }
        ],
        "user_reason": "Chiếc áo này lệch tông màu với các món còn lại trong outfit.",
    }
    analysis["recommendations"][0] = {
        "rank": 1,
        "item_id": "candidate-a",
        "overall_visual_support": "supports_recommendation",
        "visual_observations": [
            {
                "context_item_indices": [0, 2, 3],
                "dimension": "color_harmony",
                "effect": "supports_recommendation",
                "confidence": "medium",
            }
        ],
    }
    return analysis


class VlmValidatorV2Tests(unittest.TestCase):
    def test_accepts_grounded_closed_taxonomy_analysis(self):
        normalized = validate_visual_analysis_v2(valid_analysis(), evidence_fixture())
        self.assertEqual(normalized["problematic_item_index"], 1)
        self.assertEqual(normalized["problematic_item_id"], "bottom")
        self.assertEqual(
            normalized["diagnosis"]["user_reason"],
            "Chiếc áo này lệch tông màu với các món còn lại trong outfit.",
        )
        self.assertEqual(
            [(row["rank"], row["item_id"]) for row in normalized["recommendations"]],
            [(1, "candidate-a"), (2, "candidate-b"), (3, "candidate-c")],
        )

    def test_rejects_changed_problematic_item(self):
        analysis = valid_analysis()
        analysis["problematic_item_index"] = 0
        with self.assertRaisesRegex(ValueError, "change problematic_item_index"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_rejects_changed_candidate_identity(self):
        analysis = valid_analysis()
        analysis["recommendations"][0]["item_id"] = "invented-candidate"
        with self.assertRaisesRegex(ValueError, "candidate identity"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_rejects_changed_candidate_rank(self):
        analysis = valid_analysis()
        analysis["recommendations"][0]["rank"] = 2
        with self.assertRaisesRegex(ValueError, "rank/order"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_rejects_removed_recommendation(self):
        analysis = valid_analysis()
        analysis["recommendations"] = analysis["recommendations"][:2]
        with self.assertRaisesRegex(ValueError, "authoritative Top-3"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_rejects_extra_free_text_key(self):
        analysis = valid_analysis()
        analysis["explanation"] = "Bạn nên thay bằng một đôi khác."
        with self.assertRaisesRegex(ValueError, "keys must be exactly"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_rejects_free_text_hidden_in_enum_field(self):
        analysis = valid_analysis()
        analysis["diagnosis"]["visual_observations"][0]["effect"] = (
            "Đôi này nhìn không hợp, nên thay đi"
        )
        with self.assertRaisesRegex(ValueError, "must be one of"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_rejects_scoring_language_in_user_reason(self):
        analysis = valid_analysis()
        analysis["diagnosis"]["user_reason"] = "Điểm outfit này thấp hơn vì áo không hợp."
        with self.assertRaisesRegex(ValueError, "internal scoring term"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_requires_user_reason_for_grounded_diagnosis(self):
        analysis = valid_analysis()
        analysis["diagnosis"]["user_reason"] = ""
        with self.assertRaisesRegex(ValueError, "user_reason is required"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_diagnosis_observation_must_include_problematic_item(self):
        analysis = valid_analysis()
        analysis["diagnosis"]["visual_observations"][0]["item_indices"] = [0, 2]
        with self.assertRaisesRegex(ValueError, "reference the problematic item"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_diagnosis_observation_must_be_relational(self):
        analysis = valid_analysis()
        analysis["diagnosis"]["visual_observations"][0]["item_indices"] = [1]
        with self.assertRaisesRegex(ValueError, "plus at least one other original outfit item"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_recommendation_context_cannot_include_problematic_item(self):
        analysis = valid_analysis()
        analysis["recommendations"][0]["visual_observations"][0][
            "context_item_indices"
        ] = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, "remaining original outfit context"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_recommendation_context_cannot_reference_unknown_item(self):
        analysis = valid_analysis()
        analysis["recommendations"][0]["visual_observations"][0][
            "context_item_indices"
        ] = [0, 99]
        with self.assertRaisesRegex(ValueError, "remaining original outfit context"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_empty_observations_require_ambiguous_overall(self):
        analysis = valid_analysis()
        analysis["recommendations"][1]["visual_observations"] = []
        analysis["recommendations"][1]["overall_visual_support"] = (
            "supports_recommendation"
        )
        with self.assertRaisesRegex(ValueError, "must be ambiguous when observations are empty"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_support_overall_requires_supporting_observation(self):
        analysis = valid_analysis()
        analysis["recommendations"][0]["overall_visual_support"] = (
            "supports_recommendation"
        )
        analysis["recommendations"][0]["visual_observations"][0]["effect"] = (
            "ambiguous"
        )
        with self.assertRaisesRegex(ValueError, "requires a supporting observation"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_rejects_exact_cloned_high_confidence_top3_analysis(self):
        analysis = valid_analysis()
        for position, item_id in enumerate(("candidate-a", "candidate-b", "candidate-c")):
            analysis["recommendations"][position] = {
                "rank": position + 1,
                "item_id": item_id,
                "overall_visual_support": "supports_recommendation",
                "visual_observations": [
                    {
                        "context_item_indices": [0, 2, 3],
                        "dimension": "color_harmony",
                        "effect": "supports_recommendation",
                        "confidence": "high",
                    },
                    {
                        "context_item_indices": [0, 2, 3],
                        "dimension": "formality_alignment",
                        "effect": "supports_recommendation",
                        "confidence": "high",
                    },
                ],
            }
        with self.assertRaisesRegex(ValueError, "exact cloned high-confidence pattern"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_allows_identical_ambiguous_low_confidence_top3_analysis(self):
        analysis = expected_output_shape_v2(evidence_fixture())
        validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_rejects_wrong_taxonomy_for_recommendation(self):
        analysis = valid_analysis()
        analysis["recommendations"][0]["visual_observations"][0]["effect"] = (
            "supports_loo"
        )
        with self.assertRaisesRegex(ValueError, "must be one of"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_rejects_wrong_limitations(self):
        analysis = valid_analysis()
        analysis["limitations"] = []
        with self.assertRaisesRegex(ValueError, "must exactly match"):
            validate_visual_analysis_v2(analysis, evidence_fixture())

    def test_validator_does_not_mutate_input(self):
        analysis = valid_analysis()
        before = copy.deepcopy(analysis)
        validate_visual_analysis_v2(analysis, evidence_fixture())
        self.assertEqual(analysis, before)


if __name__ == "__main__":
    unittest.main()
