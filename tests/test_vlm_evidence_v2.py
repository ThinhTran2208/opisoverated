"""Tests for VLM Evidence V2 recommendation grounding."""

import copy
import json
import math
import unittest

from src.vlm.schema import build_vlm_evidence, validate_vlm_evidence
from src.vlm.schema_v2 import (
    CANONICAL_RECOMMENDATION_VERSION,
    EVIDENCE_SCHEMA_VERSION_V2,
    GROUNDING_RULES_V2,
    RECOMMENDATION_RANKING_SEMANTICS,
    RECOMMENDATION_SCORE_SEMANTICS,
    build_recommendation_evidence,
    build_vlm_evidence_v2,
    canonical_evidence_json_v2,
    validate_vlm_evidence_v2,
)


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
        {
            "item_id": "candidate-d",
            "compatibility_logit": -0.1,
            "improvement_logit": 0.3,
            "category_id_used": 2,
            "used_category_fallback": False,
        },
    ]
    return {
        "status": "ok",
        "recommendation_version": CANONICAL_RECOMMENDATION_VERSION,
        "items": public_items,
        "internal_metadata": {
            "problematic_item_index": 1,
            "loo_protocol_version": "loo-diagnostic-v1",
            "retrieval": {
                "retrieval_scope": "exact_master_category",
                "used_core7_fallback": False,
            },
            "reranked_candidates": reranked,
        },
    }


def build_evidence():
    return build_vlm_evidence_v2(
        loo_fixture(),
        recommendation_fixture(),
        sample_id="demo-neg",
        item_ids=["top", "bottom", "shoe", "bag"],
        coarse_categories=["TOP", "BOTTOM", "SHOES", "BAG"],
    )


class RecommendationEvidenceV2Tests(unittest.TestCase):
    def test_v2_builder_preserves_v1_and_adds_authoritative_top3(self):
        evidence = build_evidence()

        self.assertEqual(evidence["schema_version"], EVIDENCE_SCHEMA_VERSION_V2)
        self.assertEqual(evidence["grounding_rules"], list(GROUNDING_RULES_V2))
        recommendation = evidence["recommendation"]
        self.assertEqual(recommendation["status"], "available")
        self.assertEqual(
            recommendation["version"], CANONICAL_RECOMMENDATION_VERSION
        )
        self.assertEqual(recommendation["problematic_item_index"], 1)
        self.assertEqual(recommendation["problematic_item_id"], "bottom")
        self.assertEqual(
            recommendation["ranking_semantics"], RECOMMENDATION_RANKING_SEMANTICS
        )
        self.assertEqual(
            recommendation["score_semantics"], RECOMMENDATION_SCORE_SEMANTICS
        )
        self.assertEqual(
            [row["item_id"] for row in recommendation["items"]],
            ["candidate-a", "candidate-b", "candidate-c"],
        )
        self.assertNotIn("image_url", recommendation["items"][0])
        self.assertAlmostEqual(
            recommendation["items"][0]["improvement_logit"], 1.2
        )

        # V1 remains a separate frozen contract and still says not implemented.
        v1 = build_vlm_evidence(
            loo_fixture(),
            sample_id="demo-neg",
            item_ids=["top", "bottom", "shoe", "bag"],
            coarse_categories=["TOP", "BOTTOM", "SHOES", "BAG"],
        )
        self.assertEqual(
            v1["recommendation"], {"status": "not_implemented", "items": []}
        )
        validate_vlm_evidence(v1)

    def test_canonical_json_round_trips_through_v2_validator(self):
        evidence = build_evidence()
        serialized = canonical_evidence_json_v2(evidence)
        loaded = json.loads(serialized)
        self.assertEqual(validate_vlm_evidence_v2(loaded), loaded)

    def test_rejects_recommendation_problematic_index_mismatch(self):
        recommendation = recommendation_fixture()
        recommendation["internal_metadata"]["problematic_item_index"] = 0
        with self.assertRaisesRegex(ValueError, "does not match LOO"):
            build_vlm_evidence_v2(
                loo_fixture(),
                recommendation,
                sample_id="demo-neg",
                item_ids=["top", "bottom", "shoe", "bag"],
                coarse_categories=["TOP", "BOTTOM", "SHOES", "BAG"],
            )

    def test_rejects_public_top3_not_matching_reranker(self):
        recommendation = recommendation_fixture()
        recommendation["items"][0]["item_id"] = "different-item"
        with self.assertRaisesRegex(ValueError, "does not match frozen scorer"):
            build_evidence_from_recommendation(recommendation)

    def test_rejects_duplicate_candidate_ids(self):
        recommendation = recommendation_fixture()
        recommendation["items"][1]["item_id"] = "candidate-a"
        recommendation["internal_metadata"]["reranked_candidates"][1]["item_id"] = (
            "candidate-a"
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            build_evidence_from_recommendation(recommendation)

    def test_rejects_candidate_already_in_outfit(self):
        recommendation = recommendation_fixture()
        recommendation["items"][0]["item_id"] = "top"
        recommendation["internal_metadata"]["reranked_candidates"][0]["item_id"] = (
            "top"
        )
        with self.assertRaisesRegex(ValueError, "already be in the outfit"):
            build_evidence_from_recommendation(recommendation)

    def test_rejects_candidate_category_mismatch(self):
        recommendation = recommendation_fixture()
        recommendation["items"][2]["coarse_category"] = "SHOES"
        with self.assertRaisesRegex(ValueError, "must match the LOO problematic category"):
            build_evidence_from_recommendation(recommendation)

    def test_rejects_inconsistent_improvement_logit(self):
        recommendation = recommendation_fixture()
        recommendation["internal_metadata"]["reranked_candidates"][0][
            "improvement_logit"
        ] = 999.0
        with self.assertRaisesRegex(ValueError, "inconsistent with the LOO/scorer baseline"):
            build_evidence_from_recommendation(recommendation)

    def test_rejects_nonfinite_candidate_score(self):
        recommendation = recommendation_fixture()
        recommendation["internal_metadata"]["reranked_candidates"][0][
            "compatibility_logit"
        ] = math.inf
        with self.assertRaisesRegex(ValueError, "must be finite"):
            build_evidence_from_recommendation(recommendation)

    def test_rejects_evaluation_leakage_from_recommendation_source(self):
        recommendation = recommendation_fixture()
        recommendation["internal_metadata"]["ground_truth_item_id"] = "candidate-a"
        with self.assertRaisesRegex(ValueError, "forbidden evaluation leakage"):
            build_evidence_from_recommendation(recommendation)

    def test_rejects_wrong_recommendation_version(self):
        recommendation = recommendation_fixture()
        recommendation["recommendation_version"] = "experimental-v3"
        with self.assertRaisesRegex(ValueError, "frozen category-aware-hybrid-v2"):
            build_evidence_from_recommendation(recommendation)

    def test_serialized_validator_rejects_candidate_rerank(self):
        evidence = build_evidence()
        tampered = copy.deepcopy(evidence)
        tampered["recommendation"]["items"][0], tampered["recommendation"]["items"][1] = (
            tampered["recommendation"]["items"][1],
            tampered["recommendation"]["items"][0],
        )
        tampered["recommendation"]["items"][0]["rank"] = 1
        tampered["recommendation"]["items"][1]["rank"] = 2
        with self.assertRaisesRegex(ValueError, "rank order is inconsistent"):
            validate_vlm_evidence_v2(tampered)

    def test_build_recommendation_section_can_be_validated_independently(self):
        base = build_vlm_evidence(
            loo_fixture(),
            sample_id="demo-neg",
            item_ids=["top", "bottom", "shoe", "bag"],
            coarse_categories=["TOP", "BOTTOM", "SHOES", "BAG"],
        )
        section = build_recommendation_evidence(
            recommendation_fixture(), base_evidence=base
        )
        self.assertEqual(len(section["items"]), 3)
        self.assertEqual(section["items"][0]["rank"], 1)


def build_evidence_from_recommendation(recommendation):
    return build_vlm_evidence_v2(
        loo_fixture(),
        recommendation,
        sample_id="demo-neg",
        item_ids=["top", "bottom", "shoe", "bag"],
        coarse_categories=["TOP", "BOTTOM", "SHOES", "BAG"],
    )


if __name__ == "__main__":
    unittest.main()
