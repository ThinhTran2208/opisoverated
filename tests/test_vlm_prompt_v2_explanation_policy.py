"""Regression tests for the VLM V2 explanation-role prompt policy."""

import json
import unittest

from src.vlm.prompt_v2 import append_repair_request_v2, build_qwen_messages_v2
from src.vlm.schema_v2 import build_vlm_evidence_v2


def evidence_fixture():
    loo = {
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
        {"item_id": "candidate-a", "compatibility_logit": 0.8, "improvement_logit": 1.2},
        {"item_id": "candidate-b", "compatibility_logit": 0.5, "improvement_logit": 0.9},
        {"item_id": "candidate-c", "compatibility_logit": 0.1, "improvement_logit": 0.5},
    ]
    recommendation = {
        "status": "ok",
        "recommendation_version": "category-aware-hybrid-v2",
        "items": public_items,
        "internal_metadata": {
            "problematic_item_index": 1,
            "loo_protocol_version": "loo-diagnostic-v1",
            "reranked_candidates": reranked,
        },
    }
    return build_vlm_evidence_v2(
        loo,
        recommendation,
        sample_id="prompt-policy-v2-demo",
        item_ids=["top", "bottom", "shoe", "bag"],
        coarse_categories=["TOP", "BOTTOM", "SHOES", "BAG"],
    )


def messages_fixture():
    return build_qwen_messages_v2(
        evidence_fixture(),
        [f"https://example.com/outfit-{index}.jpg" for index in range(4)],
        {
            "candidate-a": "https://example.com/a.jpg",
            "candidate-b": "https://example.com/b.jpg",
            "candidate-c": "https://example.com/c.jpg",
        },
        min_pixels=262144,
        max_pixels=262144,
        must_exist=False,
    )


class VlmPromptV2ExplanationPolicyTests(unittest.TestCase):
    def test_prompt_defines_qwen_as_evidence_extractor_not_second_decision_maker(self):
        text = json.dumps(messages_fixture(), ensure_ascii=False)
        self.assertIn("NOT a second decision-maker", text)
        self.assertIn("already fixed", text)
        self.assertIn("extract useful", text)
        self.assertIn("not to recreate the ranking", text)

    def test_prompt_prefers_grounded_support_but_forbids_forced_justification(self):
        text = json.dumps(messages_fixture(), ensure_ascii=False)
        self.assertIn("First look for one concrete", text)
        self.assertIn("ambiguous with an empty visual_observations list", text)
        self.assertIn("does NOT mean the candidate should be removed or reranked", text)
        self.assertIn("Raw numerical scorer, LOO, and recommendation values are intentionally omitted", text)
        self.assertIn("Do not derive visual labels from rank", text)

    def test_repair_prompt_keeps_same_explanation_policy(self):
        repaired = append_repair_request_v2(
            messages_fixture(),
            raw_response='{"bad":"payload"}',
            validation_error="invalid schema",
        )
        repair_text = repaired[-1]["content"][0]["text"]
        self.assertIn("grounded positive visual relation", repair_text)
        self.assertIn("ambiguous with an empty visual_observations list", repair_text)
        self.assertIn("negative filler", repair_text)
        self.assertIn("imagined scores", repair_text)


if __name__ == "__main__":
    unittest.main()
