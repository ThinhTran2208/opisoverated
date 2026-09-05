"""Tests for VLM V2 image binding and constrained Qwen prompt."""

import json
import tempfile
import unittest
from pathlib import Path

from src.vlm.prompt_v2 import (
    PROMPT_CONTEXT_SCHEMA_VERSION_V2,
    SYSTEM_PROMPT_V2,
    TWO_ITEM_EXTRAPOLATION_LIMITATION,
    VISUAL_ANALYSIS_SCHEMA_VERSION_V2,
    append_repair_request_v2,
    build_prompt_context_v2,
    build_qwen_messages_v2,
    expected_output_shape_v2,
    required_limitations_v2,
)
from src.vlm.schema_v2 import build_vlm_evidence_v2


def loo_fixture(*, extrapolation=False):
    if extrapolation:
        return {
            "protocol_version": "loo-diagnostic-v1",
            "original_item_count": 3,
            "full_logit": -0.4,
            "without_item_logits": [-0.3, 0.2, -0.35],
            "deltas_without_minus_full": [0.1, 0.6, 0.05],
            "ranked_item_indices": [1, 0, 2],
            "problematic_item_index": 1,
            "problematic_item_id": "bottom",
            "uses_two_item_extrapolation": True,
        }
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


def evidence_fixture(*, extrapolation=False):
    item_ids = ["top", "bottom", "shoe"]
    categories = ["TOP", "BOTTOM", "SHOES"]
    if not extrapolation:
        item_ids.append("bag")
        categories.append("BAG")
    return build_vlm_evidence_v2(
        loo_fixture(extrapolation=extrapolation),
        recommendation_fixture(),
        sample_id="prompt-v2-demo",
        item_ids=item_ids,
        coarse_categories=categories,
    )


def recommendation_images():
    return {
        "candidate-c": "https://example.com/c.jpg",
        "candidate-a": "https://example.com/a.jpg",
        "candidate-b": "https://example.com/b.jpg",
    }


def _collect_keys(value):
    keys = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key)
            keys.update(_collect_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_collect_keys(child))
    return keys


class VlmPromptV2Tests(unittest.TestCase):
    def test_binds_original_crops_and_exact_top3_candidate_images(self):
        evidence = evidence_fixture()
        messages = build_qwen_messages_v2(
            evidence,
            [f"https://example.com/outfit-{index}.jpg" for index in range(4)],
            recommendation_images(),
            min_pixels=262144,
            max_pixels=262144,
            must_exist=False,
        )

        self.assertEqual(len(messages), 2)
        user_content = messages[1]["content"]
        image_rows = [row for row in user_content if row["type"] == "image"]
        text = "\n".join(
            row["text"] for row in user_content if row["type"] == "text"
        )

        self.assertEqual(len(image_rows), 7)
        self.assertIn("VISUAL INPUT GROUP: ORIGINAL OUTFIT ITEM CROPS", text)
        self.assertIn("item_index=1, item_id=bottom", text)
        self.assertIn("problematic_item=true", text)
        self.assertIn("VISUAL INPUT GROUP: AUTHORITATIVE RECOMMENDATION CANDIDATES", text)
        self.assertIn("rank=1, item_id=candidate-a", text)
        self.assertIn("rank=2, item_id=candidate-b", text)
        self.assertIn("rank=3, item_id=candidate-c", text)
        self.assertIn("Do not change this candidate identity or rank", text)
        self.assertIn(PROMPT_CONTEXT_SCHEMA_VERSION_V2, text)
        self.assertIn("Raw scorer, LOO, and recommendation numerical values are deliberately omitted", text)
        self.assertIn("no other free-text fields", text)
        self.assertIn("user_reason", text)
        self.assertIn("language about scores", SYSTEM_PROMPT_V2)

        self.assertEqual(
            [row["image"] for row in image_rows[-3:]],
            [
                "https://example.com/a.jpg",
                "https://example.com/b.jpg",
                "https://example.com/c.jpg",
            ],
        )

    def test_binds_full_original_outfit_image_before_item_crops(self):
        evidence = evidence_fixture()
        with tempfile.TemporaryDirectory() as directory:
            original_ref = Path(directory) / "original-outfit.png"
            original_ref.write_bytes(b"fake-image")
            messages = build_qwen_messages_v2(
                evidence,
                [f"https://example.com/outfit-{index}.jpg" for index in range(4)],
                recommendation_images(),
                min_pixels=262144,
                max_pixels=262144,
                must_exist=False,
                original_image_ref=original_ref,
            )

        user_content = messages[1]["content"]
        image_rows = [row for row in user_content if row["type"] == "image"]
        text = "\n".join(row["text"] for row in user_content if row["type"] == "text")
        self.assertEqual(len(image_rows), 8)
        self.assertIn("VISUAL INPUT GROUP: FULL ORIGINAL OUTFIT IMAGE", text)
        self.assertEqual(image_rows[0]["image"], original_ref.as_uri())

    def test_prompt_context_projects_out_all_raw_score_keys(self):
        context = build_prompt_context_v2(evidence_fixture())
        self.assertEqual(context["schema_version"], PROMPT_CONTEXT_SCHEMA_VERSION_V2)
        keys = _collect_keys(context)
        for forbidden_score_key in (
            "compatibility_logit",
            "improvement_logit",
            "full_logit",
            "without_item_logits",
            "without_item_logit",
            "loo_delta",
            "deltas_without_minus_full",
            "top1_top2_delta_gap",
            "score_semantics",
            "ranking_semantics",
        ):
            self.assertNotIn(forbidden_score_key, keys)

    def test_real_prompt_does_not_embed_raw_numeric_score_fields(self):
        messages = build_qwen_messages_v2(
            evidence_fixture(),
            [f"https://example.com/outfit-{index}.jpg" for index in range(4)],
            recommendation_images(),
            min_pixels=262144,
            max_pixels=262144,
            must_exist=False,
        )
        prompt_text = json.dumps(messages, ensure_ascii=False)
        for raw_json_field in (
            '"compatibility_logit":',
            '"improvement_logit":',
            '"full_logit":',
            '"without_item_logits":',
            '"loo_delta":',
            '"top1_top2_delta_gap":',
        ):
            self.assertNotIn(raw_json_field, prompt_text)

    def test_recommendation_image_keys_must_exactly_match_top3_ids(self):
        evidence = evidence_fixture()
        with self.assertRaisesRegex(ValueError, "exactly match authoritative Top-3"):
            build_qwen_messages_v2(
                evidence,
                [f"https://example.com/outfit-{index}.jpg" for index in range(4)],
                {
                    "candidate-a": "https://example.com/a.jpg",
                    "candidate-b": "https://example.com/b.jpg",
                    "wrong-id": "https://example.com/wrong.jpg",
                },
                min_pixels=262144,
                max_pixels=262144,
                must_exist=False,
            )

    def test_requires_one_crop_per_original_item(self):
        with self.assertRaisesRegex(ValueError, "one crop image per original item"):
            build_qwen_messages_v2(
                evidence_fixture(),
                ["https://example.com/only-one.jpg"],
                recommendation_images(),
                min_pixels=262144,
                max_pixels=262144,
                must_exist=False,
            )

    def test_expected_shape_preserves_problematic_and_candidate_identity(self):
        evidence = evidence_fixture()
        shape = expected_output_shape_v2(evidence)

        self.assertEqual(shape["schema_version"], VISUAL_ANALYSIS_SCHEMA_VERSION_V2)
        self.assertEqual(shape["problematic_item_index"], 1)
        self.assertEqual(shape["problematic_item_id"], "bottom")
        self.assertEqual(
            [(row["rank"], row["item_id"]) for row in shape["recommendations"]],
            [
                (1, "candidate-a"),
                (2, "candidate-b"),
                (3, "candidate-c"),
            ],
        )
        serialized = json.dumps(shape, ensure_ascii=False)
        for forbidden_free_text_key in (
            "headline",
            "explanation",
            "recommendation_text",
            "reason",
            "description",
        ):
            self.assertNotIn(f'"{forbidden_free_text_key}"', serialized)

    def test_recommendation_context_excludes_problematic_original_item(self):
        messages = build_qwen_messages_v2(
            evidence_fixture(),
            [f"https://example.com/outfit-{index}.jpg" for index in range(4)],
            recommendation_images(),
            min_pixels=262144,
            max_pixels=262144,
            must_exist=False,
        )
        prompt_text = json.dumps(messages, ensure_ascii=False)
        self.assertIn("remaining original outfit context indices: [0, 2, 3]", prompt_text)
        self.assertIn(
            "Do not use the problematic original item as recommendation context",
            prompt_text,
        )

    def test_two_item_extrapolation_disclosure_is_preserved(self):
        evidence = evidence_fixture(extrapolation=True)
        limitations = required_limitations_v2(evidence)
        self.assertIn(TWO_ITEM_EXTRAPOLATION_LIMITATION, limitations)

        messages = build_qwen_messages_v2(
            evidence,
            [f"https://example.com/outfit-{index}.jpg" for index in range(3)],
            recommendation_images(),
            min_pixels=262144,
            max_pixels=262144,
            must_exist=False,
        )
        self.assertIn(
            TWO_ITEM_EXTRAPOLATION_LIMITATION,
            json.dumps(messages, ensure_ascii=False),
        )

    def test_non_extrapolation_does_not_request_false_disclosure(self):
        limitations = required_limitations_v2(evidence_fixture())
        self.assertNotIn(TWO_ITEM_EXTRAPOLATION_LIMITATION, limitations)

    def test_repair_request_forbids_identity_or_rank_changes(self):
        messages = build_qwen_messages_v2(
            evidence_fixture(),
            [f"https://example.com/outfit-{index}.jpg" for index in range(4)],
            recommendation_images(),
            min_pixels=262144,
            max_pixels=262144,
            must_exist=False,
        )
        repaired = append_repair_request_v2(
            messages,
            raw_response='{"bad":"payload"}',
            validation_error="invalid schema",
        )
        repair_text = repaired[-1]["content"][0]["text"]
        self.assertIn("Do not change the problematic item", repair_text)
        self.assertIn("recommendation candidate identities", repair_text)
        self.assertIn("recommendation ranks", repair_text)
        self.assertIn("Do not infer visual labels from rank or imagined scores", repair_text)

    def test_rejects_invalid_pixel_budget(self):
        with self.assertRaisesRegex(ValueError, "1 <= min_pixels <= max_pixels"):
            build_qwen_messages_v2(
                evidence_fixture(),
                [f"https://example.com/outfit-{index}.jpg" for index in range(4)],
                recommendation_images(),
                min_pixels=300,
                max_pixels=200,
                must_exist=False,
            )


if __name__ == "__main__":
    unittest.main()
