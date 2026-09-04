"""Tests for deterministic VLM V2 rendering and handoff boundaries."""

import json
import unittest

from src.vlm.config_v2 import VLM_PROTOCOL_VERSION_V2, validate_vlm_config_v2
from src.vlm.pipeline_v2 import (
    EXPLANATION_SCHEMA_VERSION_V2,
    HANDOFF_SCHEMA_VERSION_V2,
    RUN_SCHEMA_VERSION_V2,
    VLMExplanationPipelineV2,
    build_handoff_result_v2,
    render_explanation_vi_v2,
)
from src.vlm.prompt_v2 import expected_output_shape_v2
from src.vlm.schema_v2 import build_vlm_evidence_v2
from src.vlm.user_renderer_v2 import USER_FACING_SCHEMA_VERSION_V2


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
        sample_id="pipeline-v2-demo",
        item_ids=["top", "bottom", "shoe", "bag"],
        coarse_categories=["TOP", "BOTTOM", "SHOES", "BAG"],
    )


def config_fixture(*, include_raw_response=True):
    return {
        "protocol_version": VLM_PROTOCOL_VERSION_V2,
        "model": {
            "id": "Qwen/Qwen3-VL-4B-Instruct",
            "dtype": "float16",
            "device_map": "auto",
            "require_cuda": True,
        },
        "vision": {
            "min_pixels": 262144,
            "max_pixels": 262144,
            "image_patch_size": 16,
        },
        "generation": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.05,
            "max_validation_retries": 1,
        },
        "output": {"language": "vi", "include_raw_response": include_raw_response},
    }


def outfit_images():
    return [f"https://example.com/outfit-{index}.jpg" for index in range(4)]


def recommendation_images():
    return {
        "candidate-c": "https://example.com/c.jpg",
        "candidate-a": "https://example.com/a.jpg",
        "candidate-b": "https://example.com/b.jpg",
    }


class FakeBackend:
    model_id = "Qwen/Qwen3-VL-4B-Instruct"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, messages, generation):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class VlmPipelineV2Tests(unittest.TestCase):
    def test_internal_renderer_keeps_audit_scores(self):
        evidence = evidence_fixture()
        analysis = expected_output_shape_v2(evidence)
        rendered = render_explanation_vi_v2(analysis, evidence)

        self.assertEqual(rendered["schema_version"], EXPLANATION_SCHEMA_VERSION_V2)
        self.assertEqual(rendered["problematic_item_index"], 1)
        self.assertEqual(rendered["problematic_item_id"], "bottom")
        self.assertEqual(
            [(row["rank"], row["item_id"]) for row in rendered["recommendations"]],
            [(1, "candidate-a"), (2, "candidate-b"), (3, "candidate-c")],
        )
        self.assertIn("LOO", rendered["headline"])
        self.assertIn("không phải xác suất", rendered["recommendations"][0]["score_summary"])
        self.assertIn("Qwen chỉ bổ sung", rendered["explanation"])

    def test_handoff_is_score_free_but_preserves_ids_rank_and_visual_evidence(self):
        evidence = evidence_fixture()
        rendered = render_explanation_vi_v2(expected_output_shape_v2(evidence), evidence)
        handoff = build_handoff_result_v2(
            rendered,
            model_id="Qwen/Qwen3-VL-4B-Instruct",
            generation_attempts=1,
        )

        self.assertEqual(handoff["schema_version"], HANDOFF_SCHEMA_VERSION_V2)
        self.assertEqual(handoff["protocol_version"], VLM_PROTOCOL_VERSION_V2)
        self.assertEqual(len(handoff["recommendations"]), 3)
        self.assertEqual(
            [row["item_id"] for row in handoff["recommendations"]],
            ["candidate-a", "candidate-b", "candidate-c"],
        )
        serialized = json.dumps(handoff, ensure_ascii=False)
        for forbidden in (
            "compatibility_logit",
            "improvement_logit",
            "without_item_logit",
            "loo_delta",
            "score_summary",
            '"model_id"',
            '"generation_attempts"',
            '"raw_response"',
            '"evidence"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_pipeline_returns_internal_run_score_free_handoff_and_user_facing(self):
        evidence = evidence_fixture()
        valid_response = json.dumps(expected_output_shape_v2(evidence), ensure_ascii=False)
        backend = FakeBackend([valid_response])
        pipeline = VLMExplanationPipelineV2(backend, config_fixture())

        run = pipeline.explain(
            evidence,
            outfit_images(),
            recommendation_images(),
            must_exist=False,
        )

        self.assertEqual(run["schema_version"], RUN_SCHEMA_VERSION_V2)
        self.assertEqual(run["protocol_version"], VLM_PROTOCOL_VERSION_V2)
        self.assertEqual(run["generation_attempts"], 1)
        self.assertEqual(run["handoff"]["schema_version"], HANDOFF_SCHEMA_VERSION_V2)
        self.assertEqual(run["user_facing"]["schema_version"], USER_FACING_SCHEMA_VERSION_V2)
        self.assertEqual(run["handoff"]["problematic_item_id"], "bottom")
        self.assertEqual(
            [row["item_id"] for row in run["handoff"]["recommendations"]],
            ["candidate-a", "candidate-b", "candidate-c"],
        )
        public_serialized = json.dumps(
            {"handoff": run["handoff"], "user_facing": run["user_facing"]},
            ensure_ascii=False,
        )
        self.assertNotIn("compatibility_logit", public_serialized)
        self.assertNotIn("improvement_logit", public_serialized)
        self.assertNotIn("score_summary", public_serialized)
        self.assertIn("raw_response", run)
        self.assertIn("compatibility_logit", json.dumps(run["evidence"]))
        self.assertEqual(backend.calls, 1)

    def test_pipeline_repairs_one_invalid_model_response(self):
        evidence = evidence_fixture()
        valid_response = json.dumps(expected_output_shape_v2(evidence), ensure_ascii=False)
        backend = FakeBackend(['{"bad":"payload"}', valid_response])
        pipeline = VLMExplanationPipelineV2(backend, config_fixture())

        run = pipeline.explain(
            evidence,
            outfit_images(),
            recommendation_images(),
            must_exist=False,
        )
        self.assertEqual(run["generation_attempts"], 2)
        self.assertEqual(backend.calls, 2)

    def test_pipeline_fails_after_repair_budget_is_exhausted(self):
        evidence = evidence_fixture()
        backend = FakeBackend(['{"bad":"payload"}', '{"still":"bad"}'])
        pipeline = VLMExplanationPipelineV2(backend, config_fixture())

        with self.assertRaisesRegex(ValueError, "failed validation after 2 attempt"):
            pipeline.explain(
                evidence,
                outfit_images(),
                recommendation_images(),
                must_exist=False,
            )

    def test_v2_config_keeps_shared_runtime_parameters_but_uses_v2_generation_budget(self):
        normalized = validate_vlm_config_v2(config_fixture())
        self.assertEqual(normalized["protocol_version"], VLM_PROTOCOL_VERSION_V2)
        self.assertEqual(normalized["vision"]["max_pixels"], 262144)
        self.assertEqual(normalized["generation"]["max_new_tokens"], 1024)
        self.assertEqual(normalized["generation"]["max_validation_retries"], 1)

        wrong = config_fixture()
        wrong["protocol_version"] = "vlm-explanation-v1"
        with self.assertRaisesRegex(ValueError, "vlm-explanation-v2"):
            validate_vlm_config_v2(wrong)


if __name__ == "__main__":
    unittest.main()
