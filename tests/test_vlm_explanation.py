"""Unit tests for grounded VLM evidence, constrained analysis, and rendering."""

import copy
import json
import unittest
from pathlib import Path

from src.vlm.config import CANONICAL_MODEL_ID, load_vlm_config, validate_vlm_config
from src.vlm.pipeline import (
    VLMExplanationPipeline,
    extract_json_object,
    render_explanation_vi,
    validate_visual_analysis,
)
from src.vlm.prompt import (
    BASE_REQUIRED_LIMITATIONS,
    TWO_ITEM_EXTRAPOLATION_LIMITATION,
    build_qwen_messages,
    required_limitations,
)
from src.vlm.schema import build_vlm_evidence, validate_vlm_evidence


def loo_fixture(*, extrapolation=False):
    without_logits = [-0.3, 0.2, -0.35]
    deltas = [0.1, 0.6, 0.05]
    if not extrapolation:
        without_logits.append(-0.5)
        deltas.append(-0.1)
    return {
        "protocol_version": "loo-diagnostic-v1",
        "original_item_count": len(without_logits),
        "full_logit": -0.4,
        "without_item_logits": without_logits,
        "deltas_without_minus_full": deltas,
        "ranked_item_indices": [1, 0, 2] + ([] if extrapolation else [3]),
        "problematic_item_index": 1,
        "problematic_item_id": "bottom",
        "uses_two_item_extrapolation": extrapolation,
    }


def evidence_fixture(*, extrapolation=False):
    item_ids = ["top", "bottom", "shoe"]
    categories = ["TOP", "BOTTOM", "SHOES"]
    if not extrapolation:
        item_ids.append("bag")
        categories.append("BAG")
    return build_vlm_evidence(
        loo_fixture(extrapolation=extrapolation),
        sample_id="demo_neg",
        item_ids=item_ids,
        coarse_categories=categories,
    )


def analysis_fixture(evidence=None, *, problem_index=1, problem_id="bottom"):
    evidence = evidence or evidence_fixture()
    return {
        "schema_version": "vlm-visual-analysis-v1",
        "problematic_item_index": problem_index,
        "problematic_item_id": problem_id,
        "overall_visual_support": "supports_loo",
        "visual_observations": [
            {
                "item_indices": [problem_index, 0],
                "dimension": "style_coherence",
                "effect": "supports_loo",
                "confidence": "medium",
            }
        ],
        "limitations": list(required_limitations(evidence)),
    }


def hidden_recommendation_fixture():
    """The exact class of payload that passed the old free-text validator."""

    return {
        "schema_version": "vlm-explanation-v1",
        "problematic_item_index": 1,
        "problematic_item_id": "bottom",
        "headline": "Đôi giày là item kém phù hợp nhất.",
        "evidence_summary": ["LOO xác định item 1 có ảnh hưởng lớn nhất."],
        "visual_observations": [],
        "explanation": (
            "Bạn nên thay đôi giày này bằng một đôi sneaker trắng tối giản."
        ),
        "uncertainty_note": "Đây là đánh giá của model.",
        "limitations": list(BASE_REQUIRED_LIMITATIONS),
    }


class VlmEvidenceTests(unittest.TestCase):
    def test_builder_freezes_diagnosis_and_omits_ground_truth(self):
        evidence = evidence_fixture()

        self.assertEqual(evidence["diagnosis"]["problematic_item_index"], 1)
        self.assertEqual(evidence["diagnosis"]["problematic_item_id"], "bottom")
        self.assertEqual(evidence["recommendation"]["status"], "not_implemented")
        serialized = json.dumps(evidence)
        for forbidden in (
            "swapped_item_index",
            "negative_metadata",
            "top1_correct",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_builder_rejects_target_leakage(self):
        loo = loo_fixture()
        loo["target_swapped_item_index"] = 1
        with self.assertRaisesRegex(ValueError, "target leakage"):
            build_vlm_evidence(
                loo,
                sample_id="demo",
                item_ids=["top", "bottom", "shoe", "bag"],
                coarse_categories=["TOP", "BOTTOM", "SHOES", "BAG"],
            )

    def test_builder_rejects_inconsistent_ranking(self):
        loo = loo_fixture()
        loo["ranked_item_indices"] = [0, 1, 2, 3]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            build_vlm_evidence(
                loo,
                sample_id="demo",
                item_ids=["top", "bottom", "shoe", "bag"],
                coarse_categories=["TOP", "BOTTOM", "SHOES", "BAG"],
            )

    def test_validator_rejects_recommendation_payload(self):
        evidence = evidence_fixture()
        evidence["recommendation"] = {
            "status": "implemented",
            "items": [{"item_id": "invented"}],
        }
        with self.assertRaisesRegex(ValueError, "not implemented"):
            validate_vlm_evidence(evidence)

    def test_builder_rejects_noncanonical_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "checkpoint must be"):
            build_vlm_evidence(
                loo_fixture(),
                sample_id="demo",
                item_ids=["top", "bottom", "shoe", "bag"],
                coarse_categories=["TOP", "BOTTOM", "SHOES", "BAG"],
                checkpoint="experimental/best.pt",
            )

    def test_validator_rechecks_logit_delta_consistency(self):
        evidence = evidence_fixture()
        evidence["diagnosis"]["ranked_items"][0]["without_item_logit"] += 0.2
        with self.assertRaisesRegex(ValueError, "inconsistent with scorer logits"):
            validate_vlm_evidence(evidence)


class VlmPromptTests(unittest.TestCase):
    def test_prompt_maps_images_and_requests_no_free_text(self):
        messages = build_qwen_messages(
            evidence_fixture(),
            [f"https://example.com/{index}.jpg" for index in range(4)],
            min_pixels=262144,
            max_pixels=262144,
            must_exist=False,
        )
        user_content = messages[1]["content"]
        image_rows = [row for row in user_content if row["type"] == "image"]
        prompt_text = "\n".join(
            row["text"] for row in user_content if row["type"] == "text"
        )

        self.assertEqual(len(image_rows), 4)
        self.assertIn("item_index=1, item_id=bottom", prompt_text)
        self.assertIn("no free-text fields", prompt_text)
        self.assertIn("recommendation_not_implemented", prompt_text)
        self.assertNotIn("swapped_item_index", prompt_text)

    def test_prompt_requires_one_image_per_item(self):
        with self.assertRaisesRegex(ValueError, "one image per item"):
            build_qwen_messages(
                evidence_fixture(),
                ["https://example.com/only-one.jpg"],
                min_pixels=262144,
                max_pixels=262144,
                must_exist=False,
            )

    def test_prompt_conditionally_requests_two_item_disclosure(self):
        evidence = evidence_fixture(extrapolation=True)
        messages = build_qwen_messages(
            evidence,
            [f"https://example.com/{index}.jpg" for index in range(3)],
            min_pixels=262144,
            max_pixels=262144,
            must_exist=False,
        )
        prompt_text = json.dumps(messages)
        self.assertIn(TWO_ITEM_EXTRAPOLATION_LIMITATION, prompt_text)


class VlmOutputTests(unittest.TestCase):
    def test_json_extraction_accepts_one_markdown_fence(self):
        payload = analysis_fixture()
        extracted = extract_json_object(
            "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
        )
        self.assertEqual(extracted["problematic_item_id"], "bottom")

    def test_json_extraction_rejects_recommendation_before_object(self):
        raw = (
            "Bạn nên thay bằng sneaker trắng.\n"
            + json.dumps(analysis_fixture(), ensure_ascii=False)
        )
        with self.assertRaisesRegex(ValueError, "text before"):
            extract_json_object(raw)

    def test_json_extraction_rejects_duplicate_keys_hiding_free_text(self):
        raw = (
            '{"schema_version":"vlm-visual-analysis-v1",'
            '"schema_version":"Bạn nên thay bằng sneaker trắng"}'
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            extract_json_object(raw)

    def test_validator_rejects_changed_problematic_item(self):
        output = analysis_fixture(problem_index=0, problem_id="top")
        with self.assertRaisesRegex(ValueError, "change problematic_item_index"):
            validate_visual_analysis(output, evidence_fixture())

    def test_validator_rejects_boolean_problematic_item_index(self):
        output = analysis_fixture(problem_index=True, problem_id="bottom")
        with self.assertRaisesRegex(ValueError, "change problematic_item_index"):
            validate_visual_analysis(output, evidence_fixture())

    def test_validator_rejects_hidden_recommendation_in_old_free_text_schema(self):
        with self.assertRaisesRegex(ValueError, "keys must be exactly"):
            validate_visual_analysis(
                hidden_recommendation_fixture(), evidence_fixture()
            )

    def test_validator_rejects_natural_language_inside_enum_field(self):
        output = analysis_fixture()
        output["visual_observations"][0]["dimension"] = (
            "Bạn nên thay bằng sneaker trắng"
        )
        with self.assertRaisesRegex(ValueError, "dimension must be one of"):
            validate_visual_analysis(output, evidence_fixture())

    def test_empty_observations_must_be_visually_ambiguous(self):
        output = analysis_fixture()
        output["visual_observations"] = []
        with self.assertRaisesRegex(ValueError, "must be ambiguous"):
            validate_visual_analysis(output, evidence_fixture())

    def test_pipeline_repairs_one_invalid_response_and_renders_text(self):
        class FakeBackend:
            model_id = CANONICAL_MODEL_ID

            def __init__(self):
                self.responses = [
                    json.dumps(hidden_recommendation_fixture(), ensure_ascii=False),
                    json.dumps(analysis_fixture(), ensure_ascii=False),
                ]
                self.call_count = 0

            def generate(self, messages, generation):
                del messages, generation
                response = self.responses[self.call_count]
                self.call_count += 1
                return response

        repo_root = Path(__file__).resolve().parents[1]
        config = load_vlm_config(
            repo_root / "configs/vlm_qwen3_vl_4b_instruct_v1.json"
        )
        backend = FakeBackend()
        pipeline = VLMExplanationPipeline(backend, config)
        result = pipeline.explain(
            evidence_fixture(),
            [f"https://example.com/{index}.jpg" for index in range(4)],
            must_exist=False,
        )

        self.assertEqual(backend.call_count, 2)
        self.assertEqual(result["generation_attempts"], 2)
        self.assertEqual(result["visual_analysis"]["problematic_item_id"], "bottom")
        self.assertEqual(result["explanation"]["problematic_item_id"], "bottom")
        self.assertNotIn("sneaker", json.dumps(result["explanation"]))
        self.assertEqual(len(result["evidence_sha256"]), 64)

    def test_two_item_case_requires_machine_readable_disclosure(self):
        evidence = evidence_fixture(extrapolation=True)
        output = analysis_fixture(evidence)
        output["limitations"].remove(TWO_ITEM_EXTRAPOLATION_LIMITATION)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_visual_analysis(output, evidence)

    def test_non_extrapolation_case_rejects_false_disclosure(self):
        evidence = evidence_fixture(extrapolation=False)
        output = analysis_fixture(evidence)
        output["limitations"].append(TWO_ITEM_EXTRAPOLATION_LIMITATION)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_visual_analysis(output, evidence)

    def test_renderer_explicitly_discloses_two_item_extrapolation(self):
        evidence = evidence_fixture(extrapolation=True)
        explanation = render_explanation_vi(analysis_fixture(evidence), evidence)

        self.assertIn(
            TWO_ITEM_EXTRAPOLATION_LIMITATION, explanation["limitations"]
        )
        self.assertIn("subset còn 2 item", explanation["uncertainty_note"])
        self.assertIn("ngoài phân phối huấn luyện", explanation["uncertainty_note"])

    def test_renderer_uses_only_validated_analysis_not_model_prose(self):
        evidence = evidence_fixture()
        analysis = analysis_fixture(evidence)
        explanation = render_explanation_vi(analysis, evidence)

        self.assertEqual(explanation["schema_version"], "vlm-explanation-v1")
        self.assertIn("LOO xếp item 1", explanation["headline"])
        self.assertIn("taxonomy đóng", explanation["explanation"])
        self.assertNotIn(
            TWO_ITEM_EXTRAPOLATION_LIMITATION, explanation["limitations"]
        )


class VlmConfigTests(unittest.TestCase):
    def test_canonical_config_targets_qwen3_vl_4b_fp16(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = load_vlm_config(
            repo_root / "configs/vlm_qwen3_vl_4b_instruct_v1.json"
        )
        self.assertEqual(config["model"]["id"], CANONICAL_MODEL_ID)
        self.assertEqual(config["model"]["dtype"], "float16")
        self.assertFalse(config["generation"]["do_sample"])

    def test_config_rejects_experimental_generation_override(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = load_vlm_config(
            repo_root / "configs/vlm_qwen3_vl_4b_instruct_v1.json"
        )
        config["generation"]["do_sample"] = True
        with self.assertRaisesRegex(ValueError, "deterministic"):
            validate_vlm_config(config)


if __name__ == "__main__":
    unittest.main()
