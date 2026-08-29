"""Unit tests for grounded VLM evidence, prompting, and output validation."""

import copy
import json
import unittest
from pathlib import Path

from src.vlm.config import CANONICAL_MODEL_ID, load_vlm_config, validate_vlm_config
from src.vlm.pipeline import (
    VLMExplanationPipeline,
    extract_json_object,
    validate_explanation,
)
from src.vlm.prompt import REQUIRED_LIMITATIONS, build_qwen_messages
from src.vlm.schema import build_vlm_evidence, validate_vlm_evidence


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


def evidence_fixture():
    return build_vlm_evidence(
        loo_fixture(),
        sample_id="demo_neg",
        item_ids=["top", "bottom", "shoe", "bag"],
        coarse_categories=["TOP", "BOTTOM", "SHOES", "BAG"],
    )


def explanation_fixture(problem_index=1, problem_id="bottom"):
    return {
        "schema_version": "vlm-explanation-v1",
        "problematic_item_index": problem_index,
        "problematic_item_id": problem_id,
        "headline": "Món đồ ở vị trí 1 có ảnh hưởng âm rõ nhất.",
        "evidence_summary": [
            "LOO xếp item bottom đứng đầu theo mức tăng logit khi loại bỏ."
        ],
        "visual_observations": [
            {
                "item_indices": [0, 1],
                "observation": "Hai món có độ tương phản màu sắc dễ nhận thấy.",
            }
        ],
        "explanation": "Kết quả này diễn giải tín hiệu của scorer và LOO.",
        "uncertainty_note": "Độ chắc chắn của LOO chưa được hiệu chỉnh.",
        "limitations": list(REQUIRED_LIMITATIONS),
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
    def test_prompt_maps_every_image_to_canonical_item(self):
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


class VlmOutputTests(unittest.TestCase):
    def test_json_extraction_accepts_one_markdown_fence(self):
        payload = explanation_fixture()
        extracted = extract_json_object(
            "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
        )
        self.assertEqual(extracted["problematic_item_id"], "bottom")

    def test_validator_rejects_changed_problematic_item(self):
        output = explanation_fixture(problem_index=0, problem_id="top")
        with self.assertRaisesRegex(ValueError, "change problematic_item_index"):
            validate_explanation(output, evidence_fixture())

    def test_pipeline_repairs_one_invalid_response(self):
        class FakeBackend:
            model_id = CANONICAL_MODEL_ID

            def __init__(self):
                self.responses = [
                    json.dumps(explanation_fixture(0, "top"), ensure_ascii=False),
                    json.dumps(explanation_fixture(), ensure_ascii=False),
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
        self.assertEqual(result["explanation"]["problematic_item_id"], "bottom")
        self.assertEqual(len(result["evidence_sha256"]), 64)

    def test_validator_requires_all_limitations(self):
        output = copy.deepcopy(explanation_fixture())
        output["limitations"].remove("recommendation_not_implemented")
        with self.assertRaisesRegex(ValueError, "required limitations"):
            validate_explanation(output, evidence_fixture())


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
