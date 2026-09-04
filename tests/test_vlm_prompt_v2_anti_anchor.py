"""Regression guards for VLM V2 prompt anchoring.

These tests protect behavior, not incidental source formatting. The real Qwen
prompt must describe the output contract without embedding a populated visual
answer such as ambiguous/style_coherence/low. The populated fixture helper is
kept only for deterministic fake-backend tests.
"""

import inspect
import unittest

import src.vlm.prompt_v2 as prompt_v2


def _squash_whitespace(text: str) -> str:
    return " ".join(text.split())


class VlmPromptV2AntiAnchorTests(unittest.TestCase):
    def test_system_prompt_requires_image_grounded_labels(self):
        text = _squash_whitespace(prompt_v2.SYSTEM_PROMPT_V2)
        self.assertIn("There is deliberately no example visual answer", text)
        self.assertIn("Determine every visual dimension, effect, confidence", text)
        self.assertIn("from the supplied images", text)
        self.assertIn("Do not use a default label", text)
        self.assertIn(
            "Inspect the diagnosis and each recommendation candidate independently",
            text,
        )

    def test_real_prompt_builder_does_not_call_populated_fixture(self):
        source = inspect.getsource(prompt_v2.build_qwen_messages_v2)
        self.assertNotIn("expected_output_shape_v2(", source)
        self.assertNotIn("REQUESTED SHAPE", source)
        self.assertIn("_output_contract_text_v2", source)

    def test_output_contract_source_contains_no_seeded_visual_json(self):
        source = inspect.getsource(prompt_v2._output_contract_text_v2)
        # Allowed token lists are expected. What must not exist is a populated
        # semantic JSON answer that Qwen can copy verbatim.
        self.assertNotIn('"overall_visual_support": "ambiguous"', source)
        self.assertNotIn('"dimension": "style_coherence"', source)
        self.assertNotIn('"effect": "ambiguous"', source)
        self.assertNotIn('"confidence": "low"', source)

    def test_fixture_helper_is_documented_test_only(self):
        doc = prompt_v2.expected_output_shape_v2.__doc__ or ""
        self.assertIn("unit tests and fake", doc)
        self.assertIn("MUST NOT be embedded", doc)


if __name__ == "__main__":
    unittest.main()
