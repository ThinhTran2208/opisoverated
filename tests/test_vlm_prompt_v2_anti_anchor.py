"""Regression guards for VLM V2 prompt anchoring.

The real Qwen prompt must describe the schema without embedding a populated
visual-analysis answer such as ambiguous/style_coherence/low. The populated
fixture helper remains available only for deterministic fake-backend tests.
"""

import inspect
import unittest

import src.vlm.prompt_v2 as prompt_v2


class VlmPromptV2AntiAnchorTests(unittest.TestCase):
    def test_system_prompt_requires_image_grounded_labels(self):
        text = prompt_v2.SYSTEM_PROMPT_V2
        self.assertIn("There is deliberately no example visual answer", text)
        self.assertIn("Determine every", text)
        self.assertIn("from the supplied\n    images", text)
        self.assertIn("Do not use a default label", text)
        self.assertIn("Inspect the diagnosis and each recommendation candidate independently", text)

    def test_real_prompt_builder_does_not_embed_populated_fixture(self):
        source = inspect.getsource(prompt_v2.build_qwen_messages_v2)
        self.assertNotIn("expected_output_shape_v2(", source)
        self.assertNotIn("REQUESTED SHAPE", source)
        self.assertIn("_output_contract_text_v2", source)
        self.assertIn("no populated", source)

    def test_output_contract_builder_contains_no_seeded_visual_answer(self):
        source = inspect.getsource(prompt_v2._output_contract_text_v2)
        # Allowed token lists are expected. What must not exist is a populated
        # semantic JSON example that models can copy verbatim.
        self.assertNotIn('"overall_visual_support": "ambiguous"', source)
        self.assertNotIn('"dimension": "style_coherence"', source)
        self.assertNotIn('"effect": "ambiguous"', source)
        self.assertNotIn('"confidence": "low"', source)
        # inspect.getsource() sees adjacent Python string literals before they are
        # concatenated at runtime, so assert on the contiguous literal fragment.
        self.assertIn("not default to the first index", source)
        self.assertIn("Evaluate each candidate independently", source)

    def test_fixture_helper_is_documented_test_only(self):
        doc = prompt_v2.expected_output_shape_v2.__doc__ or ""
        self.assertIn("unit tests and fake", doc)
        self.assertIn("MUST NOT be embedded", doc)


if __name__ == "__main__":
    unittest.main()
