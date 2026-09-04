# VLM V2 Renderer and Handoff

## Completed baseline flow

```text
Frozen scorer + LOO
        +
Recommendation V2 Top-3
        ↓
vlm-evidence-v2
        ↓
original garment crops + Top-3 candidate images
        ↓
Qwen3-VL closed-taxonomy visual analysis
        ↓
validate_visual_analysis_v2(...)
        ↓
render_explanation_vi_v2(...)
        ↓
vlm-explanation-v2
        ↓
vlm-handoff-v2
```

The optional full original outfit image is not part of this baseline. It remains
a future ablation.

## Deterministic renderer

Qwen does not write user-facing prose. It returns only closed enum tokens and
authoritative copied IDs/ranks. `render_explanation_vi_v2(...)` converts the
validated analysis into Vietnamese templates for:

- LOO diagnosis evidence and visual support;
- each authoritative Recommendation V2 Top-3 candidate;
- candidate scorer/improvement logits with explicit non-probability semantics;
- visual observations for color, pattern, silhouette, formality, and style;
- uncertainty and required limitations.

A visual contradiction never changes the upstream decision. It is rendered as a
visible disagreement while keeping LOO and Recommendation V2 authoritative.

## End-to-end VLM V2 wrapper

Use:

```python
from src.vlm import VLMExplanationPipelineV2, load_vlm_config_v2

config = load_vlm_config_v2("configs/vlm_qwen3_vl_4b_instruct_v2.json")
pipeline = VLMExplanationPipelineV2(qwen_backend, config)

run = pipeline.explain(
    evidence,
    outfit_image_refs,
    recommendation_image_refs,
)
```

`recommendation_image_refs` is keyed by authoritative candidate `item_id`, not
position, to prevent accidental rank/image mismatch.

The wrapper performs one allowed repair retry using the same generation policy
as V1, then returns an internal `vlm-run-v2` record containing evidence hash,
validated visual analysis, deterministic explanation, and deploy handoff.

## Deploy-facing output

The stable compact output is:

```python
run["handoff"]
```

with schema `vlm-handoff-v2`. It contains:

- protocol/model/generation-attempt metadata;
- problematic item identity;
- rendered diagnosis;
- exactly three authoritative recommendation rows in frozen rank order;
- final deterministic explanation and limitations.

It intentionally excludes raw Qwen text, private structured evidence, and the
internal visual-analysis payload.

## Frozen runtime configuration

V2 uses `configs/vlm_qwen3_vl_4b_instruct_v2.json` with the same Qwen runtime
settings already used by V1:

- `Qwen/Qwen3-VL-4B-Instruct`;
- float16 / CUDA path;
- 262,144 pixels per image;
- greedy decoding;
- one schema-repair retry.

Only the explanation protocol is versioned to `vlm-explanation-v2` so V1 and V2
results cannot be confused.

## Remaining verification

Implementation is complete, but real-model verification is still required before
freezing the branch for handoff:

1. run the full unit-test suite;
2. run real Qwen3-VL on VALID samples through actual Recommendation V2 output;
3. inspect validator failure/repair rate;
4. manually review Vietnamese output and visual usefulness;
5. fix any integration/runtime issues found by execution.

The known low-severity mapping leakage-hardening note in `schema_v2.py` is still
left for final review: forbidden top-level evaluation fields on a mapping input
should ideally hard-fail before allowed fields are extracted. The current code
drops those extra top-level fields before evidence reaches Qwen, so this note is
contract hardening rather than a known runtime leakage path.
