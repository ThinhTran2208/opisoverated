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
internal deterministic explanation / handoff
        ↓
render_user_facing_vi_v2(...)
        ↓
vlm-user-facing-v2
```

The optional full original outfit image is not part of this baseline. It remains
a future ablation.

## Internal renderer vs end-user renderer

The VLM pipeline intentionally keeps two different presentation layers.

### Internal/debug layer

`render_explanation_vi_v2(...)` and `run["handoff"]` preserve implementation
metadata useful for debugging, integration, and audit.  They may contain scorer
or LOO terminology and should not be shown directly to a normal user.

### End-user layer

`render_user_facing_vi_v2(...)` is the deploy-facing Vietnamese UI payload. It
keeps the authoritative problematic item identity and Recommendation Top-3 item
IDs/ranks so the frontend can bind the correct images, but the prose deliberately
hides implementation vocabulary such as LOO, Qwen, logits, validator names, and
probability semantics.

Use:

```python
from src.vlm import render_user_facing_vi_v2

user_result = render_user_facing_vi_v2(
    run["visual_analysis"],
    run["evidence"],
)
```

The result schema is `vlm-user-facing-v2` and contains:

- the problematic item index, ID, category, headline, and plain-language reason;
- a short sentence telling the user to replace that item;
- exactly three authoritative replacement candidates in frozen rank order;
- one concise visual reason for each candidate;
- a plain-language caution when visual evidence and the compatibility decision do
  not fully agree.

Raw `compatibility_logit`, `improvement_logit`, score summaries, model names, and
internal validation details are intentionally absent from this user-facing
payload.

Example shape:

```text
Item 3 (túi) là món đồ có vấn đề nhất trong outfit.
Bạn nên thử thay Item 3 (túi) bằng một trong ba gợi ý bên dưới.

Gợi ý 1: thay Item 3 bằng món này.
→ Màu sắc hài hòa với các món còn lại.

Gợi ý 2: thay Item 3 bằng món này.
→ Hình ảnh chưa cho thấy ưu điểm thị giác đủ rõ.

Gợi ý 3: thay Item 3 bằng món này.
→ Phom dáng giúp outfit cân đối hơn.
```

A visual contradiction never changes the upstream diagnosis or recommendation
rank.  The user-facing renderer instead phrases disagreement as uncertainty, for
example that the image itself does not show an obvious mismatch and the result
should be treated as a suggestion rather than a certain conclusion.

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

The wrapper performs one allowed repair retry and returns the internal
`vlm-run-v2` record containing evidence hash, validated visual analysis,
deterministic internal explanation, and the existing integration handoff.  For
end-user UI, render `vlm-user-facing-v2` from the validated `visual_analysis` and
`evidence` as shown above.

## Runtime configuration

V2 uses `configs/vlm_qwen3_vl_4b_instruct_v2.json` with:

- `Qwen/Qwen3-VL-4B-Instruct`;
- float16 / CUDA path;
- canonical 262,144 pixels per image;
- greedy decoding;
- one schema-repair retry;
- a V2 generation budget sized for diagnosis plus three recommendation rows.

T4 notebook runs may use a lower per-image pixel budget as a functional-test
override.  That override is not the canonical deploy configuration.

## Verification status

Real Qwen execution has successfully passed the V2 prompt, validator, renderer,
and Recommendation Top-3 grounding on NB10.  The user-facing renderer is a
separate deterministic layer and does not require another Qwen generation to
render an already validated run.

The known low-severity mapping leakage-hardening note in `schema_v2.py` is still
left for final review: forbidden top-level evaluation fields on a mapping input
should ideally hard-fail before allowed fields are extracted. The current code
drops those extra top-level fields before evidence reaches Qwen, so this note is
contract hardening rather than a known runtime leakage path.
