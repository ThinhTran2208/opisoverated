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
metadata useful for debugging, integration, and audit. They may contain scorer
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

- one final plain-language `text` string suitable for direct display;
- the problematic item identity/category and structured copy for UI layout;
- exactly three authoritative replacement candidates in frozen rank order;
- a safe display name derived from coarse category, e.g. `Mẫu túi 1`, `Mẫu túi 2`, `Mẫu túi 3`;
- one concise visual reason for each candidate;
- a plain-language caution when visual evidence and the compatibility decision do not fully agree.

The candidate `item_id` and `rank` stay in the machine payload only so the frontend can bind the correct recommendation image. The renderer does not ask Qwen to invent a specific subtype such as shoulder bag vs backpack; the displayed image is the source of truth for the exact visual item.

Raw `compatibility_logit`, `improvement_logit`, score summaries, model names, and internal validation details are intentionally absent from `vlm-user-facing-v2`. They may remain in internal/debug artifacts and must not be rendered to the end user.

Example final text:

```text
Chiếc túi hiện tại được hệ thống đánh giá là món kém phù hợp nhất với outfit.
Tuy nhiên, khi nhìn riêng về mặt thị giác, món này không có dấu hiệu lệch outfit quá rõ.
Bạn có thể thử thay món này bằng một trong ba mẫu túi bên dưới để outfit hài hòa hơn.
Mẫu túi 1 phù hợp về mặt thị giác vì màu sắc hài hòa với các món còn lại.
Mẫu túi 2 chưa cho thấy ưu điểm thị giác đủ rõ để nổi bật hơn các lựa chọn còn lại.
Mẫu túi 3 phù hợp về mặt thị giác vì phom dáng giúp outfit cân đối hơn.
```

A visual contradiction never changes the upstream diagnosis or recommendation
rank. The user-facing renderer instead phrases disagreement as uncertainty, for
example that the image itself does not show an obvious mismatch and the result
should be treated as a suggestion rather than a certain conclusion.

## Frontend image binding

Recommendation images are a frontend/deploy responsibility. The frontend should
bind each image using the authoritative candidate `item_id` and `rank` from the
same Recommendation V2 result. The visible card can therefore show, for example:

```text
[Mẫu túi 1 image]
Mẫu túi 1
Màu sắc hài hòa với các món còn lại.
```

The VLM does not generate recommendation images and does not change candidate
identity or order.

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
deterministic internal explanation, and the existing integration handoff. For
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
override. That override is not the canonical deploy configuration.

## Verification status

Real Qwen execution has successfully passed the V2 prompt, validator, renderer,
and Recommendation Top-3 grounding on NB10. The user-facing renderer is a
separate deterministic layer and does not require another Qwen generation to
render an already validated run.

The known low-severity mapping leakage-hardening note in `schema_v2.py` is still
left for final review: forbidden top-level evaluation fields on a mapping input
should ideally hard-fail before allowed fields are extracted. The current code
drops those extra top-level fields before evidence reaches Qwen, so this note is
contract hardening rather than a known runtime leakage path.
