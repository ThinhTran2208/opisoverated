# VLM V2 Renderer and Handoff

## Baseline flow

```text
Frozen scorer + LOO
        +
Recommendation V2 Top-3
        ↓
vlm-evidence-v2
        ↓
original garment crops + Top-3 candidate images
        ↓
Qwen3-VL closed-taxonomy visual evidence
        ↓
validate_visual_analysis_v2(...)
        ↓
internal QA/debug analysis
        ↓
render_user_facing_vi_v2(...)
        ↓
vlm-user-facing-v2
```

The optional full original outfit image is not part of this baseline. It remains
a future ablation.

## Decision authority vs visual evidence

The production contract is intentionally asymmetric:

- frozen scorer + LOO decide the problematic original item;
- Recommendation V2 decides the authoritative Top-3 candidate identities and rank;
- Qwen does **not** make a second fashion decision and does **not** rerank anything;
- Qwen only extracts image-grounded visual evidence that can help explain the already-fixed decisions.

The internal Qwen schema still allows `supports_*`, `ambiguous`, and `contradicts_*`
labels for QA. Those labels are not a user-facing vote on the upstream result.
`ambiguous` or `contradicts_*` never authorizes the VLM to remove a candidate,
change rank, or replace the diagnosed item.

Prompt policy now asks Qwen to first seek one concrete positive visible relation
for each recommendation candidate. If no clear positive reason is visible, Qwen
should return `ambiguous` with no filler observations rather than inventing a
justification or producing weak negative commentary. Clear contradictions remain
available only as internal QA evidence.

## Internal/debug layer

`render_explanation_vi_v2(...)`, raw visual analysis, evidence, and the existing
internal handoff may preserve scorer/LOO terminology, logits, confidence labels,
and visual disagreement for debugging or audit. They must not be rendered
directly to a normal user.

## End-user layer

`render_user_facing_vi_v2(...)` is the deploy-facing Vietnamese UI payload.

Use:

```python
from src.vlm import render_user_facing_vi_v2

user_result = render_user_facing_vi_v2(
    run["visual_analysis"],
    run["evidence"],
)
```

The result schema is `vlm-user-facing-v2` and contains:

- one final `text` string suitable for direct display;
- the authoritative problematic item identity/category;
- exactly three authoritative replacement candidates in frozen rank order;
- safe display names derived from coarse category, e.g. `Mẫu túi 1`, `Mẫu túi 2`, `Mẫu túi 3`;
- an optional positive visual reason when Qwen provides grounded support;
- a ranking-based fallback sentence when Qwen cannot provide a positive visual reason;
- machine-only `item_id` and `rank` so frontend can bind the correct recommendation image.

User-facing prose deliberately hides implementation vocabulary such as LOO,
Qwen, logits, validator names, and raw confidence taxonomy. Raw
`compatibility_logit`, `improvement_logit`, and score summaries are absent from
`vlm-user-facing-v2`.

Internal visual disagreement is also not surfaced as a counter-argument to the
user. If Qwen marks a recommendation `ambiguous` or `contradicts_recommendation`,
the renderer does not say that the recommendation is weak or that the system
disagrees with itself. It keeps the authoritative recommendation and simply
omits an unsupported visual justification.

Example for a case where all three candidate improvement logits are positive:

```text
Chiếc túi hiện tại là món được đánh giá kém phù hợp nhất và được ưu tiên thay trong outfit.
Đây là món được ưu tiên thay để cải thiện độ phù hợp tổng thể của outfit.
Cả ba mẫu túi bên dưới đều được đánh giá phù hợp hơn khi thay cho chiếc túi hiện tại.
Mẫu túi 1 là lựa chọn được xếp hạng đầu tiên; màu sắc phối hợp tốt với các món còn lại.
Mẫu túi 2 là phương án thay thế thứ hai và được đánh giá phù hợp hơn chiếc túi hiện tại.
Mẫu túi 3 là phương án thay thế thứ ba; phom dáng giúp tổng thể outfit cân đối hơn.
Bạn có thể tham khảo ba phương án trên để thay cho chiếc túi hiện tại và chọn mẫu phù hợp nhất với sở thích của mình.
```

If a candidate does not improve on the current outfit score, the renderer does
not falsely claim it is better; it only says that it is one of the highest-ranked
available replacement candidates.

## Frontend image binding

Recommendation images are a frontend/deploy responsibility. The frontend binds
each recommendation image using the authoritative candidate `item_id` and `rank`
from the same Recommendation V2 result.

For example:

```text
[Mẫu túi 1 image]
Mẫu túi 1
Màu sắc phối hợp tốt với các món còn lại.
```

The VLM does not generate recommendation images and does not change candidate
identity or order.

## Runtime configuration

V2 uses `configs/vlm_qwen3_vl_4b_instruct_v2.json` with:

- `Qwen/Qwen3-VL-4B-Instruct`;
- float16 / CUDA path;
- canonical 262,144 pixels per image;
- greedy decoding;
- one schema-repair retry;
- V2 generation budget for diagnosis plus three recommendation rows.

T4 notebook runs may use a lower per-image pixel budget as a functional-test
override. That override is not the canonical deploy configuration.

## Verification status

A real Qwen NB10 smoke test previously passed schema validation and Top-3
grounding. The explanation-role prompt and user-facing renderer were subsequently
revised after manual review showed that surfacing visual disagreement produced
confusing end-user copy.

Therefore the revised prompt policy still requires another real-Qwen smoke run,
followed by batch failure/repair-rate measurement and manual quality review before
final freeze.

The known low-severity mapping leakage-hardening note in `schema_v2.py` remains:
forbidden top-level evaluation fields on a Mapping input should ideally hard-fail
before allowed fields are extracted. The current code drops those extra fields
before evidence reaches Qwen, so this is contract hardening rather than a known
runtime leakage path.
