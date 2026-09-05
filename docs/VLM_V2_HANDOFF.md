# VLM V2 Renderer and Handoff

## Baseline flow

```text
Frozen scorer + LOO
        +
Recommendation V2 Top-3
        ↓
full internal vlm-evidence-v2
(raw logits kept for audit/validation)
        ↓
score-free vlm-prompt-context-v2
        +
full original outfit image + original garment crops + Top-3 candidate images
        ↓
Qwen3-VL closed-taxonomy visual evidence
        ↓
validate_visual_analysis_v2(...)
        ↓
internal QA/debug explanation
        +
score-free integration handoff
        +
vlm-user-facing-v2
```

The full original outfit image is sent when available. It supplies whole-outfit
composition and layering context; positional crops remain the authoritative
item-level visual bindings.

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

Prompt policy asks Qwen to first seek one concrete positive visible relation for
each recommendation candidate. If no clear positive reason is visible, Qwen
returns `ambiguous` with no filler observations rather than inventing a
justification or producing weak negative commentary. Clear contradictions remain
available only as internal QA evidence.

## Raw-score boundary

Full `vlm-evidence-v2` keeps scorer/LOO/recommendation numerical values because
they are needed for deterministic validation, audit, and the ranking-based
fallback logic used by the renderer.

Those raw values are **not sent to Qwen**. `build_prompt_context_v2(...)` projects
the full evidence into `vlm-prompt-context-v2`, which contains only:

- original item index / ID / coarse category;
- authoritative problematic item index / ID / category;
- the two-item-extrapolation flag when applicable;
- Top-3 recommendation rank / ID / category metadata.

No `compatibility_logit`, `improvement_logit`, LOO delta, without-item score,
checkpoint score, or other raw numerical scorer state is present in the prompt
context. This reduces numerical anchoring and keeps visual evidence image-driven.

## Leakage contract

For plain Mapping recommendation inputs, the **entire raw nested mapping is
scanned before public/internal fields are projected**. Any forbidden benchmark or
evaluation field such as `original_item_id`, `ground_truth_item_id`, `hit_at_3`,
`mrr`, etc. causes a fail-fast `ValueError`.

This prevents adapter projection from silently dropping benchmark fields before
the leakage scanner can see them.

## Internal/debug layer

The following remain internal/audit artifacts and may contain technical terms or
raw scores:

```text
run["evidence"]
run["visual_analysis"]
run["explanation"]
run["raw_response"]   # only when configured
```

`render_explanation_vi_v2(...)` is explicitly an internal renderer. It may still
show LOO deltas, compatibility logits, Qwen taxonomy, and confidence labels for
debugging.

## Score-free integration handoff

`run["handoff"]` uses schema `vlm-handoff-v2` and is now score-free. It preserves
only integration-safe data such as:

- problematic item ID/index;
- Top-3 item IDs, rank, category metadata;
- validated visual summaries/observations;
- machine-readable limitations.

It does **not** contain raw compatibility/improvement logits, LOO deltas, score
summaries, model ID, generation attempts, raw response, or full evidence.

## End-user layer

`run["user_facing"]` is produced automatically by the V2 pipeline. It is the
payload intended for the UI. The same payload can still be recreated directly:

```python
from src.vlm import render_user_facing_vi_v2

user_result = render_user_facing_vi_v2(
    run["visual_analysis"],
    run["evidence"],
)
```

The result schema is `vlm-user-facing-v2` and contains:

- one final `text` string suitable for direct display;
- the authoritative item identity/category that should be replaced;
- exactly three authoritative replacement candidates in frozen rank order;
- safe display names derived from coarse category, e.g. `Mẫu túi 1`, `Mẫu túi 2`, `Mẫu túi 3`;
- a positive visual reason only when Qwen provides grounded support;
- a ranking-based fallback sentence when Qwen cannot provide a positive visual reason;
- machine-only `item_id` and `rank` so frontend can bind the correct recommendation image.

User-facing prose hides implementation vocabulary such as LOO, Qwen, logits,
validator names, and raw confidence taxonomy. Internal visual disagreement is
not surfaced as a counter-argument to the user.

Example for the bag case used during manual review:

```text
Ba mẫu túi bên dưới đều là những phương án phù hợp hơn với outfit khi thay cho chiếc túi hiện tại.
Mẫu túi 1 là lựa chọn ưu tiên nhất, với màu sắc phối hợp tốt với các món còn lại.
Mẫu túi 2 là lựa chọn thứ hai và cũng phù hợp hơn với outfit so với chiếc túi hiện tại.
Mẫu túi 3 là lựa chọn thứ ba, với phom dáng giúp tổng thể outfit cân đối hơn.
Bạn có thể tham khảo ba mẫu trên để thay cho chiếc túi hiện tại và chọn phương án phù hợp nhất với sở thích của mình.
```

If a candidate does not improve according to the internal selection logic, the
renderer does not make a numeric claim; it only says that it is one of the
prioritized replacement candidates.

## Frontend image binding

Recommendation images are a frontend/deploy responsibility. The frontend binds
each recommendation image using the authoritative candidate `item_id` and `rank`
from the same Recommendation V2 result.

For example:

```text
[Mẫu túi 1 image]
Mẫu túi 1
Mẫu túi 1 là lựa chọn ưu tiên nhất, với màu sắc phối hợp tốt với các món còn lại.
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

For a local/demo deployment, the team can use whichever GPU the active Colab Pro
session provides. If the canonical 262,144-pixel budget does not fit a given GPU,
a lower visual budget can be used as an explicit runtime override for the demo;
this is an infrastructure choice and does not change the VLM evidence contract.

## Verification status

A real Qwen NB10 smoke test previously passed schema validation and Top-3
grounding. After manual review, the prompt, leakage boundary, prompt projection,
handoff boundary, and user-facing renderer were revised.

Before final freeze, run the revised prompt on real Qwen again, then measure
batch failure/repair rate and perform manual quality review across multiple
samples. Production HTTP wiring can proceed against `run["handoff"]` and
`run["user_facing"]` while that validation is being completed.
