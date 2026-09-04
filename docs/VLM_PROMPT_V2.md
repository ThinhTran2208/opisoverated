# VLM Prompt V2 — crop-only baseline with Recommendation Top-3

## Scope

The V2 pipeline validates full internal `vlm-evidence-v2`, then projects only the
visual identities/categories/ranks needed by Qwen into a score-free
`vlm-prompt-context-v2`:

```text
full vlm-evidence-v2
(raw scorer / LOO / recommendation values retained internally)
        ↓
build_prompt_context_v2(...)
        ↓
score-free vlm-prompt-context-v2
+ one crop per original outfit item
+ exactly three authoritative recommendation images
        ↓
Qwen3-VL constrained visual analysis V2
        ↓
deterministic validator
        ↓
internal renderer + score-free handoff + user-facing renderer
```

The full original outfit image is intentionally outside this baseline and can be
studied later as a separate ablation.

## Image binding

Original outfit crops are positional because `item_index` is authoritative.
Recommendation images are supplied as a mapping keyed by candidate `item_id`:

```python
{
    candidate_item_id: image_ref,
    ...
}
```

The key set must match the authoritative Top-3 exactly. The prompt builder then
orders candidate images by frozen Recommendation V2 rank. This prevents a valid
set of three images from being accidentally attached to the wrong ranks.

## Authority boundary

```text
LOO controls problematic item
Recommendation V2 controls candidate IDs and rank
Frozen scorer controls numerical compatibility scores
Qwen only extracts visible relations
```

Qwen is not a second decision-maker. It may record support, ambiguity, or a clear
contradiction as internal QA evidence, but none of those labels can change the
problematic item or Top-3 candidates/rank.

## Score-free prompt context

`build_prompt_context_v2(...)` exposes only:

```text
items:
- item_index
- item_id
- coarse_category

diagnosis:
- problematic_item_index
- problematic_item_id
- problematic_category
- uses_two_item_extrapolation

recommendation.items:
- rank
- item_id
- master_category
- coarse_category
```

The following remain internal and are not serialized into Qwen prompt context:

```text
compatibility_logit
improvement_logit
full_logit
without_item_logits / without_item_logit
loo_delta / deltas_without_minus_full
top1_top2_delta_gap
checkpoint / scorer score state
```

This keeps Qwen's visual evidence image-driven and avoids numerical anchoring.
The prompt explicitly says not to reconstruct or guess hidden numerical values.

## Closed visual taxonomy

Visible dimensions:

```text
color_harmony
pattern_coherence
silhouette_balance
formality_alignment
style_coherence
```

Diagnosis effects / overall labels:

```text
supports_loo
ambiguous
contradicts_loo
```

Recommendation effects / overall labels:

```text
supports_recommendation
ambiguous
contradicts_recommendation
```

Confidence:

```text
low
medium
high
```

Qwen does not write user-facing prose. Free-text fields such as `headline`,
`reason`, `description`, `recommendation_text`, or `explanation` are not part of
`vlm-visual-analysis-v2`.

## Explanation-role policy

For diagnosis, Qwen first looks for a visible relation that helps explain why the
already-fixed problematic item fits the outfit less well. If no grounded support
is visible, it should return `ambiguous` instead of inventing a reason.

For each Recommendation V2 candidate, Qwen evaluates the candidate independently
against the remaining original outfit context and first looks for one concrete
positive visual relation. If none is clear, it should return `ambiguous` with an
empty observation list rather than filler or a fabricated justification.

A recommendation candidate being `ambiguous` internally does not remove it or
rerank it. Candidate identity and rank remain authoritative upstream output.

## Recommendation context rule

A recommendation replaces the LOO-selected problematic item. Candidate visual
observations therefore may reference only the remaining original outfit items.
For a four-item outfit where item 2 is problematic, the allowed context is:

```text
[0, 1, 3]
```

The problematic original item is never used as recommendation context.

## Anti-anchoring rule

The real Qwen prompt deliberately contains **no populated visual-analysis
example**. In particular, it does not seed a default answer such as:

```text
ambiguous / style_coherence / low
```

Instead, it describes the exact JSON contract and lists allowed tokens. The
model is instructed to determine visual dimension, effect, confidence, and
overall support from the supplied images, to evaluate each candidate
independently, and not to mechanically reuse the first context item.

`expected_output_shape_v2(...)` still exists only as a deterministic fixture for
unit tests and fake backends; `build_qwen_messages_v2(...)` must not embed it in
the real model prompt.

## Required limitations

The machine output still carries fixed protocol disclosures:

```text
compatibility_logit_is_not_probability
recommendation_scores_are_not_probabilities
recommendation_identity_and_rank_are_authoritative
vlm_visual_observations_are_inferences
```

These are protocol tokens only; the corresponding raw score values are not
included in Qwen's prompt context.

For three-item outfits, where LOO scores two-item subsets, this token is added:

```text
loo_uses_two_item_extrapolation
```

## Implemented downstream checks

`validate_visual_analysis_v2(...)` hard-fails output that changes the problematic
item, candidate identity, Top-3 rank, allowed context references, taxonomy, or
required limitations.

`render_explanation_vi_v2(...)` is internal/debug and may retain technical audit
information. `render_user_facing_vi_v2(...)` creates the final Vietnamese UI
payload and does not expose raw score fields or internal model debate.

## API

```python
from src.vlm import build_prompt_context_v2, build_qwen_messages_v2

prompt_context = build_prompt_context_v2(evidence)

messages = build_qwen_messages_v2(
    evidence,
    outfit_image_refs,
    {
        "candidate-a": candidate_a_image,
        "candidate-b": candidate_b_image,
        "candidate-c": candidate_c_image,
    },
    min_pixels=262144,
    max_pixels=262144,
)
```

The canonical config remains `262144` pixels per image. For a local/demo run, a
lower pixel budget can be used as an explicit runtime override if the active GPU
cannot fit the canonical visual budget; this does not change the evidence or
prompt contract.
