# VLM Prompt V2 — crop-only baseline with Recommendation Top-3

## Scope

This layer consumes `vlm-evidence-v2` and binds visual inputs for Qwen3-VL:

```text
vlm-evidence-v2
+ one crop per original outfit item
+ exactly three authoritative recommendation images
        ↓
Qwen3-VL constrained visual analysis V2
        ↓
deterministic validator
        ↓
deterministic Vietnamese renderer
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
Qwen only classifies visible relations
```

Qwen may visually support, contradict, or remain ambiguous about the upstream
decisions, but it may not change them.

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

## Numerical evidence

Scorer logits, LOO deltas, and recommendation improvement logits are
uncalibrated model outputs. Qwen is explicitly told not to interpret them as
probabilities, percentages, objective fashion truth, or as a substitute for
visual inspection.

## Required limitations

Baseline disclosures:

```text
compatibility_logit_is_not_probability
recommendation_scores_are_not_probabilities
recommendation_identity_and_rank_are_authoritative
vlm_visual_observations_are_inferences
```

For three-item outfits, where LOO scores two-item subsets, this token is added:

```text
loo_uses_two_item_extrapolation
```

## Implemented downstream checks

`validate_visual_analysis_v2(...)` hard-fails output that changes the problematic
item, candidate identity, Top-3 rank, allowed context references, taxonomy, or
required limitations. `render_explanation_vi_v2(...)` then creates Vietnamese
user-facing text from validated enum values only.

## API

```python
from src.vlm import build_qwen_messages_v2

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

The canonical config remains `262144` pixels per image. NB10 may use a lower
pixel budget only as a T4 functional-test override; that does not change this
protocol contract.
