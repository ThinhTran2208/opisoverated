# VLM Prompt V2 — crop-only baseline with Recommendation Top-3

## Scope

This layer consumes `vlm-evidence-v2` and binds visual inputs for Qwen3-VL.
It intentionally keeps the baseline visual contract small and deterministic:

```text
vlm-evidence-v2
+ one crop per original outfit item
+ exactly three authoritative recommendation images
        ↓
Qwen3-VL constrained visual analysis V2
```

The original full outfit image is **not** part of this baseline. It can be tested
later on a separate ablation branch without changing the canonical crop-only V2
contract.

## Image binding

Original outfit crops remain positional because V1 already freezes the canonical
`item_index` order.

Recommendation images are not positional. The caller must provide a mapping:

```python
{
    candidate_item_id: image_ref,
    ...
}
```

The key set must match the authoritative Top-3 item IDs exactly. The prompt layer
then binds images in rank order from `vlm-evidence-v2`.

This prevents a caller from accidentally supplying the right three images in the
wrong recommendation order.

## Visual input groups

Qwen receives two explicitly labeled groups.

### Original outfit item crops

Each image is preceded by:

```text
ORIGINAL OUTFIT ITEM:
item_index=...
item_id=...
coarse_category=...
problematic_item=true|false
```

### Recommendation candidates

Each image is preceded by:

```text
AUTHORITATIVE RECOMMENDATION CANDIDATE:
rank=1|2|3
item_id=...
master_category=...
coarse_category=...
```

Candidate identity and rank are explicitly declared authoritative.

## Closed output taxonomy

Qwen does not produce user-facing prose. It must return
`vlm-visual-analysis-v2` with:

```text
problematic item identity
+ diagnosis visual support
+ exactly three recommendation rows
+ closed visual observations
+ required limitations
```

Visible dimensions are frozen to:

```text
color_harmony
pattern_coherence
silhouette_balance
formality_alignment
style_coherence
```

Diagnosis effect values:

```text
supports_loo
ambiguous
contradicts_loo
```

Recommendation effect values:

```text
supports_recommendation
ambiguous
contradicts_recommendation
```

Confidence values:

```text
low
medium
high
```

No free-text `headline`, `reason`, `description`, `recommendation_text`, or
`explanation` field exists in the requested model output.

## Recommendation context rule

A recommendation candidate is intended to replace the LOO-selected problematic
item. Therefore recommendation visual observations may reference only the
**remaining original outfit item indices** and must not use the problematic
original item as context.

For example, if item 2 is problematic in a four-item outfit, recommendation
observations may reference only:

```text
[0, 1, 3]
```

This keeps the visual question aligned with the actual replacement operation.

## Authority boundary

The VLM may visually support, contradict, or remain ambiguous about upstream
model decisions, but it may not change them:

```text
LOO controls problematic item
Recommendation V2 controls candidate IDs and rank
Frozen scorer controls numerical compatibility scores
Qwen only classifies visible relations
```

The prompt explicitly forbids Qwen from inventing, removing, replacing, or
reranking candidates.

## Score semantics

The prompt repeats that all scorer/LOO/recommendation logits are uncalibrated
model outputs. Qwen may not interpret them as probabilities, percentages, or
objective fashion truth.

## Required limitations

Baseline V2 requires these machine-readable disclosures:

```text
compatibility_logit_is_not_probability
recommendation_scores_are_not_probabilities
recommendation_identity_and_rank_are_authoritative
vlm_visual_observations_are_inferences
```

For three-item outfits where LOO scores two-item subsets, the existing V1 token
is additionally required:

```text
loo_uses_two_item_extrapolation
```

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

## Next layer

The next implementation step is deterministic validation of
`vlm-visual-analysis-v2` followed by the Vietnamese renderer. The validator must
recheck problematic identity, recommendation IDs/ranks, observation item
references, closed enums, and exact limitation disclosure before any prose is
rendered.
