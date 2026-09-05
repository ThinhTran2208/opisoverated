# VLM Recommendation Evidence V2

## Scope

This is the first VLM V2 integration boundary. It does **not** change the frozen
Recommendation V2 algorithm and it does **not** change VLM V1.

Canonical upstream recommendation remains:

```text
feat/recommendation-rank-diagnostics-v2
category-aware-hybrid-v2
```

The new flow starts from authoritative runtime outputs:

```text
Frozen V5 scorer + LOO
        +
Recommendation V2 RecommendationResult
        ↓
build_vlm_evidence_v2(...)
        ↓
vlm-evidence-v2
```

Prompt/image binding and VLM output V2 are intentionally the next layer. This
commit only freezes what recommendation information a future VLM is allowed to
receive.

## Why a new schema

`vlm-evidence-v1` deliberately requires:

```json
{
  "recommendation": {
    "status": "not_implemented",
    "items": []
  }
}
```

That V1 contract remains unchanged. Recommendation is added only through the new
`vlm-evidence-v2` schema.

## Recommendation evidence

The V2 recommendation section is:

```json
{
  "status": "available",
  "version": "category-aware-hybrid-v2",
  "problematic_item_index": 2,
  "problematic_item_id": "...",
  "ranking_semantics": "frozen_scorer_descending_compatibility_logit",
  "score_semantics": "uncalibrated_logits_not_probabilities",
  "items": [
    {
      "rank": 1,
      "item_id": "...",
      "master_category": "...",
      "coarse_category": "SHOES",
      "compatibility_logit": 0.42,
      "improvement_logit": 0.73
    }
  ]
}
```

Exactly three recommendation rows are required.

`image_url` is intentionally not copied into structured evidence. Candidate
images will be bound explicitly in the VLM prompt layer so visual inputs cannot
be confused with textual evidence.

## Grounding checks

`build_recommendation_evidence(...)` cross-checks the public Recommendation V2
Top-3 against `RecommendationResult.internal_metadata` before serializing it.
It hard-fails unless:

- recommendation version is exactly `category-aware-hybrid-v2`;
- recommendation problematic index equals the LOO-selected index;
- public ranks are exactly 1, 2, 3;
- public Top-3 item IDs equal the first three frozen-scorer reranked candidates;
- candidate IDs are unique and are not already in the current outfit;
- candidate Core-7 category equals the problematic item's Core-7 category;
- candidate logits are finite;
- `improvement_logit == candidate_compatibility_logit - original_outfit_logit`;
- Top-3 order is consistent with frozen-scorer descending compatibility logit.

The serialized V2 validator rechecks all invariants that can be checked from the
evidence alone.

## Evaluation leakage policy

Synthetic benchmark/evaluation data is not explanation evidence. The V2 builder
rejects fields such as:

```text
label
negative_metadata
swapped_item_index
target_swapped_item_index
original_item_id
ground_truth_item_id
replacement_item_id
Hit@K / MRR fields
source_split
```

Therefore Recommendation V2 evaluation artifacts must not be fed to VLM V2.
The intended source is the runtime `RecommendationResult` returned by the frozen
recommendation pipeline.

## API

```python
from src.vlm import build_vlm_evidence_v2

evidence = build_vlm_evidence_v2(
    loo_result,
    recommendation_result,
    sample_id=sample_id,
    item_ids=item_ids,
    coarse_categories=coarse_categories,
)
```

## Next layer

With this evidence contract frozen, the next VLM V2 step is:

```text
vlm-evidence-v2
+ full original outfit image (when available)
+ original garment crop images
+ Top-3 candidate images
        ↓
Qwen3-VL constrained visual analysis V2
        ↓
strict validator
        ↓
deterministic Vietnamese renderer
```

The full original outfit image is used for whole-outfit context. Crops remain
positional for exact item binding, and candidate images remain keyed by
authoritative item ID.
