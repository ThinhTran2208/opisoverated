# Recommendation V2 — Category-Aware Hybrid Retrieval

## Scope

Recommendation V2 keeps the frozen FashionCLIP embeddings, LOO diagnosis integration, and frozen V5 compatibility scorer from V1. The main model-logic change is candidate retrieval order, with a runtime robustness fallback when exact master-category metadata is unavailable.

```text
problematic item
  -> if exact master_category is available:
       build eligible pool from the same master_category
     else:
       fall back to the same Core-7 coarse category
  -> exclude current outfit items
  -> require metadata + embedding + image availability
  -> item-query cosine Top-200 inside the selected category pool
  -> context-centroid cosine Top-200 inside the selected category pool
  -> union + de-duplicate
  -> frozen V5 scorer reranking
  -> public Top-3
```

## Why V2 exists

V1 performed global cosine Top-200 before exact `master_category` filtering. This could spend most of the Top-200 budget on categories that cannot replace the problematic item. V2 applies the replacement-category constraint first so both retrieval channels spend their search budget inside an eligible replacement pool.

## Category selection policy

Preferred path:

```text
exact master_category available
→ exact master-category pool
```

Runtime fallback:

```text
exact master_category unavailable
but Core-7 category is known
→ Core-7 category pool
```

If neither exact master category nor a valid Core-7 category is available, recommendation cannot proceed.

The offline Polyvore one-item-swap benchmark normally has exact master-category metadata, so Core-7 fallback is a runtime robustness mechanism rather than the normal benchmark path.

## Two retrieval channels

Item channel:

```text
query = FashionCLIP embedding(problematic item)
```

Context channel:

```text
context = mean(FashionCLIP embeddings of all non-problematic outfit items)
context = L2_normalize(context)
```

Both cosine searches are performed only inside the selected category-constrained pool. Each retains Top-200, then the two lists are unioned and de-duplicated. The full union may contain up to roughly 400 candidates.

## Reranking

Every full-union candidate replaces the problematic position in the complete outfit. The frozen scorer computes `compatibility_logit` and reranks the full union. Public output is Top-3.

Frozen scorer checkpoint SHA-256:

```text
7b3d0b6e0d44e3de517565f5725ded198bbc762b02a4dece26a58ee145cfed9c
```

## Offline benchmark terminology

For each synthetic negative query:

```text
problematic position = negative_metadata.swapped_item_index
ground truth item    = negative_metadata.original_item_id
current swapped item = negative_metadata.replacement_item_id
```

The ground-truth item is only the original item removed by the synthetic corruption process. It is not assumed to be the unique valid recommendation.

## Conditional Hit@3 experiment branch

This branch changes only the evaluation view. The recommendation algorithm, retrieval candidates, frozen scorer, and public Top-3 remain unchanged.

Branch:

```text
feat/recommendation-conditional-hit-v2
```

The previous branch keeps the original unconditional Hit@1 / Hit@3 / MRR metrics unchanged:

```text
feat/recommendation-category-aware-v2
```

### Why conditional Hit@3

Unconditional Hit@3 divides by every benchmark query, including queries where the exact ground-truth item was never present in the candidate set given to the scorer. That mixes candidate-generation failure with reranker behavior.

Conditional Hit@3 instead asks:

> When the exact reference item is present in the full retrieval union actually seen by the scorer, how often does the scorer rank it in the final Top-3?

Formula:

```text
Conditional Hit@3
=
# queries where GT is in final Top-3
----------------------------------
# queries where GT is in FULL retrieval union
```

The denominator uses the full de-duplicated union, not `Hybrid Recall@200`. This matters because the scorer reranks the full union, while `Hybrid Recall@200` inspects only the first 200 candidates under the deterministic hybrid ordering.

The evaluator also reports:

```text
full_union_gt_coverage
= # queries where GT is in full union / # valid queries
```

This separates the two stages:

```text
retrieval: can the exact reference be generated?
reranking: conditional on generation, can the scorer place it in Top-3?
```

### Interpretation limitation

Conditional Hit@3 is still an exact-reference recovery diagnostic, not recommendation accuracy. A candidate different from the original item can still be a valid or better replacement, but it will not count as a hit under this synthetic single-reference benchmark.

## Metrics on this branch

Main retrieval diagnostics:

- item-only Recall@50 / Recall@100 / Recall@200;
- context-only Recall@50 / Recall@100 / Recall@200;
- hybrid Top-200 Recall@50 / Recall@100 / Recall@200.

Reranking diagnostic:

- full-union GT coverage;
- Conditional exact-reference Hit@3.

Additional internal diagnostic:

- Replacement Success Rate at fixed epsilon `0.0`, interpreted only as frozen-scorer self-consistency.

The old unconditional Hit@1 / Hit@3 / MRR remain available on the original category-aware V2 branch and are intentionally not the headline reranking metrics on this experimental branch.

## Frozen V2 algorithm config

```text
recommendation_version = category-aware-hybrid-v2
retrieval_scope        = exact_master_category_before_cosine
runtime_fallback       = core7_when_master_category_unavailable
top_k_problematic      = 200
top_k_context          = 200
final_k                = 3
embedding_version      = fashionclip-512-l2-v1
scorer_version         = type_aware_pairwise_v1
```

Config file:

```text
configs/recommendation_category_aware_v2.json
```

This conditional-metric branch does not retrain or alter FashionCLIP, the scorer, LOO, retrieval logic, or recommendation output.
