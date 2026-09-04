# Recommendation V2 — Category-Aware Hybrid Retrieval

## Scope

Recommendation V2 keeps the frozen FashionCLIP embeddings, LOO diagnosis integration, and frozen V5 compatibility scorer from V1. The only model-logic change is candidate retrieval order.

```text
problematic item
  -> resolve exact master_category
  -> build eligible candidate pool from the same master_category
  -> exclude current outfit items
  -> require embedding + image availability
  -> item-query cosine Top-200 inside that category pool
  -> context-centroid cosine Top-200 inside that category pool
  -> union + de-duplicate
  -> frozen V5 scorer reranking
  -> public Top-3
```

## Why V2 exists

V1 performed global cosine Top-200 before exact `master_category` filtering. This could spend most of the Top-200 budget on categories that cannot replace the problematic item. V2 applies the replacement-category constraint first so both retrieval channels spend their full search budget inside the eligible category.

## Two retrieval channels

### Item channel

```text
query = FashionCLIP embedding(problematic item)
```

Cosine similarity is computed only against eligible items in the same exact `master_category`, then Top-200 are retained.

### Context channel

```text
context = mean(FashionCLIP embeddings of all non-problematic outfit items)
context = L2_normalize(context)
```

Cosine similarity is computed between this context vector and the same category-constrained candidate pool, then Top-200 are retained.

The two Top-200 lists are unioned and de-duplicated. The union may contain fewer than 400 items because the channels can overlap or the category pool can contain fewer than 200 eligible items.

## Reranking

Every hybrid candidate replaces the problematic position in the complete outfit. The frozen scorer computes `compatibility_logit` for each candidate outfit. Final ranking is descending `compatibility_logit`; public output is Top-3.

`improvement_logit = candidate_logit - baseline_logit` remains internal metadata. Since all candidates share the same baseline, sorting by candidate logit and sorting by improvement give the same order.

Frozen scorer checkpoint SHA-256:

```text
7b3d0b6e0d44e3de517565f5725ded198bbc762b02a4dece26a58ee145cfed9c
```

## Runtime versus evaluation

Runtime can use the existing LOO diagnosis path:

```text
outfit -> LOO -> problematic index -> Recommendation V2 -> Top-3
```

The current offline component evaluation does not evaluate LOO. It uses the known synthetic `swapped_item_index` so recommendation retrieval/reranking can be measured independently.

## Offline benchmark terminology

The evaluator is a Polyvore synthetic one-item-swap recovery benchmark, not the human-annotated EVALUATION3 A-Test2000 protocol.

For each negative query:

```text
problematic position = negative_metadata.swapped_item_index
ground truth item    = negative_metadata.original_item_id
current swapped item = negative_metadata.replacement_item_id
```

Ground truth therefore means the original item that was removed when the synthetic negative outfit was created. It is not assumed to be the only valid replacement.

## Metrics retained for time-constrained V2 comparison

V2 keeps the existing metrics so V1 and V2 can be compared directly:

- item-only Recall@50 / Recall@100 / Recall@200;
- context-only Recall@50 / Recall@100 / Recall@200;
- hybrid Recall@50 / Recall@100 / Recall@200;
- final Hit@1 / Hit@3 / MRR;
- Replacement Success Rate with fixed epsilon `0.0`.

Interpretation limitation:

- Recall@K measures whether the original swapped-out item is recovered by retrieval;
- Hit@1 / Hit@3 / MRR measure exact original-item recovery after scorer reranking;
- they do not directly measure human preference or prove that other higher-ranked replacements are bad;
- Replacement Success Rate is scorer self-consistency, not an independent human-quality metric.

For the project report, retrieval Recall@200 should be emphasized as the main diagnostic for whether V2 fixes the V1 candidate-generation bottleneck. Hit@1/Hit@3/MRR remain useful exact-recovery diagnostics but should not be called recommendation accuracy.

## Frozen V2 config

```text
recommendation_version = category-aware-hybrid-v2
retrieval_scope        = exact_master_category_before_cosine
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

## Evaluation order

Because V1 test results have already been viewed, V2 should be treated as a new development iteration. Preferred process under the remaining time budget:

```text
1. run V2 on validation and verify retrieval behavior
2. freeze V2 implementation/config
3. run the existing one-item-swap test metrics once for the V2 comparison
4. report V1 vs V2 transparently
```

Do not describe the V2 test run as an untouched first-look test selected independently of V1 results.
