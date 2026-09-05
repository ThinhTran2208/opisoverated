# Recommendation V2 — Category-Aware Hybrid Retrieval

## Frozen status

Recommendation V2 is frozen for the final report on the canonical branch:

```text
feat/recommendation-rank-diagnostics-v2
```

The previous experimental branch name `feat/recommendation-conditional-hit-v2` is deprecated because Conditional Hit@3 is no longer part of the main reporting protocol.

The evaluation code used for the recorded VALID/TEST runs is commit:

```text
3472abf7a39ae9fc51683bcb19b5dcd3ac3e8ec4
```

Canonical frozen results are recorded in:

```text
artifacts/recommendation_v2_rankdiag_freeze.json
```

This freeze changes reporting/evaluation only. FashionCLIP, retrieval logic, frozen scorer, LOO integration and public Top-3 recommendation output are unchanged.

## Pipeline

```text
problematic item
  -> exact same master_category pool when available
     else Core-7 fallback at runtime
  -> exclude current outfit items
  -> require metadata + embedding + image availability
  -> item-query cosine Top-200
  -> context-centroid cosine Top-200
  -> union + de-duplicate
  -> frozen V5 scorer reranking of the full union
  -> public Top-3
```

V1 performed global cosine Top-200 before exact master-category filtering. V2 applies the replacement-category constraint before cosine retrieval so the Top-200 budget is spent inside an eligible replacement pool.

Frozen scorer checkpoint SHA-256:

```text
7b3d0b6e0d44e3de517565f5725ded198bbc762b02a4dece26a58ee145cfed9c
```

## Benchmark terminology

For each Polyvore synthetic negative query:

```text
problematic position = negative_metadata.swapped_item_index
ground truth item    = negative_metadata.original_item_id
current swapped item = negative_metadata.replacement_item_id
```

The GT is only the original item removed by the synthetic corruption process. It is not assumed to be the unique valid replacement. Therefore all recommendation metrics below are exact-reference diagnostics, not human recommendation accuracy.

## Frozen reporting policy

### Primary retrieval metrics

Report in the main table:

- Hybrid Recall@200;
- Full-union GT coverage.

Supporting channel diagnostics may also show Item-only Recall@200 and Context-only Recall@200.

Definitions:

```text
Hybrid Recall@200
= fraction of valid queries whose exact GT is in the first 200 candidates
  under the deterministic hybrid ordering.

Full-union GT coverage
= fraction of valid queries whose exact GT is anywhere in the complete
  de-duplicated candidate union actually passed to the frozen scorer.
```

### Secondary reranking metrics

Report in the main evaluation discussion:

- GT rank improved ratio;
- GT rank worsened ratio;
- median rank change (`pre_rank - post_rank`).

These are computed only on queries where GT exists in the full union, so retrieval failure is not attributed to the scorer.

```text
rank change = pre-rerank GT rank - post-rerank GT rank
```

Positive values mean the frozen scorer moved the exact reference upward.

### Diagnostic-only metrics

Keep in raw artifacts/code but do not headline in the main report:

- Replacement Success Rate;
- conditional MRR before/after/gain;
- old unconditional Hit@1 / Hit@3 / MRR;
- experimental Conditional Hit@3.

The MRR fields remain available for audit and deeper analysis but are intentionally omitted from the concise report table.

## Frozen VALID result

`N = 1,142`, excluded = 0.

### Retrieval

| Metric | VALID |
| --- | ---: |
| Item-only Recall@200 | 14.80% |
| Context-only Recall@200 | 36.87% |
| Hybrid Recall@200 | 31.79% |
| Full-union GT coverage | 40.89% |

### Frozen scorer reranking

Among the 467 queries where the exact GT is in the full union:

| Metric | VALID |
| --- | ---: |
| GT rank improved | 60.39% |
| GT rank unchanged | 1.50% |
| GT rank worsened | 38.12% |
| Median rank change | +16 positions |

Replacement Success Rate = 99.36% (diagnostic only).

## Frozen TEST result

`N = 2,327`, excluded = 0.

### Retrieval

| Metric | TEST |
| --- | ---: |
| Item-only Recall@200 | 17.58% |
| Context-only Recall@200 | 39.62% |
| Hybrid Recall@200 | 33.95% |
| Full-union GT coverage | 44.13% |

### Frozen scorer reranking

Among the 1,027 queries where the exact GT is in the full union:

| Metric | TEST |
| --- | ---: |
| GT rank improved | 58.71% |
| GT rank unchanged | 1.85% |
| GT rank worsened | 39.44% |
| Median rank change | +13 positions |

Replacement Success Rate = 99.28% (diagnostic only).

## Interpretation for the report

Recommended concise interpretation:

> Category-aware V2 substantially improves exact-reference candidate recovery compared with V1. On the frozen TEST split, Hybrid Recall@200 reaches 33.95% and the full candidate union contains the exact reference in 44.13% of queries. Conditional on the reference being available to the reranker, the frozen scorer moves it upward in 58.71% of cases, with a median improvement of 13 positions. These results are exact-reference diagnostics on a synthetic one-swap benchmark and are not human recommendation accuracy.

Do not describe Hit@K, rank diagnostics, Replacement Success or any other exact-reference metric as recommendation accuracy.

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
evaluation_protocol    = polyvore-one-item-swap-recovery-v2-rank-diagnostics
```

Config file:

```text
configs/recommendation_category_aware_v2.json
```
