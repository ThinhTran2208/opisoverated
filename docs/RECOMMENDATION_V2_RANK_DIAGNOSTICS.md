# Recommendation V2 — Reranking Rank Diagnostics

This branch keeps Recommendation V2 retrieval, the frozen FashionCLIP embeddings, and the frozen V5 compatibility scorer unchanged. It changes only the offline reranking evaluation protocol.

## Why replace headline Hit@K

The Polyvore one-item-swap benchmark supplies a single exact reference: `negative_metadata.original_item_id`. That item is the original item removed to create the synthetic negative outfit. It is not guaranteed to be the only valid replacement.

Therefore exact Hit@1/Hit@3 can under-credit plausible alternatives. This branch keeps exact-reference evaluation diagnostic, but measures whether the scorer improves the reference item's rank when the reference is actually present in the candidate set.

## Retrieval metrics

Primary candidate-generation diagnostics remain:

- item-only Recall@50 / @100 / @200;
- context-only Recall@50 / @100 / @200;
- hybrid Top-200 Recall@50 / @100 / @200;
- Full Candidate-Union GT Coverage.

Full Candidate-Union GT Coverage is:

```text
# valid queries where original_item_id is anywhere in the complete union
-----------------------------------------------------------------------
# valid queries
```

The complete union is the de-duplicated union of the item-query Top-200 and context-query Top-200 that is actually passed to the frozen scorer.

## Conditional reranking diagnostics

Reranking diagnostics are computed only on queries where the exact GT is present in the complete candidate union.

Before-rerank GT rank is the deterministic union order produced by `retrieval.py`:

```text
sort by best available channel rank
then item_id as deterministic tie-break
```

After-rerank GT rank is the rank produced by the frozen compatibility scorer over exactly the same candidate set.

For each eligible query:

```text
rank_change = pre_rerank_rank - post_rerank_rank
```

Interpretation:

- positive: scorer moved the exact reference upward;
- zero: rank unchanged;
- negative: scorer moved the exact reference downward.

Reported summaries:

- GT rank improved / unchanged / worsened rates;
- mean rank change;
- median rank change;
- Conditional MRR before scorer;
- Conditional MRR after scorer;
- Conditional MRR gain.

Conditional MRR uses the same eligible query set before and after reranking, so retrieval failures are not counted against the scorer.

## Limitation

These metrics remain exact-reference diagnostics. They do not measure human recommendation quality because a candidate ranked above the original swapped-out item may still be a valid or better replacement.

Replacement Success Rate remains a scorer self-consistency diagnostic only, because the scorer both selects and evaluates the candidates.
