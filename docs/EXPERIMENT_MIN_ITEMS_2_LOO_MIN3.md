# Experiment — Scorer min 2 items, LOO min 3 original items

**Branch:** `exp/min-items-2-loo-min3`

## Goal

Test whether the existing Type-aware Pairwise compatibility scorer can safely
extend its input domain from 3–8 items to 2–8 items. This allows Leave-One-Out
(LOO) diagnosis on a 3-item outfit, because each LOO residual contains 2 items
and remains scoreable.

The experiment does **not** redefine LOO diagnosis for original 2-item outfits.
With only two original items, removing either side leaves a single item, which
has no pairwise compatibility relation for the current scorer and makes
problematic-item localization ambiguous.

## Experiment contract

```text
Compatibility scorer:
    minimum original/input outfit = 2 items
    maximum = 8 items

LOO diagnosis:
    minimum original outfit = 3 items

LOO residual:
    n original items -> n-1 items scored by compatibility scorer
    n=3 -> residual size=2 -> valid
    n=2 -> diagnosis rejected
```

## Data-processing changes

The branch changes the Core-7 positive-cleaning default from 3 to 2 items.
`prepare_core7_dataset_v2.py` imports the base cleaning implementation, so a
fresh branch process uses the new two-item default.

The raw compatibility builder already allowed `DEFAULT_MIN_ITEMS_PER_KIT = 2`.
The important filter is the Core-7 post-DROP recount: an outfit is retained when
at least two Core-7 items remain.

Embedding validation still uses the same frozen FashionCLIP contract. If any
newly retained item is missing/invalid in the embedding cache, regenerate or
extend the cache before scorer training. When a repair step is needed, use
`repair_split_min2` from `src.data.validate_core7_embeddings_min2` so an outfit
is retained whenever at least two valid items remain.

## Scorer-ready dataset build

The canonical frozen V2 builder is intentionally not overwritten by this
experiment. Build the experimental scorer-ready artifacts with:

```python
from src.data.build_core7_scorer_dataset_min2 import build_scorer_dataset_min2

result = build_scorer_dataset_min2(
    data_dir=...,
    output_dir=...,
    embedding_report_path=...,
    category_mapping_path=...,
    embedding_cache_path=...,
    embedding_manifest_path=...,
    repo_root=...,
)
```

The branch's example runtime paths are isolated from canonical V2:

```text
core7_dir        = data/core7_drop_v2_min2_exp
scorer_ready_dir = data/scorer_ready_v2_min2_exp
```

Do not point this experiment at the frozen canonical `core7_drop_v2` or
`scorer_ready_v2` directories.

## Compatibility scorer

Both scorer YAML profiles use:

```yaml
data:
  min_items: 2
  max_items: 8
```

`src/scorer/dataset.py` accepts and collates 2-item samples. Their pair mask has
exactly one valid unordered pair. `src/scorer/train.py` also validates
`data.min_items = 2`, so the experiment cannot silently fall back to the old
3-item training boundary.

The scorer architecture itself does not need a structural change: for two real
items the existing Type-aware Pairwise scorer computes one valid pair, runs the
Pair MLP, mean-aggregates that single score, and returns one
`compatibility_logit`.

## LOO diagnosis

Use `src.diagnosis.loo.build_loo_subsets`.

```python
from src.diagnosis.loo import build_loo_subsets

subsets = build_loo_subsets(items)
```

The helper rejects original outfits with fewer than 3 items. A 3-item outfit
produces three 2-item residuals, all valid scorer inputs.

## Evaluation protocol for this experiment

Report scorer metrics overall **and by outfit length**:

```text
ROC-AUC: n=2, n=3, n>=4, overall
2-way FITB: n=2, n=3, n>=4, overall
```

For a fair comparison against the current baseline, also compare both models on
an identical shared subset such as `n>=3` (or `n>=4` when comparing old LOO
coverage). Otherwise an overall metric change can be caused by a changed length
distribution rather than a better/worse scorer.

Report diagnosis only for original outfits `n>=3`:

```text
LOO Top-1 Localization Accuracy: n=3, n=4, n>=5, overall n>=3
LOO Hit@2: same buckets
```

Do not include original 2-item outfits in LOO localization metrics.

## Merge criterion

Treat this branch as an ablation until the following are demonstrated:

1. two-item scorer path is finite and stable;
2. scorer performance on the shared `n>=3` subset does not regress materially;
3. LOO on original 3-item outfits works with 2-item residuals;
4. canonical V2 artifacts remain unchanged;
5. experimental artifacts and checkpoints record the branch/commit and artifact hashes.
