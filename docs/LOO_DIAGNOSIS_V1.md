# LOO Diagnosis V1 — validation experiment

## Status

This is a downstream, validation-only experiment on frozen V5 `best.pt`.
It does not change the canonical scorer training contract.

Canonical invariants that remain unchanged:

- scorer/config/checkpoint `min_items = 3`;
- train and validation collators accept only 3–8 item outfits;
- BCE-only, shuffled batches, FP32, and checkpoint selection remain untouched;
- no checkpoint parameters or metadata are rewritten;
- the test split is not loaded.

## Why a separate inference path is needed

For a three-item outfit, Leave-One-Out produces three subsets with two items.
The pairwise model can compute one valid item pair, but frozen V5 was trained
only on outfits with at least three items. A normal scorer forward therefore
continues to reject the subset.

The experimental path accepts two items only when all of these conditions hold:

1. the model is in `eval()` mode;
2. the caller explicitly passes `diagnostic_min_items=2`;
3. the loaded model still has canonical `min_items=3`.

Training with the override hard-fails. A normal inference call without the
override also hard-fails on two-item outfits.

## LOO protocol

For each validation negative outfit `O`:

1. score the complete outfit;
2. construct every `O \ x_i` subset;
3. score the complete outfit and all subsets in one batched forward pass;
4. compute `delta_i = score(O \ x_i) - score(O)`;
5. rank item indices by descending delta;
6. compare Top-1 and Top-2 with `negative_metadata.swapped_item_index`.

The target index is used only after neural scores have been produced.

## Reporting rule

Always report:

- overall LOO Top-1 Localization Accuracy;
- overall Hit@2;
- a separate original-size-3 extrapolation summary;
- a separate original-size-4+ summary whose removed subsets remain canonical
  3+ item inputs;
- the same metrics grouped by original outfit size;
- count of original three-item outfits that required two-item extrapolation.

Do not merge the size-3 result into a claimed in-distribution metric without
showing the grouped table. The `uses_two_item_extrapolation` field is present
in every per-sample record.

## Run

Open and run:

`notebooks/experiments/NB7_loo_diagnosis_v1.ipynb`

The notebook:

- checks out branch `feat/diagnosis-loo-v1`;
- runs LOO and scorer regression tests;
- loads only the validation split;
- loads the canonical frozen V5 `best.pt`;
- runs one smoke example;
- evaluates all 1,142 validation negatives;
- saves a detailed JSON report under
  `ML_Final/diagnosis_runs/loo_diagnostic_v1_v5_seed42/`.

## Interpretation

This branch answers whether frozen V5 is useful for LOO without retraining.
It does not prove that two-item scoring is in-distribution.

If size-3 localization is unstable while size 4+ is useful, keep size 4+ as the
clean frozen-V5 LOO result and treat size 3 as an experimental coverage result.
A true canonical two-item solution would require a separately versioned
model/data change and retraining; changing the existing YAML from 3 to 2 is not
a valid migration for the current checkpoint.
