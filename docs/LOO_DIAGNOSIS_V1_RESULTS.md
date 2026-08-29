# LOO Diagnosis V1 — validation results

## Run provenance

This result was produced by `notebooks/experiments/NB7_loo_diagnosis_v1.ipynb` on the frozen V5 scorer.

- protocol: `loo-diagnostic-v1`
- split: `valid` only
- git head used for the run: `35122750c8aacb0d7782e6120a56c6755a7431ce`
- checkpoint: `artifacts/checkpoints/type_aware_pairwise_v1/final_val_auc_v5_seed42/best.pt`
- checkpoint epoch: `52`
- checkpoint best validation ROC-AUC: `0.6905082489625538`
- canonical scorer `min_items`: `3`
- validation samples loaded: `2,284`
- paired families / validation negatives evaluated: `1,142`
- test split: **not loaded**

The notebook completed without cell errors. The LOO test suite passed 5/5 tests and the scorer regression suite passed 19/19 tests.

The detailed per-sample report is stored outside Git in:

`ML_Final/diagnosis_runs/loo_diagnostic_v1_v5_seed42/validation_loo_report.json`

The detailed report has 1,142 per-negative records and is intentionally kept as a generated Drive artifact rather than committed to the repository.

## Primary diagnosis result

| Scope | Samples | LOO Top-1 Localization Accuracy | Hit@2 | Two-item extrapolation |
|---|---:|---:|---:|---:|
| Overall | 1,142 | 0.5350 | 0.7688 | 358 |
| Original size = 3 | 358 | 0.5922 | 0.8659 | 358 |
| Original size >= 4 | 784 | 0.5089 | 0.7245 | 0 |

Equivalent counts for the overall result:

- Top-1 correct: `611 / 1,142`
- Hit@2: `878 / 1,142`
- Top-1 incorrect: `531 / 1,142`

The size-3 rows must be interpreted as **two-item extrapolation**, because removing one item from a three-item outfit creates a two-item subset while frozen V5 was trained with `min_items = 3`.

The clean in-distribution frozen-V5 diagnosis result is therefore the **original size >= 4** row:

- LOO Top-1 Localization Accuracy: `0.5089285714`
- Hit@2: `0.7244897959`
- samples: `784`

## Result by original outfit size

| Original items | Samples | Top-1 | Hit@2 | Mean target delta | Mean max delta |
|---:|---:|---:|---:|---:|---:|
| 3 | 358 | 0.5922 | 0.8659 | 1.5555 | 2.4323 |
| 4 | 460 | 0.5457 | 0.7717 | 1.0558 | 1.6786 |
| 5 | 264 | 0.4508 | 0.6477 | 0.6741 | 1.2242 |
| 6 | 42 | 0.4524 | 0.6667 | 0.5941 | 1.0019 |
| 7 | 10 | 0.6000 | 0.8000 | 0.7279 | 0.9486 |
| 8 | 8 | 0.5000 | 0.7500 | 0.8172 | 1.0257 |

The size-7 and size-8 groups are very small and should not be over-interpreted independently.

## Validation checks performed

The saved JSON report was checked for internal consistency:

- `len(records) == 1,142`;
- group sample counts sum to 1,142;
- size-3 count is exactly 358;
- size-4+ count is exactly 784;
- all 358 size-3 records are marked `uses_two_item_extrapolation = true`;
- no size-4+ record uses two-item extrapolation;
- reported Top-1 and Hit@2 values exactly match recomputation from per-sample records;
- target and predicted indices are within the corresponding outfit bounds;
- each record contains exactly one LOO delta per original item.

## Interpretation

The run confirms that the LOO implementation executes correctly on the frozen V5 checkpoint and produces measurable localization signal without retraining.

For reporting and model decisions, keep the two scopes distinct:

- **size >= 4:** canonical/in-distribution LOO evaluation for frozen V5;
- **size = 3:** experimental coverage result using the explicit eval-only two-item extrapolation path.

No acceptance threshold for diagnosis is defined by the current project metrics contract, so these values should be reported as measured validation performance rather than converted into a pass/fail model-quality claim.
