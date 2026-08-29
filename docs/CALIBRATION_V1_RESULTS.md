# Calibration V1 — frozen V5 scorer

## Status

Calibration V1 is a post-hoc, validation-only mapping for the frozen canonical scorer. It does not modify scorer weights or checkpoint metadata, and the test split was not loaded.

The Data Contract requires product-facing `compatibility_score` to be produced by a versioned calibration step rather than interpreting raw logits as percentages. Calibration V1 implements that boundary as:

```text
compatibility_logit
    -> sigmoid(scale * logit + bias)
    -> calibrated compatibility in [0, 1]
    -> rounded compatibility_score in [0, 100]
```

The scale is constrained to be strictly positive, so calibration preserves scorer ranking.

## Frozen inputs

- scorer version: `type_aware_pairwise_v1`
- checkpoint: `artifacts/checkpoints/type_aware_pairwise_v1/final_val_auc_v5_seed42/best.pt`
- checkpoint SHA-256: `7b3d0b6e0d44e3de517565f5725ded198bbc762b02a4dece26a58ee145cfed9c`
- checkpoint epoch: `52`
- checkpoint best validation ROC-AUC: `0.6905082489625538`
- dataset version: `polyvore1000-core7-compat-v2`
- embedding version: `fashionclip-512-l2-v1`
- category mapping version: `core7-v2`
- validation samples: `2,284` = 1,142 positive + 1,142 paired negative
- test split: **not loaded**

## Calibration method

Version: `platt-logistic-v1`

Method: positive-scale Platt calibration.

The final artifact was fit on all 2,284 frozen validation rows after scorer/model selection had already been locked.

Final parameters:

```text
scale = 0.47217959118640485
bias  = -0.17733224823027438
```

Artifact:

`artifacts/calibration/type_aware_pairwise_v1/platt_logistic_v1.json`

## Calibration metrics on the full fit set

These values describe calibration behavior on the same frozen validation rows used for the final fit. They are provenance/debug metrics, not a held-out generalization claim.

| Mapping | NLL | Brier | ECE-10 | Mean probability | Accuracy @ 0.5 |
|---|---:|---:|---:|---:|---:|
| Raw `sigmoid(logit)` | 0.697292 | 0.238633 | 0.114192 | 0.567220 | 0.633100 |
| Calibration V1 | 0.634974 | 0.222356 | 0.010863 | 0.500000 | 0.637040 |

## Deterministic holdout sanity check

Before refitting the final artifact on all validation rows, a calibration-only sanity check split the 1,142 paired families deterministically by SHA-256 of the positive sample ID. Families whose hash bucket was `0 mod 5` were held out; both members of a positive/negative family always stayed together.

- calibration-fit rows: `1,834`
- calibration-holdout rows: `450`
- holdout was never used by the fitter
- test split was not used

Fit-only parameters:

```text
scale = 0.4863434021173121
bias  = -0.16981528424097045
```

Holdout metrics:

| Mapping | NLL | Brier | ECE-10 | Mean probability | Accuracy @ 0.5 |
|---|---:|---:|---:|---:|---:|
| Raw `sigmoid(logit)` | 0.729318 | 0.246703 | 0.140872 | 0.584704 | 0.617778 |
| Fit-only Platt mapping | 0.646594 | 0.227000 | 0.048283 | 0.514690 | 0.635556 |

This sanity check supports using the monotonic Platt mapping as Calibration V1. The project currently defines no hard calibration acceptance threshold, so the numbers are recorded rather than converted into an arbitrary PASS/FAIL quality threshold.

## Product interpretation

A returned score such as `78` means only that the versioned calibration layer mapped the frozen scorer output to `78/100` compatibility score. It must not be described as “78% objectively fashionable” or as a probability that human stylists will approve the outfit.
