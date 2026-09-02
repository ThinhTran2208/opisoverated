# EVALUATION3 guarded-ECC consensus verifier experiment

Branch: `feat/evaluation3-ecc-neardup-verifier`

Base: `feat/evaluation3-rgb-neardup-verifier`

## Why another experiment?

NB10C showed that simply replacing grayscale SSIM with RGB SSIM did not remove the dominant failure mode: many visually same-image / near-identical pairs still landed in manual review. Color information is useful, but it is not the main nuisance when the same catalog image has been resized, re-encoded, shifted, or slightly brightness-adjusted.

The new hypothesis is therefore **registration + multiple independent vetoes**, not another single similarity threshold.

## Pipeline

```text
E3 image
  |
  +-- grayscale pHash --> BK-tree global retrieval (scorer-seen Polyvore)
                              |
                              +-- candidates with pHash Hamming <= 4
                                           |
                                           v
                              RGB foreground normalization
                              (crop white bg + preserve aspect)
                                           |
                                           v
                              guarded Euclidean ECC alignment
                              (translation + rotation only)
                                           |
                                           v
                              multi-evidence verification
                              - ECC correlation
                              - RGB SSIM
                              - grayscale SSIM
                              - foreground IoU
                              - edge SSIM
                              - mean Lab color delta
                              - interior MAE
                              - worst interior patch MAE
                                           |
                              DUP / MANUAL / NON
```

## Why Euclidean ECC, not affine/homography?

A duplicate verifier should tolerate small registration nuisance but should not be able to *warp a different garment into looking the same*. Therefore this experiment deliberately allows only translation + rotation after foreground normalization. Scale is already mostly normalized by the foreground crop + aspect-preserving fit. Non-uniform scale, shear and projective transforms are not allowed.

## Why the patch residual?

Global SSIM/MAE can hide a small meaningful difference such as a pocket, buckle, logo or waistband because the changed region is a small fraction of the full garment. The verifier therefore measures RGB residual on the eroded foreground interior in 32x32 patches. A single strongly changed patch can veto auto-DUP even when global SSIM is high.

This also reduces the influence of silhouette-boundary interpolation noise caused by resizing/alignment.

## Trial decision rule

NB10D uses a nuisance-tolerant trial configuration:

- `auto_rgb_ssim_min = 0.82`
- `auto_edge_ssim_min = 0.72`

These two global metrics are intentionally not extremely strict because JPEG/re-encoding can depress them. Auto-DUP still requires ALL of the following default vetoes to pass:

- ECC correlation >= 0.90
- foreground IoU >= 0.90
- mean Lab delta <= 8
- interior MAE <= 0.05
- worst interior patch MAE <= 0.12

The values are **EXPERIMENTAL_UNCALIBRATED**. They exist to create a useful first distribution, not to be declared canonical.

## Manual queue behavior improvements

The branch also fixes two workflow problems from earlier experiments:

1. `manual_review_previews/` is cleared at the start of prepare, so stale `PAIRxxxx.jpg` from a previous run cannot mix with the current run.
2. Manual candidates are emitted only when the outfit has no automatic duplicate. If any item in an outfit is already confirmed auto-DUP, the outfit is contaminated and redundant manual previews are suppressed.

## Calibration/debug output

`evaluation3_candidate_evidence.csv` contains evidence for every pHash candidate, including auto-DUP, manual and non-duplicate candidate decisions. This is the file to use for real threshold calibration after human labels are available.

The audit also emits outfit-level pre-review manifests for `DUPLICATE`, `MANUAL_REVIEW`, and `NON_DUPLICATE`.

## What to look for in the first real run

The experiment is useful only if the manual queue qualitatively changes:

- same-image JPEG/resize/placement variants should move from MANUAL toward DUPLICATE;
- same-shape but clearly different-color items should NOT auto-DUP;
- small meaningful local detail changes should remain MANUAL rather than being erased by registration;
- manual pair count should drop, especially for outfits already containing an auto-DUP item.

If these do not improve on real data, the next candidates are image-copy-specific learned/perceptual features (e.g. LPIPS/DINO) or keypoint/geometric matching as additional evidence, not further blind tuning of a single SSIM threshold.
