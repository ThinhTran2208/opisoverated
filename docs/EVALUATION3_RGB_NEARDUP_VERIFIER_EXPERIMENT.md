# EVALUATION3 RGB near-duplicate verifier experiment

Branch: `feat/evaluation3-rgb-neardup-verifier`

Base branch: `feat/evaluation3-phash-ssim-overlap-v2`

## Why this experiment exists

Manual review from grayscale SSIM contained three qualitatively different cases:

1. Same image / same garment with tiny nuisance changes (resize, placement, JPEG, brightness) that should usually be DUPLICATE.
2. Very similar shape but different color, which should not become an automatic duplicate only because grayscale removes chroma.
3. Truly ambiguous near-matches with small but meaningful garment-detail differences, which are appropriate for MANUAL_REVIEW.

The old pipeline used grayscale twice: pHash retrieval and grayscale SSIM verification. This experiment keeps grayscale only where it is useful and cheap: candidate retrieval.

## Experimental pipeline

```text
E3 RGB source image
        |
        +--> grayscale pHash --> BK-tree search over scorer-seen Polyvore items
                                   |
                                   +--> candidates with Hamming distance <= 4
                                                |
                                                v
                                     reopen original RGB pair
                                                |
                                     foreground bbox on white bg
                                                |
                                     aspect-preserving resize
                                     + centered white canvas
                                                |
                                                v
                                             RGB SSIM
                                                |
                              DUPLICATE / MANUAL_REVIEW / NON_DUPLICATE
```

pHash is intentionally still grayscale because it is only a retrieval gate. The final verifier sees RGB.

## What changed

- Added `src/evaluation/evaluation3_rgb_verifier.py`.
- Added foreground normalization for catalog-style white-background images.
- Added RGB SSIM using `channel_axis=2`.
- Added `pair_evidence()` for future calibration/debugging:
  - RGB SSIM
  - grayscale SSIM
  - mean foreground Lab color distance
  - simple edge similarity
- Added `NB10C_evaluation3_rgb_aligned_verifier.ipynb`.
- Added synthetic tests for scale/position normalization and color sensitivity.

## What did NOT change

- pHash recipe and BK-tree candidate retrieval remain inherited from the v2 branch.
- Scorer-seen train/valid/test identity logic remains unchanged.
- Exact-pixel, ID collision and dHash are not used in NB10C.
- Final model-clean vs strict-clean semantics remain inherited from the v2 branch.

## Threshold status

RGB verification changes the score distribution. Therefore old grayscale thresholds (`0.92/0.90`, `0.90/0.88`, `0.85/0.83`, etc.) must not be treated as calibrated RGB thresholds.

NB10C starts conservatively with:

- `RGB SSIM >= 0.95` -> automatic duplicate
- `0.85 <= RGB SSIM < 0.95` -> manual review
- lower scores -> non-duplicate candidate result

These are **trial values only**. The intended next step is to human-label representative pairs and compare score distributions before freezing any threshold.

## Important caveats

1. Foreground detection assumes a white or near-white catalog background. It falls back to the full image when too little foreground is detected, but unusual backgrounds still need inspection.
2. Foreground crop + aspect-preserving fit handles scale and placement nuisance, not arbitrary geometric warps.
3. RGB SSIM still cannot understand semantic garment details. The extra color/edge evidence is currently exposed for analysis but not yet part of the automatic decision rule.
4. If meaningful DUPLICATE/NON_DUPLICATE overlap remains after RGB normalization, the next candidates to test are ORB/SIFT-style geometric matching or a learned perceptual metric (LPIPS/DINO) as additional evidence. FashionCLIP can be an auxiliary semantic-similarity feature, but should not be the sole duplicate criterion.
