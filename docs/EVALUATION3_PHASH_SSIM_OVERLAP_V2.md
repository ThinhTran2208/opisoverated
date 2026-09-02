# EVALUATION3 pHash + SSIM overlap audit v2

This path is separate from the original dHash-based NB10 audit and does not use outfit/item ID collisions as overlap evidence.

## Frozen three-state protocol

Before manual review, every EVALUATION3 ↔ Polyvore image pair is assigned exactly one state:

- `DUPLICATE`: `pHash distance <= 4` and `SSIM >= 0.92`.
- `MANUAL_REVIEW`: `pHash distance <= 4` and `0.90 <= SSIM < 0.92`.
- `NON_DUPLICATE`: everything else.

Manual-review rows are resolved to exactly one final label: `DUPLICATE` or `NON_DUPLICATE`.

No `UNCERTAIN`, `SKIP`, or `SAME_PRODUCT_DIFFERENT_IMAGE` class is part of the final cleaning contract. Broken/missing files remain a data-completeness error, not a duplicate class.

## Two notebook variants

### NB10A — pHash + SSIM only

`NB10A_evaluation3_phash_ssim_only.ipynb` is the canonical candidate that follows the calibrated pHash + SSIM rule only.

### NB10B — pHash + SSIM + exact-pixel ablation

`NB10B_evaluation3_phash_ssim_exact_pixel.ipynb` adds a normalized decoded-pixel SHA256 shortcut:

- exact normalized pixel match -> `DUPLICATE` immediately;
- otherwise use the same pHash + SSIM rule as NB10A.

Exact-pixel matches are exported separately so the team can decide whether the shortcut belongs in the final evaluation protocol. This variant is an ablation, not a replacement for NB10A.

Because identical decoded pixels should also have pHash distance 0 and SSIM 1.0, the exact-pixel shortcut is expected mainly to change evidence/runtime, not final classification, unless an implementation/data edge case exists.

## Image implementation

The implementation records these details in every summary artifact:

- pHash: 64-bit DCT pHash, `hash_size=8`, `highfreq_factor=4`.
- SSIM preprocessing: EXIF transpose -> grayscale -> 256x256 LANCZOS resize.
- SSIM: structural similarity with `data_range=255`.

The thresholds `4 / 0.90 / 0.92` were calibrated outside this module. They should only be called official after confirming that the calibration run used the same pHash and SSIM implementation/preprocessing.

## Split semantics

The audit uses every unique Polyvore item actually present in the scorer-ready JSONL splits.

- `model_clean`: excludes confirmed duplicate overlap with model-development splits (`train`, `valid` by default).
- `strict_clean`: excludes confirmed duplicate overlap with all supplied Polyvore splits, including `test`.

A test-only duplicate can therefore remain in `model_clean` but cannot remain in `strict_clean`.

## Manual review workflow

The prepare pass writes:

- `evaluation3_auto_duplicates.csv`
- `evaluation3_manual_review_BLIND.csv`
- `evaluation3_manual_review_BLIND.xlsx`
- `evaluation3_manual_review_KEY.csv`
- `evaluation3_manual_review_BLIND.html`
- `manual_review_previews/`
- `evaluation3_overlap_audit_pre_review.jsonl`
- `evaluation3_full.jsonl`

The blind workbook only exposes `pair_id`, `preview_file`, and `human_label`. The allowed final labels are `DUPLICATE` and `NON_DUPLICATE`.

The finalize pass will not create official clean manifests until every manual pair is resolved and the Polyvore/EVALUATION3 image inputs are complete. Once ready, it writes:

- `evaluation3_overlap_summary.json`
- `evaluation3_overlap_audit.jsonl`
- `evaluation3_model_clean.jsonl`
- `evaluation3_strict_clean.jsonl`

## Drive isolation

Both notebooks write outside the teammate's calibration folder:

`MyDrive/evaluation3_overlap_phash_ssim_v2/`

with separate subfolders:

- `phash_ssim_only/`
- `with_exact_pixel/`
