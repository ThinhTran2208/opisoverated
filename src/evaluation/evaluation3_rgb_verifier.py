# -*- coding: utf-8 -*-
"""Experimental RGB near-duplicate verifier for EVALUATION3.

This branch deliberately keeps the existing grayscale pHash retrieval stage and
replaces only the SSIM verification representation:

1. pHash remains grayscale and is used only to retrieve a small candidate set.
2. Candidate images are re-opened in RGB.
3. A simple white-background foreground box is estimated independently on both
   images.
4. The garment crop is resized with aspect ratio preserved and centered on a
   white square canvas.
5. RGB SSIM is computed on the normalized pair.

The goal is to remove nuisance differences such as object scale / placement
while preserving color information that grayscale SSIM throws away.

IMPORTANT: thresholds from the old grayscale-SSIM calibration are NOT valid for
this verifier. Notebook defaults are intentionally marked as exploratory and
must be recalibrated from human-labeled pairs before becoming canonical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import sobel
from skimage.metrics import structural_similarity

from src.evaluation import evaluation3_phash_ssim as legacy


DEFAULT_RGB_VERIFIER_SIZE = 256
DEFAULT_BACKGROUND_DELTA = 14
DEFAULT_FOREGROUND_MARGIN_RATIO = 0.06
DEFAULT_MIN_FOREGROUND_FRACTION = 0.001


def _foreground_mask(
    array: np.ndarray,
    *,
    background_delta: int = DEFAULT_BACKGROUND_DELTA,
) -> np.ndarray:
    """Return a conservative non-white foreground mask for an RGB uint8 image."""

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB array, got shape={array.shape}")
    distance_from_white = np.max(
        255 - array.astype(np.int16),
        axis=2,
    )
    return distance_from_white >= int(background_delta)


def normalize_rgb_for_verification(
    image,
    *,
    size: int = DEFAULT_RGB_VERIFIER_SIZE,
    background_delta: int = DEFAULT_BACKGROUND_DELTA,
    margin_ratio: float = DEFAULT_FOREGROUND_MARGIN_RATIO,
    min_foreground_fraction: float = DEFAULT_MIN_FOREGROUND_FRACTION,
) -> np.ndarray:
    """Normalize one image for RGB near-duplicate verification.

    The normalization is intentionally simple and auditable. It is designed for
    catalog-style product images that are usually presented on a white or nearly
    white background. If foreground detection is unreliable, it falls back to
    the full image instead of inventing a crop.
    """

    Image, ImageOps = legacy._require_pillow()
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    raw = np.asarray(normalized, dtype=np.uint8)
    mask = _foreground_mask(raw, background_delta=background_delta)

    total_pixels = max(1, mask.size)
    foreground_fraction = float(mask.sum()) / float(total_pixels)

    crop = normalized
    if mask.any() and foreground_fraction >= float(min_foreground_fraction):
        ys, xs = np.nonzero(mask)
        left = int(xs.min())
        right = int(xs.max()) + 1
        top = int(ys.min())
        bottom = int(ys.max()) + 1

        box_width = max(1, right - left)
        box_height = max(1, bottom - top)
        margin = int(round(max(box_width, box_height) * float(margin_ratio)))

        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(normalized.width, right + margin)
        bottom = min(normalized.height, bottom + margin)
        crop = normalized.crop((left, top, right, bottom))

    target = int(size)
    if target <= 0:
        raise ValueError("size must be > 0")

    resampling = getattr(Image, "Resampling", Image)
    width, height = crop.size
    scale = min(target / max(1, width), target / max(1, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = crop.resize((new_width, new_height), resampling.LANCZOS)

    canvas = Image.new("RGB", (target, target), "white")
    x = (target - new_width) // 2
    y = (target - new_height) // 2
    canvas.paste(resized, (x, y))
    return np.asarray(canvas, dtype=np.uint8)


def rgb_ssim_score(left: np.ndarray, right: np.ndarray) -> float:
    """RGB SSIM for two normalized uint8 arrays."""

    if left.shape != right.shape:
        raise ValueError(
            f"Verifier arrays must have the same shape: {left.shape} vs {right.shape}"
        )
    if left.ndim != 3 or left.shape[2] != 3:
        raise ValueError(f"Expected RGB verifier arrays, got shape={left.shape}")
    return float(
        structural_similarity(
            left,
            right,
            data_range=255,
            channel_axis=2,
        )
    )


def pair_evidence(
    left: np.ndarray,
    right: np.ndarray,
    *,
    background_delta: int = DEFAULT_BACKGROUND_DELTA,
) -> dict[str, float]:
    """Return interpretable evidence for calibration/debugging.

    These extra metrics are intentionally not used by the current decision rule
    yet. They are emitted so the branch can be extended without guessing:

    - rgb_ssim: color-aware structural similarity.
    - grayscale_ssim: shape/luminance similarity for comparison with RGB.
    - mean_lab_delta: coarse color difference on estimated foreground pixels.
    - edge_similarity: simple edge-map agreement after normalization.
    """

    if left.shape != right.shape:
        raise ValueError(
            f"Verifier arrays must have the same shape: {left.shape} vs {right.shape}"
        )

    rgb_score = rgb_ssim_score(left, right)

    left_float = left.astype(np.float32) / 255.0
    right_float = right.astype(np.float32) / 255.0
    left_gray = rgb2gray(left_float)
    right_gray = rgb2gray(right_float)
    gray_score = float(
        structural_similarity(left_gray, right_gray, data_range=1.0)
    )

    left_edges = sobel(left_gray)
    right_edges = sobel(right_gray)
    denominator = float(np.mean(np.abs(left_edges) + np.abs(right_edges)))
    if denominator <= 1e-8:
        edge_similarity = 1.0
    else:
        edge_similarity = 1.0 - float(
            np.mean(np.abs(left_edges - right_edges)) / denominator
        )
        edge_similarity = float(np.clip(edge_similarity, 0.0, 1.0))

    left_mask = _foreground_mask(left, background_delta=background_delta)
    right_mask = _foreground_mask(right, background_delta=background_delta)
    left_lab = rgb2lab(left_float)
    right_lab = rgb2lab(right_float)

    def mean_lab(lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.any():
            return np.mean(lab[mask], axis=0)
        return np.mean(lab.reshape(-1, 3), axis=0)

    left_mean = mean_lab(left_lab, left_mask)
    right_mean = mean_lab(right_lab, right_mask)
    mean_lab_delta = float(np.linalg.norm(left_mean - right_mean))

    return {
        "rgb_ssim": rgb_score,
        "grayscale_ssim": gray_score,
        "mean_lab_delta": mean_lab_delta,
        "edge_similarity": edge_similarity,
    }


def prepare_overlap_audit_rgb(
    *,
    evaluation3_root: Path | str,
    development_split_paths: Mapping[str, Path | str],
    output_dir: Path | str,
    polyvore_image_root: Path | str | None = None,
    polyvore_hf_dataset: str | None = None,
    hf_split_mapping: Mapping[str, str] | None = None,
    annotations_path: Path | str | None = None,
    annotation_sheet: str = "CMT",
    metadata_path: Path | str | None = None,
    metadata_sheet: str = "Num",
    evaluation3_groups=None,
    model_development_splits=legacy.DEFAULT_MODEL_DEVELOPMENT_SPLITS,
    phash_threshold: int = 4,
    rgb_ssim_auto_threshold: float = 0.95,
    rgb_ssim_manual_lower_bound: float = 0.85,
    verifier_size: int = DEFAULT_RGB_VERIFIER_SIZE,
    background_delta: int = DEFAULT_BACKGROUND_DELTA,
    foreground_margin_ratio: float = DEFAULT_FOREGROUND_MARGIN_RATIO,
    allow_incomplete_image_index: bool = False,
):
    """Run the existing pHash audit with aligned RGB SSIM verification.

    The legacy implementation is reused for retrieval, indexing, manifests,
    manual-review outputs, and finalization semantics. Only the verification
    image representation and SSIM function are swapped during this call.

    This function is intentionally NOT re-entrant/thread-safe because it
    temporarily replaces two module-level verifier functions in the legacy
    implementation. Colab usage is sequential, which is the supported mode for
    this experimental branch.
    """

    if not 0 <= rgb_ssim_manual_lower_bound < rgb_ssim_auto_threshold <= 1:
        raise ValueError(
            "Require 0 <= RGB manual lower bound < RGB auto threshold <= 1"
        )

    original_image_to_ssim_array = legacy.image_to_ssim_array
    original_ssim_score = legacy.ssim_score

    def rgb_array_adapter(image, *, size=verifier_size):
        return normalize_rgb_for_verification(
            image,
            size=size,
            background_delta=background_delta,
            margin_ratio=foreground_margin_ratio,
        )

    try:
        legacy.image_to_ssim_array = rgb_array_adapter
        legacy.ssim_score = rgb_ssim_score

        summary, paths = legacy.prepare_overlap_audit(
            evaluation3_root=evaluation3_root,
            development_split_paths=development_split_paths,
            output_dir=output_dir,
            polyvore_image_root=polyvore_image_root,
            polyvore_hf_dataset=polyvore_hf_dataset,
            hf_split_mapping=hf_split_mapping,
            annotations_path=annotations_path,
            annotation_sheet=annotation_sheet,
            metadata_path=metadata_path,
            metadata_sheet=metadata_sheet,
            evaluation3_groups=evaluation3_groups,
            model_development_splits=model_development_splits,
            use_exact_pixel=False,
            phash_threshold=phash_threshold,
            ssim_auto_threshold=rgb_ssim_auto_threshold,
            ssim_manual_lower_bound=rgb_ssim_manual_lower_bound,
            ssim_size=verifier_size,
            allow_incomplete_image_index=allow_incomplete_image_index,
        )
    finally:
        legacy.image_to_ssim_array = original_image_to_ssim_array
        legacy.ssim_score = original_ssim_score

    configuration = summary.setdefault("configuration", {})
    configuration.update(
        {
            "protocol": "phash-rgb-aligned-ssim-v0-experimental",
            "verifier": "RGB SSIM after foreground crop + aspect-preserving fit",
            "verifier_preprocess": (
                "EXIF transpose -> RGB -> conservative non-white foreground bbox "
                f"(delta={background_delta}) -> margin={foreground_margin_ratio:.3f} "
                f"-> aspect-preserving fit on white {verifier_size}x{verifier_size} canvas"
            ),
            "rgb_ssim_auto_duplicate_threshold": rgb_ssim_auto_threshold,
            "rgb_ssim_manual_lower_bound": rgb_ssim_manual_lower_bound,
            "calibration_status": "EXPERIMENTAL_UNCALIBRATED",
            "calibration_warning": (
                "Do not reuse grayscale-SSIM thresholds as canonical RGB thresholds. "
                "Human-label representative pairs and recalibrate first."
            ),
        }
    )
    configuration.pop("ssim_preprocess", None)
    legacy.write_json(summary, paths["summary"])
    return summary, paths


finalize_overlap_audit = legacy.finalize_overlap_audit
