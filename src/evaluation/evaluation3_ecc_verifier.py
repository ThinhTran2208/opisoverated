# -*- coding: utf-8 -*-
"""Experimental EVALUATION3 near-duplicate verifier using guarded ECC alignment.

Design goals:

1. Keep the existing grayscale pHash + BK-tree only as a cheap global candidate
   retrieval stage over scorer-seen Polyvore items.
2. Re-open candidate pairs in RGB and normalize white-background catalog images
   by foreground crop + aspect-preserving fit.
3. Apply only a SMALL Euclidean ECC registration (translation + rotation). We do
   deliberately NOT use affine/homography alignment because a flexible warp can
   morph genuinely different garments into looking artificially similar.
4. Decide with several pieces of evidence instead of one SSIM threshold:
   RGB SSIM, grayscale SSIM, foreground IoU, edge SSIM, Lab color distance,
   interior residual MAE, and the worst interior patch residual.
5. Keep ambiguous pairs for human review. Auto-DUPLICATE is intentionally a
   consensus rule and therefore conservative.

This is an experiment, not a frozen calibration. Thresholds MUST be validated on
human-labelled real EVALUATION3/Polyvore pairs before becoming canonical.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import cv2
import numpy as np
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import sobel
from skimage.metrics import structural_similarity

from src.evaluation import evaluation3_phash_ssim as legacy
from src.evaluation.evaluation3_rgb_verifier import normalize_rgb_for_verification


DUPLICATE = legacy.DUPLICATE
MANUAL_REVIEW = legacy.MANUAL_REVIEW
NON_DUPLICATE = legacy.NON_DUPLICATE

DEFAULT_VERIFIER_SIZE = 256
DEFAULT_BACKGROUND_DELTA = 14
DEFAULT_ECC_ITERATIONS = 80
DEFAULT_ECC_EPSILON = 1e-5
DEFAULT_MAX_TRANSLATION_FRACTION = 0.08
DEFAULT_MAX_ROTATION_DEGREES = 4.0
DEFAULT_INTERIOR_ERODE_PIXELS = 7
DEFAULT_PATCH_SIZE = 32


@dataclass(frozen=True)
class EccDecisionThresholds:
    """Trial thresholds for the consensus verifier.

    The defaults are deliberately explicit and are emitted into the summary.
    They are NOT frozen/calibrated values.
    """

    auto_ecc_correlation_min: float = 0.90
    auto_rgb_ssim_min: float = 0.90
    auto_foreground_iou_min: float = 0.90
    auto_edge_ssim_min: float = 0.90
    auto_mean_lab_delta_max: float = 8.0
    auto_interior_mae_max: float = 0.05
    auto_patch_mae_max: float = 0.12

    manual_rgb_ssim_min: float = 0.76
    manual_gray_ssim_min: float = 0.82
    manual_foreground_iou_min: float = 0.72
    manual_edge_ssim_min: float = 0.72
    manual_ecc_correlation_min: float = 0.86


DEFAULT_THRESHOLDS = EccDecisionThresholds()


def _foreground_mask(
    rgb: np.ndarray,
    *,
    background_delta: int = DEFAULT_BACKGROUND_DELTA,
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape={rgb.shape}")
    distance_from_white = np.max(255 - rgb.astype(np.int16), axis=2)
    return distance_from_white >= int(background_delta)


def _gray_float(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _edge_ssim(left_gray: np.ndarray, right_gray: np.ndarray) -> float:
    left_edge = sobel(left_gray)
    right_edge = sobel(right_gray)
    lo = float(min(left_edge.min(), right_edge.min()))
    hi = float(max(left_edge.max(), right_edge.max()))
    data_range = max(hi - lo, 1e-6)
    return float(
        structural_similarity(left_edge, right_edge, data_range=data_range)
    )


def _mean_lab(rgb_float: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = rgb2lab(rgb_float)
    if mask.any():
        return np.mean(lab[mask], axis=0)
    return np.mean(lab.reshape(-1, 3), axis=0)


def _interior_mask(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    *,
    erode_pixels: int = DEFAULT_INTERIOR_ERODE_PIXELS,
) -> np.ndarray:
    intersection = (left_mask & right_mask).astype(np.uint8)
    if int(erode_pixels) <= 0:
        return intersection.astype(bool)
    kernel_size = max(1, int(erode_pixels))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(intersection, kernel, iterations=1)
    if eroded.any():
        return eroded.astype(bool)
    return intersection.astype(bool)


def _patch_residual_stats(
    per_pixel_mae: np.ndarray,
    interior: np.ndarray,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    min_patch_foreground_fraction: float = 0.50,
) -> tuple[float, float, int]:
    """Return (max patch MAE, p90 patch MAE, usable patch count).

    This is intended to catch small but meaningful local detail changes that a
    global SSIM/MAE can wash out. Boundary-only interpolation noise is reduced by
    measuring on an eroded garment interior mask.
    """

    patch = max(8, int(patch_size))
    scores: list[float] = []
    height, width = per_pixel_mae.shape
    for top in range(0, height, patch):
        for left in range(0, width, patch):
            local_mask = interior[top : top + patch, left : left + patch]
            if local_mask.size == 0:
                continue
            if float(local_mask.mean()) < float(min_patch_foreground_fraction):
                continue
            local_diff = per_pixel_mae[top : top + patch, left : left + patch]
            scores.append(float(np.mean(local_diff[local_mask])))
    if not scores:
        values = per_pixel_mae[interior]
        fallback = float(np.mean(values)) if values.size else 1.0
        return fallback, fallback, 0
    ordered = np.asarray(sorted(scores), dtype=np.float32)
    return float(ordered[-1]), float(np.quantile(ordered, 0.90)), len(scores)


def align_rgb_ecc(
    template_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    *,
    max_iterations: int = DEFAULT_ECC_ITERATIONS,
    epsilon: float = DEFAULT_ECC_EPSILON,
    max_translation_fraction: float = DEFAULT_MAX_TRANSLATION_FRACTION,
    max_rotation_degrees: float = DEFAULT_MAX_ROTATION_DEGREES,
) -> tuple[np.ndarray, dict[str, object]]:
    """Register ``moving_rgb`` to ``template_rgb`` with guarded Euclidean ECC.

    Euclidean ECC can correct residual translation/rotation after foreground
    normalization but cannot scale, shear, or arbitrarily warp one garment into
    another. That limitation is intentional: a verifier must not erase genuine
    garment-shape differences simply to obtain a larger similarity score.
    """

    if template_rgb.shape != moving_rgb.shape:
        raise ValueError(
            f"ECC arrays must have equal shape: {template_rgb.shape} vs "
            f"{moving_rgb.shape}"
        )
    if template_rgb.ndim != 3 or template_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB arrays, got shape={template_rgb.shape}")

    height, width = template_rgb.shape[:2]
    template = cv2.GaussianBlur(_gray_float(template_rgb), (5, 5), 0)
    moving = cv2.GaussianBlur(_gray_float(moving_rgb), (5, 5), 0)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(max_iterations),
        float(epsilon),
    )

    try:
        correlation, warp = cv2.findTransformECC(
            template,
            moving,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            1,
        )
    except cv2.error:
        return moving_rgb.copy(), {
            "alignment_success": False,
            "alignment_reason": "ecc_failed",
            "ecc_correlation": 0.0,
            "rotation_degrees": 0.0,
            "translation_x": 0.0,
            "translation_y": 0.0,
        }

    a00, a01, tx = map(float, warp[0])
    a10, a11, ty = map(float, warp[1])
    rotation = math.degrees(math.atan2(a10, a00))
    translation_ok = (
        abs(tx) <= float(max_translation_fraction) * width
        and abs(ty) <= float(max_translation_fraction) * height
    )
    rotation_ok = abs(rotation) <= float(max_rotation_degrees)
    accepted = bool(translation_ok and rotation_ok)

    info = {
        "alignment_success": accepted,
        "alignment_reason": "accepted" if accepted else "transform_guard_rejected",
        "ecc_correlation": float(correlation),
        "rotation_degrees": float(rotation),
        "translation_x": float(tx),
        "translation_y": float(ty),
        "warp_matrix": warp.tolist(),
    }
    if not accepted:
        return moving_rgb.copy(), info

    aligned = cv2.warpAffine(
        moving_rgb,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return aligned, info


def aligned_pair_evidence(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    *,
    background_delta: int = DEFAULT_BACKGROUND_DELTA,
    interior_erode_pixels: int = DEFAULT_INTERIOR_ERODE_PIXELS,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> dict[str, object]:
    """Compute interpretable evidence after conservative pairwise registration."""

    aligned_right, alignment = align_rgb_ecc(left_rgb, right_rgb)

    left_float = left_rgb.astype(np.float32) / 255.0
    right_float = aligned_right.astype(np.float32) / 255.0
    left_gray = rgb2gray(left_float)
    right_gray = rgb2gray(right_float)

    left_mask = _foreground_mask(left_rgb, background_delta=background_delta)
    right_mask = _foreground_mask(
        aligned_right, background_delta=background_delta
    )
    union = left_mask | right_mask
    intersection = left_mask & right_mask
    foreground_iou = float(intersection.sum() / max(1, union.sum()))

    rgb_ssim = float(
        structural_similarity(
            left_rgb,
            aligned_right,
            data_range=255,
            channel_axis=2,
        )
    )
    gray_ssim = float(
        structural_similarity(left_gray, right_gray, data_range=1.0)
    )
    edge_ssim = _edge_ssim(left_gray, right_gray)

    left_lab_mean = _mean_lab(left_float, left_mask)
    right_lab_mean = _mean_lab(right_float, right_mask)
    mean_lab_delta = float(np.linalg.norm(left_lab_mean - right_lab_mean))

    per_pixel_mae = np.mean(np.abs(left_float - right_float), axis=2)
    interior = _interior_mask(
        left_mask,
        right_mask,
        erode_pixels=interior_erode_pixels,
    )
    if interior.any():
        interior_mae = float(np.mean(per_pixel_mae[interior]))
    elif union.any():
        interior_mae = float(np.mean(per_pixel_mae[union]))
    else:
        interior_mae = float(np.mean(per_pixel_mae))

    patch_mae_max, patch_mae_p90, patch_count = _patch_residual_stats(
        per_pixel_mae,
        interior,
        patch_size=patch_size,
    )

    return {
        **alignment,
        "rgb_ssim": rgb_ssim,
        "gray_ssim": gray_ssim,
        "foreground_iou": foreground_iou,
        "edge_ssim": edge_ssim,
        "mean_lab_delta": mean_lab_delta,
        "interior_mae": interior_mae,
        "patch_mae_max": patch_mae_max,
        "patch_mae_p90": patch_mae_p90,
        "patch_count": int(patch_count),
    }


def classify_aligned_pair(
    evidence: Mapping[str, object],
    *,
    thresholds: EccDecisionThresholds = DEFAULT_THRESHOLDS,
) -> tuple[str, str]:
    """Classify one pHash candidate with a conservative consensus rule."""

    auto_duplicate = (
        bool(evidence["alignment_success"])
        and float(evidence["ecc_correlation"])
        >= thresholds.auto_ecc_correlation_min
        and float(evidence["rgb_ssim"]) >= thresholds.auto_rgb_ssim_min
        and float(evidence["foreground_iou"])
        >= thresholds.auto_foreground_iou_min
        and float(evidence["edge_ssim"]) >= thresholds.auto_edge_ssim_min
        and float(evidence["mean_lab_delta"])
        <= thresholds.auto_mean_lab_delta_max
        and float(evidence["interior_mae"])
        <= thresholds.auto_interior_mae_max
        and float(evidence["patch_mae_max"])
        <= thresholds.auto_patch_mae_max
    )
    if auto_duplicate:
        return DUPLICATE, "phash_ecc_consensus_auto"

    manual = (
        float(evidence["rgb_ssim"]) >= thresholds.manual_rgb_ssim_min
        or float(evidence["gray_ssim"]) >= thresholds.manual_gray_ssim_min
        or float(evidence["ecc_correlation"])
        >= thresholds.manual_ecc_correlation_min
        or (
            float(evidence["foreground_iou"])
            >= thresholds.manual_foreground_iou_min
            and float(evidence["edge_ssim"])
            >= thresholds.manual_edge_ssim_min
        )
    )
    if manual:
        return MANUAL_REVIEW, "phash_ecc_consensus_manual"
    return NON_DUPLICATE, "phash_ecc_consensus_non_duplicate"


class _NormalizedReferenceStore:
    """LRU cache of normalized RGB Polyvore candidates."""

    def __init__(
        self,
        base_store,
        *,
        size: int,
        background_delta: int,
        cache_size: int = 1024,
    ) -> None:
        self.base_store = base_store
        self.size = int(size)
        self.background_delta = int(background_delta)
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def array(self, item_id: str) -> np.ndarray:
        cached = self._cache.get(item_id)
        if cached is not None:
            self._cache.move_to_end(item_id)
            return cached
        image = self.base_store.load_image(item_id)
        array = normalize_rgb_for_verification(
            image,
            size=self.size,
            background_delta=self.background_delta,
        )
        self._cache[item_id] = array
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return array


def _csv_row(row: Mapping[str, object]) -> dict[str, object]:
    payload = dict(row)
    if isinstance(payload.get("polyvore_splits"), (list, tuple)):
        payload["polyvore_splits"] = "|".join(payload["polyvore_splits"])
    return payload


def _write_csv(
    rows: Sequence[Mapping[str, object]],
    path: Path,
    headers: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(headers),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))


def prepare_overlap_audit_ecc(
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
    evaluation3_groups: Iterable[str] | None = None,
    model_development_splits: Iterable[str] = legacy.DEFAULT_MODEL_DEVELOPMENT_SPLITS,
    phash_threshold: int = 4,
    verifier_size: int = DEFAULT_VERIFIER_SIZE,
    background_delta: int = DEFAULT_BACKGROUND_DELTA,
    thresholds: EccDecisionThresholds = DEFAULT_THRESHOLDS,
    allow_incomplete_image_index: bool = False,
) -> tuple[dict[str, object], dict[str, Path]]:
    """Run pHash retrieval + guarded ECC consensus verification.

    Manual previews are emitted only for outfits that have NO automatic duplicate
    pair. This avoids asking a human to review pairs from an outfit that is
    already definitively contaminated by another item.
    """

    if bool(polyvore_image_root) == bool(polyvore_hf_dataset):
        raise ValueError("Choose exactly one Polyvore image provider")
    if not 0 <= int(phash_threshold) <= 16:
        raise ValueError("phash_threshold must be between 0 and 16")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    preview_dir = destination / "manual_review_previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    e3_root = Path(evaluation3_root).expanduser().resolve()
    identity = legacy.load_development_identity(development_split_paths)
    if polyvore_image_root:
        development_items, base_store, provider_report, unresolved = (
            legacy.build_local_development_index(
                polyvore_image_root,
                identity,
                ssim_size=verifier_size,
            )
        )
    else:
        development_items, base_store, provider_report, unresolved = (
            legacy.build_huggingface_development_index(
                str(polyvore_hf_dataset),
                hf_split_mapping or {},
                identity,
                ssim_size=verifier_size,
            )
        )
    if unresolved and not allow_incomplete_image_index:
        raise ValueError(
            f"Polyvore image index is incomplete: {len(unresolved)} unresolved "
            f"scorer item IDs; examples={unresolved[:10]}"
        )

    annotation_tables = []
    if annotations_path:
        annotation_tables.append(
            legacy.load_evaluation3_annotations(
                annotations_path,
                sheet_name=annotation_sheet,
            )
        )
    if metadata_path:
        annotation_tables.append(
            legacy.load_evaluation3_annotations(
                metadata_path,
                sheet_name=metadata_sheet,
            )
        )
    annotations = (
        legacy.merge_evaluation3_annotations(*annotation_tables)
        if annotation_tables
        else None
    )
    outfits = legacy.discover_evaluation3_outfits(
        e3_root,
        annotations=annotations,
        selected_groups=evaluation3_groups,
    )
    if not outfits:
        raise ValueError("No EVALUATION3 outfits matched the requested selection")

    phash_index: MutableMapping[int, list[object]] = defaultdict(list)
    tree = legacy.BKTree()
    for item in development_items:
        if item.fingerprint.phash64 not in phash_index:
            tree.add(item.fingerprint.phash64)
        phash_index[item.fingerprint.phash64].append(item)

    normalized_store = _NormalizedReferenceStore(
        base_store,
        size=verifier_size,
        background_delta=background_delta,
    )
    model_splits = set(model_development_splits)

    audit_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    auto_rows: list[dict[str, object]] = []
    manual_key_rows: list[dict[str, object]] = []
    manual_blind_rows: list[dict[str, object]] = []
    pair_counter = 0
    suppressed_manual_candidates = 0

    Image, _ = legacy._require_pillow()

    for outfit in outfits:
        outfit_id = str(outfit["e3_outfit_id"])
        auto_duplicate_splits: set[str] = set()
        image_results: list[dict[str, object]] = []
        result_by_slot: dict[str, dict[str, object]] = {}
        pending_manual: list[dict[str, object]] = []
        outfit_candidates: list[dict[str, object]] = []

        for slot, relative_path in sorted(dict(outfit["images"]).items()):
            e3_path = e3_root / str(relative_path)
            with Image.open(e3_path) as e3_image:
                e3_image.load()
                e3_copy = e3_image.copy()

            fingerprint = legacy.fingerprint_image(e3_copy)
            e3_rgb = normalize_rgb_for_verification(
                e3_copy,
                size=verifier_size,
                background_delta=background_delta,
            )
            per_image_auto: list[dict[str, object]] = []
            per_image_decisions: Counter[str] = Counter()

            for phash_distance, phash_value in tree.query(
                fingerprint.phash64,
                int(phash_threshold),
            ):
                for reference in phash_index[phash_value]:
                    reference_rgb = normalized_store.array(reference.item_id)
                    evidence = aligned_pair_evidence(
                        e3_rgb,
                        reference_rgb,
                        background_delta=background_delta,
                    )
                    decision, method = classify_aligned_pair(
                        evidence,
                        thresholds=thresholds,
                    )
                    per_image_decisions[decision] += 1
                    base = {
                        "e3_outfit_id": outfit_id,
                        "e3_slot": slot,
                        "e3_image": str(relative_path),
                        "decision": decision,
                        "method": method,
                        "polyvore_item_id": reference.item_id,
                        "polyvore_splits": list(reference.splits),
                        "polyvore_image_source": reference.image_source,
                        "phash_distance": int(phash_distance),
                        **evidence,
                    }
                    outfit_candidates.append(base)

                    if decision == DUPLICATE:
                        auto_duplicate_splits.update(reference.splits)
                        auto_rows.append(base)
                        per_image_auto.append(base)
                    elif decision == MANUAL_REVIEW:
                        pending_manual.append(
                            {
                                "base": base,
                                "e3_path": e3_path,
                                "reference_item_id": reference.item_id,
                                "slot": slot,
                            }
                        )

            image_result = {
                "slot": slot,
                "evaluation3_image": str(relative_path),
                "evaluation3_phash64": f"{fingerprint.phash64:016x}",
                "candidate_count": int(sum(per_image_decisions.values())),
                "candidate_decisions": dict(per_image_decisions),
                "auto_duplicate_count": len(per_image_auto),
                "manual_review_pair_ids": [],
                "auto_duplicate_examples": per_image_auto[
                    : legacy.MAX_MATCH_EXAMPLES_PER_IMAGE
                ],
            }
            image_results.append(image_result)
            result_by_slot[slot] = image_result

        manual_candidate_splits: set[str] = set()
        manual_pair_ids: list[str] = []
        if auto_duplicate_splits:
            current_decision = DUPLICATE
            suppressed_manual_candidates += len(pending_manual)
        elif pending_manual:
            current_decision = MANUAL_REVIEW
            for pending in pending_manual:
                pair_counter += 1
                pair_id = f"PAIR{pair_counter:06d}"
                base = dict(pending["base"])
                manual_candidate_splits.update(base["polyvore_splits"])
                manual_pair_ids.append(pair_id)
                result_by_slot[str(pending["slot"])][
                    "manual_review_pair_ids"
                ].append(pair_id)
                preview_rel = f"manual_review_previews/{pair_id}.jpg"
                legacy.write_manual_preview(
                    Path(pending["e3_path"]),
                    base_store.load_image(str(pending["reference_item_id"])),
                    pair_id,
                    destination / preview_rel,
                )
                manual_key_rows.append(
                    {
                        "pair_id": pair_id,
                        **base,
                        "preview_file": preview_rel,
                    }
                )
                manual_blind_rows.append(
                    {
                        "pair_id": pair_id,
                        "preview_file": preview_rel,
                        "human_label": "",
                    }
                )
        else:
            current_decision = NON_DUPLICATE

        for candidate in outfit_candidates:
            candidate["outfit_decision_pre_review"] = current_decision
        candidate_rows.extend(outfit_candidates)

        audit_rows.append(
            {
                **dict(outfit),
                "overlap": {
                    "decision_pre_review": current_decision,
                    "auto_duplicate_splits": sorted(auto_duplicate_splits),
                    "manual_candidate_splits": sorted(manual_candidate_splits),
                    "manual_review_pair_ids": manual_pair_ids,
                    "image_results": image_results,
                },
            }
        )

    paths = {
        "summary": destination / "evaluation3_overlap_summary_pre_review.json",
        "audit_pre_review": destination / "evaluation3_overlap_audit_pre_review.jsonl",
        "full": destination / "evaluation3_full.jsonl",
        "candidate_evidence": destination / "evaluation3_candidate_evidence.csv",
        "auto_duplicates": destination / "evaluation3_auto_duplicates.csv",
        "manual_blind": destination / "evaluation3_manual_review_BLIND.csv",
        "manual_xlsx": destination / "evaluation3_manual_review_BLIND.xlsx",
        "manual_key": destination / "evaluation3_manual_review_KEY.csv",
        "manual_html": destination / "evaluation3_manual_review_BLIND.html",
        "duplicate_pre_review": destination / "evaluation3_DUPLICATE_pre_review.jsonl",
        "manual_pre_review": destination / "evaluation3_MANUAL_REVIEW_pre_review.jsonl",
        "non_duplicate_pre_review": destination / "evaluation3_NON_DUPLICATE_pre_review.jsonl",
    }

    legacy.write_jsonl(audit_rows, paths["audit_pre_review"])
    legacy.write_jsonl(audit_rows, paths["full"])
    legacy.write_jsonl(
        (
            row
            for row in audit_rows
            if row["overlap"]["decision_pre_review"] == DUPLICATE
        ),
        paths["duplicate_pre_review"],
    )
    legacy.write_jsonl(
        (
            row
            for row in audit_rows
            if row["overlap"]["decision_pre_review"] == MANUAL_REVIEW
        ),
        paths["manual_pre_review"],
    )
    legacy.write_jsonl(
        (
            row
            for row in audit_rows
            if row["overlap"]["decision_pre_review"] == NON_DUPLICATE
        ),
        paths["non_duplicate_pre_review"],
    )

    evidence_headers = (
        "e3_outfit_id",
        "e3_slot",
        "e3_image",
        "outfit_decision_pre_review",
        "decision",
        "method",
        "polyvore_item_id",
        "polyvore_splits",
        "polyvore_image_source",
        "phash_distance",
        "alignment_success",
        "alignment_reason",
        "ecc_correlation",
        "rotation_degrees",
        "translation_x",
        "translation_y",
        "rgb_ssim",
        "gray_ssim",
        "foreground_iou",
        "edge_ssim",
        "mean_lab_delta",
        "interior_mae",
        "patch_mae_max",
        "patch_mae_p90",
        "patch_count",
    )
    _write_csv(candidate_rows, paths["candidate_evidence"], evidence_headers)
    _write_csv(auto_rows, paths["auto_duplicates"], evidence_headers)

    blind_headers = ("pair_id", "preview_file", "human_label")
    _write_csv(manual_blind_rows, paths["manual_blind"], blind_headers)

    key_headers = (
        "pair_id",
        "preview_file",
        *evidence_headers,
    )
    _write_csv(manual_key_rows, paths["manual_key"], key_headers)
    legacy._write_manual_html(manual_blind_rows, paths["manual_html"])
    legacy._write_manual_xlsx(manual_blind_rows, paths["manual_xlsx"])

    incomplete_e3 = sum(bool(row.get("missing_slots")) for row in audit_rows)
    decision_counts = Counter(
        row["overlap"]["decision_pre_review"] for row in audit_rows
    )
    candidate_decision_counts = Counter(
        row["decision"] for row in candidate_rows
    )
    alignment_failures = sum(
        not bool(row.get("alignment_success")) for row in candidate_rows
    )

    summary = {
        "status": (
            "MANUAL_REVIEW_REQUIRED"
            if manual_blind_rows
            else "PRE_REVIEW_COMPLETE"
        ),
        "official_clean_manifests_ready": False,
        "configuration": {
            "protocol": "phash-ecc-consensus-v0-experimental",
            "phash_hamming_threshold": int(phash_threshold),
            "phash_role": "global candidate retrieval only",
            "verifier_size": int(verifier_size),
            "normalization": (
                "EXIF -> RGB -> conservative non-white foreground crop -> "
                "aspect-preserving fit on white canvas"
            ),
            "registration": (
                "OpenCV ECC MOTION_EUCLIDEAN only; translation/rotation guarded; "
                "no affine scale/shear/homography"
            ),
            "decision_rule": "multi-evidence conservative consensus",
            "thresholds": asdict(thresholds),
            "calibration_status": "EXPERIMENTAL_UNCALIBRATED",
            "model_development_splits": sorted(model_splits),
        },
        "development_data": {
            "sample_counts": identity["sample_counts"],
            "unique_item_count": len(identity["item_splits"]),
            "fingerprinted_item_count": len(development_items),
            "unresolved_item_count": len(unresolved),
            "unresolved_item_id_examples": unresolved[:50],
            "image_provider": provider_report,
        },
        "evaluation3": {
            "selected_outfit_count": len(audit_rows),
            "outfits_missing_required_images": incomplete_e3,
        },
        "candidate_pairs": {
            "total": len(candidate_rows),
            "decisions": dict(candidate_decision_counts),
            "alignment_failures": int(alignment_failures),
        },
        "decisions_pre_review": {
            "outfits_auto_duplicate": int(decision_counts[DUPLICATE]),
            "outfits_manual_review": int(decision_counts[MANUAL_REVIEW]),
            "outfits_non_duplicate": int(decision_counts[NON_DUPLICATE]),
            "manual_review_pairs": len(manual_blind_rows),
            "manual_candidates_suppressed_by_auto_duplicate_outfit": int(
                suppressed_manual_candidates
            ),
        },
        "next_step": (
            "Review the BLIND workbook and inspect candidate_evidence.csv. "
            "Do not freeze thresholds until labelled score distributions are checked."
        ),
    }
    legacy.write_json(summary, paths["summary"])
    return summary, paths


finalize_overlap_audit = legacy.finalize_overlap_audit
