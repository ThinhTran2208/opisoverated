# -*- coding: utf-8 -*-
"""Fast active-learning pilot that works directly from existing manual preview JPGs.

This intentionally avoids NB10D/full E3 feature extraction. Each preview already
contains EVALUATION3 on the left and Polyvore on the right. We split the panels,
normalize foreground, compute cheap visual features, and train classical ML.

This is a PILOT ONLY: preview-JPEG features are not a canonical overlap detector.
If the classifier cannot learn a useful boundary here, there is little reason to
run a much more expensive full-data pipeline.
"""
from __future__ import annotations

from pathlib import Path
import shutil
from typing import Sequence
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fftpack import dct
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import sobel
from skimage.metrics import structural_similarity

from src.evaluation.evaluation3_active_learning import (
    DUPLICATE, NON_DUPLICATE,
    build_models, binary_metrics, choose_triage_thresholds,
)

SAME_PRODUCT_DIFFERENT_IMAGE = "SAME_PRODUCT_DIFFERENT_IMAGE"
VALID_PREVIEW_LABELS = frozenset(
    {DUPLICATE, NON_DUPLICATE, SAME_PRODUCT_DIFFERENT_IMAGE}
)

FEATURE_COLUMNS = (
    "phash_distance", "rgb_ssim", "gray_ssim", "edge_ssim",
    "mean_lab_delta", "foreground_iou", "foreground_mae",
    "histogram_intersection", "foreground_area_ratio_delta",
    "bbox_aspect_ratio_delta",
)


def _foreground_mask(rgb: np.ndarray, delta: int = 14) -> np.ndarray:
    return np.max(255 - rgb.astype(np.int16), axis=2) >= int(delta)


def _normalize_panel(image: Image.Image, size: int = 256) -> tuple[np.ndarray, dict[str, float]]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    raw = np.asarray(image, dtype=np.uint8)
    mask = _foreground_mask(raw)
    bbox_aspect = 1.0
    area_fraction = float(mask.mean())
    crop = image
    if mask.any():
        ys, xs = np.nonzero(mask)
        l, r = int(xs.min()), int(xs.max()) + 1
        t, b = int(ys.min()), int(ys.max()) + 1
        w, h = max(1, r - l), max(1, b - t)
        bbox_aspect = float(w / h)
        margin = int(round(max(w, h) * 0.05))
        crop = image.crop((max(0,l-margin), max(0,t-margin), min(image.width,r+margin), min(image.height,b+margin)))
    w, h = crop.size
    scale = min(size / max(1,w), size / max(1,h))
    nw, nh = max(1,int(round(w*scale))), max(1,int(round(h*scale)))
    resized = crop.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size,size), "white")
    canvas.paste(resized, ((size-nw)//2, (size-nh)//2))
    return np.asarray(canvas, dtype=np.uint8), {"area_fraction": area_fraction, "bbox_aspect": bbox_aspect}


def split_preview(path: Path | str) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        w, h = rgb.size
        # canonical previews are 760x420 with panels at x=10..370 and 390..750, y=40..400
        if w >= 740 and h >= 390:
            left = rgb.crop((10, 40, min(370,w), min(400,h)))
            right = rgb.crop((max(390, w//2), 40, min(750,w), min(400,h)))
        else:
            mid = w // 2
            left = rgb.crop((0,0,mid,h))
            right = rgb.crop((mid,0,w,h))
    la, lm = _normalize_panel(left)
    ra, rm = _normalize_panel(right)
    return la, ra, lm, rm


def _phash64(rgb: np.ndarray) -> int:
    gray = Image.fromarray(rgb).convert("L").resize((32,32), Image.Resampling.LANCZOS)
    x = np.asarray(gray, dtype=np.float32)
    z = dct(dct(x, axis=0), axis=1)[:8,:8]
    med = float(np.median(z))
    value = 0
    for bit in (z > med).reshape(-1):
        value = (value << 1) | int(bool(bit))
    return value


def _hist_intersection(a: np.ndarray, b: np.ndarray) -> float:
    scores = []
    for c in range(3):
        ha, _ = np.histogram(a[...,c], bins=32, range=(0,256), density=False)
        hb, _ = np.histogram(b[...,c], bins=32, range=(0,256), density=False)
        ha = ha.astype(float); hb = hb.astype(float)
        ha /= max(1.0, ha.sum()); hb /= max(1.0, hb.sum())
        scores.append(float(np.minimum(ha,hb).sum()))
    return float(np.mean(scores))


def extract_preview_features(path: Path | str) -> dict[str, object]:
    path = Path(path)
    left, right, lm, rm = split_preview(path)
    lf, rf = left.astype(np.float32)/255.0, right.astype(np.float32)/255.0
    lg, rg = rgb2gray(lf), rgb2gray(rf)
    le, re = sobel(lg), sobel(rg)
    edge_range = max(float(max(le.max(),re.max()) - min(le.min(),re.min())), 1e-6)
    lmask, rmask = _foreground_mask(left), _foreground_mask(right)
    union, inter = lmask | rmask, lmask & rmask
    iou = float(inter.sum()/max(1,union.sum()))
    mae_map = np.mean(np.abs(lf-rf), axis=2)
    fg_mae = float(mae_map[union].mean()) if union.any() else float(mae_map.mean())
    llab, rlab = rgb2lab(lf), rgb2lab(rf)
    lmean = llab[lmask].mean(axis=0) if lmask.any() else llab.reshape(-1,3).mean(axis=0)
    rmean = rlab[rmask].mean(axis=0) if rmask.any() else rlab.reshape(-1,3).mean(axis=0)
    return {
        "pair_id": path.stem,
        "preview_file": str(path),
        "phash_distance": int((_phash64(left)^_phash64(right)).bit_count()),
        "rgb_ssim": float(structural_similarity(left,right,data_range=255,channel_axis=2)),
        "gray_ssim": float(structural_similarity(lg,rg,data_range=1.0)),
        "edge_ssim": float(structural_similarity(le,re,data_range=edge_range)),
        "mean_lab_delta": float(np.linalg.norm(lmean-rmean)),
        "foreground_iou": iou,
        "foreground_mae": fg_mae,
        "histogram_intersection": _hist_intersection(left,right),
        "foreground_area_ratio_delta": abs(float(lm["area_fraction"]-rm["area_fraction"])),
        "bbox_aspect_ratio_delta": abs(float(lm["bbox_aspect"]-rm["bbox_aspect"])),
    }


def build_preview_feature_pool(preview_dir: Path | str, *, max_pairs: int = 600, random_state: int = 42) -> pd.DataFrame:
    files = sorted(Path(preview_dir).glob("PAIR*.jpg"))
    if not files:
        raise FileNotFoundError(f"No PAIR*.jpg in {preview_dir}")
    if max_pairs and len(files) > max_pairs:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(len(files), size=max_pairs, replace=False))
        files = [files[int(i)] for i in idx]
    rows = [extract_preview_features(p) for p in files]
    return pd.DataFrame(rows)


def copy_batch_previews(batch: pd.DataFrame, destination: Path | str) -> None:
    dst = Path(destination)
    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for p in batch["preview_file"]:
        src = Path(str(p))
        if src.is_file(): shutil.copy2(src, dst/src.name)


def write_review_sheet(batch: pd.DataFrame, destination: Path | str) -> None:
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    wb=Workbook(); ws=wb.active; ws.title="review"
    ws.append(["pair_id","preview_file","human_label"])
    for r in batch.itertuples(index=False): ws.append([str(r.pair_id),str(r.preview_file),""])
    dv=DataValidation(
        type="list",
        formula1='"DUPLICATE,SAME_PRODUCT_DIFFERENT_IMAGE,NON_DUPLICATE"',
        allow_blank=True,
    )
    ws.add_data_validation(dv)
    if len(batch): dv.add(f"C2:C{len(batch)+1}")
    ws.freeze_panes="A2"; ws.column_dimensions["A"].width=16; ws.column_dimensions["B"].width=90; ws.column_dimensions["C"].width=22
    Path(destination).parent.mkdir(parents=True,exist_ok=True); wb.save(destination)


def read_labels(path: Path | str) -> dict[str,str]:
    p=Path(path)
    if not p.is_file(): return {}
    df=pd.read_excel(p) if p.suffix.lower().startswith('.xls') else pd.read_csv(p)
    out={}
    for r in df.itertuples(index=False):
        label=str(getattr(r,'human_label','')).strip().upper()
        if label in VALID_PREVIEW_LABELS: out[str(getattr(r,'pair_id')).strip()]=label
    return out


def attach_labels(pool: pd.DataFrame, review_files) -> pd.DataFrame:
    labels={}
    for f in review_files: labels.update(read_labels(f))
    out=pool.copy(); out["human_label"]=out["pair_id"].map(labels).fillna("")
    out["target"]=out["human_label"].map({
        NON_DUPLICATE: 0,
        DUPLICATE: 1,
        SAME_PRODUCT_DIFFERENT_IMAGE: 1,
    })
    return out


def xframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, FEATURE_COLUMNS].astype(float)


def fit_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    random_state: int = 42,
    *,
    sample_weight: Sequence[float] | None = None,
):
    """Fit the preview baselines, optionally with per-row training weights.

    Historical confirmed positives are much easier than the current manual
    queue.  Supporting row weights lets callers keep those positives as useful
    anchors without allowing that source to dominate the decision boundary.
    """

    models=build_models(random_state); rows=[]
    Xtr=xframe(train); ytr=train.target.astype(int).to_numpy(); Xv=xframe(validation); yv=validation.target.astype(int).to_numpy()
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (len(train),):
            raise ValueError("sample_weight must contain exactly one value per train row")
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("sample_weight values must be finite and > 0")
    fitted={}
    for name,m in models.items():
        if weights is None:
            m.fit(Xtr,ytr)
        elif hasattr(m, "named_steps"):
            m.fit(Xtr, ytr, model__sample_weight=weights)
        else:
            m.fit(Xtr, ytr, sample_weight=weights)
        idx=list(m.classes_).index(1); p=m.predict_proba(Xv)[:,idx]
        rows.append({"model":name,**binary_metrics(yv,p)}); fitted[name]=m
    report=pd.DataFrame(rows).sort_values(["roc_auc","balanced_accuracy@0.5"],ascending=False).reset_index(drop=True)
    return fitted, report


def select_uncertain(model, unlabeled: pd.DataFrame, batch_size: int=40) -> pd.DataFrame:
    if unlabeled.empty: return unlabeled.copy()
    idx=list(model.classes_).index(1); p=model.predict_proba(xframe(unlabeled))[:,idx]
    out=unlabeled.copy(); out["model_probability_duplicate"]=p; out["uncertainty"]=np.abs(p-0.5)
    return out.sort_values("uncertainty").head(batch_size).copy()


def select_resolution_batch(
    model,
    pool_rows: pd.DataFrame,
    *,
    batch_size: int = 40,
    group_column: str = "e3_outfit_id",
    labeled_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select high-yield pairs for resolving the overlap audit.

    Active-learning queries and production adjudication have different goals.
    ``select_uncertain`` improves the classifier by asking about p≈0.5 rows.
    This function instead tries to find one duplicate quickly for each unresolved
    E3 group: it removes groups that already have a human-confirmed duplicate,
    ranks the remaining pairs by P(DUPLICATE), and returns at most one pair per
    group. If group metadata is unavailable, each pair is treated as its own
    group rather than silently collapsing unrelated rows.
    """

    if pool_rows.empty or batch_size <= 0:
        return pool_rows.iloc[0:0].copy()

    pool = pool_rows.copy()
    if "pair_id" not in pool.columns:
        raise KeyError("pool_rows missing pair_id")
    if group_column in pool.columns:
        pool["_resolution_group"] = pool[group_column].astype(str)
    else:
        pool["_resolution_group"] = pool["pair_id"].astype(str)

    resolved_groups: set[str] = set()
    labeled_pair_ids: set[str] = set()
    if labeled_rows is not None and not labeled_rows.empty:
        labeled = labeled_rows.copy()
        if "pair_id" in labeled.columns:
            labeled_pair_ids = set(labeled["pair_id"].astype(str))
        if group_column in labeled.columns:
            if "target" in labeled.columns:
                positive = pd.to_numeric(labeled["target"], errors="coerce").eq(1)
            else:
                labels = labeled.get(
                    "human_label", pd.Series("", index=labeled.index)
                ).astype(str).str.strip().str.upper()
                positive = labels.isin({DUPLICATE, SAME_PRODUCT_DIFFERENT_IMAGE})
            resolved_groups = set(
                labeled.loc[positive, group_column].astype(str)
            )

    candidates = pool[
        ~pool["pair_id"].astype(str).isin(labeled_pair_ids)
        & ~pool["_resolution_group"].isin(resolved_groups)
    ].copy()
    if candidates.empty:
        return candidates.drop(columns=["_resolution_group"])

    idx = list(model.classes_).index(1)
    probability = model.predict_proba(xframe(candidates))[:, idx]
    candidates["model_probability_duplicate"] = probability
    candidates = candidates.sort_values(
        ["model_probability_duplicate", "pair_id"],
        ascending=[False, True],
    )
    candidates = candidates.drop_duplicates(
        subset=["_resolution_group"], keep="first"
    ).head(int(batch_size))
    return candidates.drop(columns=["_resolution_group"]).copy()
