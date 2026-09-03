#!/usr/bin/env python3
"""
EVALUATION3 manual-calibration queue generator.

Purpose
-------
1) Generate a FIRST 500-pair human-review batch for SSIM-threshold calibration.
2) Optionally generate a SECOND disjoint 500-pair batch only if the first batch
   is not enough.
3) Generate a separate retrieval-audit batch outside the current pHash<=4 gate
   (default pHash 6/8/10), so retrieval recall is tested BEFORE freezing SSIM.
4) Preserve and de-duplicate all existing human evidence.

The candidate search itself is intentionally NOT re-run here. We reuse the
already-scored candidate pools in EVALUATION3/phash_ssim_threshold and recompute
RGB-SSIM on selected pairs using the same 256px thumbnail+white-pad preprocessing
as eval3_test_overlap_pipeline.py.

Human-label policy used by this project:
    DUPLICATE
    NON_DUPLICATE
    SAME_PRODUCT_DIFFERENT_IMAGE -> DUPLICATE
(the latter means same visual item with only small lighting/viewpoint nuisance).

This is a calibration/audit utility, not the final contamination protocol.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps
from skimage.metrics import structural_similarity

DUPLICATE = "DUPLICATE"
NON_DUPLICATE = "NON_DUPLICATE"
SAME_PRODUCT_DIFFERENT_IMAGE = "SAME_PRODUCT_DIFFERENT_IMAGE"
SSIM_SIZE = 256
PREVIEW_SIZE = 360
RNG_SEED = 20260903


def normalize_label(value) -> str | None:
    label = str(value or "").strip().upper()
    if label in {DUPLICATE, SAME_PRODUCT_DIFFERENT_IMAGE}:
        return DUPLICATE
    if label == NON_DUPLICATE:
        return NON_DUPLICATE
    return None


def pair_uid(eval3_rel_path: str, polyvore_item_id: str) -> str:
    raw = f"{eval3_rel_path}|{polyvore_item_id}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def normalized_rgb(im: Image.Image, size: int = SSIM_SIZE) -> np.ndarray:
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def rgb_ssim(a: Image.Image, b: Image.Image) -> float:
    aa, bb = normalized_rgb(a), normalized_rgb(b)
    return float(structural_similarity(aa, bb, data_range=255, channel_axis=2))


def fit_square(im: Image.Image, size: int = PREVIEW_SIZE) -> Image.Image:
    return Image.fromarray(normalized_rgb(im, size=size))


def open_eval_image(e3_root: Path, rel_path: str) -> Image.Image:
    p = Path(str(rel_path))
    candidates = [
        e3_root / p,
        e3_root / p.name if len(p.parts) == 1 else e3_root / p.parts[-2] / p.parts[-1],
    ]
    for path in candidates:
        if path.is_file():
            old = ImageFile.LOAD_TRUNCATED_IMAGES
            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                with Image.open(path) as im:
                    im.load()
                    return ImageOps.exif_transpose(im).convert("RGB").copy()
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = old
    raise FileNotFoundError(f"E3 image not found for {rel_path}; tried {candidates}")


def load_polyvore_dataset(dataset_name: str = "codewaly/polyvore1000"):
    from datasets import load_dataset
    return load_dataset(dataset_name, split="train")


def ensure_pil(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "bytes" in value:
        import io
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def _standardize_candidate_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    needed = {"eval3_rel_path", "polyvore_item_id", "polyvore_row_idx", "phash_distance"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"{source} missing columns {sorted(missing)}")
    out = df.copy()
    out["candidate_source"] = source
    out["phash_distance"] = pd.to_numeric(out["phash_distance"], errors="coerce")
    out["polyvore_row_idx"] = pd.to_numeric(out["polyvore_row_idx"], errors="coerce")
    out["sampling_ssim_hint"] = pd.to_numeric(out.get("ssim", np.nan), errors="coerce")
    out["pair_uid"] = [pair_uid(a, b) for a, b in zip(out["eval3_rel_path"], out["polyvore_item_id"])]
    return out


def load_candidate_pool(threshold_root: Path | str) -> pd.DataFrame:
    root = Path(threshold_root)
    frames = []
    for name in ["candidate_pairs_all.csv", "hard_negative_candidates_scored.csv"]:
        p = root / name
        if p.is_file():
            frames.append(_standardize_candidate_frame(pd.read_csv(p), name))
    if not frames:
        raise FileNotFoundError(f"No candidate pool CSV under {root}")
    pool = pd.concat(frames, ignore_index=True, sort=False)
    pool["_has_hint"] = pool["sampling_ssim_hint"].notna().astype(int)
    sort_cols = ["pair_uid", "_has_hint"]
    ascending = [True, False]
    if "neighbor_rank" in pool.columns:
        pool["neighbor_rank"] = pd.to_numeric(pool["neighbor_rank"], errors="coerce")
        sort_cols.append("neighbor_rank")
        ascending.append(True)
    return (
        pool.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates("pair_uid", keep="first")
        .drop(columns="_has_hint")
        .reset_index(drop=True)
    )


def build_human_evidence(
    threshold_root: Path | str,
    *,
    hard_review_labels: pd.DataFrame | None = None,
    extra_evidence_csvs: Iterable[Path | str] = (),
) -> pd.DataFrame:
    root = Path(threshold_root)
    rows = []

    pos_path = root / "confirmed_duplicates_dedup.csv"
    if pos_path.is_file():
        pos = pd.read_csv(pos_path)
        for r in pos.to_dict("records"):
            rows.append({
                "eval3_rel_path": str(r["eval3_rel_path"]),
                "polyvore_item_id": str(r["polyvore_item_id"]),
                "human_label": DUPLICATE,
                "human_label_original": DUPLICATE,
                "evidence_source": "confirmed_duplicates_dedup",
                "phash_distance": r.get("phash_distance", np.nan),
                "ssim": r.get("ssim", np.nan),
            })

    key_path = root / "hard_negative_review_KEY.csv"
    if key_path.is_file() and hard_review_labels is not None:
        key = pd.read_csv(key_path)
        labels = hard_review_labels.copy()
        labels["pair_id"] = labels["pair_id"].astype(str).str.strip()
        labels["human_label_original"] = labels["human_label"].astype(str).str.strip().str.upper()
        labels["human_label"] = labels["human_label_original"].map(normalize_label)
        labels = labels[labels["human_label"].notna()]
        merged = key.merge(
            labels[["pair_id", "human_label", "human_label_original"]],
            on="pair_id", how="inner", validate="one_to_one",
        )
        for r in merged.to_dict("records"):
            rows.append({
                "eval3_rel_path": str(r["eval3_rel_path"]),
                "polyvore_item_id": str(r["polyvore_item_id"]),
                "human_label": str(r["human_label"]),
                "human_label_original": str(r["human_label_original"]),
                "evidence_source": "hard_negative_review",
                "phash_distance": r.get("phash_distance", np.nan),
                "ssim": r.get("ssim", np.nan),
            })

    for path in extra_evidence_csvs:
        p = Path(path)
        if not p.is_file():
            continue
        x = pd.read_csv(p)
        req = {"eval3_rel_path", "polyvore_item_id", "human_label"}
        if not req.issubset(x.columns):
            continue
        x["human_label_original"] = x["human_label"].astype(str).str.strip().str.upper()
        x["human_label"] = x["human_label_original"].map(normalize_label)
        x = x[x["human_label"].notna()]
        for r in x.to_dict("records"):
            rows.append({
                "eval3_rel_path": str(r["eval3_rel_path"]),
                "polyvore_item_id": str(r["polyvore_item_id"]),
                "human_label": str(r["human_label"]),
                "human_label_original": str(r["human_label_original"]),
                "evidence_source": p.name,
                "phash_distance": r.get("phash_distance", np.nan),
                "ssim": r.get("ssim", np.nan),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "pair_uid","eval3_rel_path","polyvore_item_id","human_label",
            "human_label_original","evidence_source","phash_distance","ssim"
        ])

    ev = pd.DataFrame(rows)
    ev["pair_uid"] = [pair_uid(a,b) for a,b in zip(ev.eval3_rel_path, ev.polyvore_item_id)]
    conflicts = ev.groupby("pair_uid")["human_label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting human labels found in evidence registry")
    return ev.drop_duplicates("pair_uid", keep="first").reset_index(drop=True)


def evidence_retrieval_report(evidence: pd.DataFrame) -> dict:
    if evidence.empty:
        return {}
    dup = evidence[evidence["human_label"].eq(DUPLICATE)].copy()
    d = pd.to_numeric(dup["phash_distance"], errors="coerce")
    counts = d.value_counts(dropna=True).sort_index().to_dict()
    outside = dup[d > 4]
    return {
        "human_duplicate_count": int(len(dup)),
        "duplicate_phash_distance_counts": {str(int(k)): int(v) for k,v in counts.items()},
        "confirmed_duplicates_outside_phash4": int(len(outside)),
        "outside_phash4_examples": outside[
            ["eval3_rel_path","polyvore_item_id","phash_distance","human_label_original","evidence_source"]
        ].head(20).to_dict("records"),
    }


def _unique_e3_greedy(df: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    work = df.sample(frac=1, random_state=rng.randrange(1, 2**31-1))
    one = work.drop_duplicates("eval3_rel_path", keep="first").head(n)
    if len(one) >= n:
        return one
    rest = work[~work["pair_uid"].isin(one["pair_uid"])].head(n-len(one))
    return pd.concat([one, rest], ignore_index=True)


def select_threshold_batch(pool, evidence, *, batch_size=500, exclude_uids=None, seed=RNG_SEED):
    rng = random.Random(seed)
    exclude = set(exclude_uids or set()) | set(evidence.get("pair_uid", []))
    x = pool[(pool["phash_distance"] <= 4) & ~pool["pair_uid"].isin(exclude)].copy()
    h = x["sampling_ssim_hint"].fillna(-1)
    strata = [
        ("boundary_088_096", (h >= .88) & (h < .96), 180),
        ("near_low_080_088", (h >= .80) & (h < .88), 80),
        ("high_096_099", (h >= .96) & (h < .99), 90),
        ("exactish_099_100", h >= .99, 80),
        ("low_control_lt080", h < .80, 70),
    ]
    chosen, used = [], set()
    for name, mask, quota in strata:
        part = x[mask & ~x["pair_uid"].isin(used)].copy()
        take = _unique_e3_greedy(part, min(quota, batch_size-sum(len(c) for c in chosen)), rng)
        if len(take):
            take["sampling_stratum"] = name
            chosen.append(take)
            used |= set(take["pair_uid"])
        if sum(len(c) for c in chosen) >= batch_size:
            break
    out = pd.concat(chosen, ignore_index=True) if chosen else x.iloc[0:0].copy()
    if len(out) < batch_size:
        rest = x[~x["pair_uid"].isin(out["pair_uid"])]
        fill = _unique_e3_greedy(rest, batch_size-len(out), rng)
        fill["sampling_stratum"] = "fallback"
        out = pd.concat([out, fill], ignore_index=True)
    return out.head(batch_size).reset_index(drop=True)


def select_retrieval_audit_batch(pool, evidence, *, batch_size=300, exclude_uids=None, max_phash=10, seed=RNG_SEED+777):
    rng = random.Random(seed)
    exclude = set(exclude_uids or set()) | set(evidence.get("pair_uid", []))
    x = pool[
        (pool["phash_distance"] > 4)
        & (pool["phash_distance"] <= max_phash)
        & ~pool["pair_uid"].isin(exclude)
    ].copy()
    x["_hint"] = x["sampling_ssim_hint"].fillna(-1)
    chunks = []
    per_band = max(1, batch_size // max(1, x["phash_distance"].nunique()))
    for dist, g in x.groupby("phash_distance"):
        g = g.sort_values("_hint", ascending=False)
        top = g.head(max(per_band*3, per_band))
        take = _unique_e3_greedy(top, min(per_band, len(top)), rng)
        take["sampling_stratum"] = f"retrieval_phash_{int(dist)}"
        chunks.append(take)
    out = pd.concat(chunks, ignore_index=True) if chunks else x.iloc[0:0].copy()
    if len(out) < batch_size:
        rest = x[~x["pair_uid"].isin(out["pair_uid"])].sort_values("_hint", ascending=False)
        fill = _unique_e3_greedy(rest, batch_size-len(out), rng)
        fill["sampling_stratum"] = "retrieval_fallback"
        out = pd.concat([out, fill], ignore_index=True)
    return out.drop(columns=["_hint"], errors="ignore").head(batch_size).reset_index(drop=True)


def render_review_batch(selected, *, e3_root, output_dir, batch_name, polyvore_dataset_name="codewaly/polyvore1000"):
    e3_root, output_dir = Path(e3_root), Path(output_dir)
    batch_dir = output_dir / batch_name
    preview_dir = batch_dir / "previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    poly = load_polyvore_dataset(polyvore_dataset_name)
    blind_rows, key_rows = [], []

    for i, r in enumerate(selected.to_dict("records"), 1):
        pid = f"{batch_name.upper()}_{i:04d}"
        preview_name = f"{pid}.jpg"
        out_path = preview_dir / preview_name
        status, fresh_ssim = "ok", np.nan
        try:
            left_raw = open_eval_image(e3_root, r["eval3_rel_path"])
            right_raw = ensure_pil(poly[int(r["polyvore_row_idx"])]["image"])
            fresh_ssim = rgb_ssim(left_raw, right_raw)
            left, right = fit_square(left_raw), fit_square(right_raw)
            gap, header = 12, 56
            canvas = Image.new("RGB", (PREVIEW_SIZE*2+gap, PREVIEW_SIZE+header), "white")
            canvas.paste(left, (0, header)); canvas.paste(right, (PREVIEW_SIZE+gap, header))
            draw = ImageDraw.Draw(canvas); font = ImageFont.load_default()
            draw.text((8,8), f"{pid} | LEFT=EVALUATION3 | RIGHT=POLYVORE", fill="black", font=font)
            draw.text((8,30), "Human label from image only; metrics are hidden.", fill="black", font=font)
            canvas.save(out_path, quality=92)
        except Exception as e:
            status = f"{type(e).__name__}: {e}"

        blind_rows.append({"pair_id": pid, "preview_file": f"previews/{preview_name}" if out_path.exists() else "", "human_label": "", "notes": ""})
        key_rows.append({
            "pair_id": pid, "pair_uid": r["pair_uid"],
            "eval3_rel_path": r["eval3_rel_path"], "polyvore_item_id": r["polyvore_item_id"],
            "polyvore_row_idx": int(r["polyvore_row_idx"]), "phash_distance": int(r["phash_distance"]),
            "sampling_ssim_hint": r.get("sampling_ssim_hint", np.nan), "rgb_ssim_recomputed": fresh_ssim,
            "sampling_stratum": r.get("sampling_stratum", ""), "candidate_source": r.get("candidate_source", ""),
            "preview_file": f"previews/{preview_name}" if out_path.exists() else "", "render_status": status,
        })

    blind, key = pd.DataFrame(blind_rows), pd.DataFrame(key_rows)
    blind_path = batch_dir / f"{batch_name}_BLIND.csv"
    blind_xlsx = batch_dir / f"{batch_name}_BLIND.xlsx"
    key_path = batch_dir / f"{batch_name}_KEY.csv"
    blind.to_csv(blind_path, index=False); key.to_csv(key_path, index=False)

    try:
        from openpyxl import Workbook
        from openpyxl.worksheet.datavalidation import DataValidation
        wb = Workbook(); ws = wb.active; ws.title = "review"
        ws.append(["pair_id", "preview_file", "human_label", "notes"])
        for r in blind.itertuples(index=False):
            ws.append([r.pair_id, r.preview_file, "", ""])
        dv = DataValidation(type="list", formula1='"DUPLICATE,NON_DUPLICATE"', allow_blank=True)
        ws.add_data_validation(dv)
        if len(blind): dv.add(f"C2:C{len(blind)+1}")
        ws.freeze_panes = "A2"
        ws.column_dimensions["A"].width = 28; ws.column_dimensions["B"].width = 55
        ws.column_dimensions["C"].width = 22; ws.column_dimensions["D"].width = 40
        wb.save(blind_xlsx)
    except Exception as e:
        print(f"[warn] Could not write XLSX review sheet: {e}")

    return {"batch_dir": batch_dir, "blind": blind_path, "blind_xlsx": blind_xlsx, "key": key_path, "previews": preview_dir}


def generate_batch(
    *, threshold_root, e3_root, output_root, batch_number=1, batch_size=500,
    hard_review_labels=None, extra_evidence_csvs=(), mode="threshold", retrieval_max_phash=10,
):
    threshold_root, output_root = Path(threshold_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pool = load_candidate_pool(threshold_root)
    evidence = build_human_evidence(threshold_root, hard_review_labels=hard_review_labels, extra_evidence_csvs=extra_evidence_csvs)
    evidence.to_csv(output_root / "human_evidence_registry.csv", index=False)

    generated = set()
    for p in output_root.rglob("*_KEY.csv"):
        try:
            x = pd.read_csv(p, usecols=["pair_uid"])
            generated |= set(x["pair_uid"].dropna().astype(str))
        except Exception:
            pass

    if mode == "threshold":
        selected = select_threshold_batch(pool, evidence, batch_size=batch_size, exclude_uids=generated, seed=RNG_SEED + batch_number)
        batch_name = f"threshold_batch_{batch_number:02d}"
    elif mode == "retrieval":
        selected = select_retrieval_audit_batch(pool, evidence, batch_size=batch_size, exclude_uids=generated, max_phash=retrieval_max_phash, seed=RNG_SEED + 1000 + batch_number)
        batch_name = f"retrieval_audit_{batch_number:02d}"
    else:
        raise ValueError("mode must be 'threshold' or 'retrieval'")

    paths = render_review_batch(selected, e3_root=e3_root, output_dir=output_root, batch_name=batch_name)
    report = {
        "batch_name": batch_name, "requested": int(batch_size), "selected": int(len(selected)), "mode": mode,
        "phash_counts": {str(int(k)): int(v) for k,v in selected["phash_distance"].value_counts().sort_index().items()},
        "sampling_strata": selected["sampling_stratum"].value_counts().to_dict(),
        "human_evidence": evidence_retrieval_report(evidence),
        "paths": {k: str(v) for k,v in paths.items()},
    }
    (Path(paths["batch_dir"]) / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def load_labeled_batch(batch_dir: Path | str) -> pd.DataFrame:
    d = Path(batch_dir)
    keys = list(d.glob("*_KEY.csv"))
    if len(keys) != 1:
        raise FileNotFoundError(f"Expected exactly one *_KEY.csv under {d}")
    key = pd.read_csv(keys[0])
    xlsx, csvs = list(d.glob("*_BLIND.xlsx")), list(d.glob("*_BLIND.csv"))
    labels = pd.read_excel(xlsx[0]) if xlsx else pd.read_csv(csvs[0])
    labels["human_label_original"] = labels["human_label"].astype(str).str.strip().str.upper()
    labels["human_label"] = labels["human_label_original"].map(normalize_label)
    labels = labels[labels["human_label"].notna()].copy()
    return key.merge(labels[["pair_id","human_label","human_label_original","notes"]], on="pair_id", how="inner", validate="one_to_one")


def collect_new_human_labels(output_root: Path | str, prefix: str = "threshold_batch_") -> pd.DataFrame:
    root = Path(output_root); frames = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)):
        try:
            x = load_labeled_batch(d); x["review_batch"] = d.name; frames.append(x)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def threshold_diagnostics(labeled: pd.DataFrame, *, phash_max: int = 4) -> pd.DataFrame:
    x = labeled.copy()
    x = x[(pd.to_numeric(x["phash_distance"], errors="coerce") <= phash_max) & x["human_label"].isin({DUPLICATE, NON_DUPLICATE})]
    y = x["human_label"].eq(DUPLICATE).to_numpy()
    s = pd.to_numeric(x["rgb_ssim_recomputed"], errors="coerce").to_numpy()
    ok = np.isfinite(s); y, s = y[ok], s[ok]
    rows = []
    for t in np.arange(0.80, 0.991, 0.005):
        pred = s >= t
        tp, fp = int(np.sum(pred & y)), int(np.sum(pred & ~y))
        fn, tn = int(np.sum(~pred & y)), int(np.sum(~pred & ~y))
        rows.append({
            "ssim_threshold": round(float(t), 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "duplicate_precision": tp/max(1,tp+fp),
            "duplicate_recall_on_review_sample": tp/max(1,tp+fn),
            "nonduplicate_specificity_on_review_sample": tn/max(1,tn+fp),
        })
    return pd.DataFrame(rows)


def labeled_retrieval_report(output_root: Path | str) -> pd.DataFrame:
    x = collect_new_human_labels(output_root, prefix="retrieval_audit_")
    if x.empty:
        return pd.DataFrame()
    x["phash_distance"] = pd.to_numeric(x["phash_distance"], errors="coerce")
    return x.groupby(["phash_distance","human_label"], dropna=False).size().reset_index(name="count").sort_values(["phash_distance","human_label"])
