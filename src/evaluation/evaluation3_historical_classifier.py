# -*- coding: utf-8 -*-
"""Historical human-label pilot for EVALUATION3 duplicate classification.

This module intentionally reuses human labels that already exist in the
`phash_ssim_threshold` review workspace:

* confirmed duplicate review rows (positive ground truth), and
* hard-negative review rows (mostly NON_DUPLICATE, with human-confirmed
  DUPLICATE and SAME_PRODUCT_DIFFERENT_IMAGE cases).

For this project, SAME_PRODUCT_DIFFERENT_IMAGE is treated as DUPLICATE because
those examples were reviewed as the same visual item with only a very small
lighting/viewpoint nuisance.

The split is group-disjoint by EVALUATION3 relative image path. This prevents
one E3 visual from leaking across train/validation/test through multiple
Polyvore candidate pairs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluation.evaluation3_active_learning import DUPLICATE, NON_DUPLICATE
from src.evaluation.evaluation3_preview_active_learning import extract_preview_features


SAME_PRODUCT_DIFFERENT_IMAGE = "SAME_PRODUCT_DIFFERENT_IMAGE"
VALID_HISTORICAL_LABELS = frozenset(
    {DUPLICATE, NON_DUPLICATE, SAME_PRODUCT_DIFFERENT_IMAGE}
)


def normalize_historical_label(value: object) -> str | None:
    label = str(value or "").strip().upper()
    if label in {DUPLICATE, SAME_PRODUCT_DIFFERENT_IMAGE}:
        return DUPLICATE
    if label == NON_DUPLICATE:
        return NON_DUPLICATE
    return None


def prepare_historical_metadata(
    *,
    historical_root: Path | str,
    hard_review_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Build one clean metadata table from the two historical review sources.

    The confirmed-positive CSV contains repeated E3 visuals because one E3
    image can have multiple known duplicate Polyvore rows. We reproduce the old
    calibration dedup policy: one positive per E3 path, then exact E3 pHash
    dedup. Human labels remain the source of truth.
    """

    root = Path(historical_root)
    positive_csv = root / "confirmed_duplicates_dedup.csv"
    hard_key_csv = root / "hard_negative_review_KEY.csv"
    if not positive_csv.is_file():
        raise FileNotFoundError(positive_csv)
    if not hard_key_csv.is_file():
        raise FileNotFoundError(hard_key_csv)

    positive = pd.read_csv(positive_csv).copy()
    required_positive = {
        "pair_id",
        "eval3_rel_path",
        "polyvore_item_id",
        "preview_file",
        "eval3_phash_hex",
    }
    missing = sorted(required_positive - set(positive.columns))
    if missing:
        raise KeyError(f"confirmed_duplicates_dedup.csv missing {missing}")

    # Match the historical report: 400 -> one/E3 path -> exact E3 pHash dedup.
    positive = positive.drop_duplicates(subset=["eval3_rel_path"], keep="first")
    positive = positive.drop_duplicates(subset=["eval3_phash_hex"], keep="first")
    positive["human_label"] = DUPLICATE
    positive["source"] = "confirmed_duplicate"

    hard_key = pd.read_csv(hard_key_csv).copy()
    if "pair_id" not in hard_review_labels.columns or "human_label" not in hard_review_labels.columns:
        raise KeyError("hard_review_labels must contain pair_id and human_label")
    labels = hard_review_labels[["pair_id", "human_label"]].copy()
    labels["pair_id"] = labels["pair_id"].astype(str).str.strip()
    labels["human_label_original"] = labels["human_label"].astype(str).str.strip().str.upper()
    labels["human_label"] = labels["human_label_original"].map(normalize_historical_label)
    labels = labels[labels["human_label"].notna()].copy()

    hard_key["pair_id"] = hard_key["pair_id"].astype(str).str.strip()
    hard = hard_key.merge(
        labels[["pair_id", "human_label", "human_label_original"]],
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    hard["source"] = "hard_negative_review"

    keep = [
        "pair_id",
        "eval3_rel_path",
        "polyvore_item_id",
        "preview_file",
        "human_label",
        "source",
    ]
    if "human_label_original" not in positive.columns:
        positive["human_label_original"] = DUPLICATE
    keep_with_original = keep + ["human_label_original"]
    combined = pd.concat(
        [positive[keep_with_original], hard[keep_with_original]],
        ignore_index=True,
    )

    # Prefix IDs because the two historical queues use different namespaces.
    combined["source_pair_id"] = combined["pair_id"].astype(str)
    combined["example_id"] = combined["source"] + ":" + combined["source_pair_id"]
    combined["group_id"] = combined["eval3_rel_path"].astype(str)
    combined["target"] = combined["human_label"].map({NON_DUPLICATE: 0, DUPLICATE: 1}).astype(int)

    # Detect exact pair conflicts before dropping duplicates. A conflicting human
    # label is a data-quality problem and must not be silently resolved.
    pair_key = ["eval3_rel_path", "polyvore_item_id"]
    conflict_counts = (
        combined.groupby(pair_key, dropna=False)["target"].nunique().reset_index(name="n")
    )
    conflicts = conflict_counts[conflict_counts["n"] > 1]
    if not conflicts.empty:
        examples = conflicts.head(10).to_dict("records")
        raise ValueError(f"Conflicting human labels for exact pairs: {examples}")

    combined = combined.drop_duplicates(subset=pair_key + ["target"], keep="first")
    return combined.reset_index(drop=True)


def _resolve_preview_path(root: Path, row: pd.Series) -> Path:
    rel = str(row["preview_file"])
    path = root / rel
    if path.is_file():
        return path
    raise FileNotFoundError(path)


def build_historical_feature_dataset(
    metadata: pd.DataFrame,
    *,
    historical_root: Path | str,
    cache_csv: Path | str | None = None,
    workers: int = 4,
) -> pd.DataFrame:
    """Extract the same cheap preview features for all historical labels."""

    if cache_csv is not None:
        cache = Path(cache_csv)
        if cache.is_file():
            cached = pd.read_csv(cache)
            expected = set(metadata["example_id"].astype(str))
            observed = set(cached.get("example_id", pd.Series(dtype=str)).astype(str))
            if expected == observed:
                return cached

    root = Path(historical_root)

    def one(index: int, row: pd.Series) -> dict[str, object]:
        path = _resolve_preview_path(root, row)
        features = extract_preview_features(path)
        return {
            "_index": index,
            "example_id": str(row["example_id"]),
            "source_pair_id": str(row["source_pair_id"]),
            "source": str(row["source"]),
            "eval3_rel_path": str(row["eval3_rel_path"]),
            "polyvore_item_id": str(row["polyvore_item_id"]),
            "group_id": str(row["group_id"]),
            "preview_file": str(path),
            "human_label": str(row["human_label"]),
            "human_label_original": str(row["human_label_original"]),
            "target": int(row["target"]),
            **{k: v for k, v in features.items() if k not in {"pair_id", "preview_file"}},
        }

    records: list[dict[str, object]] = []
    worker_count = max(1, int(workers))
    if worker_count == 1:
        for index, row in metadata.iterrows():
            records.append(one(int(index), row))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(one, int(index), row): int(index)
                for index, row in metadata.iterrows()
            }
            for future in as_completed(futures):
                records.append(future.result())

    result = pd.DataFrame(records).sort_values("_index").drop(columns=["_index"]).reset_index(drop=True)
    if cache_csv is not None:
        cache = Path(cache_csv)
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache, index=False)
    return result


def _best_group_fold(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the group-stratified fold closest to global size/class ratio."""

    y = frame["target"].astype(int).to_numpy()
    groups = frame["group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(random_state)
    )
    overall_rate = float(np.mean(y))
    target_fraction = 1.0 / float(n_splits)
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    dummy = np.zeros((len(frame), 1), dtype=np.float32)
    for train_idx, hold_idx in splitter.split(dummy, y, groups):
        hold_y = y[hold_idx]
        fraction = len(hold_idx) / max(1, len(frame))
        rate = float(np.mean(hold_y)) if len(hold_y) else 0.0
        # Penalize missing-class folds heavily.
        missing_class_penalty = 10.0 if len(np.unique(hold_y)) < 2 else 0.0
        score = (
            abs(fraction - target_fraction)
            + abs(rate - overall_rate)
            + missing_class_penalty
        )
        candidates.append((score, train_idx, hold_idx))
    candidates.sort(key=lambda item: item[0])
    _, train_idx, hold_idx = candidates[0]
    return train_idx, hold_idx


def split_historical_dataset(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Approximately 60/20/20, stratified and group-disjoint by E3 image."""

    remain_idx, test_idx = _best_group_fold(
        frame, n_splits=5, random_state=random_state
    )
    remain = frame.iloc[remain_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)

    train_idx, val_idx = _best_group_fold(
        remain, n_splits=4, random_state=random_state + 1
    )
    train = remain.iloc[train_idx].reset_index(drop=True)
    validation = remain.iloc[val_idx].reset_index(drop=True)

    sets = {
        "train": set(train["group_id"].astype(str)),
        "validation": set(validation["group_id"].astype(str)),
        "test": set(test["group_id"].astype(str)),
    }
    if sets["train"] & sets["validation"]:
        raise AssertionError("train/validation group leakage")
    if sets["train"] & sets["test"]:
        raise AssertionError("train/test group leakage")
    if sets["validation"] & sets["test"]:
        raise AssertionError("validation/test group leakage")

    return train, validation, test


def split_summary(*frames: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in frames:
        counts = frame["human_label"].value_counts().to_dict()
        rows.append(
            {
                "split": name,
                "rows": len(frame),
                "groups": frame["group_id"].nunique(),
                "DUPLICATE": int(counts.get(DUPLICATE, 0)),
                "NON_DUPLICATE": int(counts.get(NON_DUPLICATE, 0)),
                "duplicate_rate": float(frame["target"].mean()) if len(frame) else float("nan"),
            }
        )
    return pd.DataFrame(rows)
