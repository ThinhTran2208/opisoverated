# -*- coding: utf-8 -*-
"""Grouped cross-validation helpers for the S3.1 paired-ranking scorer.

This module is experimental and deliberately does not change the frozen V5
canonical path.  It supports a two-stage protocol:

1) choose ranking_weight on TRAIN only with grouped K-fold CV;
2) freeze that weight, then confirm it on canonical TRAIN/VALID with seeds
   42/43/44.

Fold grouping keeps every positive/paired-negative family together and also
keeps all families sharing the same source_kit_id in the same fold.  TEST is
never involved.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from functools import partial
from typing import Mapping, Sequence

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # keep source importable in lightweight CI
    torch = None
    DataLoader = None

from .dataset import collate_scorer_batch


class FamilySubset:
    """Subset a scorer dataset by complete pair families with remapped indices."""

    def __init__(self, base_dataset, family_positions: Sequence[int]) -> None:
        positions = [int(x) for x in family_positions]
        if not positions:
            raise ValueError("family_positions must be non-empty")
        if len(positions) != len(set(positions)):
            raise ValueError("family_positions contains duplicates")

        base_families = list(getattr(base_dataset, "pair_families", []))
        if not base_families:
            raise ValueError("base_dataset must expose non-empty pair_families")
        if min(positions) < 0 or max(positions) >= len(base_families):
            raise IndexError("family position outside base_dataset.pair_families")

        self.base_dataset = base_dataset
        self.family_positions = tuple(positions)
        self.base_indices: list[int] = []
        self.pair_families: list[tuple[int, int]] = []

        for position in self.family_positions:
            positive_index, negative_index = base_families[position]
            local_positive = len(self.base_indices)
            self.base_indices.append(int(positive_index))
            local_negative = len(self.base_indices)
            self.base_indices.append(int(negative_index))
            self.pair_families.append((local_positive, local_negative))

        if len(self.base_indices) != 2 * len(self.family_positions):
            raise RuntimeError("family subset did not preserve 1-positive/1-negative pairs")
        if len(self.base_indices) != len(set(self.base_indices)):
            raise ValueError("selected families reuse base dataset rows")

    def __len__(self) -> int:
        return len(self.base_indices)

    def __getitem__(self, index: int):
        return self.base_dataset[self.base_indices[index]]


def _family_group_key(dataset, family_position: int) -> str:
    """Return leakage-safe group key for one positive/negative family."""

    families = dataset.pair_families
    positive_index, negative_index = families[family_position]
    records = getattr(dataset, "records", None)
    if not isinstance(records, list):
        return f"family:{family_position}"

    positive = records[positive_index]
    negative = records[negative_index]
    positive_source = str(positive.get("source_kit_id", "")).strip()
    negative_source = str(negative.get("source_kit_id", "")).strip()

    # If source_kit_id is present, both rows of the family should agree.
    if positive_source and negative_source and positive_source != negative_source:
        raise ValueError(
            "paired family has inconsistent source_kit_id: "
            f"{positive_source!r} vs {negative_source!r}"
        )
    source = positive_source or negative_source
    if source:
        return f"source_kit:{source}"

    # Fallback remains family-safe even if source_kit_id is absent.
    sample_id = str(positive.get("sample_id", "")).strip()
    return f"positive:{sample_id or family_position}"


def build_grouped_family_folds(
    dataset,
    *,
    n_splits: int = 3,
    split_seed: int = 20260829,
) -> list[dict[str, object]]:
    """Create deterministic, source-kit-grouped folds over complete families.

    The split unit is a source-kit group containing one or more complete
    positive/negative families.  Groups are greedily balanced by family count.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    families = list(getattr(dataset, "pair_families", []))
    if len(families) < n_splits:
        raise ValueError("not enough pair families for requested folds")

    grouped: dict[str, list[int]] = defaultdict(list)
    for family_position in range(len(families)):
        grouped[_family_group_key(dataset, family_position)].append(family_position)
    if len(grouped) < n_splits:
        raise ValueError("not enough source-kit groups for requested folds")

    rng = random.Random(int(split_seed))
    group_items = list(grouped.items())
    rng.shuffle(group_items)
    # Stable sort after shuffle randomizes ties while placing large groups first.
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_families: list[list[int]] = [[] for _ in range(n_splits)]
    fold_counts = [0] * n_splits

    for group_key, family_positions in group_items:
        fold_index = min(range(n_splits), key=lambda i: (fold_counts[i], i))
        fold_groups[fold_index].append(group_key)
        fold_families[fold_index].extend(family_positions)
        fold_counts[fold_index] += len(family_positions)

    all_family_positions = set(range(len(families)))
    folds: list[dict[str, object]] = []
    seen_valid: set[int] = set()
    for fold_index in range(n_splits):
        valid_positions = sorted(fold_families[fold_index])
        train_positions = sorted(all_family_positions - set(valid_positions))
        if not valid_positions or not train_positions:
            raise RuntimeError("grouped fold produced empty train or validation split")
        if set(train_positions) & set(valid_positions):
            raise RuntimeError("family leakage between train and validation")
        if seen_valid & set(valid_positions):
            raise RuntimeError("validation families appear in multiple folds")
        seen_valid.update(valid_positions)

        train_groups = {
            _family_group_key(dataset, position) for position in train_positions
        }
        valid_groups = {
            _family_group_key(dataset, position) for position in valid_positions
        }
        if train_groups & valid_groups:
            raise RuntimeError("source_kit leakage between train and validation")

        folds.append(
            {
                "fold": fold_index + 1,
                "train_family_positions": train_positions,
                "valid_family_positions": valid_positions,
                "train_family_count": len(train_positions),
                "valid_family_count": len(valid_positions),
                "train_sample_count": 2 * len(train_positions),
                "valid_sample_count": 2 * len(valid_positions),
                "train_group_count": len(train_groups),
                "valid_group_count": len(valid_groups),
            }
        )

    if seen_valid != all_family_positions:
        raise RuntimeError("grouped K-fold does not cover every family exactly once")
    return folds


def build_fold_datasets(dataset, fold: Mapping[str, object]):
    """Materialize train/validation FamilySubset objects for one fold spec."""

    return (
        FamilySubset(dataset, fold["train_family_positions"]),
        FamilySubset(dataset, fold["valid_family_positions"]),
    )


def build_nonshuffled_loader(
    dataset,
    config: Mapping[str, object],
    *,
    num_workers: int = 0,
):
    """Build an ordinary FP32 evaluation loader for a family subset."""

    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for CV loaders")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    training = config["training"]
    data = config["data"]
    batch_size = int(training["batch_size"])
    max_items = int(data["max_items"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(torch.cuda.is_available()),
        collate_fn=partial(collate_scorer_batch, max_items=max_items),
        drop_last=False,
    )


def summarize_lambda_cv(rows: Sequence[Mapping[str, object]]) -> list[dict[str, float | int]]:
    """Aggregate fold metrics for each ranking weight without hidden tie-breaks."""

    by_weight: dict[float, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_weight[float(row["ranking_weight"])].append(row)

    summary: list[dict[str, float | int]] = []
    for weight in sorted(by_weight):
        group = by_weight[weight]
        aucs = [float(row["valid_roc_auc"]) for row in group]
        fitbs = [float(row["valid_fitb_2way"]) for row in group]
        margins = [float(row["mean_logit_margin"]) for row in group]
        if len(group) < 2:
            auc_std = fitb_std = margin_std = 0.0
        else:
            def sample_std(values: Sequence[float]) -> float:
                mean = sum(values) / len(values)
                return math.sqrt(
                    sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                )
            auc_std = sample_std(aucs)
            fitb_std = sample_std(fitbs)
            margin_std = sample_std(margins)

        summary.append(
            {
                "ranking_weight": weight,
                "fold_count": len(group),
                "mean_roc_auc": sum(aucs) / len(aucs),
                "std_roc_auc": auc_std,
                "mean_fitb_2way": sum(fitbs) / len(fitbs),
                "std_fitb_2way": fitb_std,
                "mean_logit_margin": sum(margins) / len(margins),
                "std_logit_margin": margin_std,
            }
        )
    return summary


def select_lambda_by_mean_auc(summary: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Select largest mean ROC-AUC; exact ties prefer the smaller lambda."""

    if not summary:
        raise ValueError("CV summary is empty")
    winner = max(
        summary,
        key=lambda row: (float(row["mean_roc_auc"]), -float(row["ranking_weight"])),
    )
    return dict(winner)
