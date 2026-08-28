# -*- coding: utf-8 -*-
"""Pure metric logic for the Scorer V1 evaluation contract."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Sequence


def _validate_prediction_vectors(
    sample_ids: Sequence[str],
    paired_positive_sample_ids: Sequence[str | None],
    labels: Sequence[int | float],
    logits: Sequence[int | float],
) -> tuple[list[str], list[str | None], list[int], list[float]]:
    lengths = {
        len(sample_ids),
        len(paired_positive_sample_ids),
        len(labels),
        len(logits),
    }
    if len(lengths) != 1:
        raise ValueError("Prediction vectors must have equal length")
    if not sample_ids:
        raise ValueError("Prediction vectors are empty")

    normalized_ids = [str(sample_id).strip() for sample_id in sample_ids]
    if any(not sample_id for sample_id in normalized_ids):
        raise ValueError("sample_ids must be non-empty strings")
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("sample_ids must be unique")

    normalized_pairs: list[str | None] = []
    normalized_labels: list[int] = []
    normalized_logits: list[float] = []
    for pair_id, label, logit in zip(
        paired_positive_sample_ids, labels, logits
    ):
        normalized_label = int(label)
        if normalized_label not in (0, 1) or float(label) != normalized_label:
            raise ValueError(f"labels must be binary 0/1, got {label!r}")
        normalized_logit = float(logit)
        if not math.isfinite(normalized_logit):
            raise ValueError(f"logits must be finite, got {logit!r}")
        normalized_pair = None if pair_id in (None, "") else str(pair_id).strip()
        normalized_pairs.append(normalized_pair)
        normalized_labels.append(normalized_label)
        normalized_logits.append(normalized_logit)

    return normalized_ids, normalized_pairs, normalized_labels, normalized_logits


def roc_auc(labels: Sequence[int | float], logits: Sequence[int | float]) -> float:
    """Compute binary ROC-AUC from raw logits, including average ranks for ties."""

    if len(labels) != len(logits) or not labels:
        raise ValueError("labels and logits must be non-empty and have equal length")

    normalized_labels: list[int] = []
    pairs: list[tuple[float, int]] = []
    for label, logit in zip(labels, logits):
        normalized_label = int(label)
        if normalized_label not in (0, 1) or float(label) != normalized_label:
            raise ValueError(f"labels must be binary 0/1, got {label!r}")
        score = float(logit)
        if not math.isfinite(score):
            raise ValueError(f"logits must be finite, got {logit!r}")
        normalized_labels.append(normalized_label)
        pairs.append((score, normalized_label))

    positive_count = sum(normalized_labels)
    negative_count = len(normalized_labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both positive and negative labels")

    pairs.sort(key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in pairs[index:end]
        )
        index = end

    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def paired_logit_margins(
    sample_ids: Sequence[str],
    paired_positive_sample_ids: Sequence[str | None],
    labels: Sequence[int | float],
    logits: Sequence[int | float],
) -> list[float]:
    """Return one ``positive_logit - negative_logit`` margin per frozen pair."""

    ids, pair_ids, normalized_labels, normalized_logits = _validate_prediction_vectors(
        sample_ids,
        paired_positive_sample_ids,
        labels,
        logits,
    )

    positives: dict[str, float] = {}
    negatives_by_positive: dict[str, list[float]] = defaultdict(list)

    for sample_id, pair_id, label, logit in zip(
        ids, pair_ids, normalized_labels, normalized_logits
    ):
        if label == 1:
            if pair_id is not None:
                raise ValueError(
                    f"Positive {sample_id} must have null paired_positive_sample_id"
                )
            positives[sample_id] = logit
        else:
            if pair_id is None:
                raise ValueError(f"Negative {sample_id} is missing paired positive ID")
            negatives_by_positive[pair_id].append(logit)

    orphan_ids = sorted(set(negatives_by_positive) - set(positives))
    if orphan_ids:
        raise ValueError(f"Negatives reference missing positives: {orphan_ids[:10]}")

    if set(positives) != set(negatives_by_positive):
        missing = sorted(set(positives) - set(negatives_by_positive))
        raise ValueError(f"Positives missing paired negatives: {missing[:10]}")

    margins: list[float] = []
    for positive_id, positive_logit in positives.items():
        negatives = negatives_by_positive[positive_id]
        if len(negatives) != 1:
            raise ValueError(
                f"Positive {positive_id} must have exactly one negative, "
                f"found {len(negatives)}"
            )
        margins.append(positive_logit - negatives[0])

    if not margins:
        raise ValueError("No complete positive/negative pair families found")
    return margins


def compute_scorer_metrics(
    sample_ids: Sequence[str],
    paired_positive_sample_ids: Sequence[str | None],
    labels: Sequence[int | float],
    logits: Sequence[int | float],
) -> dict[str, float | int]:
    """Compute the canonical Scorer V1 metrics from raw compatibility logits."""

    ids, pair_ids, normalized_labels, normalized_logits = _validate_prediction_vectors(
        sample_ids,
        paired_positive_sample_ids,
        labels,
        logits,
    )
    margins = paired_logit_margins(ids, pair_ids, normalized_labels, normalized_logits)
    return {
        "roc_auc": roc_auc(normalized_labels, normalized_logits),
        "fitb_2way": sum(margin > 0.0 for margin in margins) / len(margins),
        "mean_logit_margin": statistics.fmean(margins),
        "median_logit_margin": statistics.median(margins),
        "sample_count": len(ids),
        "paired_family_count": len(margins),
    }
