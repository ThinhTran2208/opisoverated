# -*- coding: utf-8 -*-
"""Evaluation helpers for Scorer V1.

S1 can validate metrics with synthetic predictions before the neural model is
implemented. Once S2 exists, ``evaluate_model`` applies the same metric path to
real model outputs without changing the evaluator contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, Sequence

from .metrics import compute_scorer_metrics

try:
    import torch
except ModuleNotFoundError:  # Keep prediction-only evaluation importable in CI.
    torch = None


def evaluate_predictions(
    *,
    sample_ids: Sequence[str],
    paired_positive_sample_ids: Sequence[str | None],
    labels: Sequence[int | float],
    logits: Sequence[int | float],
) -> dict[str, float | int]:
    """Evaluate already-computed raw compatibility logits."""

    return compute_scorer_metrics(
        sample_ids,
        paired_positive_sample_ids,
        labels,
        logits,
    )


def evaluate_model(model, dataloader: Iterable[Mapping[str, object]], *, device=None):
    """Run a scorer model over a DataLoader and return metrics + predictions.

    The function assumes the locked forward contract:

    ``model(item_embeddings=..., coarse_category_ids=..., item_mask=...)``
    -> ``{"compatibility_logit": Tensor[B]}``.
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for model evaluation")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    sample_ids: list[str] = []
    paired_positive_sample_ids: list[str | None] = []
    labels: list[float] = []
    logits: list[float] = []

    was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            output = model(
                item_embeddings=batch["item_embeddings"].to(device),
                coarse_category_ids=batch["coarse_category_ids"].to(device),
                item_mask=batch["item_mask"].to(device),
            )
            if not isinstance(output, Mapping) or "compatibility_logit" not in output:
                raise ValueError(
                    "Scorer output must be a mapping containing compatibility_logit"
                )
            batch_logits = output["compatibility_logit"]
            if not isinstance(batch_logits, torch.Tensor) or batch_logits.ndim != 1:
                raise ValueError("compatibility_logit must be a rank-1 Tensor [B]")
            if batch_logits.shape[0] != len(batch["sample_ids"]):
                raise ValueError("compatibility_logit batch size mismatch")
            if not torch.isfinite(batch_logits).all():
                raise ValueError("compatibility_logit contains NaN/Inf")

            sample_ids.extend(str(value) for value in batch["sample_ids"])
            paired_positive_sample_ids.extend(batch["paired_positive_sample_ids"])
            labels.extend(float(value) for value in batch["labels"].detach().cpu().tolist())
            logits.extend(float(value) for value in batch_logits.detach().cpu().tolist())

    if was_training:
        model.train()

    metrics = compute_scorer_metrics(
        sample_ids,
        paired_positive_sample_ids,
        labels,
        logits,
    )
    predictions = [
        {
            "sample_id": sample_id,
            "paired_positive_sample_id": pair_id,
            "label": label,
            "compatibility_logit": logit,
        }
        for sample_id, pair_id, label, logit in zip(
            sample_ids, paired_positive_sample_ids, labels, logits
        )
    ]
    return {"metrics": metrics, "predictions": predictions}
