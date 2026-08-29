# -*- coding: utf-8 -*-
"""LOO diagnosis evaluation for the MIN2 scorer experiment.

The compatibility scorer may consume 2-item outfits, but LOO localization only
starts from original outfits with at least 3 items. Evaluation is performed on
synthetic negative samples because ``negative_metadata.swapped_item_index`` is
the project ground truth for the problematic item.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from .loo import LOO_MIN_ORIGINAL_ITEMS, build_leave_one_out_outfits
from src.scorer.min2_experiment import collate_min2_scorer_batch

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for LOO evaluation")


def _score_batch(model, batch, device):
    with torch.no_grad():
        output = model(
            item_embeddings=batch["item_embeddings"].to(device),
            coarse_category_ids=batch["coarse_category_ids"].to(device),
            item_mask=batch["item_mask"].to(device),
            pair_mask=batch["pair_mask"].to(device),
        )
    logits = output["compatibility_logit"]
    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise ValueError("LOO scorer returned invalid compatibility logits")
    return logits.detach().cpu()


def evaluate_loo_dataset(model, dataset, *, device=None) -> dict[str, object]:
    """Compute project LOO Top-1 and Hit@2 on eligible synthetic negatives.

    For each negative outfit O, the evaluator computes C(O), then scores every
    leave-one-out residual and uses

        delta_i = C(O \\ x_i) - C(O)

    to rank candidate problematic items. Exact ties are broken deterministically
    by the lower item index and are also counted for diagnostics.
    """

    _require_torch()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    was_training = bool(getattr(model, "training", False))
    model.eval()

    predictions: list[dict[str, object]] = []
    skipped_original_size_2 = 0

    for dataset_index, record in enumerate(dataset.records):
        if int(record.get("label", -1)) != 0:
            continue

        base = dataset[dataset_index]
        item_count = len(base["item_ids"])
        if item_count < LOO_MIN_ORIGINAL_ITEMS:
            skipped_original_size_2 += 1
            continue

        metadata = base.get("negative_metadata")
        if not isinstance(metadata, Mapping) or "swapped_item_index" not in metadata:
            raise ValueError(
                f"Negative sample {base['sample_id']} is missing swapped_item_index"
            )
        gt_index = int(metadata["swapped_item_index"])
        if not 0 <= gt_index < item_count:
            raise ValueError(
                f"Invalid swapped_item_index={gt_index} for {base['sample_id']}"
            )

        original_batch = collate_min2_scorer_batch([base])
        original_logit = float(_score_batch(model, original_batch, device)[0])

        residual_index_sets = build_leave_one_out_outfits(list(range(item_count)))
        residual_samples = []
        for removed_index, kept_indices in enumerate(residual_index_sets):
            residual_samples.append(
                {
                    "sample_id": f"{base['sample_id']}__loo_remove_{removed_index}",
                    "source_kit_id": base["source_kit_id"],
                    "paired_positive_sample_id": None,
                    "item_ids": [base["item_ids"][i] for i in kept_indices],
                    "item_embeddings": base["item_embeddings"][kept_indices],
                    "coarse_category_ids": base["coarse_category_ids"][kept_indices],
                    "label": 0.0,
                    "negative_metadata": None,
                }
            )

        residual_batch = collate_min2_scorer_batch(residual_samples)
        residual_logits = _score_batch(model, residual_batch, device).tolist()
        deltas = [float(logit - original_logit) for logit in residual_logits]

        ranked_indices = sorted(range(item_count), key=lambda i: (-deltas[i], i))
        predicted_index = ranked_indices[0]
        top2_indices = ranked_indices[: min(2, item_count)]
        max_delta = max(deltas)
        max_tie_count = sum(abs(delta - max_delta) <= 1e-12 for delta in deltas)

        predictions.append(
            {
                "sample_id": base["sample_id"],
                "source_kit_id": base["source_kit_id"],
                "outfit_length": item_count,
                "gt_swapped_item_index": gt_index,
                "predicted_problematic_index": predicted_index,
                "top2_indices": top2_indices,
                "top1_correct": int(predicted_index == gt_index),
                "hit_at_2": int(gt_index in top2_indices),
                "original_logit": original_logit,
                "gt_delta": deltas[gt_index],
                "predicted_top1_delta": deltas[predicted_index],
                "max_tie_count": max_tie_count,
                "loo_deltas": deltas,
            }
        )

    if was_training:
        model.train()
    if not predictions:
        raise RuntimeError("LOO evaluation found zero eligible synthetic negatives")

    count = len(predictions)
    metrics = {
        "eligible_negative_count": count,
        "skipped_original_size_2": skipped_original_size_2,
        "loo_top1_localization_accuracy": sum(
            int(row["top1_correct"]) for row in predictions
        ) / count,
        "loo_hit_at_2": sum(int(row["hit_at_2"]) for row in predictions) / count,
        "mean_gt_delta": sum(float(row["gt_delta"]) for row in predictions) / count,
        "mean_predicted_top1_delta": sum(
            float(row["predicted_top1_delta"]) for row in predictions
        ) / count,
        "max_delta_tie_sample_count": sum(
            int(row["max_tie_count"]) > 1 for row in predictions
        ),
    }

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        grouped[int(row["outfit_length"])].append(row)

    by_length = []
    for outfit_length in sorted(grouped):
        rows = grouped[outfit_length]
        n = len(rows)
        by_length.append(
            {
                "outfit_length": outfit_length,
                "eligible_negative_count": n,
                "loo_top1_localization_accuracy": sum(
                    int(row["top1_correct"]) for row in rows
                ) / n,
                "loo_hit_at_2": sum(int(row["hit_at_2"]) for row in rows) / n,
                "mean_gt_delta": sum(float(row["gt_delta"]) for row in rows) / n,
                "tie_sample_count": sum(
                    int(row["max_tie_count"]) > 1 for row in rows
                ),
            }
        )

    return {
        "metrics": metrics,
        "by_length": by_length,
        "predictions": predictions,
    }
