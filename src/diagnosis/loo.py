# -*- coding: utf-8 -*-
"""Batched Leave-One-Out diagnosis for the frozen pairwise scorer.

The canonical scorer remains a 3--8 item model.  A three-item outfit produces
two-item subsets under Leave-One-Out, so this module uses the model's explicit
eval-only ``diagnostic_min_items=2`` path.  Those rows are marked as
extrapolation in every returned record and should be reported separately by
original outfit size.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # Keep source importable in lightweight CI.
    torch = None

from src.scorer.dataset import build_pair_mask


LOO_PROTOCOL_VERSION = "loo-diagnostic-v1"
MIN_ORIGINAL_ITEMS = 3
DIAGNOSTIC_MIN_ITEMS = 2


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for Leave-One-Out diagnosis")


def _validate_original_outfit(
    item_embeddings,
    coarse_category_ids,
    *,
    max_items: int,
) -> int:
    require_torch()
    if not isinstance(item_embeddings, torch.Tensor) or item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must have shape [N, D]")
    if not isinstance(coarse_category_ids, torch.Tensor):
        raise TypeError("coarse_category_ids must be a torch.Tensor")
    if coarse_category_ids.ndim != 1:
        raise ValueError("coarse_category_ids must have shape [N]")
    if coarse_category_ids.dtype != torch.long:
        raise ValueError("coarse_category_ids must have dtype torch.long")
    if coarse_category_ids.device != item_embeddings.device:
        raise ValueError("embeddings and category IDs must be on the same device")
    if not item_embeddings.is_floating_point():
        raise ValueError("item_embeddings must be floating point")
    if not torch.isfinite(item_embeddings).all():
        raise ValueError("item_embeddings contains NaN or Inf")

    item_count = int(item_embeddings.shape[0])
    if coarse_category_ids.shape[0] != item_count:
        raise ValueError("category count does not match embedding row count")
    if not MIN_ORIGINAL_ITEMS <= item_count <= max_items:
        raise ValueError(
            f"Original outfit must contain [{MIN_ORIGINAL_ITEMS}, {max_items}] items"
        )
    if torch.any(coarse_category_ids <= 0):
        raise ValueError("Original outfit items may not use the PAD category ID")
    return item_count


def build_loo_variant_batch(
    item_embeddings,
    coarse_category_ids,
    *,
    max_items: int = 8,
) -> dict[str, object]:
    """Build ``full + remove-each-item`` rows padded to ``max_items``.

    Row zero is the untouched outfit.  Row ``i + 1`` is the outfit with item
    ``i`` removed.  No labels or ``swapped_item_index`` are used to construct
    the neural inputs.
    """

    item_count = _validate_original_outfit(
        item_embeddings,
        coarse_category_ids,
        max_items=max_items,
    )
    embedding_dim = int(item_embeddings.shape[1])
    variant_count = item_count + 1

    embeddings = torch.zeros(
        (variant_count, max_items, embedding_dim),
        dtype=item_embeddings.dtype,
        device=item_embeddings.device,
    )
    category_ids = torch.zeros(
        (variant_count, max_items),
        dtype=torch.long,
        device=coarse_category_ids.device,
    )
    item_mask = torch.zeros(
        (variant_count, max_items),
        dtype=torch.bool,
        device=item_embeddings.device,
    )

    embeddings[0, :item_count] = item_embeddings
    category_ids[0, :item_count] = coarse_category_ids
    item_mask[0, :item_count] = True

    all_indices = torch.arange(item_count, device=item_embeddings.device)
    for removed_index in range(item_count):
        kept = all_indices != removed_index
        kept_embeddings = item_embeddings[kept]
        kept_categories = coarse_category_ids[kept]
        kept_count = item_count - 1
        row = removed_index + 1
        embeddings[row, :kept_count] = kept_embeddings
        category_ids[row, :kept_count] = kept_categories
        item_mask[row, :kept_count] = True

    return {
        "item_embeddings": embeddings,
        "coarse_category_ids": category_ids,
        "item_mask": item_mask,
        "pair_mask": build_pair_mask(item_mask),
        "removed_item_indices": [None, *range(item_count)],
        "original_item_count": item_count,
        "uses_two_item_subsets": item_count == MIN_ORIGINAL_ITEMS,
    }


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _rank_deltas(deltas) -> list[int]:
    values = [float(value) for value in deltas.detach().cpu().tolist()]
    return sorted(range(len(values)), key=lambda index: (-values[index], index))


def diagnose_outfit(
    model,
    item_embeddings,
    coarse_category_ids,
    *,
    item_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run one batched LOO forward pass and return JSON-friendly evidence."""

    require_torch()
    if model.training:
        raise ValueError("LOO diagnosis requires model.eval()")

    variants = build_loo_variant_batch(
        item_embeddings,
        coarse_category_ids,
        max_items=int(model.max_items),
    )
    item_count = int(variants["original_item_count"])
    if item_ids is not None and len(item_ids) != item_count:
        raise ValueError("item_ids length does not match the original outfit")

    device = _model_device(model)
    model_inputs = {
        key: variants[key].to(device)
        for key in (
            "item_embeddings",
            "coarse_category_ids",
            "item_mask",
            "pair_mask",
        )
    }
    diagnostic_min_items = (
        DIAGNOSTIC_MIN_ITEMS if variants["uses_two_item_subsets"] else None
    )

    with torch.inference_mode():
        logits = model(
            **model_inputs,
            diagnostic_min_items=diagnostic_min_items,
        )["compatibility_logit"]

    full_logit = logits[0]
    without_item_logits = logits[1:]
    deltas = without_item_logits - full_logit
    ranking = _rank_deltas(deltas)

    return {
        "protocol_version": LOO_PROTOCOL_VERSION,
        "original_item_count": item_count,
        "full_logit": float(full_logit.detach().cpu()),
        "without_item_logits": [
            float(value) for value in without_item_logits.detach().cpu().tolist()
        ],
        "deltas_without_minus_full": [
            float(value) for value in deltas.detach().cpu().tolist()
        ],
        "ranked_item_indices": ranking,
        "problematic_item_index": ranking[0],
        "problematic_item_id": (
            str(item_ids[ranking[0]]) if item_ids is not None else None
        ),
        "uses_two_item_extrapolation": bool(variants["uses_two_item_subsets"]),
    }


def _summarize_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    count = len(records)
    if count == 0:
        return {
            "sample_count": 0,
            "top1_localization_accuracy": None,
            "hit_at_2": None,
            "mean_target_delta": None,
            "mean_max_delta": None,
            "two_item_extrapolation_count": 0,
        }

    return {
        "sample_count": count,
        "top1_localization_accuracy": sum(bool(row["top1_correct"]) for row in records)
        / count,
        "hit_at_2": sum(bool(row["hit_at_2"]) for row in records) / count,
        "mean_target_delta": sum(float(row["target_delta"]) for row in records)
        / count,
        "mean_max_delta": sum(float(row["max_delta"]) for row in records) / count,
        "two_item_extrapolation_count": sum(
            bool(row["uses_two_item_extrapolation"]) for row in records
        ),
    }


def _score_sample_chunk(model, samples: Sequence[Mapping[str, object]]) -> list[dict]:
    variant_batches = []
    for sample in samples:
        variant_batches.append(
            build_loo_variant_batch(
                sample["item_embeddings"],
                sample["coarse_category_ids"],
                max_items=int(model.max_items),
            )
        )

    device = _model_device(model)
    combined = {
        key: torch.cat([batch[key] for batch in variant_batches], dim=0).to(device)
        for key in (
            "item_embeddings",
            "coarse_category_ids",
            "item_mask",
            "pair_mask",
        )
    }
    has_two_item_rows = any(
        bool(batch["uses_two_item_subsets"]) for batch in variant_batches
    )

    with torch.inference_mode():
        logits = model(
            **combined,
            diagnostic_min_items=(DIAGNOSTIC_MIN_ITEMS if has_two_item_rows else None),
        )["compatibility_logit"]

    results: list[dict] = []
    offset = 0
    for sample, variants in zip(samples, variant_batches):
        item_count = int(variants["original_item_count"])
        group_logits = logits[offset : offset + item_count + 1]
        offset += item_count + 1

        full_logit = group_logits[0]
        without_item_logits = group_logits[1:]
        deltas = without_item_logits - full_logit
        ranking = _rank_deltas(deltas)

        metadata = sample.get("negative_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Negative sample is missing negative_metadata")
        target_index = metadata.get("swapped_item_index")
        if isinstance(target_index, bool) or not isinstance(target_index, int):
            raise ValueError("swapped_item_index must be an integer")
        if not 0 <= target_index < item_count:
            raise ValueError("swapped_item_index is outside the outfit")

        delta_values = [float(value) for value in deltas.detach().cpu().tolist()]
        item_ids = [str(value) for value in sample.get("item_ids", [])]
        results.append(
            {
                "sample_id": str(sample.get("sample_id", "")),
                "source_kit_id": str(sample.get("source_kit_id", "")),
                "original_item_count": item_count,
                "item_ids": item_ids,
                "target_swapped_item_index": target_index,
                "predicted_problematic_item_index": ranking[0],
                "ranked_item_indices": ranking,
                "full_logit": float(full_logit.detach().cpu()),
                "without_item_logits": [
                    float(value)
                    for value in without_item_logits.detach().cpu().tolist()
                ],
                "deltas_without_minus_full": delta_values,
                "target_delta": delta_values[target_index],
                "max_delta": delta_values[ranking[0]],
                "top1_correct": ranking[0] == target_index,
                "hit_at_2": target_index in ranking[: min(2, item_count)],
                "uses_two_item_extrapolation": bool(
                    variants["uses_two_item_subsets"]
                ),
            }
        )

    return results


def evaluate_loo_localization(
    model,
    dataset,
    *,
    outfit_batch_size: int = 64,
) -> dict[str, object]:
    """Evaluate validation-negative LOO Top-1 and Hit@2.

    ``swapped_item_index`` is read only after model scores are produced.  The
    returned report is split by original outfit size so the two-item
    extrapolation used by original three-item outfits remains visible.
    """

    require_torch()
    if model.training:
        raise ValueError("LOO evaluation requires model.eval()")
    if outfit_batch_size < 1:
        raise ValueError("outfit_batch_size must be >= 1")

    negative_samples: list[Mapping[str, object]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if float(sample.get("label", -1.0)) == 0.0:
            negative_samples.append(sample)

    if not negative_samples:
        raise ValueError("Dataset contains no negative samples for LOO evaluation")

    records: list[dict] = []
    for start in range(0, len(negative_samples), outfit_batch_size):
        records.extend(
            _score_sample_chunk(
                model,
                negative_samples[start : start + outfit_batch_size],
            )
        )

    by_size_records: dict[int, list[dict]] = defaultdict(list)
    for row in records:
        by_size_records[int(row["original_item_count"])].append(row)

    return {
        "protocol_version": LOO_PROTOCOL_VERSION,
        "split_scope": "negative_samples_only",
        "overall": _summarize_records(records),
        "by_original_item_count": {
            str(size): _summarize_records(rows)
            for size, rows in sorted(by_size_records.items())
        },
        "records": records,
    }
