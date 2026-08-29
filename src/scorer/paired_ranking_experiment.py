# -*- coding: utf-8 -*-
"""Experimental V5 paired-ranking recheck.

This module deliberately leaves the frozen canonical V5 training path untouched.
It reintroduces only the S3.1 ingredients that are needed for a controlled
ablation on top of the current V5 regime:

- complete positive/negative families stay in the same shuffled train batch;
- total loss = BCEWithLogitsLoss + weight * softplus(-(pos_logit-neg_logit));
- FP32 train and FP32 validation;
- checkpoint selection remains strict validation ROC-AUC;
- the current V5 model/config (including category init) is otherwise unchanged.

It also provides a clean pure-LOO evaluator for original outfits with >=4 items,
so every leave-one-out residual still satisfies the canonical 3-item minimum.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # Keep source importable in lightweight CI.
    torch = None
    DataLoader = None

from src.scorer.checkpoint import save_checkpoint
from src.scorer.dataset import build_pair_mask, collate_scorer_batch
from src.scorer.train import (
    build_optimizer,
    evaluate_epoch,
    seed_everything,
    validate_s3_config,
)


def _require_torch() -> None:
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for paired-ranking experiments")


class PairedFamilyBatchSampler:
    """Shuffle pair families while keeping each positive/negative pair together."""

    def __init__(
        self,
        families: Sequence[tuple[int, int]],
        *,
        sample_batch_size: int,
        generator,
    ) -> None:
        _require_torch()
        if sample_batch_size < 2 or sample_batch_size % 2:
            raise ValueError("sample_batch_size must be a positive even integer")
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        if not families:
            raise ValueError("families must be non-empty")

        normalized: list[tuple[int, int]] = []
        seen: set[int] = set()
        for family in families:
            if not isinstance(family, (tuple, list)) or len(family) != 2:
                raise ValueError("Every family must be (positive_index, negative_index)")
            positive_index, negative_index = int(family[0]), int(family[1])
            if positive_index < 0 or negative_index < 0 or positive_index == negative_index:
                raise ValueError("Invalid paired-family indices")
            if positive_index in seen or negative_index in seen:
                raise ValueError("Paired families may not reuse dataset indices")
            seen.update((positive_index, negative_index))
            normalized.append((positive_index, negative_index))

        self.families = tuple(normalized)
        self.families_per_batch = sample_batch_size // 2
        self.generator = generator

    def __iter__(self):
        order = torch.randperm(len(self.families), generator=self.generator).tolist()
        for start in range(0, len(order), self.families_per_batch):
            batch: list[int] = []
            for family_position in order[start : start + self.families_per_batch]:
                batch.extend(self.families[family_position])
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.families) / self.families_per_batch)


def build_paired_train_loader(train_dataset, config: Mapping[str, object], *, num_workers: int = 0):
    """Build only the experimental paired-family train loader.

    Validation should continue to use the ordinary non-shuffled V5 loader from
    ``build_train_valid_loaders``.
    """

    _require_torch()
    validate_s3_config(config)
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    training = config["training"]
    data = config["data"]
    batch_size = int(training["batch_size"])
    seed = int(training["seed"])
    max_items = int(data["max_items"])
    if batch_size < 2 or batch_size % 2:
        raise ValueError("Paired ranking requires an even batch size >= 2")

    families = list(getattr(train_dataset, "pair_families", []))
    if len(families) * 2 != len(train_dataset):
        raise ValueError("Paired ranking requires complete 1-positive/1-negative coverage")
    covered = sorted(index for family in families for index in family)
    if covered != list(range(len(train_dataset))):
        raise ValueError("Pair families must cover every train row exactly once")

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = PairedFamilyBatchSampler(
        families,
        sample_batch_size=batch_size,
        generator=generator,
    )
    return DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(torch.cuda.is_available()),
        collate_fn=lambda rows: collate_scorer_batch(rows, max_items=max_items),
    )


def paired_logit_margins(logits, labels, sample_ids, paired_positive_sample_ids):
    """Recover positive-minus-negative margins without relying on row adjacency."""

    _require_torch()
    if logits.ndim != 1 or labels.ndim != 1 or logits.shape != labels.shape:
        raise ValueError("logits and labels must have identical shape [B]")
    if len(sample_ids) != len(logits) or len(paired_positive_sample_ids) != len(logits):
        raise ValueError("Batch metadata length mismatch")

    positives: dict[str, int] = {}
    negatives: dict[str, list[int]] = {}
    for index, (sample_id, pair_id, raw_label) in enumerate(
        zip(sample_ids, paired_positive_sample_ids, labels.detach().cpu().tolist())
    ):
        label = int(raw_label)
        sample_id = str(sample_id)
        if label == 1:
            if pair_id not in (None, ""):
                raise ValueError("Positive rows must not reference paired_positive_sample_id")
            positives[sample_id] = index
        elif label == 0:
            pair_key = str(pair_id or "")
            if not pair_key:
                raise ValueError("Negative row is missing paired_positive_sample_id")
            negatives.setdefault(pair_key, []).append(index)
        else:
            raise ValueError("labels must be binary")

    if set(positives) != set(negatives):
        raise ValueError("Paired batch does not contain complete families")

    margins = []
    for positive_id, positive_index in positives.items():
        negative_indices = negatives[positive_id]
        if len(negative_indices) != 1:
            raise ValueError("Each positive must have exactly one paired negative in batch")
        margins.append(logits[positive_index] - logits[negative_indices[0]])
    if not margins:
        raise ValueError("Paired batch contains no complete families")
    return torch.stack(margins)


def _forward(model, batch, device):
    return model(
        item_embeddings=batch["item_embeddings"].to(device, non_blocking=True),
        coarse_category_ids=batch["coarse_category_ids"].to(device, non_blocking=True),
        item_mask=batch["item_mask"].to(device, non_blocking=True),
        pair_mask=batch["pair_mask"].to(device, non_blocking=True),
    )["compatibility_logit"]


def train_one_epoch_paired(
    model,
    dataloader,
    *,
    optimizer,
    device,
    ranking_weight: float,
) -> dict[str, float | int]:
    """FP32 BCE + paired-logistic training for one epoch."""

    _require_torch()
    if not math.isfinite(ranking_weight) or ranking_weight < 0.0:
        raise ValueError("ranking_weight must be a finite non-negative value")

    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    total_sum = bce_sum = ranking_sum = 0.0
    sample_count = family_count = step_count = 0

    for batch in dataloader:
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Deliberately no autocast: V5 recheck is FP32 train + FP32 validation.
        logits = _forward(model, batch, device)
        bce_loss = criterion(logits, labels)
        margins = paired_logit_margins(
            logits,
            labels,
            batch["sample_ids"],
            batch["paired_positive_sample_ids"],
        )
        ranking_loss = torch.nn.functional.softplus(-margins).mean()
        total_loss = bce_loss + float(ranking_weight) * ranking_loss

        if not torch.isfinite(total_loss):
            raise RuntimeError("Non-finite paired-ranking training loss")
        total_loss.backward()
        optimizer.step()

        batch_samples = int(labels.shape[0])
        batch_families = int(margins.shape[0])
        total_sum += float(total_loss.detach()) * batch_samples
        bce_sum += float(bce_loss.detach()) * batch_samples
        ranking_sum += float(ranking_loss.detach()) * batch_families
        sample_count += batch_samples
        family_count += batch_families
        step_count += 1

    if sample_count == 0 or family_count == 0:
        raise RuntimeError("Paired training loader produced no data")
    return {
        "total_loss": total_sum / sample_count,
        "bce_loss": bce_sum / sample_count,
        "ranking_loss": ranking_sum / family_count,
        "sample_count": sample_count,
        "family_count": family_count,
        "step_count": step_count,
    }


def fit_paired_ranking_scorer(
    model,
    train_loader,
    valid_loader,
    *,
    config: Mapping[str, object],
    checkpoint_dir: Path | str,
    provenance: Mapping[str, object],
    ranking_weight: float,
    device=None,
) -> dict[str, object]:
    """Train a V5-compatible scorer with only the ranking objective added."""

    _require_torch()
    validate_s3_config(config)
    training = config["training"]
    if bool(training.get("mixed_precision", False)):
        raise ValueError("V5 paired-ranking recheck requires mixed_precision=false")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(training["seed"])
    seed_everything(seed)
    model.to(device)
    optimizer = build_optimizer(model, config)

    max_epochs = int(training["max_epochs"])
    patience = int(training["early_stopping_patience"])
    min_epochs = int(training["early_stopping_min_epochs"])
    min_delta = float(training.get("early_stopping_min_delta", 0.0))

    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    history: list[dict[str, object]] = []
    best_auc = float("-inf")
    best_epoch = None
    no_improve = 0
    global_step = 0
    stopped_early = False

    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(1, max_epochs + 1):
        train_stats = train_one_epoch_paired(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            ranking_weight=ranking_weight,
        )
        global_step += int(train_stats["step_count"])
        valid = evaluate_epoch(
            model,
            valid_loader,
            criterion=criterion,
            device=device,
        )
        auc = float(valid["roc_auc"])
        improved = auc > (best_auc + min_delta)
        if improved:
            best_auc = auc
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        row = {
            "epoch": epoch,
            "train_total_loss": float(train_stats["total_loss"]),
            "train_bce_loss": float(train_stats["bce_loss"]),
            "train_ranking_loss": float(train_stats["ranking_loss"]),
            "valid_loss": float(valid["loss"]),
            "valid_roc_auc": auc,
            "valid_fitb_2way": float(valid["fitb_2way"]),
            "valid_mean_logit_margin": float(valid["mean_logit_margin"]),
            "valid_median_logit_margin": float(valid["median_logit_margin"]),
            "best_valid_roc_auc": float(best_auc),
            "improved": bool(improved),
        }
        history.append(row)

        experiment_config = dict(config)
        experiment_config["experiment"] = {
            "name": "v5_paired_ranking_recheck",
            "ranking_weight": float(ranking_weight),
            "paired_batching": True,
            "fp32_train": True,
            "fp32_validation": True,
        }
        checkpoint_kwargs = dict(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            config=experiment_config,
            seed=seed,
            best_valid_roc_auc=best_auc,
            validation_metrics=valid,
            provenance=provenance,
        )
        save_checkpoint(last_path, **checkpoint_kwargs)
        if improved:
            save_checkpoint(best_path, **checkpoint_kwargs)

        print(
            f"epoch={epoch:02d} "
            f"train_total={float(train_stats['total_loss']):.6f} "
            f"train_bce={float(train_stats['bce_loss']):.6f} "
            f"train_rank={float(train_stats['ranking_loss']):.6f} "
            f"valid_loss={float(valid['loss']):.6f} "
            f"auc={auc:.5f} fitb={float(valid['fitb_2way']):.5f} "
            f"mean_margin={float(valid['mean_logit_margin']):.5f} "
            f"{'BEST' if improved else ''}"
        )

        if epoch >= min_epochs and no_improve >= patience:
            stopped_early = True
            print(
                "Early stopping: validation ROC-AUC did not improve for "
                f"{patience} epochs after minimum epoch {min_epochs}."
            )
            break

    if best_epoch is None:
        raise RuntimeError("No best checkpoint was produced")
    return {
        "best_epoch": int(best_epoch),
        "best_valid_roc_auc": float(best_auc),
        "ranking_weight": float(ranking_weight),
        "stopped_early": bool(stopped_early),
        "history": history,
        "best_path": str(best_path),
        "last_path": str(last_path),
    }


def evaluate_pure_loo_4plus(model, dataset, *, device=None) -> dict[str, object]:
    """Evaluate pure LOO only where all residuals remain canonical (original n>=4)."""

    _require_torch()
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    records: list[dict[str, object]] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        if float(sample.get("label", -1.0)) != 0.0:
            continue
        embeddings = sample["item_embeddings"]
        categories = sample["coarse_category_ids"]
        item_count = int(embeddings.shape[0])
        if item_count < 4:
            continue

        max_items = int(model.max_items)
        variant_count = item_count + 1
        batch_embeddings = torch.zeros(
            variant_count, max_items, embeddings.shape[1], dtype=embeddings.dtype
        )
        batch_categories = torch.zeros(variant_count, max_items, dtype=torch.long)
        item_mask = torch.zeros(variant_count, max_items, dtype=torch.bool)

        batch_embeddings[0, :item_count] = embeddings
        batch_categories[0, :item_count] = categories
        item_mask[0, :item_count] = True
        all_indices = torch.arange(item_count)
        for removed in range(item_count):
            keep = all_indices != removed
            kept_count = item_count - 1
            batch_embeddings[removed + 1, :kept_count] = embeddings[keep]
            batch_categories[removed + 1, :kept_count] = categories[keep]
            item_mask[removed + 1, :kept_count] = True

        pair_mask = build_pair_mask(item_mask)
        with torch.inference_mode():
            logits = model(
                item_embeddings=batch_embeddings.to(device),
                coarse_category_ids=batch_categories.to(device),
                item_mask=item_mask.to(device),
                pair_mask=pair_mask.to(device),
            )["compatibility_logit"]

        deltas = logits[1:] - logits[0]
        ranking = sorted(
            range(item_count),
            key=lambda i: (-float(deltas[i].detach().cpu()), i),
        )
        metadata = sample.get("negative_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Negative sample is missing negative_metadata")
        target = int(metadata["swapped_item_index"])
        records.append(
            {
                "sample_id": str(sample.get("sample_id", "")),
                "original_item_count": item_count,
                "target_swapped_item_index": target,
                "ranked_item_indices": ranking,
                "top1_correct": ranking[0] == target,
                "hit_at_2": target in ranking[:2],
                "hit_at_3": target in ranking[:3],
            }
        )

    if not records:
        raise ValueError("No validation negatives with original size >=4")
    count = len(records)
    return {
        "sample_count": count,
        "top1_localization_accuracy": sum(bool(r["top1_correct"]) for r in records) / count,
        "hit_at_2": sum(bool(r["hit_at_2"]) for r in records) / count,
        "hit_at_3": sum(bool(r["hit_at_3"]) for r in records) / count,
        "records": records,
    }
