# -*- coding: utf-8 -*-
"""Canonical training helpers for Type-aware Pairwise Scorer V1.

The notebook is only an experiment wrapper.  Loss, optimizer, reproducibility,
tiny-family selection, and the training loop live here so S2.5 and S3 exercise
the same implementation.
"""

from __future__ import annotations

import json
import math
import platform
import random
from importlib import metadata as importlib_metadata
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from functools import partial
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # Keep source importable in lightweight CI.
    np = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
except ModuleNotFoundError:  # Keep source importable in lightweight CI.
    torch = None
    nn = None
    DataLoader = None
    Subset = None

from .dataset import MAX_ITEMS, collate_scorer_batch, flatten_family_indices
from .evaluate import evaluate_model
from .checkpoint import (
    build_checkpoint_payload,
    restore_checkpoint,
    save_checkpoint,
    save_epoch_checkpoints,
)
from .model import SCORER_VERSION


DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
DEFAULT_TINY_FAMILY_COUNT = 32
DEFAULT_TINY_MAX_EPOCHS = 300
DEFAULT_FULL_BATCH_SIZE = 256
DEFAULT_FULL_MAX_EPOCHS = 30
DEFAULT_EARLY_STOPPING_PATIENCE = 5
BASELINE_OBJECTIVE = "bce"
PAIRED_RANKING_OBJECTIVE = "bce_plus_paired_logistic"


def require_training_dependencies() -> None:
    """Fail clearly when the canonical training dependencies are unavailable."""

    if torch is None or nn is None or DataLoader is None or Subset is None:
        raise RuntimeError("PyTorch is required for scorer training")
    if np is None:
        raise RuntimeError("NumPy is required for reproducible scorer training")


def set_reproducible_seed(seed: int, *, deterministic: bool = True):
    """Seed Python, NumPy, PyTorch CPU/CUDA, and return a loader generator."""

    require_training_dependencies()
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(device: str | object | None = None):
    """Resolve the requested device, defaulting to CUDA when available."""

    require_training_dependencies()
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _training_section(config: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    section = config.get("training", config)
    if not isinstance(section, Mapping):
        raise ValueError("config['training'] must be a mapping")
    return section


def _locked_integer(
    section: Mapping[str, object],
    key: str,
    default: int,
    *,
    minimum: int,
) -> int:
    """Read an integer config value without silently truncating floats/bools."""

    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"training.{key} must be an integer >= {minimum}")
    return value


def build_optimizer(model, config: Mapping[str, object]):
    """Build the optimizer locked by Scorer Contract V1."""

    require_training_dependencies()
    section = _training_section(config)
    optimizer_name = str(section.get("optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError("Scorer V1 supports optimizer='adamw' only")
    if str(section.get("lr_scheduler", "none")).lower() != "none":
        raise ValueError("Scorer V1 locks lr_scheduler='none'")
    if str(section.get("gradient_clipping", "none")).lower() != "none":
        raise ValueError("Scorer V1 locks gradient_clipping='none'")

    learning_rate = float(section.get("learning_rate", DEFAULT_LEARNING_RATE))
    weight_decay = float(section.get("weight_decay", DEFAULT_WEIGHT_DECAY))
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be a positive finite number")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be a non-negative finite number")

    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_full_training_loaders(
    train_dataset,
    valid_dataset,
    config: Mapping[str, object],
    *,
    num_workers: int = 0,
) -> tuple[object, object]:
    """Build the locked S3 train/validation loaders without a test loader."""

    require_training_dependencies()
    section = _training_section(config)
    data_section = config.get("data", {})
    if not isinstance(data_section, Mapping):
        raise ValueError("config['data'] must be a mapping")

    batch_size = _locked_integer(
        section,
        "batch_size",
        DEFAULT_FULL_BATCH_SIZE,
        minimum=1,
    )
    seed = _locked_integer(section, "seed", DEFAULT_SEED, minimum=0)
    max_items = int(data_section.get("max_items", MAX_ITEMS))
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if len(train_dataset) < 1 or len(valid_dataset) < 1:
        raise ValueError("Train and validation datasets must be non-empty")

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    collate_fn = partial(collate_scorer_batch, max_items=max_items)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=train_generator,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_loader, valid_loader


class PairedFamilyBatchSampler:
    """Yield shuffled batches made only from complete positive-negative families."""

    def __init__(
        self,
        families: Sequence[tuple[int, int]],
        *,
        sample_batch_size: int,
        generator,
    ) -> None:
        require_training_dependencies()
        if sample_batch_size < 2 or sample_batch_size % 2 != 0:
            raise ValueError("paired sample_batch_size must be a positive even number")
        if not isinstance(generator, torch.Generator):
            raise TypeError("paired batch sampler requires a torch.Generator")
        if not families:
            raise ValueError("paired batch sampler requires at least one family")

        normalized: list[tuple[int, int]] = []
        seen_indices: set[int] = set()
        for family_index, family in enumerate(families):
            if not isinstance(family, (list, tuple)) or len(family) != 2:
                raise ValueError(
                    f"Family {family_index} must contain exactly two indices"
                )
            positive_index, negative_index = family
            if (
                isinstance(positive_index, bool)
                or not isinstance(positive_index, int)
                or isinstance(negative_index, bool)
                or not isinstance(negative_index, int)
                or positive_index < 0
                or negative_index < 0
                or positive_index == negative_index
            ):
                raise ValueError(f"Family {family_index} has invalid sample indices")
            if positive_index in seen_indices or negative_index in seen_indices:
                raise ValueError("Paired families contain duplicate sample indices")
            seen_indices.update((positive_index, negative_index))
            normalized.append((positive_index, negative_index))

        self.families = tuple(normalized)
        self.families_per_batch = sample_batch_size // 2
        self.generator = generator

    def __iter__(self):
        order = torch.randperm(
            len(self.families), generator=self.generator
        ).tolist()
        for start in range(0, len(order), self.families_per_batch):
            batch: list[int] = []
            for position in order[start : start + self.families_per_batch]:
                batch.extend(self.families[position])
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.families) / self.families_per_batch)


def build_paired_training_loaders(
    train_dataset,
    valid_dataset,
    config: Mapping[str, object],
    *,
    num_workers: int = 0,
) -> tuple[object, object]:
    """Build an S3.1 train loader that never separates a frozen pair family."""

    require_training_dependencies()
    section = _training_section(config)
    data_section = config.get("data", {})
    if not isinstance(data_section, Mapping):
        raise ValueError("config['data'] must be a mapping")

    objective = str(section.get("objective", BASELINE_OBJECTIVE)).strip().lower()
    if objective != PAIRED_RANKING_OBJECTIVE:
        raise ValueError(
            "Paired loaders require "
            f"training.objective={PAIRED_RANKING_OBJECTIVE!r}"
        )
    if section.get("paired_batching") is not True:
        raise ValueError("Paired loaders require training.paired_batching=true")

    batch_size = _locked_integer(
        section,
        "batch_size",
        DEFAULT_FULL_BATCH_SIZE,
        minimum=2,
    )
    if batch_size % 2 != 0:
        raise ValueError("Paired training requires an even training.batch_size")
    seed = _locked_integer(section, "seed", DEFAULT_SEED, minimum=0)
    max_items = int(data_section.get("max_items", MAX_ITEMS))
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if len(train_dataset) < 2 or len(valid_dataset) < 1:
        raise ValueError("Train and validation datasets must be non-empty")

    families = list(getattr(train_dataset, "pair_families", []))
    if len(families) * 2 != len(train_dataset):
        raise ValueError(
            "Paired training requires complete one-positive/one-negative "
            "coverage of the train dataset"
        )
    covered_indices = sorted(index for family in families for index in family)
    if covered_indices != list(range(len(train_dataset))):
        raise ValueError(
            "Paired training families must cover every train row exactly once"
        )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    batch_sampler = PairedFamilyBatchSampler(
        families,
        sample_batch_size=batch_size,
        generator=train_generator,
    )
    collate_fn = partial(collate_scorer_batch, max_items=max_items)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=train_generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_loader, valid_loader


def build_tiny_overfit_loader(
    dataset,
    *,
    family_count: int = DEFAULT_TINY_FAMILY_COUNT,
    batch_size: int | None = None,
    seed: int = DEFAULT_SEED,
    max_items: int = MAX_ITEMS,
    shuffle: bool = True,
) -> tuple[object, dict[str, object]]:
    """Select reproducible complete positive-negative families for S2.5."""

    require_training_dependencies()
    if family_count < 1:
        raise ValueError("family_count must be >= 1")
    if max_items < 1:
        raise ValueError("max_items must be >= 1")

    families = list(getattr(dataset, "pair_families", []))
    if len(families) < family_count:
        raise ValueError(
            f"Dataset has {len(families)} complete families; need {family_count}"
        )

    family_rng = random.Random(seed)
    family_positions = sorted(
        family_rng.sample(range(len(families)), family_count)
    )
    selected_families = [families[index] for index in family_positions]
    sample_indices = flatten_family_indices(selected_families)
    if len(sample_indices) != 2 * family_count:
        raise ValueError("Every tiny-overfit family must contain exactly two samples")
    if len(sample_indices) != len(set(sample_indices)):
        raise ValueError("Tiny-overfit families contain duplicate sample indices")

    if batch_size is None:
        batch_size = len(sample_indices)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    subset = Subset(dataset, sample_indices)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=partial(collate_scorer_batch, max_items=max_items),
        generator=loader_generator,
        drop_last=False,
    )
    selection = {
        "seed": seed,
        "family_count": family_count,
        "sample_count": len(sample_indices),
        "family_positions": family_positions,
        "sample_indices": sample_indices,
    }
    return loader, selection


def _model_inputs(batch: Mapping[str, object], device) -> dict[str, object]:
    required = {
        "item_embeddings",
        "coarse_category_ids",
        "item_mask",
        "labels",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"Training batch is missing keys: {missing}")

    inputs = {
        "item_embeddings": batch["item_embeddings"].to(device),
        "coarse_category_ids": batch["coarse_category_ids"].to(device),
        "item_mask": batch["item_mask"].to(device),
    }
    if "pair_mask" in batch:
        inputs["pair_mask"] = batch["pair_mask"].to(device)
    return inputs


def _extract_logits(output: object, labels) -> object:
    if not isinstance(output, Mapping) or "compatibility_logit" not in output:
        raise ValueError(
            "Scorer output must be a mapping containing compatibility_logit"
        )
    logits = output["compatibility_logit"]
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise ValueError("compatibility_logit must be a rank-1 Tensor [B]")
    if logits.shape != labels.shape:
        raise ValueError(
            "compatibility_logit and labels must have identical shape [B]"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("compatibility_logit contains NaN/Inf")
    return logits


def paired_batch_logit_margins(
    logits,
    labels,
    sample_ids: Sequence[object],
    paired_positive_sample_ids: Sequence[object],
):
    """Return ``positive_logit - negative_logit`` for each complete batch family.

    Pair recovery uses IDs rather than row adjacency so shuffling inside a batch
    cannot silently change the training target.
    """

    require_training_dependencies()
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise ValueError("paired ranking logits must be a rank-1 Tensor [B]")
    if not isinstance(labels, torch.Tensor) or labels.shape != logits.shape:
        raise ValueError("paired ranking labels must match logits shape [B]")
    if len(sample_ids) != len(logits) or len(paired_positive_sample_ids) != len(
        logits
    ):
        raise ValueError("paired ranking IDs must match logits batch size")

    normalized_ids = [str(value).strip() for value in sample_ids]
    if any(not value for value in normalized_ids):
        raise ValueError("paired ranking sample_ids must be non-empty")
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("paired ranking sample_ids must be unique within a batch")

    label_values = labels.detach().cpu().tolist()
    positives: dict[str, int] = {}
    negatives: dict[str, list[int]] = {}
    for index, (sample_id, raw_label, raw_pair_id) in enumerate(
        zip(normalized_ids, label_values, paired_positive_sample_ids)
    ):
        label = int(raw_label)
        if label not in (0, 1) or float(raw_label) != label:
            raise ValueError("paired ranking labels must contain binary values 0/1")
        pair_id = (
            None
            if raw_pair_id in (None, "")
            else str(raw_pair_id).strip()
        )
        if label == 1:
            if pair_id is not None:
                raise ValueError(
                    f"Positive {sample_id} must not reference another positive"
                )
            positives[sample_id] = index
        else:
            if not pair_id:
                raise ValueError(
                    f"Negative {sample_id} is missing paired_positive_sample_id"
                )
            negatives.setdefault(pair_id, []).append(index)

    if set(positives) != set(negatives):
        missing_negatives = sorted(set(positives) - set(negatives))
        missing_positives = sorted(set(negatives) - set(positives))
        raise ValueError(
            "Paired ranking batch contains incomplete families: "
            f"missing_negatives={missing_negatives[:5]}, "
            f"missing_positives={missing_positives[:5]}"
        )

    margins = []
    for positive_id, positive_index in positives.items():
        negative_indices = negatives[positive_id]
        if len(negative_indices) != 1:
            raise ValueError(
                f"Positive {positive_id} must have exactly one paired negative"
            )
        margins.append(logits[positive_index] - logits[negative_indices[0]])
    if not margins:
        raise ValueError("Paired ranking batch contains no complete families")
    return torch.stack(margins).float()


def paired_logistic_ranking_loss(
    logits,
    labels,
    sample_ids: Sequence[object],
    paired_positive_sample_ids: Sequence[object],
):
    """Smoothly penalize non-positive paired margins with ``softplus(-margin)``."""

    margins = paired_batch_logit_margins(
        logits,
        labels,
        sample_ids,
        paired_positive_sample_ids,
    )
    return torch.nn.functional.softplus(-margins).mean()


def _autocast_context(device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def create_grad_scaler(device, *, use_amp: bool):
    """Create a CUDA scaler only when AMP is both requested and supported."""

    require_training_dependencies()
    enabled = bool(use_amp and device.type == "cuda")
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):  # Compatible with older supported torch.
        return torch.cuda.amp.GradScaler(enabled=True)


def _assert_finite_gradients(model) -> None:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("No model gradients were produced")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Model gradients contain NaN/Inf")


def train_one_epoch(
    model,
    dataloader: Iterable[Mapping[str, object]],
    optimizer,
    *,
    device=None,
    criterion=None,
    use_amp: bool = False,
    scaler=None,
    global_step: int = 0,
    paired_ranking_weight: float = 0.0,
) -> dict[str, int | float]:
    """Train one epoch with BCE and optional paired logistic ranking loss."""

    require_training_dependencies()
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.train()
    criterion = criterion or nn.BCEWithLogitsLoss()
    amp_enabled = bool(use_amp and resolved_device.type == "cuda")
    scaler = scaler if scaler is not None else create_grad_scaler(
        resolved_device, use_amp=amp_enabled
    )
    ranking_weight = float(paired_ranking_weight)
    if not math.isfinite(ranking_weight) or ranking_weight < 0.0:
        raise ValueError("paired_ranking_weight must be finite and >= 0")

    loss_sum = 0.0
    bce_loss_sum = 0.0
    ranking_loss_sum = 0.0
    paired_margin_sum = 0.0
    paired_family_count = 0
    sample_count = 0
    batch_count = 0
    for batch in dataloader:
        if not isinstance(batch, Mapping):
            raise TypeError("Each training batch must be a mapping")
        labels = batch["labels"].to(resolved_device).float()
        if labels.ndim != 1 or not torch.isfinite(labels).all():
            raise ValueError("labels must be a finite rank-1 Tensor [B]")
        if torch.any((labels != 0.0) & (labels != 1.0)):
            raise ValueError("labels must contain binary values 0/1")

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(resolved_device, amp_enabled):
            output = model(**_model_inputs(batch, resolved_device))
            logits = _extract_logits(output, labels)
            bce_loss = criterion(logits, labels)
            ranking_loss = logits.new_zeros((), dtype=torch.float32)
            margins = None
            if ranking_weight > 0.0:
                margins = paired_batch_logit_margins(
                    logits,
                    labels,
                    batch.get("sample_ids", []),
                    batch.get("paired_positive_sample_ids", []),
                )
                ranking_loss = torch.nn.functional.softplus(-margins).mean()
            loss = bce_loss + ranking_weight * ranking_loss
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError("Training loss must be one finite scalar")

        if scaler is None:
            loss.backward()
            _assert_finite_gradients(model)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _assert_finite_gradients(model)
            scaler.step(optimizer)
            scaler.update()

        current_batch_size = int(labels.shape[0])
        loss_sum += float(loss.detach().cpu()) * current_batch_size
        bce_loss_sum += float(bce_loss.detach().cpu()) * current_batch_size
        if margins is not None:
            current_family_count = int(margins.numel())
            ranking_loss_sum += (
                float(ranking_loss.detach().cpu()) * current_family_count
            )
            paired_margin_sum += float(margins.detach().sum().cpu())
            paired_family_count += current_family_count
        sample_count += current_batch_size
        batch_count += 1
        global_step += 1

    if sample_count == 0:
        raise ValueError("Training dataloader yielded no samples")
    return {
        "loss": loss_sum / sample_count,
        "bce_loss": bce_loss_sum / sample_count,
        "paired_ranking_loss": (
            ranking_loss_sum / paired_family_count
            if paired_family_count
            else 0.0
        ),
        "mean_paired_margin": (
            paired_margin_sum / paired_family_count
            if paired_family_count
            else 0.0
        ),
        "paired_family_count": paired_family_count,
        "sample_count": sample_count,
        "batch_count": batch_count,
        "global_step": global_step,
    }


def evaluate_binary_loss(
    model,
    dataloader: Iterable[Mapping[str, object]],
    *,
    device=None,
    criterion=None,
    use_amp: bool = False,
) -> float:
    """Compute sample-weighted BCEWithLogitsLoss without updating the model."""

    require_training_dependencies()
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    criterion = criterion or nn.BCEWithLogitsLoss()
    amp_enabled = bool(use_amp and resolved_device.type == "cuda")
    was_training = bool(model.training)
    model.eval()

    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(resolved_device).float()
            with _autocast_context(resolved_device, amp_enabled):
                output = model(**_model_inputs(batch, resolved_device))
                logits = _extract_logits(output, labels)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Evaluation loss contains NaN/Inf")
            current_batch_size = int(labels.shape[0])
            loss_sum += float(loss.detach().cpu()) * current_batch_size
            sample_count += current_batch_size

    if was_training:
        model.train()
    if sample_count == 0:
        raise ValueError("Evaluation dataloader yielded no samples")
    return loss_sum / sample_count


def _evaluation_snapshot(
    model, dataloader, *, device, use_amp: bool
) -> dict[str, object]:
    loss = evaluate_binary_loss(
        model,
        dataloader,
        device=device,
        use_amp=use_amp,
    )
    evaluation = evaluate_model(model, dataloader, device=device)
    metrics = dict(evaluation["metrics"])
    return {"loss": loss, **metrics}


def run_tiny_overfit(
    model,
    dataloader,
    config: Mapping[str, object],
    *,
    device=None,
    optimizer=None,
    max_epochs: int = DEFAULT_TINY_MAX_EPOCHS,
    expected_family_count: int = DEFAULT_TINY_FAMILY_COUNT,
    target_roc_auc: float = 0.99,
    target_fitb: float = 0.99,
    max_loss_ratio: float = 0.25,
    use_amp: bool = False,
) -> dict[str, object]:
    """Run the S2.5 sanity experiment and return a JSON-safe report.

    The explicit thresholds operationalize the Contract wording "tiến gần 1"
    and "loss giảm mạnh" for this run.  They are recorded in the report rather
    than silently becoming canonical S3 model-selection rules.
    """

    require_training_dependencies()
    if max_epochs < 1:
        raise ValueError("max_epochs must be >= 1")
    if expected_family_count < 1:
        raise ValueError("expected_family_count must be >= 1")
    for name, value in (
        ("target_roc_auc", target_roc_auc),
        ("target_fitb", target_fitb),
        ("max_loss_ratio", max_loss_ratio),
    ):
        if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must satisfy 0 < value <= 1")

    resolved_device = resolve_device(device)
    model.to(resolved_device)
    optimizer = optimizer or build_optimizer(model, config)
    scaler = create_grad_scaler(resolved_device, use_amp=use_amp)

    initial = _evaluation_snapshot(
        model,
        dataloader,
        device=resolved_device,
        use_amp=use_amp,
    )
    expected_sample_count = expected_family_count * 2
    if initial["paired_family_count"] != expected_family_count:
        raise ValueError(
            "Tiny overfit must contain exactly "
            f"{expected_family_count} complete pair families"
        )
    if initial["sample_count"] != expected_sample_count:
        raise ValueError(
            f"Tiny overfit must contain exactly {expected_sample_count} samples"
        )

    history: list[dict[str, object]] = [{"epoch": 0, **initial}]
    global_step = 0
    status = "FAIL"
    final = initial
    for epoch in range(1, max_epochs + 1):
        train_result = train_one_epoch(
            model,
            dataloader,
            optimizer,
            device=resolved_device,
            use_amp=use_amp,
            scaler=scaler,
            global_step=global_step,
        )
        global_step = int(train_result["global_step"])
        final = _evaluation_snapshot(
            model,
            dataloader,
            device=resolved_device,
            use_amp=use_amp,
        )
        loss_ratio = final["loss"] / initial["loss"]
        epoch_record = {
            "epoch": epoch,
            **final,
            "loss_ratio": loss_ratio,
        }
        history.append(epoch_record)

        if (
            final["roc_auc"] >= target_roc_auc
            and final["fitb_2way"] >= target_fitb
            and loss_ratio <= max_loss_ratio
        ):
            status = "PASS"
            break

    final_loss_ratio = final["loss"] / initial["loss"]
    return {
        "stage": "S2.5",
        "status": status,
        "device": str(resolved_device),
        "amp_enabled": bool(use_amp and resolved_device.type == "cuda"),
        "max_epochs": max_epochs,
        "epochs_ran": len(history) - 1,
        "global_step": global_step,
        "criteria": {
            "target_roc_auc": target_roc_auc,
            "target_fitb_2way": target_fitb,
            "max_loss_ratio": max_loss_ratio,
        },
        "initial": initial,
        "final": {**final, "loss_ratio": final_loss_ratio},
        "history": history,
    }


def _full_training_settings(config: Mapping[str, object]) -> dict[str, object]:
    section = _training_section(config)
    selection = config.get("selection", {})
    if not isinstance(selection, Mapping):
        raise ValueError("config['selection'] must be a mapping")
    experiment = config.get("experiment", {})
    if not isinstance(experiment, Mapping):
        raise ValueError("config['experiment'] must be a mapping when provided")

    max_epochs = _locked_integer(
        section,
        "max_epochs",
        DEFAULT_FULL_MAX_EPOCHS,
        minimum=1,
    )
    patience = _locked_integer(
        section,
        "early_stopping_patience",
        DEFAULT_EARLY_STOPPING_PATIENCE,
        minimum=1,
    )
    min_delta = float(section.get("early_stopping_min_delta", 0.0))
    seed = _locked_integer(section, "seed", DEFAULT_SEED, minimum=0)
    mixed_precision = section.get("mixed_precision", True)
    objective = str(section.get("objective", BASELINE_OBJECTIVE)).strip().lower()
    paired_batching = section.get("paired_batching", False)
    ranking_weight = float(section.get("paired_ranking_weight", 0.0))
    if min_delta != 0.0:
        raise ValueError("Scorer V1 locks early_stopping_min_delta=0.0")
    if not isinstance(mixed_precision, bool):
        raise ValueError("training.mixed_precision must be boolean")
    if not isinstance(paired_batching, bool):
        raise ValueError("training.paired_batching must be boolean")
    if not math.isfinite(ranking_weight) or ranking_weight < 0.0:
        raise ValueError("training.paired_ranking_weight must be finite and >= 0")
    if objective == BASELINE_OBJECTIVE:
        if paired_batching or ranking_weight != 0.0:
            raise ValueError(
                "BCE objective requires paired_batching=false and "
                "paired_ranking_weight=0"
            )
    elif objective == PAIRED_RANKING_OBJECTIVE:
        if not paired_batching or ranking_weight <= 0.0:
            raise ValueError(
                "Paired objective requires paired_batching=true and a positive "
                "paired_ranking_weight"
            )
    else:
        raise ValueError(f"Unsupported training.objective: {objective!r}")
    if str(selection.get("primary_metric", "roc_auc")) != "roc_auc":
        raise ValueError("Scorer V1 selects checkpoints by validation ROC-AUC")
    if str(selection.get("guardrail_metric", "fitb_2way")) != "fitb_2way":
        raise ValueError("Scorer V1 locks fitb_2way as the guardrail metric")

    stage = str(experiment.get("stage", "S3")).strip()
    experiment_name = str(
        experiment.get("name", "s3_full_baseline")
    ).strip()
    if not stage or not experiment_name:
        raise ValueError("Experiment stage and name must be non-empty strings")

    return {
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "seed": seed,
        "mixed_precision": mixed_precision,
        "objective": objective,
        "paired_batching": paired_batching,
        "paired_ranking_weight": ranking_weight,
        "stage": stage,
        "experiment_name": experiment_name,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
    }
    for package, key in (("scikit-learn", "scikit_learn"), ("PyYAML", "pyyaml")):
        try:
            versions[key] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[key] = "NOT_INSTALLED"
    return versions


def _capture_rng_state(train_dataloader) -> dict[str, object]:
    numpy_state = np.random.get_state()
    loader_generator = getattr(train_dataloader, "generator", None)
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "dataloader_generator_state": (
            loader_generator.get_state() if loader_generator is not None else None
        ),
    }


def _cpu_byte_rng_state(value: object, *, name: str) -> torch.Tensor:
    """Return a CPU ByteTensor accepted by PyTorch RNG restore APIs.

    ``torch.load(..., map_location="cuda")`` also moves tensors stored in
    checkpoint metadata to CUDA.  RNG restore APIs are stricter than model and
    optimizer loading: CPU and DataLoader generator states must be CPU byte
    tensors, while CUDA RNG restore also accepts its saved state from CPU.
    """

    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Checkpoint {name} must be a torch.Tensor")
    if value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError(
            f"Checkpoint {name} must be a rank-1 torch.uint8 tensor"
        )
    return value.detach().cpu().contiguous()


def _restore_rng_state(state: Mapping[str, object], train_dataloader) -> None:
    required = {
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state_all",
        "dataloader_generator_state",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Checkpoint RNG state is missing keys: {missing}")

    random.setstate(state["python_random_state"])
    numpy_state = state["numpy_random_state"]
    if not isinstance(numpy_state, Mapping):
        raise ValueError("Checkpoint NumPy RNG state must be a mapping")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(
        _cpu_byte_rng_state(
            state["torch_cpu_rng_state"], name="torch_cpu_rng_state"
        )
    )

    cuda_states = state["torch_cuda_rng_state_all"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore CUDA RNG state without CUDA")
        if not isinstance(cuda_states, (list, tuple)):
            raise ValueError(
                "Checkpoint torch_cuda_rng_state_all must be a list or tuple"
            )
        torch.cuda.set_rng_state_all(
            [
                _cpu_byte_rng_state(
                    value, name=f"torch_cuda_rng_state_all[{index}]"
                )
                for index, value in enumerate(cuda_states)
            ]
        )

    loader_state = state["dataloader_generator_state"]
    loader_generator = getattr(train_dataloader, "generator", None)
    if loader_state is not None:
        if loader_generator is None:
            raise ValueError(
                "Checkpoint has DataLoader RNG state but loader has no generator"
            )
        loader_generator.set_state(
            _cpu_byte_rng_state(
                loader_state, name="dataloader_generator_state"
            )
        )


def _resume_extra_state(payload: Mapping[str, object]) -> dict[str, object]:
    required = {
        "training_history",
        "epochs_without_improvement",
        "best_epoch",
        "best_validation_metrics",
        "rng_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"S3 checkpoint is missing resume state: {missing}")
    history = payload["training_history"]
    if not isinstance(history, list):
        raise ValueError("training_history in checkpoint must be a list")
    if history and int(history[-1].get("epoch", -1)) != int(payload["epoch"]):
        raise ValueError("Checkpoint epoch does not match training_history")
    return {
        "history": list(history),
        "epochs_without_improvement": int(
            payload["epochs_without_improvement"]
        ),
        "best_epoch": int(payload["best_epoch"]),
        "best_validation_metrics": dict(payload["best_validation_metrics"]),
        "rng_state": payload["rng_state"],
        "grad_scaler_state_dict": payload.get("grad_scaler_state_dict"),
    }


def run_full_training(
    model,
    train_dataloader,
    valid_dataloader,
    config: Mapping[str, object],
    *,
    output_dir: Path | str,
    provenance: Mapping[str, object],
    git_state: Mapping[str, object],
    device=None,
    resume: bool = False,
    require_clean_git: bool = True,
    epoch_callback=None,
) -> dict[str, object]:
    """Run canonical S3 or versioned post-baseline train/validation orchestration.

    Checkpoint selection is strictly ``valid ROC-AUC > best ROC-AUC``.  FITB is
    reported as a guardrail and never used as a hidden tie-breaker.  The test
    split cannot enter this API because only train and validation loaders are
    accepted.
    """

    require_training_dependencies()
    settings = _full_training_settings(config)
    resolved_device = resolve_device(device)
    run_root = Path(output_dir)
    best_path = run_root / "best.pt"
    last_path = run_root / "last.pt"
    history_path = run_root / "training_history.json"
    metrics_path = run_root / "validation_metrics.json"
    config_path = run_root / "run_config.json"
    summary_path = run_root / "run_summary.json"

    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    if not isinstance(git_state, Mapping):
        raise TypeError("git_state must be a mapping")
    if "git_commit" not in git_state or "git_tree_clean" not in git_state:
        raise ValueError("git_state must contain git_commit and git_tree_clean")
    if require_clean_git and git_state["git_tree_clean"] is not True:
        raise RuntimeError("Canonical S3 training requires a clean Git tree")

    set_reproducible_seed(int(settings["seed"]))
    model.to(resolved_device)
    optimizer = build_optimizer(model, config)
    scaler = create_grad_scaler(
        resolved_device,
        use_amp=bool(settings["mixed_precision"]),
    )

    # Validate every locked checkpoint field before spending time on epoch 1.
    build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=0,
        config=config,
        provenance=provenance,
        git_commit=str(git_state["git_commit"]),
        git_tree_clean=bool(git_state["git_tree_clean"]),
        seed=int(settings["seed"]),
        best_valid_roc_auc=0.0,
        validation_metrics={},
    )

    state_paths = (
        best_path,
        last_path,
        history_path,
        metrics_path,
        config_path,
        summary_path,
    )
    if not resume and any(path.exists() for path in state_paths):
        existing = [str(path) for path in state_paths if path.exists()]
        raise FileExistsError(
            "S3 output already contains run state; use resume=True or a new "
            f"output directory: {existing}"
        )
    if resume and not last_path.is_file():
        raise FileNotFoundError(f"Cannot resume without {last_path}")

    history: list[dict[str, object]] = []
    global_step = 0
    start_epoch = 1
    best_valid_roc_auc = -math.inf
    best_epoch = 0
    best_validation_metrics: dict[str, object] = {}
    last_validation_metrics: dict[str, object] = {}
    epochs_without_improvement = 0

    if resume:
        expected = {
            "scorer_version": SCORER_VERSION,
            "config": dict(config),
            "git_commit": str(git_state["git_commit"]),
            "git_tree_clean": bool(git_state["git_tree_clean"]),
            **dict(provenance),
        }
        payload = restore_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            expected=expected,
            map_location=resolved_device,
        )
        resume_state = _resume_extra_state(payload)
        history = resume_state["history"]
        epochs_without_improvement = resume_state[
            "epochs_without_improvement"
        ]
        best_epoch = resume_state["best_epoch"]
        best_validation_metrics = resume_state["best_validation_metrics"]
        best_valid_roc_auc = float(payload["best_valid_roc_auc"])
        last_validation_metrics = dict(payload["validation_metrics"])
        global_step = int(payload["global_step"])
        start_epoch = int(payload["epoch"]) + 1
        if scaler is not None and resume_state["grad_scaler_state_dict"] is not None:
            scaler.load_state_dict(resume_state["grad_scaler_state_dict"])
        _restore_rng_state(resume_state["rng_state"], train_dataloader)
        # A disconnect may happen after last.pt but before best.pt/JSON logs.
        # The complete resume state in last.pt is therefore the recovery source.
        if best_epoch == int(payload["epoch"]):
            save_checkpoint(best_path, payload)

    run_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        "stage": settings["stage"],
        "experiment_name": settings["experiment_name"],
        "scorer_version": SCORER_VERSION,
        "config": dict(config),
        "provenance": dict(provenance),
        "git_state": dict(git_state),
        "device": str(resolved_device),
        "runtime_versions": _runtime_versions(),
        "amp_enabled": bool(
            settings["mixed_precision"] and resolved_device.type == "cuda"
        ),
    }
    _write_json_atomic(config_path, run_config)
    if resume:
        _write_json_atomic(history_path, {"epochs": history})
        _write_json_atomic(
            metrics_path,
            {
                "best_epoch": best_epoch,
                "best_valid_roc_auc": best_valid_roc_auc,
                "best_validation_metrics": best_validation_metrics,
                "last_validation_metrics": last_validation_metrics,
            },
        )

    status = (
        "EARLY_STOPPED"
        if epochs_without_improvement >= int(settings["patience"])
        else "COMPLETED_MAX_EPOCHS"
    )
    for epoch in range(start_epoch, int(settings["max_epochs"]) + 1):
        if status == "EARLY_STOPPED":
            break
        train_result = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            device=resolved_device,
            use_amp=bool(settings["mixed_precision"]),
            scaler=scaler,
            global_step=global_step,
            paired_ranking_weight=float(settings["paired_ranking_weight"]),
        )
        global_step = int(train_result["global_step"])
        validation = _evaluation_snapshot(
            model,
            valid_dataloader,
            device=resolved_device,
            use_amp=bool(settings["mixed_precision"]),
        )
        last_validation_metrics = dict(validation)
        current_auc = float(validation["roc_auc"])
        if not math.isfinite(current_auc) or not 0.0 <= current_auc <= 1.0:
            raise RuntimeError("Validation ROC-AUC must be finite and within [0, 1]")
        improved = current_auc > best_valid_roc_auc
        if improved:
            best_valid_roc_auc = current_auc
            best_epoch = epoch
            best_validation_metrics = dict(validation)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(train_result["loss"]),
            "train_bce_loss": float(
                train_result.get("bce_loss", train_result["loss"])
            ),
            "train_paired_ranking_loss": float(
                train_result.get("paired_ranking_loss", 0.0)
            ),
            "train_mean_paired_margin": float(
                train_result.get("mean_paired_margin", 0.0)
            ),
            "valid_loss": float(validation["loss"]),
            "valid_roc_auc": current_auc,
            "valid_fitb_2way": float(validation["fitb_2way"]),
            "valid_mean_logit_margin": float(
                validation["mean_logit_margin"]
            ),
            "valid_median_logit_margin": float(
                validation["median_logit_margin"]
            ),
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(epoch_record)

        extra_state = {
            "training_history": history,
            "epochs_without_improvement": epochs_without_improvement,
            "best_epoch": best_epoch,
            "best_validation_metrics": best_validation_metrics,
            "rng_state": _capture_rng_state(train_dataloader),
            "grad_scaler_state_dict": (
                scaler.state_dict() if scaler is not None else None
            ),
        }
        checkpoint_payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            config=config,
            provenance=provenance,
            git_commit=str(git_state["git_commit"]),
            git_tree_clean=bool(git_state["git_tree_clean"]),
            seed=int(settings["seed"]),
            best_valid_roc_auc=best_valid_roc_auc,
            validation_metrics=validation,
            extra_state=extra_state,
        )
        save_epoch_checkpoints(
            run_root,
            checkpoint_payload,
            is_best=improved,
        )
        _write_json_atomic(history_path, {"epochs": history})
        _write_json_atomic(
            metrics_path,
            {
                "best_epoch": best_epoch,
                "best_valid_roc_auc": best_valid_roc_auc,
                "best_validation_metrics": best_validation_metrics,
                "last_validation_metrics": last_validation_metrics,
            },
        )

        if epoch_callback is not None:
            epoch_callback(dict(epoch_record))
        if epochs_without_improvement >= int(settings["patience"]):
            status = "EARLY_STOPPED"
            break

    epochs_completed = int(history[-1]["epoch"]) if history else start_epoch - 1
    summary = {
        "stage": settings["stage"],
        "experiment_name": settings["experiment_name"],
        "objective": settings["objective"],
        "paired_ranking_weight": settings["paired_ranking_weight"],
        "status": status,
        "scorer_version": SCORER_VERSION,
        "epochs_completed": epochs_completed,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_valid_roc_auc": best_valid_roc_auc,
        "best_validation_metrics": best_validation_metrics,
        "last_validation_metrics": last_validation_metrics,
        "early_stopping_patience": int(settings["patience"]),
        "output_paths": {
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "training_history": str(history_path),
            "validation_metrics": str(metrics_path),
            "run_config": str(config_path),
            "run_summary": str(summary_path),
        },
    }
    _write_json_atomic(summary_path, summary)
    return {**summary, "history": history}


__all__: Sequence[str] = (
    "BASELINE_OBJECTIVE",
    "PAIRED_RANKING_OBJECTIVE",
    "PairedFamilyBatchSampler",
    "build_optimizer",
    "build_full_training_loaders",
    "build_paired_training_loaders",
    "build_tiny_overfit_loader",
    "create_grad_scaler",
    "evaluate_binary_loss",
    "paired_batch_logit_margins",
    "paired_logistic_ranking_loss",
    "resolve_device",
    "run_full_training",
    "run_tiny_overfit",
    "set_reproducible_seed",
    "train_one_epoch",
)
