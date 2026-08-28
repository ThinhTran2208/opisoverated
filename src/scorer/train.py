# -*- coding: utf-8 -*-
"""Canonical training helpers for Type-aware Pairwise Scorer V1.

The notebook is only an experiment wrapper.  Loss, optimizer, reproducibility,
tiny-family selection, and the training loop live here so S2.5 and S3 exercise
the same implementation.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from functools import partial

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


DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
DEFAULT_TINY_FAMILY_COUNT = 32
DEFAULT_TINY_MAX_EPOCHS = 300


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
) -> dict[str, int | float]:
    """Train one epoch with canonical BCE-with-logits semantics."""

    require_training_dependencies()
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.train()
    criterion = criterion or nn.BCEWithLogitsLoss()
    amp_enabled = bool(use_amp and resolved_device.type == "cuda")
    scaler = scaler if scaler is not None else create_grad_scaler(
        resolved_device, use_amp=amp_enabled
    )

    loss_sum = 0.0
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
            loss = criterion(logits, labels)
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
        sample_count += current_batch_size
        batch_count += 1
        global_step += 1

    if sample_count == 0:
        raise ValueError("Training dataloader yielded no samples")
    return {
        "loss": loss_sum / sample_count,
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


def _tiny_snapshot(model, dataloader, *, device, use_amp: bool) -> dict[str, object]:
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

    initial = _tiny_snapshot(
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
        final = _tiny_snapshot(
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


__all__: Sequence[str] = (
    "build_optimizer",
    "build_tiny_overfit_loader",
    "create_grad_scaler",
    "evaluate_binary_loss",
    "resolve_device",
    "run_tiny_overfit",
    "set_reproducible_seed",
    "train_one_epoch",
)
