# -*- coding: utf-8 -*-
"""Training utilities for Type-aware Pairwise Scorer V1.

The primary training path remains deliberately simple:
- BCEWithLogitsLoss
- AdamW
- standard sample-level shuffled training batches
- optional AMP for training when explicitly enabled in config
- canonical validation always in FP32
- validation ROC-AUC checkpoint selection
- patience-based early stopping with an optional minimum-epoch floor
- best.pt + last.pt checkpoints in external artifact storage

The minimum-epoch floor is useful for this scorer because validation ROC-AUC
can be noisy early while the strongest FP32 run continued improving near the
old 30-epoch boundary.
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Mapping

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # Keep source importable in lightweight CI.
    torch = None
    DataLoader = None

from .checkpoint import build_runtime_provenance, save_checkpoint
from .dataset import build_datasets_from_runtime, collate_scorer_batch
from .evaluate import evaluate_predictions
from .model import SCORER_VERSION, TypeAwarePairwiseScorer


class TrainingContractError(ValueError):
    """Raised when scorer training config violates supported semantics."""


def require_torch() -> None:
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for scorer training")


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch for reproducible scorer runs."""

    require_torch()
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _training_config(config: Mapping[str, object]) -> Mapping[str, object]:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TrainingContractError("config['training'] must be a mapping")
    return training


def validate_s3_config(config: Mapping[str, object]) -> None:
    """Reject silent drift away from supported scorer-training semantics."""

    if not isinstance(config, Mapping):
        raise TrainingContractError("config must be a mapping")

    model_config = config.get("model")
    data_config = config.get("data")
    training = _training_config(config)
    selection = config.get("selection")

    if not isinstance(model_config, Mapping) or model_config.get("name") != SCORER_VERSION:
        raise TrainingContractError(f"model.name must be {SCORER_VERSION!r}")
    if not isinstance(data_config, Mapping):
        raise TrainingContractError("config['data'] must be a mapping")
    if int(data_config.get("min_items", -1)) != 2:
        raise TrainingContractError("Min-items-2 experiment locks data.min_items = 2")
    if int(data_config.get("max_items", -1)) != 8:
        raise TrainingContractError("Scorer V1 locks data.max_items = 8")

    if str(training.get("optimizer", "")).lower() != "adamw":
        raise TrainingContractError("Scorer V1 supports optimizer=adamw")
    if str(training.get("lr_scheduler", "")).lower() != "none":
        raise TrainingContractError("Scorer V1 currently supports lr_scheduler=none")
    if str(training.get("gradient_clipping", "")).lower() != "none":
        raise TrainingContractError("Scorer V1 currently supports gradient_clipping=none")

    max_epochs = int(training.get("max_epochs", 0))
    if max_epochs < 1 or max_epochs > 100:
        raise TrainingContractError("max_epochs must be in [1, 100]")

    patience = int(training.get("early_stopping_patience", 0))
    if patience < 1:
        raise TrainingContractError("early_stopping_patience must be >= 1")

    min_epochs = int(training.get("early_stopping_min_epochs", 1))
    if min_epochs < 1 or min_epochs > max_epochs:
        raise TrainingContractError(
            "early_stopping_min_epochs must satisfy 1 <= value <= max_epochs"
        )

    if not isinstance(selection, Mapping):
        raise TrainingContractError("config['selection'] must be a mapping")
    if str(selection.get("primary_metric", "")) != "roc_auc":
        raise TrainingContractError("Scorer V1 selects checkpoints by validation ROC-AUC")
    if str(selection.get("guardrail_metric", "")) != "fitb_2way":
        raise TrainingContractError("Scorer V1 guardrail metric must be fitb_2way")


def build_train_valid_loaders(
    runtime_paths,
    config: Mapping[str, object],
    *,
    num_workers: int = 0,
    pin_memory: bool | None = None,
):
    """Build full frozen train/valid datasets and fresh standard loaders."""

    require_torch()
    validate_s3_config(config)
    training = _training_config(config)
    data_config = config["data"]

    seed = int(training["seed"])
    batch_size = int(training["batch_size"])
    if batch_size < 1:
        raise TrainingContractError("training.batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    datasets, embedding_store = build_datasets_from_runtime(
        runtime_paths,
        splits=("train", "valid"),
    )

    max_items = int(data_config["max_items"])
    collate_fn = partial(collate_scorer_batch, max_items=max_items)

    # Use a DataLoader-local generator so sample order is reproducible and can
    # be recreated by rebuilding the loader from the same seed.
    generator = torch.Generator()
    generator.manual_seed(seed)

    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        generator=generator,
    )
    valid_loader = DataLoader(
        datasets["valid"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return {
        "datasets": datasets,
        "embedding_store": embedding_store,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
    }


def build_optimizer(model, config: Mapping[str, object]):
    """Build AdamW from scorer config."""

    require_torch()
    training = _training_config(config)
    if str(training.get("optimizer", "")).lower() != "adamw":
        raise TrainingContractError("Only AdamW is supported in Scorer V1")

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )


def _autocast_context(device, enabled: bool):
    """Training-only autocast context."""

    enabled = bool(enabled and device.type == "cuda")
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _build_grad_scaler(enabled: bool):
    enabled = bool(enabled and torch.cuda.is_available())
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _forward_batch(model, batch, device):
    return model(
        item_embeddings=batch["item_embeddings"].to(device, non_blocking=True),
        coarse_category_ids=batch["coarse_category_ids"].to(device, non_blocking=True),
        item_mask=batch["item_mask"].to(device, non_blocking=True),
        pair_mask=batch["pair_mask"].to(device, non_blocking=True),
    )["compatibility_logit"]


def train_one_epoch(
    model,
    dataloader,
    *,
    criterion,
    optimizer,
    device,
    scaler=None,
    mixed_precision: bool = False,
) -> tuple[float, int]:
    """Train one standard shuffled BCE epoch."""

    require_torch()
    model.train()
    loss_sum = 0.0
    sample_count = 0
    step_count = 0

    for batch in dataloader:
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, mixed_precision):
            logits = _forward_batch(model, batch, device)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss encountered")

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = int(labels.shape[0])
        loss_sum += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        step_count += 1

    if sample_count == 0:
        raise RuntimeError("Training DataLoader produced zero samples")
    return loss_sum / sample_count, step_count


def evaluate_epoch(
    model,
    dataloader,
    *,
    criterion,
    device,
    mixed_precision: bool = False,
) -> dict[str, object]:
    """Evaluate validation logits and metrics in FP32.

    ``mixed_precision`` is retained for call-site compatibility but is
    intentionally ignored. Validation logits, BCE, ROC-AUC, FITB and margins
    must use a non-autocast forward pass because FP16 can quantize small paired
    margins into exact ties.
    """

    require_torch()
    del mixed_precision
    was_training = bool(model.training)
    model.eval()

    sample_ids: list[str] = []
    pair_ids: list[str | None] = []
    labels_all: list[float] = []
    logits_all: list[float] = []
    loss_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device, non_blocking=True)
            logits = _forward_batch(model, batch, device)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite validation output encountered")

            count = int(labels.shape[0])
            loss_sum += float(loss.detach().item()) * count
            sample_count += count

            sample_ids.extend(str(value) for value in batch["sample_ids"])
            pair_ids.extend(batch["paired_positive_sample_ids"])
            labels_all.extend(float(value) for value in labels.detach().cpu().tolist())
            logits_all.extend(float(value) for value in logits.detach().cpu().tolist())

    if was_training:
        model.train()
    if sample_count == 0:
        raise RuntimeError("Validation DataLoader produced zero samples")

    metrics = evaluate_predictions(
        sample_ids=sample_ids,
        paired_positive_sample_ids=pair_ids,
        labels=labels_all,
        logits=logits_all,
    )
    return {
        "loss": loss_sum / sample_count,
        **metrics,
    }


def fit_scorer(
    model,
    train_loader,
    valid_loader,
    *,
    config: Mapping[str, object],
    checkpoint_dir: Path | str,
    provenance: Mapping[str, object],
    device=None,
) -> dict[str, object]:
    """Train BCE scorer and select best checkpoint by validation ROC-AUC."""

    require_torch()
    validate_s3_config(config)
    training = _training_config(config)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    seed = int(training["seed"])
    seed_everything(seed)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, config)

    requested_amp = bool(training.get("mixed_precision", False))
    mixed_precision = bool(requested_amp and device.type == "cuda")
    scaler = _build_grad_scaler(mixed_precision)

    max_epochs = int(training["max_epochs"])
    patience = int(training["early_stopping_patience"])
    min_epochs = int(training.get("early_stopping_min_epochs", 1))
    min_delta = float(training.get("early_stopping_min_delta", 0.0))

    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    history: list[dict[str, object]] = []
    best_valid_roc_auc = float("-inf")
    best_epoch: int | None = None
    epochs_without_improvement = 0
    global_step = 0
    stopped_early = False

    for epoch in range(1, max_epochs + 1):
        train_loss, steps = train_one_epoch(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            mixed_precision=mixed_precision,
        )
        global_step += steps

        valid_metrics = evaluate_epoch(
            model,
            valid_loader,
            criterion=criterion,
            device=device,
        )
        valid_roc_auc = float(valid_metrics["roc_auc"])

        improved = valid_roc_auc > (best_valid_roc_auc + min_delta)
        if improved:
            best_valid_roc_auc = valid_roc_auc
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "valid_loss": float(valid_metrics["loss"]),
            "valid_roc_auc": valid_roc_auc,
            "valid_fitb_2way": float(valid_metrics["fitb_2way"]),
            "valid_mean_logit_margin": float(valid_metrics["mean_logit_margin"]),
            "valid_median_logit_margin": float(valid_metrics["median_logit_margin"]),
            "best_valid_roc_auc": float(best_valid_roc_auc),
            "improved": bool(improved),
        }
        history.append(row)

        common_checkpoint_kwargs = dict(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            config=config,
            seed=seed,
            best_valid_roc_auc=best_valid_roc_auc,
            validation_metrics=valid_metrics,
            provenance=provenance,
        )
        save_checkpoint(last_path, **common_checkpoint_kwargs)
        if improved:
            save_checkpoint(best_path, **common_checkpoint_kwargs)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.6f} "
            f"valid_loss={float(valid_metrics['loss']):.6f} "
            f"auc={valid_roc_auc:.5f} "
            f"fitb={float(valid_metrics['fitb_2way']):.5f} "
            f"mean_margin={float(valid_metrics['mean_logit_margin']):.5f} "
            f"{'BEST' if improved else ''}"
        )

        if epoch >= min_epochs and epochs_without_improvement >= patience:
            stopped_early = True
            print(
                "Early stopping: validation ROC-AUC did not improve for "
                f"{patience} epochs after minimum epoch {min_epochs}."
            )
            break

    if best_epoch is None or not best_path.is_file():
        raise RuntimeError("Training finished without producing best.pt")

    return {
        "best_epoch": best_epoch,
        "best_valid_roc_auc": best_valid_roc_auc,
        "epochs_ran": len(history),
        "stopped_early": stopped_early,
        "history": history,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "device": str(device),
        "mixed_precision_active": mixed_precision,
        "training_precision": "amp_fp16" if mixed_precision else "fp32",
        "validation_precision": "fp32",
    }


def run_baseline_training(
    runtime_paths,
    config: Mapping[str, object],
    *,
    checkpoint_dir: Path | str,
    repo_root: Path | str,
    num_workers: int = 0,
    device=None,
) -> dict[str, object]:
    """One-call scorer training entry point for notebook/CLI wrappers."""

    require_torch()
    validate_s3_config(config)
    seed = int(_training_config(config)["seed"])
    seed_everything(seed)

    loaders = build_train_valid_loaders(
        runtime_paths,
        config,
        num_workers=num_workers,
    )
    model = TypeAwarePairwiseScorer.from_config(config)
    provenance = build_runtime_provenance(runtime_paths, repo_root)

    result = fit_scorer(
        model,
        loaders["train_loader"],
        loaders["valid_loader"],
        config=config,
        checkpoint_dir=checkpoint_dir,
        provenance=provenance,
        device=device,
    )
    result.update(
        {
            "model": model,
            "datasets": loaders["datasets"],
            "embedding_store": loaders["embedding_store"],
            "provenance": provenance,
        }
    )
    return result
