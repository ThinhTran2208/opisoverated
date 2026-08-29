# -*- coding: utf-8 -*-
"""Training/runtime path for the 2-item compatibility-scorer experiment.

The frozen scorer modules keep their canonical 3-item defaults. This experiment
module explicitly supplies min_items=2 to the dataset, collator and model while
reusing the same Type-aware Pairwise V1 architecture and V5 optimization path.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Mapping, Optional, Sequence

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    torch = None
    DataLoader = None

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from src.data.min2_experiment import (
    EXPERIMENT_DATASET_VERSION,
    EXPERIMENT_TAG,
    LOO_MIN_ORIGINAL_ITEMS,
    MAX_SCORER_ITEMS,
    MIN_SCORER_ITEMS,
    scorer_ready_path,
)
from src.data.runtime_paths import load_runtime_paths

from . import checkpoint as checkpoint_utils
from .dataset import (
    EMBEDDING_DIM,
    EmbeddingStore,
    ScorerDataset,
    build_pair_mask,
    metadata_split_path,
)
from .model import SCORER_VERSION, TypeAwarePairwiseScorer
from .train import (
    _build_grad_scaler,
    build_optimizer,
    evaluate_epoch,
    seed_everything,
    train_one_epoch,
)


class Min2ExperimentContractError(ValueError):
    """Raised when a MIN2 experiment config violates the branch contract."""


def require_torch() -> None:
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for the MIN2 scorer experiment")


def validate_min2_config(config: Mapping[str, object]) -> None:
    if not isinstance(config, Mapping):
        raise Min2ExperimentContractError("config must be a mapping")

    experiment = config.get("experiment")
    model = config.get("model")
    data = config.get("data")
    training = config.get("training")
    selection = config.get("selection")

    if not isinstance(experiment, Mapping):
        raise Min2ExperimentContractError("config['experiment'] must be a mapping")
    if experiment.get("dataset_version") != EXPERIMENT_DATASET_VERSION:
        raise Min2ExperimentContractError(
            f"experiment.dataset_version must be {EXPERIMENT_DATASET_VERSION!r}"
        )
    if int(experiment.get("scorer_min_items", -1)) != MIN_SCORER_ITEMS:
        raise Min2ExperimentContractError("experiment.scorer_min_items must be 2")
    if int(experiment.get("loo_min_original_items", -1)) != LOO_MIN_ORIGINAL_ITEMS:
        raise Min2ExperimentContractError("experiment.loo_min_original_items must be 3")

    if not isinstance(model, Mapping) or model.get("name") != SCORER_VERSION:
        raise Min2ExperimentContractError(f"model.name must be {SCORER_VERSION!r}")
    if not isinstance(data, Mapping):
        raise Min2ExperimentContractError("config['data'] must be a mapping")
    if int(data.get("min_items", -1)) != MIN_SCORER_ITEMS:
        raise Min2ExperimentContractError("data.min_items must be 2")
    if int(data.get("max_items", -1)) != MAX_SCORER_ITEMS:
        raise Min2ExperimentContractError("data.max_items must be 8")

    if not isinstance(training, Mapping):
        raise Min2ExperimentContractError("config['training'] must be a mapping")
    if str(training.get("optimizer", "")).lower() != "adamw":
        raise Min2ExperimentContractError("optimizer must remain adamw")
    if str(training.get("lr_scheduler", "")).lower() != "none":
        raise Min2ExperimentContractError("lr_scheduler must remain none")
    if str(training.get("gradient_clipping", "")).lower() != "none":
        raise Min2ExperimentContractError("gradient_clipping must remain none")
    if bool(training.get("mixed_precision", True)):
        raise Min2ExperimentContractError("canonical MIN2 experiment uses FP32")

    max_epochs = int(training.get("max_epochs", 0))
    min_epochs = int(training.get("early_stopping_min_epochs", 0))
    patience = int(training.get("early_stopping_patience", 0))
    if max_epochs < 1 or min_epochs < 1 or min_epochs > max_epochs:
        raise Min2ExperimentContractError("invalid max/min epoch settings")
    if patience < 1:
        raise Min2ExperimentContractError("early_stopping_patience must be >= 1")

    if not isinstance(selection, Mapping):
        raise Min2ExperimentContractError("config['selection'] must be a mapping")
    if selection.get("primary_metric") != "roc_auc":
        raise Min2ExperimentContractError("primary metric must remain roc_auc")
    if selection.get("guardrail_metric") != "fitb_2way":
        raise Min2ExperimentContractError("guardrail metric must remain fitb_2way")


def collate_min2_scorer_batch(
    samples: Sequence[Mapping[str, object]],
    *,
    max_items: int = MAX_SCORER_ITEMS,
) -> dict:
    """Pad scorer samples while allowing exactly two real items."""

    require_torch()
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    if max_items < MIN_SCORER_ITEMS:
        raise ValueError("max_items must be >= 2")

    batch_size = len(samples)
    embeddings = torch.zeros(
        (batch_size, max_items, EMBEDDING_DIM), dtype=torch.float32
    )
    category_ids = torch.zeros((batch_size, max_items), dtype=torch.long)
    item_mask = torch.zeros((batch_size, max_items), dtype=torch.bool)
    labels = torch.empty(batch_size, dtype=torch.float32)

    for batch_index, sample in enumerate(samples):
        sample_embeddings = sample["item_embeddings"]
        sample_categories = sample["coarse_category_ids"]
        if not isinstance(sample_embeddings, torch.Tensor) or sample_embeddings.ndim != 2:
            raise ValueError("item_embeddings must be rank-2 tensors")
        if sample_embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Expected embedding dim {EMBEDDING_DIM}")
        if not isinstance(sample_categories, torch.Tensor) or sample_categories.ndim != 1:
            raise ValueError("coarse_category_ids must be rank-1 tensors")

        item_count = int(sample_embeddings.shape[0])
        if not MIN_SCORER_ITEMS <= item_count <= max_items:
            raise ValueError(
                f"Sample item count {item_count} outside "
                f"[{MIN_SCORER_ITEMS}, {max_items}]"
            )
        if sample_categories.shape[0] != item_count:
            raise ValueError("Category count does not match embedding row count")

        embeddings[batch_index, :item_count] = sample_embeddings.float()
        category_ids[batch_index, :item_count] = sample_categories.long()
        item_mask[batch_index, :item_count] = True
        labels[batch_index] = float(sample["label"])

    return {
        "item_embeddings": embeddings,
        "coarse_category_ids": category_ids,
        "item_mask": item_mask,
        "pair_mask": build_pair_mask(item_mask),
        "labels": labels,
        "sample_ids": [str(sample["sample_id"]) for sample in samples],
        "source_kit_ids": [str(sample.get("source_kit_id", "")) for sample in samples],
        "paired_positive_sample_ids": [
            sample.get("paired_positive_sample_id") for sample in samples
        ],
        "item_ids": [list(sample["item_ids"]) for sample in samples],
        "negative_metadata": [sample.get("negative_metadata") for sample in samples],
    }


def build_min2_datasets(runtime_paths, *, splits=("train", "valid")):
    """Load MIN2 scorer-ready artifacts with an explicit two-item lower bound."""

    store = EmbeddingStore(runtime_paths.embedding_cache)
    datasets = {
        split: ScorerDataset(
            scorer_ready_path(runtime_paths.scorer_ready_dir, split),
            metadata_split_path(runtime_paths.core7_dir, split),
            embedding_store=store,
            min_items=MIN_SCORER_ITEMS,
            max_items=MAX_SCORER_ITEMS,
        )
        for split in splits
    }
    return datasets, store


def build_train_valid_loaders_min2(
    runtime_paths,
    config: Mapping[str, object],
    *,
    num_workers: int = 0,
    pin_memory: bool | None = None,
):
    """Build reproducible train/valid loaders for the MIN2 experiment."""

    require_torch()
    validate_min2_config(config)
    training = config["training"]
    batch_size = int(training["batch_size"])
    seed = int(training["seed"])
    if batch_size < 1:
        raise Min2ExperimentContractError("training.batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    datasets, embedding_store = build_min2_datasets(
        runtime_paths, splits=("train", "valid")
    )
    collate_fn = partial(
        collate_min2_scorer_batch,
        max_items=int(config["data"]["max_items"]),
    )

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


def build_min2_provenance(runtime_paths, repo_root: Path | str) -> dict[str, object]:
    """Bind a checkpoint to MIN2 experiment manifests instead of frozen V2."""

    dataset_manifest = Path(runtime_paths.scorer_ready_dir) / "dataset_manifest_min2_exp_v1.json"
    embedding_manifest = Path(runtime_paths.embedding_manifest)
    for path in (dataset_manifest, embedding_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    git = checkpoint_utils.inspect_git_provenance(repo_root)
    return {
        "dataset_version": EXPERIMENT_DATASET_VERSION,
        "category_mapping_version": "core7-v2",
        "negative_protocol_version": "negative-v1",
        "embedding_version": "fashionclip-512-l2-v1",
        "dataset_manifest_sha256": checkpoint_utils.sha256_file(dataset_manifest),
        "embedding_manifest_sha256": checkpoint_utils.sha256_file(embedding_manifest),
        **git,
    }


def fit_min2_scorer(
    model,
    train_loader,
    valid_loader,
    *,
    config: Mapping[str, object],
    checkpoint_dir: Path | str,
    provenance: Mapping[str, object],
    device=None,
) -> dict[str, object]:
    """Train the MIN2 scorer with the unchanged V5 BCE/ROC-AUC protocol."""

    require_torch()
    validate_min2_config(config)
    training = config["training"]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    seed = int(training["seed"])
    seed_everything(seed)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, config)
    mixed_precision = False
    scaler = _build_grad_scaler(False)

    max_epochs = int(training["max_epochs"])
    patience = int(training["early_stopping_patience"])
    min_epochs = int(training["early_stopping_min_epochs"])
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

        checkpoint_kwargs = dict(
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
        checkpoint_utils.save_checkpoint(last_path, **checkpoint_kwargs)
        if improved:
            checkpoint_utils.save_checkpoint(best_path, **checkpoint_kwargs)

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"valid_loss={float(valid_metrics['loss']):.6f} "
            f"auc={valid_roc_auc:.5f} "
            f"fitb={float(valid_metrics['fitb_2way']):.5f} "
            f"{'BEST' if improved else ''}"
        )

        if epoch >= min_epochs and epochs_without_improvement >= patience:
            stopped_early = True
            break

    return {
        "experiment": EXPERIMENT_TAG,
        "dataset_version": EXPERIMENT_DATASET_VERSION,
        "min_scorer_items": MIN_SCORER_ITEMS,
        "loo_min_original_items": LOO_MIN_ORIGINAL_ITEMS,
        "best_valid_roc_auc": best_valid_roc_auc,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "history": history,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }


def load_config(path: Path | str) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load scorer YAML config")
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Scorer config must contain a YAML mapping")
    validate_min2_config(payload)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Type-aware Pairwise on MIN2 data")
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=Path("configs/data_paths.min2_experiment.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scorer_type_aware_pairwise_min2_experiment.yaml"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/checkpoints/type_aware_pairwise_v1/min2_exp_v1_seed42"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    runtime_paths = load_runtime_paths(config_path=args.paths_config)
    config = load_config(args.config)

    seed_everything(int(config["training"]["seed"]))
    model = TypeAwarePairwiseScorer.from_config(config)
    loaders = build_train_valid_loaders_min2(
        runtime_paths,
        config,
        num_workers=args.num_workers,
    )
    provenance = build_min2_provenance(runtime_paths, runtime_paths.repo_root)
    result = fit_min2_scorer(
        model,
        loaders["train_loader"],
        loaders["valid_loader"],
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        provenance=provenance,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
