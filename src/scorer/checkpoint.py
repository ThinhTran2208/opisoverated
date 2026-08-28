# -*- coding: utf-8 -*-
"""Versioned checkpoint helpers for Type-aware Pairwise Scorer V1."""

from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Keep schema helpers importable in lightweight CI.
    torch = None

from .dataset import (
    CATEGORY_MAPPING_VERSION,
    DATASET_VERSION,
    EMBEDDING_VERSION,
)
from .model import SCORER_VERSION


NEGATIVE_PROTOCOL_VERSION = "negative-v1"

REQUIRED_CHECKPOINT_KEYS = {
    "scorer_version",
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "global_step",
    "config",
    "dataset_version",
    "category_mapping_version",
    "negative_protocol_version",
    "embedding_version",
    "dataset_manifest_sha256",
    "embedding_manifest_sha256",
    "git_commit",
    "git_tree_clean",
    "seed",
    "best_valid_roc_auc",
    "validation_metrics",
}

PROVENANCE_KEYS = (
    "dataset_version",
    "category_mapping_version",
    "negative_protocol_version",
    "embedding_version",
    "dataset_manifest_sha256",
    "embedding_manifest_sha256",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for scorer checkpoints")


def capture_git_state(repo_root: Path | str) -> dict[str, object]:
    """Return the exact commit and clean-tree flag for checkpoint provenance."""

    root = Path(repo_root).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"],
        text=True,
    )
    if not _GIT_COMMIT_PATTERN.fullmatch(commit):
        raise RuntimeError(f"Could not resolve an exact Git commit at {root}")
    return {"git_commit": commit, "git_tree_clean": not bool(status.strip())}


def canonical_provenance(
    *,
    dataset_manifest_sha256: str,
    embedding_manifest_sha256: str,
) -> dict[str, str]:
    """Build the frozen version/hash block required by the checkpoint schema."""

    return {
        "dataset_version": DATASET_VERSION,
        "category_mapping_version": CATEGORY_MAPPING_VERSION,
        "negative_protocol_version": NEGATIVE_PROTOCOL_VERSION,
        "embedding_version": EMBEDDING_VERSION,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "embedding_manifest_sha256": embedding_manifest_sha256,
    }


def _validate_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")


def validate_checkpoint_payload(payload: Mapping[str, object]) -> None:
    """Hard-fail if a checkpoint violates the locked V1 schema."""

    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a mapping")
    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(payload))
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")

    expected_versions = {
        "scorer_version": SCORER_VERSION,
        "dataset_version": DATASET_VERSION,
        "category_mapping_version": CATEGORY_MAPPING_VERSION,
        "negative_protocol_version": NEGATIVE_PROTOCOL_VERSION,
        "embedding_version": EMBEDDING_VERSION,
    }
    for key, expected in expected_versions.items():
        if payload[key] != expected:
            raise ValueError(
                f"Checkpoint {key} mismatch: expected {expected!r}, "
                f"got {payload[key]!r}"
            )

    if not isinstance(payload["model_state_dict"], Mapping):
        raise ValueError("model_state_dict must be a mapping")
    if not isinstance(payload["optimizer_state_dict"], Mapping):
        raise ValueError("optimizer_state_dict must be a mapping")
    if not isinstance(payload["config"], Mapping):
        raise ValueError("config must be a mapping")
    if not isinstance(payload["validation_metrics"], Mapping):
        raise ValueError("validation_metrics must be a mapping")

    for key in ("epoch", "global_step", "seed"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")

    best_auc = payload["best_valid_roc_auc"]
    if isinstance(best_auc, bool) or not isinstance(best_auc, (int, float)):
        raise ValueError("best_valid_roc_auc must be numeric")
    if not math.isfinite(float(best_auc)) or not 0.0 <= float(best_auc) <= 1.0:
        raise ValueError("best_valid_roc_auc must be finite and within [0, 1]")

    _validate_sha256(
        "dataset_manifest_sha256", payload["dataset_manifest_sha256"]
    )
    _validate_sha256(
        "embedding_manifest_sha256", payload["embedding_manifest_sha256"]
    )

    git_commit = payload["git_commit"]
    if not isinstance(git_commit, str) or not _GIT_COMMIT_PATTERN.fullmatch(
        git_commit
    ):
        raise ValueError("git_commit must be an exact lowercase 40-character SHA")
    if not isinstance(payload["git_tree_clean"], bool):
        raise ValueError("git_tree_clean must be boolean")


def build_checkpoint_payload(
    *,
    model,
    optimizer,
    epoch: int,
    global_step: int,
    config: Mapping[str, object],
    provenance: Mapping[str, object],
    git_commit: str,
    git_tree_clean: bool,
    seed: int,
    best_valid_roc_auc: float,
    validation_metrics: Mapping[str, object],
) -> dict[str, object]:
    """Create and validate one canonical checkpoint payload."""

    require_torch()
    missing_provenance = sorted(set(PROVENANCE_KEYS) - set(provenance))
    if missing_provenance:
        raise ValueError(f"Provenance is missing keys: {missing_provenance}")

    payload: dict[str, object] = {
        "scorer_version": SCORER_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": dict(config),
        "dataset_version": provenance["dataset_version"],
        "category_mapping_version": provenance["category_mapping_version"],
        "negative_protocol_version": provenance["negative_protocol_version"],
        "embedding_version": provenance["embedding_version"],
        "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
        "embedding_manifest_sha256": provenance["embedding_manifest_sha256"],
        "git_commit": git_commit,
        "git_tree_clean": git_tree_clean,
        "seed": seed,
        "best_valid_roc_auc": float(best_valid_roc_auc),
        "validation_metrics": dict(validation_metrics),
    }
    validate_checkpoint_payload(payload)
    return payload


def save_checkpoint(path: Path | str, payload: Mapping[str, object]) -> Path:
    """Validate and write a checkpoint to the requested artifact path."""

    require_torch()
    validate_checkpoint_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), destination)
    return destination


def save_epoch_checkpoints(
    output_dir: Path | str,
    payload: Mapping[str, object],
    *,
    is_best: bool,
) -> dict[str, Path | None]:
    """Always write last.pt and update best.pt only on ROC-AUC improvement."""

    root = Path(output_dir)
    last_path = save_checkpoint(root / "last.pt", payload)
    best_path = save_checkpoint(root / "best.pt", payload) if is_best else None
    return {"last": last_path, "best": best_path}


def load_checkpoint(path: Path | str, *, map_location: str | object = "cpu"):
    """Load and validate a checkpoint without mutating a model."""

    require_torch()
    source = Path(path)
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:  # Older supported PyTorch versions.
        payload = torch.load(source, map_location=map_location)
    validate_checkpoint_payload(payload)
    return payload


def _assert_expected_values(
    payload: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    unknown = sorted(set(expected) - set(payload))
    if unknown:
        raise ValueError(f"Unknown expected checkpoint keys: {unknown}")
    mismatches = {
        key: {"expected": value, "actual": payload[key]}
        for key, value in expected.items()
        if payload[key] != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint provenance/config mismatch: {mismatches}")


def restore_checkpoint(
    path: Path | str,
    *,
    model,
    optimizer=None,
    expected: Mapping[str, object] | None = None,
    map_location: str | object = "cpu",
    strict: bool = True,
):
    """Validate provenance first, then restore model and optional optimizer."""

    payload = load_checkpoint(path, map_location=map_location)
    if expected is not None:
        _assert_expected_values(payload, expected)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


__all__: Sequence[str] = (
    "NEGATIVE_PROTOCOL_VERSION",
    "PROVENANCE_KEYS",
    "REQUIRED_CHECKPOINT_KEYS",
    "build_checkpoint_payload",
    "canonical_provenance",
    "capture_git_state",
    "load_checkpoint",
    "restore_checkpoint",
    "save_checkpoint",
    "save_epoch_checkpoints",
    "validate_checkpoint_payload",
)
