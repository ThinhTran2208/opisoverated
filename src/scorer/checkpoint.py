# -*- coding: utf-8 -*-
"""Checkpoint/provenance utilities for Scorer V1.

Large checkpoints live in external artifact storage. Git stores only code/config
and later a small canonical scorer reference. This module implements the locked
checkpoint schema from ``docs/SCORER_CONTRACT_V1.md``.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Mapping

try:
    import torch
except ModuleNotFoundError:  # Keep provenance helpers importable in lightweight CI.
    torch = None


SCORER_VERSION = "type_aware_pairwise_v1"
DATASET_VERSION = "polyvore1000-core7-compat-v2"
CATEGORY_MAPPING_VERSION = "core7-v2"
NEGATIVE_PROTOCOL_VERSION = "negative-v1"
EMBEDDING_VERSION = "fashionclip-512-l2-v1"


class CheckpointContractError(ValueError):
    """Raised when a checkpoint or provenance payload violates the V1 contract."""


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for scorer checkpoint operations")


def sha256_file(path: Path | str) -> str:
    """Return SHA-256 for a local artifact without loading it into memory."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_git_provenance(repo_root: Path | str) -> dict[str, object]:
    """Return live Git commit + clean-tree state for the training code checkout."""

    root = Path(repo_root).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot inspect Git provenance at {root}: {error}") from error

    return {
        "git_commit": commit,
        "git_tree_clean": not bool(status.strip()),
    }


def build_runtime_provenance(runtime_paths, repo_root: Path | str) -> dict[str, object]:
    """Bind a scorer run to the frozen data/embedding manifests and Git commit."""

    dataset_manifest = Path(runtime_paths.scorer_ready_dir) / "dataset_manifest_v2.json"
    embedding_manifest = Path(runtime_paths.embedding_manifest)

    for path in (dataset_manifest, embedding_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    git = inspect_git_provenance(repo_root)
    return {
        "dataset_version": DATASET_VERSION,
        "category_mapping_version": CATEGORY_MAPPING_VERSION,
        "negative_protocol_version": NEGATIVE_PROTOCOL_VERSION,
        "embedding_version": EMBEDDING_VERSION,
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "embedding_manifest_sha256": sha256_file(embedding_manifest),
        **git,
    }


def _require_keys(payload: Mapping[str, object], keys: tuple[str, ...], *, name: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise CheckpointContractError(f"{name} missing required keys: {missing}")


def validate_provenance(
    checkpoint_provenance: Mapping[str, object],
    current_provenance: Mapping[str, object],
    *,
    require_clean_git: bool = False,
) -> None:
    """Hard-fail if a checkpoint is resumed/evaluated against different artifacts."""

    required = (
        "dataset_version",
        "category_mapping_version",
        "negative_protocol_version",
        "embedding_version",
        "dataset_manifest_sha256",
        "embedding_manifest_sha256",
        "git_commit",
        "git_tree_clean",
    )
    _require_keys(checkpoint_provenance, required, name="checkpoint provenance")
    _require_keys(current_provenance, required, name="current provenance")

    artifact_keys = required[:6]
    mismatches = {
        key: (checkpoint_provenance[key], current_provenance[key])
        for key in artifact_keys
        if checkpoint_provenance[key] != current_provenance[key]
    }
    if mismatches:
        raise CheckpointContractError(
            "Checkpoint frozen-artifact provenance mismatch: " + repr(mismatches)
        )

    if require_clean_git and not bool(current_provenance["git_tree_clean"]):
        raise CheckpointContractError("Canonical scorer operation requires a clean Git tree")


def build_checkpoint_payload(
    *,
    model,
    optimizer,
    epoch: int,
    global_step: int,
    config: Mapping[str, object],
    seed: int,
    best_valid_roc_auc: float,
    validation_metrics: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Create the canonical V1 checkpoint payload."""

    require_torch()
    _require_keys(
        provenance,
        (
            "dataset_version",
            "category_mapping_version",
            "negative_protocol_version",
            "embedding_version",
            "dataset_manifest_sha256",
            "embedding_manifest_sha256",
            "git_commit",
            "git_tree_clean",
        ),
        name="provenance",
    )

    return {
        "scorer_version": SCORER_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": dict(config),
        "dataset_version": provenance["dataset_version"],
        "category_mapping_version": provenance["category_mapping_version"],
        "negative_protocol_version": provenance["negative_protocol_version"],
        "embedding_version": provenance["embedding_version"],
        "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
        "embedding_manifest_sha256": provenance["embedding_manifest_sha256"],
        "git_commit": provenance["git_commit"],
        "git_tree_clean": bool(provenance["git_tree_clean"]),
        "seed": int(seed),
        "best_valid_roc_auc": float(best_valid_roc_auc),
        "validation_metrics": dict(validation_metrics),
    }


def save_checkpoint(path: Path | str, **payload_kwargs) -> Path:
    """Atomically save a canonical scorer checkpoint on the local/external filesystem."""

    require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint_payload(**payload_kwargs)

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(
    path: Path | str,
    *,
    model=None,
    optimizer=None,
    map_location="cpu",
    current_provenance: Mapping[str, object] | None = None,
    require_clean_git: bool = False,
) -> dict[str, object]:
    """Load V1 checkpoint, optionally restoring model/optimizer and validating provenance."""

    require_torch()
    source = Path(path)
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        payload = torch.load(source, map_location=map_location)

    if not isinstance(payload, Mapping):
        raise CheckpointContractError("Checkpoint must contain a mapping")
    payload = dict(payload)

    _require_keys(
        payload,
        (
            "scorer_version",
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "global_step",
            "config",
            "seed",
            "best_valid_roc_auc",
            "validation_metrics",
            "dataset_version",
            "category_mapping_version",
            "negative_protocol_version",
            "embedding_version",
            "dataset_manifest_sha256",
            "embedding_manifest_sha256",
            "git_commit",
            "git_tree_clean",
        ),
        name="checkpoint",
    )
    if payload["scorer_version"] != SCORER_VERSION:
        raise CheckpointContractError(
            f"Expected scorer_version={SCORER_VERSION!r}, got {payload['scorer_version']!r}"
        )

    if current_provenance is not None:
        checkpoint_provenance = {
            key: payload[key]
            for key in (
                "dataset_version",
                "category_mapping_version",
                "negative_protocol_version",
                "embedding_version",
                "dataset_manifest_sha256",
                "embedding_manifest_sha256",
                "git_commit",
                "git_tree_clean",
            )
        }
        validate_provenance(
            checkpoint_provenance,
            current_provenance,
            require_clean_git=require_clean_git,
        )

    if model is not None:
        model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    return payload
