# -*- coding: utf-8 -*-
"""CLI to reproduce Calibration V1 from the frozen validation split only."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.calibration.platt import calibration_metrics, fit_platt_calibrator, save_calibrator
from src.scorer.checkpoint import load_checkpoint, sha256_file
from src.scorer.dataset import ScorerDataset, collate_scorer_batch
from src.scorer.evaluate import evaluate_model
from src.scorer.model import TypeAwarePairwiseScorer


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to fit Calibration V1")


def fit_from_validation(
    *,
    checkpoint_path: Path,
    samples_path: Path,
    metadata_path: Path,
    embedding_cache_path: Path,
    output_path: Path,
    batch_size: int = 256,
    device: str = "cpu",
):
    """Score the frozen validation set and fit monotonic Platt calibration."""

    _require_torch()
    dataset = ScorerDataset(
        samples_path,
        metadata_path,
        embedding_cache_path,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_scorer_batch,
    )

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    model = TypeAwarePairwiseScorer.from_config(payload["config"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()

    evaluated = evaluate_model(model, loader, device=torch.device(device))
    predictions = evaluated["predictions"]
    logits = [row["compatibility_logit"] for row in predictions]
    labels = [row["label"] for row in predictions]

    raw_metrics = calibration_metrics(logits, labels)
    calibrator = fit_platt_calibrator(
        logits,
        labels,
        scorer_version=str(payload["scorer_version"]),
        metadata={
            "fit_split": "valid",
            "test_split_loaded": False,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(payload["epoch"]),
            "checkpoint_best_valid_roc_auc": float(payload["best_valid_roc_auc"]),
            "scorer_seed": int(payload["seed"]),
            "dataset_version": payload["dataset_version"],
            "category_mapping_version": payload["category_mapping_version"],
            "negative_protocol_version": payload["negative_protocol_version"],
            "embedding_version": payload["embedding_version"],
            "dataset_manifest_sha256": payload["dataset_manifest_sha256"],
            "embedding_manifest_sha256": payload["embedding_manifest_sha256"],
            "scorer_checkpoint_git_commit": payload["git_commit"],
            "raw_sigmoid_metrics": raw_metrics,
        },
    )
    save_calibrator(calibrator, output_path)
    return calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    calibrator = fit_from_validation(
        checkpoint_path=args.checkpoint,
        samples_path=args.samples,
        metadata_path=args.metadata,
        embedding_cache_path=args.embedding_cache,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Saved Calibration V1: {args.output}")
    print(f"scale={calibrator.scale:.12f} bias={calibrator.bias:.12f}")
    print("TEST SPLIT WAS NOT LOADED.")


if __name__ == "__main__":
    main()
