#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portable runner for frozen EVALUATION3 Test2000 human-alignment metrics.

Unlike the historical Colab script, this runner does not mount/search Google
Drive. Pass the frozen artifacts explicitly.

The default bootstrap backend reproduces the frozen NumPy/SciPy implementation.
The core module itself remains importable without those packages, and its
reference backend is standard-library only for small unit tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from src.evaluation.eval3_human_metrics import evaluate_subset


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_status") != "FROZEN":
        raise ValueError("Protocol artifact must have protocol_status=FROZEN")
    return protocol


def join_inputs(
    predictions: list[dict[str, str]],
    human_labels: list[dict[str, str]],
    manifest: list[dict[str, str]],
) -> list[dict[str, object]]:
    pred = {str(r["outfit_id"]): r for r in predictions}
    human = {str(r["outfit_id"]): r for r in human_labels}

    if len(pred) != len(predictions):
        raise ValueError("Duplicate outfit_id in predictions")
    if len(human) != len(human_labels):
        raise ValueError("Duplicate outfit_id in human labels")

    joined = []
    for row in manifest:
        oid = str(row["outfit_id"])
        if oid not in pred:
            raise ValueError(f"Missing prediction for outfit {oid}")
        if oid not in human:
            raise ValueError(f"Missing human label for outfit {oid}")

        cmt = int(human[oid]["cmt_original"])
        quality = int(human[oid]["human_ordinal_quality"])
        expected_quality = 4 - cmt
        if quality != expected_quality:
            raise ValueError(
                f"Human ordinal mapping mismatch for {oid}: "
                f"{quality} != {expected_quality}"
            )

        joined.append(
            {
                "outfit_id": oid,
                "compatibility_logit": float(
                    pred[oid]["compatibility_logit"]
                ),
                "human_label": str(human[oid]["human_label"]),
                "human_ordinal_quality": quality,
                "subset_full": parse_bool(row["subset_full"]),
                "subset_no_trainvalid_image_overlap": parse_bool(
                    row["subset_no_trainvalid_image_overlap"]
                ),
                "subset_no_full_image_overlap": parse_bool(
                    row["subset_no_full_image_overlap"]
                ),
            }
        )

    return joined


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--human-labels", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-backend",
        choices=("frozen", "reference"),
        default="frozen",
        help="Use 'frozen' to reproduce reported NumPy/SciPy bootstrap CIs.",
    )
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    expected_manifest_sha = protocol["frozen_artifacts"]["evaluation_manifest"]["sha256"]
    actual_manifest_sha = sha256_file(args.evaluation_manifest)
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(
            "Evaluation manifest SHA256 mismatch: "
            f"{actual_manifest_sha} != {expected_manifest_sha}"
        )

    joined = join_inputs(
        read_csv(args.predictions),
        read_csv(args.human_labels),
        read_csv(args.evaluation_manifest),
    )

    bootstrap = protocol["metrics"]["bootstrap_policy"]
    subset_specs = [
        protocol["evaluation_subsets"]["no_full_image_overlap"],
        protocol["evaluation_subsets"]["no_trainvalid_image_overlap"],
        protocol["evaluation_subsets"]["full"],
    ]

    main_rows = []
    class_rows = []
    pair_rows = []

    for spec in subset_specs:
        mask_name = spec["membership_column"]
        rows = [r for r in joined if bool(r[mask_name])]
        if len(rows) != int(spec["count"]):
            raise ValueError(
                f"{spec['name']} count mismatch: {len(rows)} != {spec['count']}"
            )

        main, classwise, pairwise = evaluate_subset(
            rows,
            str(spec["name"]),
            bootstrap,
            bootstrap_backend=args.bootstrap_backend,
        )
        main_rows.append(main)
        class_rows.extend(classwise)
        pair_rows.extend(pairwise)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    write_csv(out / "EVAL3-Test2000-metrics-main.csv", main_rows)
    write_csv(
        out / "EVAL3-Test2000-classwise-logit-summary.csv",
        class_rows,
    )
    write_csv(
        out / "EVAL3-Test2000-pairwise-ordering.csv",
        pair_rows,
    )

    result = {
        "protocol_version": protocol["protocol_version"],
        "protocol_status": protocol["protocol_status"],
        "evaluation_dataset": protocol["evaluation_split_name"],
        "bootstrap": bootstrap,
        "subset_results": main_rows,
        "classwise_logit_summary": class_rows,
        "pairwise_ordering": pair_rows,
        "inputs": {
            "predictions": {
                "path": str(args.predictions),
                "sha256": sha256_file(args.predictions),
            },
            "human_labels": {
                "path": str(args.human_labels),
                "sha256": sha256_file(args.human_labels),
            },
            "evaluation_manifest": {
                "path": str(args.evaluation_manifest),
                "sha256": actual_manifest_sha,
            },
            "protocol_artifact": {
                "path": str(args.protocol),
                "sha256": sha256_file(args.protocol),
            },
        },
    }

    (out / "EVAL3-Test2000-metrics-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("EVAL3 human-alignment metrics: PASS")
    for row in main_rows:
        print(
            f"{row['subset']}: N={row['N_total']} "
            f"tau-b={row['kendall_tau_b']:.6f} "
            f"rho={row['spearman_rho']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
