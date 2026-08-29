# -*- coding: utf-8 -*-
"""End-to-end data helpers for the 2-item scorer experiment.

This module deliberately does not mutate the frozen Core-7 V2 artifacts. It
reuses the frozen Core-7 category mapping and Negative V1 semantics, while
regenerating positives/negatives with a scorer minimum of two items under an
experiment-specific dataset version and artifact directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from . import build_core7_scorer_dataset as scorer_builder
from .prepare_core7_dataset_v2 import prepare_clean_positive_split_v2
from .runtime_paths import load_runtime_paths
from .validate_core7_embeddings import validate_core7_embedding_coverage


SPLITS = ("train", "valid", "test")
MIN_SCORER_ITEMS = 2
MAX_SCORER_ITEMS = 8
LOO_MIN_ORIGINAL_ITEMS = 3
EXPERIMENT_DATASET_VERSION = "polyvore1000-core7-compat-min2-exp-v1"
EXPERIMENT_TAG = "min2_exp_v1"
NEGATIVE_VERSION = scorer_builder.NEGATIVE_VERSION
NEGATIVE_TYPE = scorer_builder.NEGATIVE_TYPE
CATEGORY_MAPPING_VERSION = scorer_builder.CATEGORY_MAPPING_VERSION
ITEM_METADATA_VERSION = scorer_builder.ITEM_METADATA_VERSION
DEFAULT_SEED = scorer_builder.DEFAULT_SEED


def category_clean_path(core7_dir: Path | str, split: str) -> Path:
    return Path(core7_dir) / f"category_clean_{split}.jsonl"


def metadata_path(core7_dir: Path | str, split: str) -> Path:
    return Path(core7_dir) / f"core7_item_metadata_v1_{split}.jsonl"


def scorer_ready_path(scorer_ready_dir: Path | str, split: str) -> Path:
    return Path(scorer_ready_dir) / f"scorer_ready_{EXPERIMENT_TAG}_{split}.jsonl"


def negative_path(scorer_ready_dir: Path | str, split: str) -> Path:
    return Path(scorer_ready_dir) / f"negative_{EXPERIMENT_TAG}_{split}.jsonl"


def embedding_report_path(core7_dir: Path | str) -> Path:
    return Path(core7_dir) / f"embedding_validation_{EXPERIMENT_TAG}.json"


def prepare_min2_positives(
    runtime_paths,
    *,
    mapping_path: Path | str,
    debug_limit: int | None = None,
) -> dict:
    """Run Core-7 category cleaning with the experiment minimum of two items."""

    runtime_paths.core7_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    for split in SPLITS:
        reports[split] = prepare_clean_positive_split_v2(
            split=split,
            output_path=category_clean_path(runtime_paths.core7_dir, split),
            item_metadata_output_path=metadata_path(runtime_paths.core7_dir, split),
            mapping_path=mapping_path,
            min_items=MIN_SCORER_ITEMS,
            debug_limit=debug_limit,
        )
        if int(reports[split].get("min_items", -1)) != MIN_SCORER_ITEMS:
            raise RuntimeError(f"split={split} did not use min_items={MIN_SCORER_ITEMS}")
    return {
        "experiment": EXPERIMENT_TAG,
        "dataset_version": EXPERIMENT_DATASET_VERSION,
        "min_scorer_items": MIN_SCORER_ITEMS,
        "max_scorer_items": MAX_SCORER_ITEMS,
        "loo_min_original_items": LOO_MIN_ORIGINAL_ITEMS,
        "splits": reports,
    }


def validate_min2_embeddings(
    runtime_paths,
    *,
    mapping_path: Path | str,
    report_path: Path | str | None = None,
) -> dict:
    """Validate embedding/metadata coverage for the regenerated MIN2 positives."""

    positives = {
        split: category_clean_path(runtime_paths.core7_dir, split) for split in SPLITS
    }
    metadata = {split: metadata_path(runtime_paths.core7_dir, split) for split in SPLITS}
    destination = Path(report_path) if report_path else embedding_report_path(runtime_paths.core7_dir)

    report = validate_core7_embedding_coverage(
        mapping_path=mapping_path,
        cache_path=runtime_paths.embedding_cache,
        manifest_path=runtime_paths.embedding_manifest,
        positives_by_split=positives,
        metadata_by_split=metadata,
        report_path=destination,
    )
    report["experiment"] = EXPERIMENT_TAG
    report["dataset_version"] = EXPERIMENT_DATASET_VERSION
    report["minimum_outfit_items"] = MIN_SCORER_ITEMS
    report["loo_min_original_items"] = LOO_MIN_ORIGINAL_ITEMS

    # The shared validator writes before the experiment annotations are added,
    # so rewrite the report to keep the on-disk evidence self-describing.
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def _write_experiment_manifests(
    *,
    runtime_paths,
    records_by_split: Mapping[str, Sequence[dict]],
    validation_report: Mapping[str, object],
    embedding_report_file: Path,
    seed: int,
    git_provenance: Mapping[str, object],
) -> tuple[Path, Path]:
    output_dir = Path(runtime_paths.scorer_ready_dir)
    split_entries: dict[str, dict] = {}
    for split in SPLITS:
        scorer_file = scorer_ready_path(output_dir, split)
        neg_file = negative_path(output_dir, split)
        meta_file = metadata_path(runtime_paths.core7_dir, split)
        records = records_by_split[split]
        split_entries[split] = {
            "scorer_file": str(scorer_file),
            "scorer_sha256": scorer_builder.sha256_file(scorer_file),
            "negative_file": str(neg_file),
            "negative_sha256": scorer_builder.sha256_file(neg_file),
            "item_metadata_file": str(meta_file),
            "item_metadata_sha256": scorer_builder.sha256_file(meta_file),
            "sample_count": len(records),
            "positive_count": sum(int(row["label"]) == 1 for row in records),
            "negative_count": sum(int(row["label"]) == 0 for row in records),
        }

    split_manifest = {
        "manifest_version": "split-manifest-min2-exp-v1",
        "dataset_version": EXPERIMENT_DATASET_VERSION,
        "split_policy": "official_train_valid_test",
        "minimum_outfit_items": MIN_SCORER_ITEMS,
        "splits": split_entries,
    }
    dataset_manifest = {
        "manifest_version": "dataset-manifest-min2-exp-v1",
        "dataset_version": EXPERIMENT_DATASET_VERSION,
        "status": validation_report["status"],
        "experimental": True,
        "base_dataset_version": scorer_builder.DATASET_VERSION_V2,
        "category_mapping_version": CATEGORY_MAPPING_VERSION,
        "item_metadata_version": ITEM_METADATA_VERSION,
        "negative_protocol_version": NEGATIVE_VERSION,
        "negative_type": NEGATIVE_TYPE,
        "negative_seed": seed,
        "minimum_outfit_items": MIN_SCORER_ITEMS,
        "maximum_outfit_items": MAX_SCORER_ITEMS,
        "loo_min_original_items": LOO_MIN_ORIGINAL_ITEMS,
        "embedding_validation_report": str(embedding_report_file),
        "embedding_validation_report_sha256": scorer_builder.sha256_file(
            embedding_report_file
        ),
        "embedding_manifest": str(runtime_paths.embedding_manifest),
        "embedding_manifest_sha256": scorer_builder.sha256_file(
            runtime_paths.embedding_manifest
        ),
        "git_commit": git_provenance.get("git_commit"),
        "git_tree_clean": git_provenance.get("git_tree_clean"),
        "splits": split_entries,
    }

    split_manifest_path = output_dir / "split_manifest_min2_exp_v1.json"
    dataset_manifest_path = output_dir / "dataset_manifest_min2_exp_v1.json"
    scorer_builder.write_json(split_manifest, split_manifest_path)
    scorer_builder.write_json(dataset_manifest, dataset_manifest_path)
    return split_manifest_path, dataset_manifest_path


def build_min2_scorer_dataset(
    runtime_paths,
    *,
    mapping_path: Path | str,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict:
    """Generate Negative V1 pairs and validate scorer-ready data at min_items=2."""

    source_dir = Path(runtime_paths.core7_dir)
    destination = Path(runtime_paths.scorer_ready_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_file = embedding_report_path(source_dir)

    protected = [
        *(negative_path(destination, split) for split in SPLITS),
        *(scorer_ready_path(destination, split) for split in SPLITS),
        destination / "final_validation_min2_exp_v1.json",
        destination / "split_manifest_min2_exp_v1.json",
        destination / "dataset_manifest_min2_exp_v1.json",
    ]
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite MIN2 experiment artifacts: "
            + repr([str(path) for path in existing])
        )

    embedding_report = scorer_builder.read_json(report_file)
    input_verification = scorer_builder.verify_embedding_report_inputs(
        embedding_report,
        data_dir=source_dir,
        category_mapping_path=mapping_path,
        embedding_cache_path=runtime_paths.embedding_cache,
        embedding_manifest_path=runtime_paths.embedding_manifest,
    )

    records_by_split: dict[str, list[dict]] = {}
    metadata_by_split: dict[str, list[dict]] = {}
    sampling_reports: dict[str, dict] = {}
    merge_reports: dict[str, dict] = {}

    for split_index, split in enumerate(SPLITS):
        positives = scorer_builder.read_jsonl(category_clean_path(source_dir, split))
        metadata = scorer_builder.read_jsonl(metadata_path(source_dir, split))
        negatives, sampling_report = scorer_builder.generate_negative_records(
            positives,
            metadata,
            split=split,
            seed=seed + split_index,
        )
        scorer_records, merge_report = scorer_builder.merge_positive_negative_families(
            positives, negatives
        )

        scorer_builder.write_jsonl(negatives, negative_path(destination, split))
        scorer_builder.write_jsonl(
            scorer_records, scorer_ready_path(destination, split)
        )
        sampling_report["merge"] = merge_report
        scorer_builder.write_json(
            sampling_report,
            destination / f"negative_sampling_{EXPERIMENT_TAG}_{split}_report.json",
        )

        records_by_split[split] = scorer_records
        metadata_by_split[split] = metadata
        sampling_reports[split] = sampling_report
        merge_reports[split] = merge_report

    final_report = scorer_builder.validate_all_splits(
        records_by_split,
        metadata_by_split,
        embedding_report=embedding_report,
        sampling_reports=sampling_reports,
        min_items=MIN_SCORER_ITEMS,
    )
    final_report["dataset_version"] = EXPERIMENT_DATASET_VERSION
    final_report["base_dataset_version"] = scorer_builder.DATASET_VERSION_V2
    final_report["experimental"] = True
    final_report["min_items"] = MIN_SCORER_ITEMS
    final_report["max_items"] = MAX_SCORER_ITEMS
    final_report["loo_min_original_items"] = LOO_MIN_ORIGINAL_ITEMS
    final_report["embedding_input_verification"] = input_verification
    final_report["merge_reports"] = merge_reports
    final_validation_path = destination / "final_validation_min2_exp_v1.json"
    scorer_builder.write_json(final_report, final_validation_path)

    git_provenance = scorer_builder.inspect_git_provenance(runtime_paths.repo_root)
    split_manifest_path, dataset_manifest_path = _write_experiment_manifests(
        runtime_paths=runtime_paths,
        records_by_split=records_by_split,
        validation_report=final_report,
        embedding_report_file=report_file,
        seed=seed,
        git_provenance=git_provenance,
    )

    return {
        "status": final_report["status"],
        "dataset_version": EXPERIMENT_DATASET_VERSION,
        "min_scorer_items": MIN_SCORER_ITEMS,
        "max_scorer_items": MAX_SCORER_ITEMS,
        "loo_min_original_items": LOO_MIN_ORIGINAL_ITEMS,
        "final_validation_path": str(final_validation_path),
        "split_manifest_path": str(split_manifest_path),
        "dataset_manifest_path": str(dataset_manifest_path),
    }


def _default_mapping(runtime_paths) -> Path:
    return runtime_paths.repo_root / "configs" / "category_mapping_core7_v2.json"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MIN2 data experiment")
    parser.add_argument(
        "stage",
        choices=("prepare", "validate", "build", "all"),
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=Path("configs/data_paths.min2_experiment.json"),
    )
    parser.add_argument("--mapping", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--debug-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    runtime_paths = load_runtime_paths(config_path=args.paths_config)
    mapping = args.mapping or _default_mapping(runtime_paths)

    result: dict[str, object] = {
        "experiment": EXPERIMENT_TAG,
        "dataset_version": EXPERIMENT_DATASET_VERSION,
    }
    if args.stage in ("prepare", "all"):
        result["prepare"] = prepare_min2_positives(
            runtime_paths,
            mapping_path=mapping,
            debug_limit=args.debug_limit,
        )
    if args.stage in ("validate", "all"):
        result["validate"] = validate_min2_embeddings(
            runtime_paths,
            mapping_path=mapping,
        )
    if args.stage in ("build", "all"):
        result["build"] = build_min2_scorer_dataset(
            runtime_paths,
            mapping_path=mapping,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    build_result = result.get("build")
    if isinstance(build_result, Mapping):
        return 0 if build_result.get("status") == "READY_TO_TRAIN" else 1
    validate_result = result.get("validate")
    if isinstance(validate_result, Mapping):
        return 0 if validate_result.get("pass") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
