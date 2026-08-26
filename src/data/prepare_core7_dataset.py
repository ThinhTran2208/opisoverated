# -*- coding: utf-8 -*-
"""Create Core-7 category-clean positive outfits before negative sampling.

This module owns only the category-drop stage of data processing:

1. load the official Polyvore1000 split;
2. map every ``master_category`` to one Core-7 type or ``DROP``;
3. remove items mapped to ``DROP``;
4. recompute outfit length and keep outfits with at least ``min_items``;
5. export canonical category-clean positive JSONL;
6. export a versioned Core-7 item-metadata JSONL for every serialized item.

The item-metadata artifact preserves both the source ``master_category`` and
project-defined ``coarse_category``. This keeps category information outside
of the outfit JSONL without losing it before serialization, and gives the next
negative-sampling stage a stable lookup keyed by ``item_id``.

This is an intermediate data stage. Image decoding and embedding-coverage
validation still have to remove invalid items and recompute outfit length
before the dataset can be marked ``READY_TO_TRAIN``.

It deliberately does **not** create negatives. After item removal, the old
negative records and ``swapped_item_index`` values are no longer valid. The
negative generator must consume the clean positive output plus the matching
item-metadata artifact and create fresh negative provenance.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .build_compatibility_dataset import (
    DATASET_NAME,
    build_item_indexes,
    load_required_datasets,
)


CORE_CATEGORIES = (
    "TOP",
    "BOTTOM",
    "DRESS",
    "OUTERWEAR",
    "SHOES",
    "BAG",
    "HAT",
)
DROP_CATEGORY = "DROP"
ALLOWED_MAPPING_VALUES = frozenset((*CORE_CATEGORIES, DROP_CATEGORY))

DEFAULT_MIN_OUTFIT_ITEMS = 3
DEFAULT_MAPPING_PATH = Path("configs/category_mapping_core7_v1.json")
DEFAULT_OUTPUT_DIR = Path("data/processed/core7_v1")
CORE7_ITEM_METADATA_VERSION = "core7-item-metadata-v1"


def load_category_mapping(
    path: Path | str,
) -> Tuple[dict, Dict[str, str]]:
    """Load and validate the versioned master-to-coarse mapping."""

    mapping_path = Path(path)
    with mapping_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(payload, dict):
        raise ValueError("Category mapping JSON must contain an object")

    raw_mapping = payload.get("mapping", payload)
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ValueError("Category mapping is empty or has invalid shape")

    mapping: Dict[str, str] = {}
    invalid_values: Dict[str, object] = {}

    for raw_master, raw_coarse in raw_mapping.items():
        master = str(raw_master).strip()
        coarse = str(raw_coarse).strip().upper()

        if not master:
            raise ValueError("Category mapping contains an empty master category")
        if coarse not in ALLOWED_MAPPING_VALUES:
            invalid_values[master] = raw_coarse
            continue
        if master in mapping:
            raise ValueError(f"Duplicate master category in mapping: {master}")

        mapping[master] = coarse

    if invalid_values:
        raise ValueError(
            "Category mapping contains invalid decisions: "
            f"{invalid_values}. Allowed values: "
            f"{sorted(ALLOWED_MAPPING_VALUES)}"
        )

    metadata = {key: value for key, value in payload.items() if key != "mapping"}
    metadata.setdefault("mapping_version", mapping_path.stem)
    metadata.setdefault("status", "unknown")

    declared_counts = metadata.get("decision_counts")
    if declared_counts is not None:
        if not isinstance(declared_counts, dict):
            raise ValueError("decision_counts must be a JSON object")

        actual_counts = dict(Counter(mapping.values()))
        normalized_declared_counts = {
            str(decision).strip().upper(): int(count)
            for decision, count in declared_counts.items()
        }
        if normalized_declared_counts != actual_counts:
            raise ValueError(
                "decision_counts does not match mapping values: "
                f"declared={normalized_declared_counts}, actual={actual_counts}"
            )

    return metadata, mapping


def validate_mapping_coverage(
    master_categories: Iterable[str],
    mapping: Mapping[str, str],
) -> dict:
    """Require an explicit KEEP/DROP decision for every observed category."""

    observed = {str(category).strip() for category in master_categories}
    observed.discard("")
    mapped = set(mapping)

    missing = sorted(observed - mapped)
    unused = sorted(mapped - observed)

    report = {
        "observed_master_category_count": len(observed),
        "mapped_master_category_count": len(mapped),
        "missing_master_categories": missing,
        "unused_mapping_categories": unused,
        "coverage": 1.0 if not observed else len(observed - set(missing)) / len(observed),
    }

    if missing:
        raise ValueError(
            "Category mapping is incomplete. Missing master categories: "
            f"{missing}"
        )

    return report


def filter_items_by_core_category(
    kit_to_items: Mapping[str, Sequence[dict]],
    mapping: Mapping[str, str],
) -> Tuple[Dict[str, List[dict]], dict]:
    """Drop non-Core-7 items while preserving item order inside each kit."""

    observed_categories = {
        item["category"] for items in kit_to_items.values() for item in items
    }
    coverage_report = validate_mapping_coverage(observed_categories, mapping)

    filtered: Dict[str, List[dict]] = {}
    master_before = Counter()
    master_after = Counter()
    coarse_after = Counter()
    dropped_master = Counter()

    raw_item_count = 0
    kept_item_count = 0

    for kit_id, outfit_items in kit_to_items.items():
        kept_items: List[dict] = []

        for item in outfit_items:
            raw_item_count += 1
            master = item["category"]
            decision = mapping[master]
            master_before[master] += 1

            if decision == DROP_CATEGORY:
                dropped_master[master] += 1
                continue

            clean_item = dict(item)
            clean_item["master_category"] = master
            clean_item["coarse_category"] = decision
            kept_items.append(clean_item)

            kept_item_count += 1
            master_after[master] += 1
            coarse_after[decision] += 1

        filtered[str(kit_id)] = kept_items

    report = {
        **coverage_report,
        "raw_item_count": raw_item_count,
        "kept_item_count": kept_item_count,
        "dropped_item_count": raw_item_count - kept_item_count,
        "kept_item_rate": 0.0 if raw_item_count == 0 else kept_item_count / raw_item_count,
        "master_category_count_before": len(master_before),
        "master_category_count_after": len(master_after),
        "coarse_category_item_counts": dict(sorted(coarse_after.items())),
        "dropped_master_category_item_counts": dict(sorted(dropped_master.items())),
    }

    return filtered, report


def _take_debug_subset(rows, debug_limit: Optional[int]):
    if debug_limit is None:
        return rows
    if debug_limit < 1:
        raise ValueError("debug_limit must be >= 1 when provided")

    limit = min(debug_limit, len(rows))
    if hasattr(rows, "select"):
        return rows.select(range(limit))
    return list(rows)[:limit]


def build_clean_positive_samples(
    kits_ds,
    raw_kit_to_items: Mapping[str, Sequence[dict]],
    filtered_kit_to_items: Mapping[str, Sequence[dict]],
    *,
    min_items: int = DEFAULT_MIN_OUTFIT_ITEMS,
    debug_limit: Optional[int] = None,
) -> Tuple[List[dict], dict]:
    """Build canonical positive samples after category filtering."""

    if min_items < 1:
        raise ValueError("min_items must be >= 1")

    kits_to_process = _take_debug_subset(kits_ds, debug_limit)
    clean_samples: List[dict] = []
    length_before = Counter()
    length_after = Counter()

    stats = {
        "kits_processed": 0,
        "kits_missing_id": 0,
        "kits_missing_items": 0,
        "outfits_kept": 0,
        "outfits_dropped_below_min_items": 0,
        "items_removed_from_kept_and_dropped_outfits": 0,
        "min_items": min_items,
    }

    for row in kits_to_process:
        stats["kits_processed"] += 1
        kit_id = str(row.get("kit_id", "")).strip()

        if not kit_id:
            stats["kits_missing_id"] += 1
            continue

        raw_items = list(raw_kit_to_items.get(kit_id, []))
        clean_items = list(filtered_kit_to_items.get(kit_id, []))

        if not raw_items:
            stats["kits_missing_items"] += 1
            continue

        length_before[len(raw_items)] += 1
        length_after[len(clean_items)] += 1
        stats["items_removed_from_kept_and_dropped_outfits"] += (
            len(raw_items) - len(clean_items)
        )

        if len(clean_items) < min_items:
            stats["outfits_dropped_below_min_items"] += 1
            continue

        clean_samples.append(
            {
                "sample_id": f"{kit_id}_pos",
                "source_kit_id": kit_id,
                "paired_positive_sample_id": None,
                "items": [item["item_id"] for item in clean_items],
                "label": 1,
                "negative_metadata": None,
            }
        )
        stats["outfits_kept"] += 1

    stats["outfit_length_distribution_before"] = {
        str(length): count for length, count in sorted(length_before.items())
    }
    stats["outfit_length_distribution_after"] = {
        str(length): count for length, count in sorted(length_after.items())
    }
    stats["outfit_keep_rate"] = (
        0.0
        if stats["kits_processed"] == 0
        else stats["outfits_kept"] / stats["kits_processed"]
    )

    return clean_samples, stats


def build_core7_item_metadata(
    samples: Sequence[dict],
    filtered_kit_to_items: Mapping[str, Sequence[dict]],
    *,
    split: str,
    mapping_version: str,
    item_metadata_version: str = CORE7_ITEM_METADATA_VERSION,
) -> List[dict]:
    """Build item lookup rows for exactly the items serialized in positives.

    Keeping the metadata universe equal to the serialized positive universe
    prevents debug/full runs from silently exporting unrelated candidate items.
    Later image/embedding cleaning can safely filter this artifact by the final
    clean item IDs before negative sampling.
    """

    records: List[dict] = []
    seen_item_ids = set()

    for sample in samples:
        source_kit_id = sample["source_kit_id"]
        indexed_items = {
            item["item_id"]: item
            for item in filtered_kit_to_items.get(source_kit_id, [])
        }

        for item_id in sample["items"]:
            if item_id in seen_item_ids:
                continue

            item = indexed_items.get(item_id)
            if item is None:
                raise ValueError(
                    "Cannot build Core-7 item metadata. Item missing from filtered "
                    f"lookup: {item_id}"
                )

            records.append(
                {
                    "item_metadata_version": item_metadata_version,
                    "category_mapping_version": mapping_version,
                    "split": split,
                    "item_id": item_id,
                    "source_kit_id": source_kit_id,
                    "slot_index": item.get("slot_index"),
                    "master_category": item["master_category"],
                    "coarse_category": item["coarse_category"],
                }
            )
            seen_item_ids.add(item_id)

    return records


def validate_core7_item_metadata(
    records: Sequence[dict],
    samples: Sequence[dict],
    *,
    split: str,
    mapping_version: str,
    item_metadata_version: str = CORE7_ITEM_METADATA_VERSION,
) -> dict:
    """Validate item-metadata schema, versions, and exact sample coverage."""

    expected_item_ids = {
        item_id for sample in samples for item_id in sample.get("items", [])
    }
    seen_item_ids = set()
    duplicate_item_ids = []
    invalid_records = []

    for row_number, record in enumerate(records, start=1):
        item_id = record.get("item_id")
        reasons = []

        if not isinstance(item_id, str) or not item_id:
            reasons.append("invalid_item_id")
        elif item_id in seen_item_ids:
            duplicate_item_ids.append(item_id)
        else:
            seen_item_ids.add(item_id)

        if record.get("item_metadata_version") != item_metadata_version:
            reasons.append("item_metadata_version_mismatch")
        if record.get("category_mapping_version") != mapping_version:
            reasons.append("category_mapping_version_mismatch")
        if record.get("split") != split:
            reasons.append("split_mismatch")
        if not isinstance(record.get("source_kit_id"), str) or not record.get(
            "source_kit_id"
        ):
            reasons.append("invalid_source_kit_id")
        if not isinstance(record.get("master_category"), str) or not record.get(
            "master_category"
        ):
            reasons.append("invalid_master_category")
        if record.get("coarse_category") not in CORE_CATEGORIES:
            reasons.append("invalid_coarse_category")

        if reasons:
            invalid_records.append(
                {
                    "row_number": row_number,
                    "item_id": item_id,
                    "reasons": reasons,
                }
            )

    missing_from_metadata = sorted(expected_item_ids - seen_item_ids)
    extra_in_metadata = sorted(seen_item_ids - expected_item_ids)

    report = {
        "item_count": len(records),
        "expected_item_count": len(expected_item_ids),
        "duplicate_item_id_count": len(duplicate_item_ids),
        "invalid_record_count": len(invalid_records),
        "missing_from_metadata_count": len(missing_from_metadata),
        "extra_in_metadata_count": len(extra_in_metadata),
        "pass": not duplicate_item_ids
        and not invalid_records
        and not missing_from_metadata
        and not extra_in_metadata,
    }

    if not report["pass"]:
        raise ValueError(
            "Core-7 item metadata validation failed: "
            + json.dumps(report, ensure_ascii=False)
        )

    return report


def validate_clean_positive_samples(
    samples: Sequence[dict],
    item_to_coarse: Mapping[str, str],
    *,
    min_items: int = DEFAULT_MIN_OUTFIT_ITEMS,
) -> dict:
    """Validate the clean-positive contract before writing JSONL."""

    seen_sample_ids = set()
    duplicate_sample_ids = []
    invalid_samples = []
    missing_category_items = []

    for row_number, sample in enumerate(samples, start=1):
        sample_id = sample.get("sample_id")
        source_kit_id = sample.get("source_kit_id")
        items = sample.get("items")

        reasons = []
        if not isinstance(sample_id, str) or not sample_id.endswith("_pos"):
            reasons.append("invalid_sample_id")
        elif sample_id in seen_sample_ids:
            duplicate_sample_ids.append(sample_id)
        else:
            seen_sample_ids.add(sample_id)

        if not isinstance(source_kit_id, str) or not source_kit_id:
            reasons.append("invalid_source_kit_id")
        elif sample_id != f"{source_kit_id}_pos":
            reasons.append("sample_id_source_kit_mismatch")
        if (
            "paired_positive_sample_id" not in sample
            or sample["paired_positive_sample_id"] is not None
        ):
            reasons.append("paired_positive_sample_id_must_be_null")
        if sample.get("label") != 1:
            reasons.append("label_must_be_1")
        if (
            "negative_metadata" not in sample
            or sample["negative_metadata"] is not None
        ):
            reasons.append("negative_metadata_must_be_null")
        if not isinstance(items, list) or len(items) < min_items:
            reasons.append("outfit_below_min_items")
        elif len(items) != len(set(items)):
            reasons.append("duplicate_item_in_outfit")
        else:
            for item_id in items:
                if item_id not in item_to_coarse:
                    missing_category_items.append(item_id)

        if reasons:
            invalid_samples.append(
                {
                    "row_number": row_number,
                    "sample_id": sample_id,
                    "reasons": reasons,
                }
            )

    report = {
        "sample_count": len(samples),
        "duplicate_sample_id_count": len(duplicate_sample_ids),
        "invalid_sample_count": len(invalid_samples),
        "missing_category_item_count": len(set(missing_category_items)),
        "pass": not duplicate_sample_ids
        and not invalid_samples
        and not missing_category_items,
    }

    if not report["pass"]:
        raise ValueError(
            "Clean positive validation failed: "
            + json.dumps(report, ensure_ascii=False)
        )

    return report


def write_jsonl(records: Iterable[dict], output_path: Path | str) -> int:
    """Write compact UTF-8 JSONL and return the number of records."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
            stream.write("\n")
            count += 1

    return count


def default_item_metadata_output_path(
    output_path: Path | str,
    *,
    split: str,
) -> Path:
    """Return the default sibling path for the versioned item metadata."""

    output_dir = Path(output_path).parent
    return output_dir / f"core7_item_metadata_v1_{split}.jsonl"


def prepare_clean_positive_split(
    *,
    split: str,
    output_path: Path | str,
    item_metadata_output_path: Path | str | None = None,
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
    dataset_name: str = DATASET_NAME,
    min_items: int = DEFAULT_MIN_OUTFIT_ITEMS,
    debug_limit: Optional[int] = None,
) -> dict:
    """Run the complete Core-7 positive-cleaning stage for one split."""

    print("=" * 72)
    print(f"CORE-7 CLEAN POSITIVES - split={split}")
    print("=" * 72)

    mapping_metadata, mapping = load_category_mapping(mapping_path)
    mapping_version = str(mapping_metadata["mapping_version"])
    metadata_output_path = (
        Path(item_metadata_output_path)
        if item_metadata_output_path is not None
        else default_item_metadata_output_path(output_path, split=split)
    )

    print("Mapping version:", mapping_version)
    print("Mapping status :", mapping_metadata["status"])
    print("Item metadata  :", CORE7_ITEM_METADATA_VERSION)
    print("Minimum items  :", min_items)
    print()

    items_ds, kits_ds = load_required_datasets(
        dataset_name=dataset_name,
        split=split,
    )
    raw_kit_to_items, _, _ = build_item_indexes(items_ds)

    filtered_kit_to_items, filter_report = filter_items_by_core_category(
        raw_kit_to_items,
        mapping,
    )
    samples, outfit_report = build_clean_positive_samples(
        kits_ds,
        raw_kit_to_items,
        filtered_kit_to_items,
        min_items=min_items,
        debug_limit=debug_limit,
    )

    item_to_coarse = {
        item["item_id"]: item["coarse_category"]
        for outfit_items in filtered_kit_to_items.values()
        for item in outfit_items
    }
    validation_report = validate_clean_positive_samples(
        samples,
        item_to_coarse,
        min_items=min_items,
    )

    item_metadata = build_core7_item_metadata(
        samples,
        filtered_kit_to_items,
        split=split,
        mapping_version=mapping_version,
    )
    item_metadata_validation = validate_core7_item_metadata(
        item_metadata,
        samples,
        split=split,
        mapping_version=mapping_version,
    )

    written = write_jsonl(samples, output_path)
    if written != len(samples):
        raise RuntimeError("Written record count does not match sample count")

    metadata_written = write_jsonl(item_metadata, metadata_output_path)
    if metadata_written != len(item_metadata):
        raise RuntimeError("Written item metadata count does not match item count")

    report = {
        "processing_stage": "category_filtered_positives",
        "ready_to_train": False,
        "dataset_name": dataset_name,
        "split": split,
        "mapping_version": mapping_version,
        "mapping_status": mapping_metadata["status"],
        "item_metadata_version": CORE7_ITEM_METADATA_VERSION,
        "min_items": min_items,
        "debug_limit": debug_limit,
        "output_path": str(Path(output_path)),
        "item_metadata_output_path": str(metadata_output_path),
        "filter": filter_report,
        "outfits": outfit_report,
        "validation": validation_report,
        "item_metadata_validation": item_metadata_validation,
    }

    print("Items kept       :", f"{filter_report['kept_item_count']:,}")
    print("Items dropped    :", f"{filter_report['dropped_item_count']:,}")
    print("Outfits kept     :", f"{outfit_report['outfits_kept']:,}")
    print(
        "Outfits dropped  :",
        f"{outfit_report['outfits_dropped_below_min_items']:,}",
    )
    print("Metadata items   :", f"{len(item_metadata):,}")
    print("Validation       : PASS")
    print("Positive output  :", output_path)
    print("Metadata output  :", metadata_output_path)
    print()

    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create Core-7 category-clean Polyvore1000 positives and versioned "
            "Core-7 item metadata"
        )
    )
    parser.add_argument("--split", required=True, choices=("train", "valid", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--item-metadata-output", type=Path, default=None)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--min-items", type=int, default=DEFAULT_MIN_OUTFIT_ITEMS)
    parser.add_argument("--debug-limit", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    report = prepare_clean_positive_split(
        split=args.split,
        output_path=args.output,
        item_metadata_output_path=args.item_metadata_output,
        mapping_path=args.mapping,
        dataset_name=args.dataset_name,
        min_items=args.min_items,
        debug_limit=args.debug_limit,
    )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
