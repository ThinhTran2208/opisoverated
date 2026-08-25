# -*- coding: utf-8 -*-
"""Build scorer-ready Polyvore1000 compatibility samples.

This module contains repository/core data-processing logic only. It has no
Google Colab or Google Drive dependency. The canonical output follows
DATA_CONTRACT_VI.md:

Positive sample::

    {
        "sample_id": "<source_kit_id>_pos",
        "source_kit_id": "<source_kit_id>",
        "items": [...],
        "label": 1,
        "negative_metadata": null
    }

Negative sample::

    {
        "sample_id": "<source_kit_id>_neg_<n>",
        "source_kit_id": "<source_kit_id>",
        "items": [...],
        "label": 0,
        "negative_metadata": {...}
    }

Typical repository usage:

    python -m src.data.build_compatibility_dataset \
        --split train \
        --output data/processed/polyvore1000_compatibility_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple



# ============================================================
# DEFAULT CONFIG
# ============================================================

DATASET_NAME = "codewaly/polyvore1000"
ITEMS_CONFIG = "items"
KITS_CONFIG = "kits"
DEFAULT_SPLIT = "train"
DEFAULT_SEED = 42
DEFAULT_NEGATIVES_PER_OUTFIT = 1
DEFAULT_MIN_ITEMS_PER_KIT = 2
DEFAULT_OUTPUT_DIR = Path("data/processed")


# ============================================================
# DATASET LOADING
# ============================================================


def load_required_datasets(
    dataset_name: str = DATASET_NAME,
    split: str = DEFAULT_SPLIT,
):
    """Load only the fields needed to build compatibility samples.

    Items:
        item_id
        master_category

    Kits:
        kit_id
    """

    print("=" * 70)
    print("STEP 1 - LOAD DATASETS")
    print("=" * 70)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The core pipeline requires the Hugging Face 'datasets' package. "
            "Install project dependencies before running dataset generation."
        ) from exc

    items_ds = load_dataset(
        dataset_name,
        ITEMS_CONFIG,
        split=split,
    )

    kits_ds = load_dataset(
        dataset_name,
        KITS_CONFIG,
        split=split,
    )

    print(f"Items columns before pruning: {items_ds.column_names}")
    print(f"Kits columns before pruning : {kits_ds.column_names}")

    required_item_columns = {
        "item_id",
        "master_category",
    }
    required_kit_columns = {"kit_id"}

    missing_item_columns = required_item_columns - set(items_ds.column_names)
    missing_kit_columns = required_kit_columns - set(kits_ds.column_names)

    if missing_item_columns:
        raise KeyError(
            "Missing columns in items dataset: "
            f"{sorted(missing_item_columns)}"
        )

    if missing_kit_columns:
        raise KeyError(
            "Missing columns in kits dataset: "
            f"{sorted(missing_kit_columns)}"
        )

    # Drop image and every unused field before iterating.
    items_ds = items_ds.select_columns([
        "item_id",
        "master_category",
    ])
    kits_ds = kits_ds.select_columns(["kit_id"])

    print(f"Items columns after pruning: {items_ds.column_names}")
    print(f"Kits columns after pruning : {kits_ds.column_names}")
    print(f"Raw items: {len(items_ds):,}")
    print(f"Raw kits : {len(kits_ds):,}")
    print()

    return items_ds, kits_ds


# ============================================================
# ITEM ID PARSER
# ============================================================


def parse_item_id(item_id: str) -> Tuple[str, Optional[int]]:
    """Parse ``{kit_id}_{slot}`` item IDs.

    Example:
        ``214181831_1 -> ("214181831", 1)``
    """

    kit_id, separator, slot = item_id.rpartition("_")

    if not separator or not kit_id:
        raise ValueError(f"Invalid item_id format: {item_id}")

    try:
        slot_index = int(slot)
    except ValueError:
        slot_index = None

    return kit_id, slot_index


# ============================================================
# BUILD INDEXES
# ============================================================


def build_item_indexes(items_ds):
    """Build lookup indexes and validate/deduplicate raw item rows."""

    print("=" * 70)
    print("STEP 2 - BUILD INDEXES")
    print("=" * 70)

    kit_to_items = defaultdict(list)
    category_to_items = defaultdict(list)
    item_to_category = {}

    seen_item_ids = set()

    stats = {
        "unique_items": 0,
        "duplicate_items": 0,
        "malformed_item_ids": 0,
        "missing_fields": 0,
    }

    for row in items_ds:
        raw_item_id = row.get("item_id")
        raw_category = row.get("master_category")

        if raw_item_id is None or raw_category is None:
            stats["missing_fields"] += 1
            continue

        item_id = str(raw_item_id).strip()
        category = str(raw_category).strip()

        if not item_id or not category:
            stats["missing_fields"] += 1
            continue

        if item_id in seen_item_ids:
            stats["duplicate_items"] += 1
            continue

        try:
            kit_id, slot_index = parse_item_id(item_id)
        except ValueError:
            stats["malformed_item_ids"] += 1
            continue

        seen_item_ids.add(item_id)

        item_info = {
            "item_id": item_id,
            "kit_id": kit_id,
            "category": category,
            "slot_index": slot_index,
        }

        item_to_category[item_id] = category
        category_to_items[category].append(item_info)
        kit_to_items[kit_id].append(item_info)
        stats["unique_items"] += 1

    for items in kit_to_items.values():
        items.sort(
            key=lambda item: (
                item["slot_index"] is None,
                item["slot_index"] if item["slot_index"] is not None else 0,
                item["item_id"],
            )
        )

    print(f"Unique items       : {stats['unique_items']:,}")
    print(f"Reconstructed kits : {len(kit_to_items):,}")
    print(f"Categories         : {len(category_to_items):,}")
    print(f"Duplicate items    : {stats['duplicate_items']:,}")
    print(f"Malformed IDs      : {stats['malformed_item_ids']:,}")
    print(f"Missing fields     : {stats['missing_fields']:,}")
    print()

    return kit_to_items, category_to_items, item_to_category


# ============================================================
# NEGATIVE SAMPLING
# ============================================================


def find_eligible_swap_positions(
    outfit_items: List[dict],
    category_to_items: Dict[str, List[dict]],
):
    """Return swap positions that have at least one valid replacement.

    A replacement must be:
    - from the same ``master_category``;
    - a different item;
    - from a different kit;
    - not already present in the current outfit.
    """

    outfit_item_ids = {item["item_id"] for item in outfit_items}

    eligible_positions = []
    candidates_by_position = {}

    for position, original_item in enumerate(outfit_items):
        category = original_item["category"]
        original_kit_id = original_item["kit_id"]
        candidates = category_to_items.get(category, [])

        valid_candidates = [
            candidate
            for candidate in candidates
            if (
                candidate["item_id"] not in outfit_item_ids
                and candidate["kit_id"] != original_kit_id
            )
        ]

        if valid_candidates:
            eligible_positions.append(position)
            candidates_by_position[position] = valid_candidates

    return eligible_positions, candidates_by_position



def create_negative_sample(
    outfit_items: List[dict],
    category_to_items: Dict[str, List[dict]],
    rng: random.Random,
    used_pairs=None,
):
    """Create one category-preserving one-item-swap negative sample."""

    eligible_positions, candidates_by_position = find_eligible_swap_positions(
        outfit_items,
        category_to_items,
    )

    if not eligible_positions:
        return None, None, "no_valid_candidate"

    if used_pairs is None:
        used_pairs = set()

    positions = eligible_positions.copy()
    rng.shuffle(positions)

    for swap_index in positions:
        original_item = outfit_items[swap_index]
        candidates = candidates_by_position[swap_index].copy()
        rng.shuffle(candidates)

        for replacement in candidates:
            pair = (
                original_item["item_id"],
                replacement["item_id"],
            )

            if pair in used_pairs:
                continue

            used_pairs.add(pair)

            negative_items = [item.copy() for item in outfit_items]
            negative_items[swap_index] = replacement.copy()

            # Canonical provenance fields required by Data Contract V1.
            negative_metadata = {
                "negative_type": "same_category_different_kit",
                "swapped_item_index": swap_index,
                "original_item_id": original_item["item_id"],
                "replacement_item_id": replacement["item_id"],
                "swap_category": original_item["category"],
                "replacement_kit_id": replacement["kit_id"],
            }

            return negative_items, negative_metadata, None

    return None, None, "duplicate_negative_pair"


# ============================================================
# GENERATE DATASET
# ============================================================


def _iter_kits(kits_ds, debug_limit: Optional[int]):
    """Select a debug subset without coupling the core pipeline to Colab."""

    if debug_limit is None:
        return kits_ds

    limit = min(debug_limit, len(kits_ds))

    if hasattr(kits_ds, "select"):
        return kits_ds.select(range(limit))

    # Fallback keeps helpers testable with ordinary Python iterables.
    return list(kits_ds)[:limit]



def generate_dataset(
    kits_ds,
    kit_to_items,
    category_to_items,
    output_file: Path | str,
    *,
    seed: int = DEFAULT_SEED,
    negatives_per_outfit: int = DEFAULT_NEGATIVES_PER_OUTFIT,
    min_items_per_kit: int = DEFAULT_MIN_ITEMS_PER_KIT,
    debug_limit: Optional[int] = None,
):
    """Generate canonical positive and synthetic negative JSONL records."""

    print("=" * 70)
    print("STEP 3 - GENERATE DATASET")
    print("=" * 70)

    if negatives_per_outfit < 0:
        raise ValueError("negatives_per_outfit must be >= 0")
    if min_items_per_kit < 1:
        raise ValueError("min_items_per_kit must be >= 1")
    if debug_limit is not None and debug_limit < 1:
        raise ValueError("debug_limit must be >= 1 when provided")

    rng = random.Random(seed)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kits_to_process = _iter_kits(kits_ds, debug_limit)

    if debug_limit is None:
        print(f"FULL MODE: {len(kits_ds):,} kits")
    else:
        print(f"DEBUG MODE: {len(kits_to_process):,} kits")

    print(f"Negatives per outfit: {negatives_per_outfit}")
    print(f"Output file: {output_path}")
    print()

    stats = {
        "kits_processed": 0,
        "kits_with_items": 0,
        "kits_without_items": 0,
        "kits_too_small": 0,
        "positive_samples": 0,
        "negative_samples": 0,
        "negative_attempts": 0,
        "failed_no_valid_candidate": 0,
        "failed_duplicate_pair": 0,
        "negative_generation_complete": 0,
    }

    with output_path.open("w", encoding="utf-8") as f:
        for kit in kits_to_process:
            stats["kits_processed"] += 1

            kit_id = str(kit.get("kit_id", "")).strip()

            if not kit_id:
                stats["kits_without_items"] += 1
                continue

            outfit_items = kit_to_items.get(kit_id, [])

            if not outfit_items:
                stats["kits_without_items"] += 1
                continue

            if len(outfit_items) < min_items_per_kit:
                stats["kits_too_small"] += 1
                continue

            stats["kits_with_items"] += 1

            positive_sample = {
                "sample_id": f"{kit_id}_pos",
                "source_kit_id": kit_id,
                "items": [item["item_id"] for item in outfit_items],
                "label": 1,
                "negative_metadata": None,
            }

            f.write(
                json.dumps(
                    positive_sample,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            f.write("\n")
            stats["positive_samples"] += 1

            used_pairs = set()
            negatives_generated = 0

            for negative_number in range(negatives_per_outfit):
                stats["negative_attempts"] += 1

                negative_items, negative_metadata, failure_reason = (
                    create_negative_sample(
                        outfit_items=outfit_items,
                        category_to_items=category_to_items,
                        rng=rng,
                        used_pairs=used_pairs,
                    )
                )

                if negative_items is None:
                    if failure_reason == "no_valid_candidate":
                        stats["failed_no_valid_candidate"] += 1
                    elif failure_reason == "duplicate_negative_pair":
                        stats["failed_duplicate_pair"] += 1
                    continue

                negative_sample = {
                    "sample_id": f"{kit_id}_neg_{negative_number + 1}",
                    "source_kit_id": kit_id,
                    "items": [item["item_id"] for item in negative_items],
                    "label": 0,
                    "negative_metadata": negative_metadata,
                }

                f.write(
                    json.dumps(
                        negative_sample,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                f.write("\n")

                stats["negative_samples"] += 1
                negatives_generated += 1

            if negatives_generated == negatives_per_outfit:
                stats["negative_generation_complete"] += 1

    positive = stats["positive_samples"]
    negative = stats["negative_samples"]
    attempts = stats["negative_attempts"]

    print()
    print("=" * 70)
    print("STEP 4 - GENERATION SUMMARY")
    print("=" * 70)
    print(f"Kits processed              : {stats['kits_processed']:,}")
    print(f"Kits with valid items       : {stats['kits_with_items']:,}")
    print(f"Kits without items          : {stats['kits_without_items']:,}")
    print(f"Kits too small              : {stats['kits_too_small']:,}")
    print()
    print(f"Positive samples            : {positive:,}")
    print(f"Negative samples            : {negative:,}")
    print(f"Negative attempts           : {attempts:,}")
    print(
        "Failed: no valid candidate  : "
        f"{stats['failed_no_valid_candidate']:,}"
    )
    print(
        "Failed: duplicate pair      : "
        f"{stats['failed_duplicate_pair']:,}"
    )
    print()

    if positive:
        print(f"Negative / Positive ratio   : {negative / positive:.3f}")

    if attempts:
        print(f"Negative generation rate    : {(negative / attempts) * 100:.2f}%")

    print(
        "Complete kits               : "
        f"{stats['negative_generation_complete']:,}"
    )
    print(f"Output file                 : {output_path}")
    print()

    return stats


# ============================================================
# VERIFY CANONICAL OUTPUT
# ============================================================


def _validate_record_shape(record: dict, line_number: int) -> None:
    required_fields = {
        "sample_id",
        "source_kit_id",
        "items",
        "label",
        "negative_metadata",
    }

    missing = required_fields - set(record)
    if missing:
        raise ValueError(
            f"Line {line_number}: missing canonical fields {sorted(missing)}"
        )

    sample_id = record["sample_id"]
    source_kit_id = record["source_kit_id"]
    items = record["items"]
    label = record["label"]
    negative_metadata = record["negative_metadata"]

    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"Line {line_number}: invalid sample_id")

    if not isinstance(source_kit_id, str) or not source_kit_id.strip():
        raise ValueError(f"Line {line_number}: invalid source_kit_id")

    if not isinstance(items, list) or not items:
        raise ValueError(f"Line {line_number}: items must be a non-empty list")

    if any(not isinstance(item_id, str) or not item_id for item_id in items):
        raise ValueError(f"Line {line_number}: items contains invalid item_id")

    if label not in {0, 1}:
        raise ValueError(f"Line {line_number}: label must be 0 or 1")

    if label == 1:
        if negative_metadata is not None:
            raise ValueError(
                f"Line {line_number}: positive sample must have negative_metadata=null"
            )
        if not sample_id.endswith("_pos"):
            raise ValueError(
                f"Line {line_number}: positive sample_id should end with '_pos'"
            )
    else:
        if not isinstance(negative_metadata, dict):
            raise ValueError(
                f"Line {line_number}: negative sample must have negative_metadata object"
            )

        required_negative_fields = {
            "negative_type",
            "swapped_item_index",
            "original_item_id",
            "replacement_item_id",
            "swap_category",
            "replacement_kit_id",
        }
        missing_negative = required_negative_fields - set(negative_metadata)
        if missing_negative:
            raise ValueError(
                f"Line {line_number}: negative_metadata missing "
                f"{sorted(missing_negative)}"
            )

        if negative_metadata["negative_type"] != "same_category_different_kit":
            raise ValueError(
                f"Line {line_number}: unsupported negative_type "
                f"{negative_metadata['negative_type']!r}"
            )

        swap_index = negative_metadata["swapped_item_index"]
        if not isinstance(swap_index, int) or not (0 <= swap_index < len(items)):
            raise ValueError(
                f"Line {line_number}: swapped_item_index out of range"
            )

        if items[swap_index] != negative_metadata["replacement_item_id"]:
            raise ValueError(
                f"Line {line_number}: replacement_item_id does not match items[swapped_item_index]"
            )


def verify_output(
    output_file: Path | str,
    *,
    item_to_category: Optional[Dict[str, str]] = None,
    kit_to_items: Optional[Dict[str, List[dict]]] = None,
):
    """Validate canonical JSONL output line-by-line.

    When ``item_to_category`` and ``kit_to_items`` are provided, this also
    validates the V1 negative provenance rules against the source data.
    """

    print("=" * 70)
    print("STEP 5 - VERIFY OUTPUT")
    print("=" * 70)

    output_path = Path(output_file)
    if not output_path.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print("Output exists: YES")
    print(f"File size    : {file_size_mb:.2f} MB")

    record_count = 0
    blank_lines = 0
    seen_sample_ids = set()

    with output_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                blank_lines += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number}: expected object, "
                    f"got {type(record).__name__}"
                )

            _validate_record_shape(record, line_number)

            sample_id = record["sample_id"]
            if sample_id in seen_sample_ids:
                raise ValueError(
                    f"Line {line_number}: duplicate sample_id {sample_id!r}"
                )
            seen_sample_ids.add(sample_id)

            if item_to_category is not None:
                missing_item_metadata = [
                    item_id
                    for item_id in record["items"]
                    if item_id not in item_to_category
                ]
                if missing_item_metadata:
                    raise ValueError(
                        f"Line {line_number}: missing master_category for "
                        f"{missing_item_metadata[:5]}"
                    )

            if record["label"] == 0:
                metadata = record["negative_metadata"]
                source_kit_id = record["source_kit_id"]
                original_item_id = metadata["original_item_id"]
                replacement_item_id = metadata["replacement_item_id"]
                replacement_kit_id = metadata["replacement_kit_id"]
                swap_category = metadata["swap_category"]

                if replacement_kit_id == source_kit_id:
                    raise ValueError(
                        f"Line {line_number}: replacement must come from a different kit"
                    )

                if item_to_category is not None:
                    original_category = item_to_category.get(original_item_id)
                    replacement_category = item_to_category.get(replacement_item_id)

                    if original_category != swap_category:
                        raise ValueError(
                            f"Line {line_number}: original item category does not match swap_category"
                        )
                    if replacement_category != swap_category:
                        raise ValueError(
                            f"Line {line_number}: replacement violates same-category V1 rule"
                        )

                if kit_to_items is not None:
                    original_outfit_ids = {
                        item["item_id"]
                        for item in kit_to_items.get(source_kit_id, [])
                    }

                    if original_item_id not in original_outfit_ids:
                        raise ValueError(
                            f"Line {line_number}: original_item_id not in source outfit"
                        )
                    if replacement_item_id in original_outfit_ids:
                        raise ValueError(
                            f"Line {line_number}: replacement already exists in original outfit"
                        )

            record_count += 1

    print(f"JSONL records : {record_count:,}")
    if blank_lines:
        print(f"Blank lines   : {blank_lines:,}")
    print("Canonical JSONL validation: PASSED")
    print()

    return {
        "record_count": record_count,
        "blank_lines": blank_lines,
        "unique_sample_ids": len(seen_sample_ids),
    }


# ============================================================
# END-TO-END ORCHESTRATOR
# ============================================================


def build_compatibility_dataset(
    *,
    output_file: Path | str,
    dataset_name: str = DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
    negatives_per_outfit: int = DEFAULT_NEGATIVES_PER_OUTFIT,
    min_items_per_kit: int = DEFAULT_MIN_ITEMS_PER_KIT,
    debug_limit: Optional[int] = None,
):
    """Run loading, indexing, generation, and canonical validation."""

    items_ds, kits_ds = load_required_datasets(
        dataset_name=dataset_name,
        split=split,
    )

    kit_to_items, category_to_items, item_to_category = build_item_indexes(
        items_ds
    )

    generation_stats = generate_dataset(
        kits_ds=kits_ds,
        kit_to_items=kit_to_items,
        category_to_items=category_to_items,
        output_file=output_file,
        seed=seed,
        negatives_per_outfit=negatives_per_outfit,
        min_items_per_kit=min_items_per_kit,
        debug_limit=debug_limit,
    )

    verification_stats = verify_output(
        output_file,
        item_to_category=item_to_category,
        kit_to_items=kit_to_items,
    )

    return {
        "generation": generation_stats,
        "verification": verification_stats,
    }


# ============================================================
# CLI
# ============================================================


def _default_output_for_split(split: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"polyvore1000_compatibility_{split}.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build canonical Polyvore1000 compatibility JSONL data."
    )
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help=f"Hugging Face dataset name (default: {DATASET_NAME})",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split (default: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. Default: "
            "data/processed/polyvore1000_compatibility_<split>.jsonl"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--negatives-per-outfit",
        type=int,
        default=DEFAULT_NEGATIVES_PER_OUTFIT,
    )
    parser.add_argument(
        "--min-items-per-kit",
        type=int,
        default=DEFAULT_MIN_ITEMS_PER_KIT,
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=None,
        help="Process only the first N kits. Omit for a full run.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_file = args.output or _default_output_for_split(args.split)

    stats = build_compatibility_dataset(
        output_file=output_file,
        dataset_name=args.dataset_name,
        split=args.split,
        seed=args.seed,
        negatives_per_outfit=args.negatives_per_outfit,
        min_items_per_kit=args.min_items_per_kit,
        debug_limit=args.debug_limit,
    )

    print("=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)
    return stats


if __name__ == "__main__":
    main()
