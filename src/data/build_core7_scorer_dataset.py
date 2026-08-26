# -*- coding: utf-8 -*-
"""Build and freeze the Core-7 scorer-ready benchmark (dataset V1).

Inputs per official split:

* ``category_clean_{split}.jsonl`` (final clean positives after NB3 pass);
* ``core7_item_metadata_v1_{split}.jsonl``;
* one passing ``core7_embedding_validation_report.json``.

Outputs:

* one negative-only JSONL per split;
* one interleaved positive + negative scorer-ready JSONL per split;
* negative-sampling reports;
* a final validation report;
* split and dataset manifests with SHA-256 hashes.

Negative V1 is frozen as one same-master-category, different-kit replacement
per positive outfit. Candidate pools never cross official splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence


SPLITS = ("train", "valid", "test")
DATASET_VERSION = "polyvore1000-core7-compat-v1"
NEGATIVE_VERSION = "negative-v1"
NEGATIVE_TYPE = "same_category_different_kit"
DEFAULT_SEED = 42
DEFAULT_MIN_ITEMS = 3
MAX_EXAMPLES = 50


def read_json(path: Path | str) -> dict:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {source}")
    return payload


def write_json(payload: Mapping[str, object], path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def read_jsonl(path: Path | str) -> list[dict]:
    source = Path(path)
    records: list[dict] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {source}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at {source}:{line_number}"
                )
            records.append(record)
    return records


def write_jsonl(records: Iterable[dict], path: Path | str) -> int:
    destination = Path(path)
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


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata_indexes(
    metadata: Sequence[dict],
    *,
    split: str,
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Build stable item and master-category indexes for one official split."""

    item_by_id: dict[str, dict] = {}
    category_to_items: dict[str, list[dict]] = defaultdict(list)

    for row_number, raw_record in enumerate(metadata, start=1):
        record = dict(raw_record)
        item_id = str(record.get("item_id", "")).strip()
        source_kit_id = str(record.get("source_kit_id", "")).strip()
        master_category = str(record.get("master_category", "")).strip()
        coarse_category = str(record.get("coarse_category", "")).strip()
        record_split = str(record.get("split", "")).strip()

        if not all((item_id, source_kit_id, master_category, coarse_category)):
            raise ValueError(
                f"Metadata row {row_number} in split={split} has missing fields"
            )
        if record_split != split:
            raise ValueError(
                f"Metadata item {item_id} says split={record_split}, expected {split}"
            )
        if str(record.get("category_mapping_version", "")) != "core7-v1":
            raise ValueError(
                f"Metadata item {item_id} does not use category_mapping_version=core7-v1"
            )
        if str(record.get("item_metadata_version", "")) != "core7-item-metadata-v1":
            raise ValueError(
                f"Metadata item {item_id} does not use item_metadata_version="
                "core7-item-metadata-v1"
            )
        if item_id in item_by_id:
            raise ValueError(f"Duplicate metadata item_id in split={split}: {item_id}")

        record["item_id"] = item_id
        record["source_kit_id"] = source_kit_id
        record["master_category"] = master_category
        record["coarse_category"] = coarse_category
        record["split"] = split
        item_by_id[item_id] = record
        category_to_items[master_category].append(record)

    for pool in category_to_items.values():
        pool.sort(key=lambda item: item["item_id"])

    return item_by_id, dict(category_to_items)


def _choose_replacement(
    pool: Sequence[dict],
    *,
    original_item_ids: set[str],
    source_kit_id: str,
    rng: random.Random,
) -> Optional[dict]:
    """Choose one valid candidate without materializing a filtered large pool."""

    if not pool:
        return None

    start = rng.randrange(len(pool))
    for offset in range(len(pool)):
        candidate = pool[(start + offset) % len(pool)]
        if candidate["item_id"] in original_item_ids:
            continue
        if candidate["source_kit_id"] == source_kit_id:
            continue
        return candidate
    return None


def create_negative_for_positive(
    positive: Mapping[str, object],
    *,
    item_by_id: Mapping[str, dict],
    category_to_items: Mapping[str, Sequence[dict]],
    rng: random.Random,
    negative_number: int = 1,
) -> tuple[Optional[dict], Optional[str]]:
    """Create one deterministic, category-preserving negative for a positive."""

    sample_id = str(positive.get("sample_id", "")).strip()
    source_kit_id = str(positive.get("source_kit_id", "")).strip()
    original_items = [str(item_id) for item_id in positive.get("items", [])]
    if not sample_id or not source_kit_id or not original_items:
        return None, "invalid_positive_shape"
    if int(positive.get("label", -1)) != 1:
        return None, "input_is_not_positive"

    missing_metadata = [
        item_id for item_id in original_items if item_id not in item_by_id
    ]
    if missing_metadata:
        return None, "positive_missing_metadata"

    original_item_ids = set(original_items)
    positions = list(range(len(original_items)))
    rng.shuffle(positions)

    for swapped_item_index in positions:
        original_item_id = original_items[swapped_item_index]
        original_metadata = item_by_id[original_item_id]
        master_category = original_metadata["master_category"]
        replacement = _choose_replacement(
            category_to_items.get(master_category, ()),
            original_item_ids=original_item_ids,
            source_kit_id=source_kit_id,
            rng=rng,
        )
        if replacement is None:
            continue

        replacement_item_id = replacement["item_id"]
        negative_items = list(original_items)
        negative_items[swapped_item_index] = replacement_item_id
        negative = {
            "sample_id": f"{source_kit_id}_neg_{negative_number}",
            "source_kit_id": source_kit_id,
            "paired_positive_sample_id": sample_id,
            "items": negative_items,
            "label": 0,
            "negative_metadata": {
                "negative_type": NEGATIVE_TYPE,
                "swapped_item_index": swapped_item_index,
                "original_item_id": original_item_id,
                "replacement_item_id": replacement_item_id,
                "swap_category": master_category,
                "replacement_kit_id": replacement["source_kit_id"],
            },
        }
        return negative, None

    return None, "no_valid_same_master_category_candidate"


def generate_negative_records(
    positives: Sequence[dict],
    metadata: Sequence[dict],
    *,
    split: str,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], dict]:
    """Generate exactly one V1 negative attempt for every clean positive."""

    item_by_id, category_to_items = build_metadata_indexes(metadata, split=split)
    rng = random.Random(seed)
    negatives: list[dict] = []
    failure_counts: Counter[str] = Counter()
    failure_examples: list[dict] = []
    swapped_category_counts: Counter[str] = Counter()
    swapped_index_counts: Counter[int] = Counter()

    for positive in positives:
        negative, failure_reason = create_negative_for_positive(
            positive,
            item_by_id=item_by_id,
            category_to_items=category_to_items,
            rng=rng,
        )
        if negative is None:
            reason = failure_reason or "unknown"
            failure_counts[reason] += 1
            if len(failure_examples) < MAX_EXAMPLES:
                failure_examples.append(
                    {
                        "sample_id": positive.get("sample_id"),
                        "source_kit_id": positive.get("source_kit_id"),
                        "reason": reason,
                    }
                )
            continue

        negatives.append(negative)
        negative_metadata = negative["negative_metadata"]
        swapped_category_counts[negative_metadata["swap_category"]] += 1
        swapped_index_counts[negative_metadata["swapped_item_index"]] += 1

    positive_count = len(positives)
    negative_count = len(negatives)
    report = {
        "processing_stage": "negative_sampling",
        "negative_version": NEGATIVE_VERSION,
        "negative_type": NEGATIVE_TYPE,
        "split": split,
        "seed": seed,
        "negatives_per_positive": 1,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "failed_positive_count": positive_count - negative_count,
        "generation_coverage": (
            1.0 if positive_count == 0 else negative_count / positive_count
        ),
        "failure_counts": dict(sorted(failure_counts.items())),
        "failure_examples": failure_examples,
        "swapped_master_category_counts": dict(
            sorted(swapped_category_counts.items())
        ),
        "swapped_index_counts": {
            str(index): count
            for index, count in sorted(swapped_index_counts.items())
        },
        "pass": positive_count > 0 and negative_count == positive_count,
    }
    return negatives, report


def merge_positive_negative_families(
    positives: Sequence[dict],
    negatives: Sequence[dict],
) -> tuple[list[dict], dict]:
    """Interleave each positive with its negative and preserve paired families."""

    negatives_by_positive: dict[str, list[dict]] = defaultdict(list)
    for negative in negatives:
        pair_id = str(negative.get("paired_positive_sample_id", ""))
        negatives_by_positive[pair_id].append(negative)

    scorer_records: list[dict] = []
    missing_pair_examples: list[str] = []
    multiple_pair_examples: list[str] = []

    for raw_positive in positives:
        positive = dict(raw_positive)
        positive_id = str(positive.get("sample_id", ""))
        family_negatives = negatives_by_positive.get(positive_id, [])
        if len(family_negatives) == 0:
            if len(missing_pair_examples) < MAX_EXAMPLES:
                missing_pair_examples.append(positive_id)
            continue
        if len(family_negatives) != 1:
            if len(multiple_pair_examples) < MAX_EXAMPLES:
                multiple_pair_examples.append(positive_id)
            continue

        positive["paired_positive_sample_id"] = None
        scorer_records.append(positive)
        scorer_records.append(dict(family_negatives[0]))

    paired_positive_count = len(scorer_records) // 2
    report = {
        "processing_stage": "merge_scorer_dataset",
        "input_positive_count": len(positives),
        "input_negative_count": len(negatives),
        "paired_positive_count": paired_positive_count,
        "output_sample_count": len(scorer_records),
        "missing_negative_count": len(positives) - paired_positive_count,
        "missing_negative_examples": missing_pair_examples,
        "multiple_negative_family_examples": multiple_pair_examples,
        "pass": (
            len(positives) == len(negatives) == paired_positive_count
            and not multiple_pair_examples
        ),
    }
    return scorer_records, report


def _add_issue(
    issue_counts: Counter[str],
    issue_examples: list[dict],
    code: str,
    *,
    split: str,
    sample_id: str,
    detail: str,
) -> None:
    issue_counts[code] += 1
    if len(issue_examples) < MAX_EXAMPLES:
        issue_examples.append(
            {
                "code": code,
                "split": split,
                "sample_id": sample_id,
                "detail": detail,
            }
        )


def validate_scorer_split(
    records: Sequence[dict],
    metadata: Sequence[dict],
    *,
    split: str,
    min_items: int = DEFAULT_MIN_ITEMS,
) -> dict:
    """Reconstruct every pair and independently validate all V1 invariants."""

    item_by_id, _ = build_metadata_indexes(metadata, split=split)
    issue_counts: Counter[str] = Counter()
    issue_examples: list[dict] = []
    sample_ids: list[str] = []
    positives: dict[str, dict] = {}
    negatives: list[dict] = []
    label_counts: Counter[int] = Counter()
    outfit_length_counts: Counter[int] = Counter()

    for row_number, record in enumerate(records, start=1):
        sample_id = str(record.get("sample_id", "")).strip()
        source_kit_id = str(record.get("source_kit_id", "")).strip()
        items = record.get("items")
        label = record.get("label")
        sample_ids.append(sample_id)

        required_sample_fields = {
            "sample_id",
            "source_kit_id",
            "paired_positive_sample_id",
            "items",
            "label",
            "negative_metadata",
        }
        missing_sample_fields = sorted(required_sample_fields - set(record))
        if missing_sample_fields:
            _add_issue(
                issue_counts,
                issue_examples,
                "missing_sample_fields",
                split=split,
                sample_id=sample_id or f"line:{row_number}",
                detail=str(missing_sample_fields),
            )

        if not sample_id or not source_kit_id:
            _add_issue(
                issue_counts,
                issue_examples,
                "invalid_identity",
                split=split,
                sample_id=sample_id or f"line:{row_number}",
                detail="sample_id/source_kit_id is empty",
            )
        if not isinstance(items, list) or len(items) < min_items:
            _add_issue(
                issue_counts,
                issue_examples,
                "outfit_below_min_items",
                split=split,
                sample_id=sample_id,
                detail=f"items length is {len(items) if isinstance(items, list) else 'invalid'}",
            )
            continue

        item_ids = [str(item_id) for item_id in items]
        outfit_length_counts[len(item_ids)] += 1
        if len(set(item_ids)) != len(item_ids):
            _add_issue(
                issue_counts,
                issue_examples,
                "duplicate_item_inside_outfit",
                split=split,
                sample_id=sample_id,
                detail="items contains duplicate IDs",
            )
        missing_metadata = [
            item_id for item_id in item_ids if item_id not in item_by_id
        ]
        if missing_metadata:
            _add_issue(
                issue_counts,
                issue_examples,
                "missing_item_metadata",
                split=split,
                sample_id=sample_id,
                detail=str(missing_metadata[:10]),
            )

        if label not in (0, 1):
            _add_issue(
                issue_counts,
                issue_examples,
                "invalid_label",
                split=split,
                sample_id=sample_id,
                detail=f"label={label!r}",
            )
            continue
        label_counts[int(label)] += 1

        normalized_record = dict(record)
        normalized_record["items"] = item_ids
        if label == 1:
            positives[sample_id] = normalized_record
            if sample_id != f"{source_kit_id}_pos":
                _add_issue(
                    issue_counts,
                    issue_examples,
                    "positive_sample_id_mismatch",
                    split=split,
                    sample_id=sample_id,
                    detail=f"expected={source_kit_id}_pos",
                )
            if record.get("negative_metadata") is not None:
                _add_issue(
                    issue_counts,
                    issue_examples,
                    "positive_has_negative_metadata",
                    split=split,
                    sample_id=sample_id,
                    detail="negative_metadata must be null",
                )
            if record.get("paired_positive_sample_id") is not None:
                _add_issue(
                    issue_counts,
                    issue_examples,
                    "positive_has_pair_reference",
                    split=split,
                    sample_id=sample_id,
                    detail="paired_positive_sample_id must be null",
                )
            for item_id in item_ids:
                metadata_row = item_by_id.get(item_id)
                if metadata_row and metadata_row["source_kit_id"] != source_kit_id:
                    _add_issue(
                        issue_counts,
                        issue_examples,
                        "positive_item_wrong_source_kit",
                        split=split,
                        sample_id=sample_id,
                        detail=f"item={item_id}",
                    )
        else:
            negatives.append(normalized_record)

    duplicate_sample_ids = sorted(
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if sample_id and count > 1
    )
    for sample_id in duplicate_sample_ids:
        _add_issue(
            issue_counts,
            issue_examples,
            "duplicate_sample_id",
            split=split,
            sample_id=sample_id,
            detail="sample_id appears more than once",
        )

    negative_family_counts: Counter[str] = Counter()
    swapped_category_counts: Counter[str] = Counter()
    for negative in negatives:
        sample_id = negative["sample_id"]
        source_kit_id = str(negative.get("source_kit_id", ""))
        pair_id = str(negative.get("paired_positive_sample_id", ""))
        negative_family_counts[pair_id] += 1
        positive = positives.get(pair_id)
        if sample_id != f"{source_kit_id}_neg_1":
            _add_issue(
                issue_counts,
                issue_examples,
                "negative_sample_id_mismatch",
                split=split,
                sample_id=sample_id,
                detail=f"expected={source_kit_id}_neg_1",
            )
        if positive is None:
            _add_issue(
                issue_counts,
                issue_examples,
                "negative_missing_positive_pair",
                split=split,
                sample_id=sample_id,
                detail=f"paired_positive_sample_id={pair_id!r}",
            )
            continue
        if source_kit_id != str(positive.get("source_kit_id", "")):
            _add_issue(
                issue_counts,
                issue_examples,
                "pair_source_kit_mismatch",
                split=split,
                sample_id=sample_id,
                detail=f"negative={source_kit_id}, positive={positive.get('source_kit_id')}",
            )

        positive_items = positive["items"]
        negative_items = negative["items"]
        if len(positive_items) != len(negative_items):
            _add_issue(
                issue_counts,
                issue_examples,
                "pair_length_mismatch",
                split=split,
                sample_id=sample_id,
                detail=f"positive={len(positive_items)}, negative={len(negative_items)}",
            )
            continue
        differences = [
            index
            for index, (positive_item, negative_item) in enumerate(
                zip(positive_items, negative_items)
            )
            if positive_item != negative_item
        ]
        if len(differences) != 1:
            _add_issue(
                issue_counts,
                issue_examples,
                "negative_not_exactly_one_swap",
                split=split,
                sample_id=sample_id,
                detail=f"different_indices={differences}",
            )
            continue

        metadata_block = negative.get("negative_metadata")
        if not isinstance(metadata_block, dict):
            _add_issue(
                issue_counts,
                issue_examples,
                "negative_metadata_missing",
                split=split,
                sample_id=sample_id,
                detail="negative_metadata must be an object",
            )
            continue

        required_negative_fields = {
            "negative_type",
            "swapped_item_index",
            "original_item_id",
            "replacement_item_id",
            "swap_category",
            "replacement_kit_id",
        }
        missing_fields = sorted(required_negative_fields - set(metadata_block))
        if missing_fields:
            _add_issue(
                issue_counts,
                issue_examples,
                "negative_metadata_missing_fields",
                split=split,
                sample_id=sample_id,
                detail=str(missing_fields),
            )
            continue

        swapped_index = metadata_block["swapped_item_index"]
        if not isinstance(swapped_index, int) or not (
            0 <= swapped_index < len(positive_items)
        ):
            _add_issue(
                issue_counts,
                issue_examples,
                "invalid_swapped_item_index",
                split=split,
                sample_id=sample_id,
                detail=f"swapped_item_index={swapped_index!r}",
            )
            continue
        actual_index = differences[0]
        if swapped_index != actual_index:
            _add_issue(
                issue_counts,
                issue_examples,
                "swapped_index_mismatch",
                split=split,
                sample_id=sample_id,
                detail=f"metadata={swapped_index}, actual={actual_index}",
            )

        original_item_id = str(metadata_block["original_item_id"])
        replacement_item_id = str(metadata_block["replacement_item_id"])
        replacement_kit_id = str(metadata_block["replacement_kit_id"])
        swap_category = str(metadata_block["swap_category"])
        if positive_items[actual_index] != original_item_id:
            _add_issue(
                issue_counts,
                issue_examples,
                "original_item_mismatch",
                split=split,
                sample_id=sample_id,
                detail=f"metadata={original_item_id}, actual={positive_items[actual_index]}",
            )
        if negative_items[actual_index] != replacement_item_id:
            _add_issue(
                issue_counts,
                issue_examples,
                "replacement_item_mismatch",
                split=split,
                sample_id=sample_id,
                detail=f"metadata={replacement_item_id}, actual={negative_items[actual_index]}",
            )
        if replacement_item_id in positive_items:
            _add_issue(
                issue_counts,
                issue_examples,
                "replacement_already_in_positive",
                split=split,
                sample_id=sample_id,
                detail=f"replacement={replacement_item_id}",
            )
        if str(metadata_block["negative_type"]) != NEGATIVE_TYPE:
            _add_issue(
                issue_counts,
                issue_examples,
                "wrong_negative_type",
                split=split,
                sample_id=sample_id,
                detail=str(metadata_block["negative_type"]),
            )

        original_metadata = item_by_id.get(original_item_id)
        replacement_metadata = item_by_id.get(replacement_item_id)
        if original_metadata and replacement_metadata:
            original_category = original_metadata["master_category"]
            replacement_category = replacement_metadata["master_category"]
            if original_category != replacement_category:
                _add_issue(
                    issue_counts,
                    issue_examples,
                    "replacement_master_category_mismatch",
                    split=split,
                    sample_id=sample_id,
                    detail=f"{original_category!r} != {replacement_category!r}",
                )
            if swap_category != original_category:
                _add_issue(
                    issue_counts,
                    issue_examples,
                    "swap_category_metadata_mismatch",
                    split=split,
                    sample_id=sample_id,
                    detail=f"metadata={swap_category!r}, actual={original_category!r}",
                )
            if replacement_metadata["source_kit_id"] != replacement_kit_id:
                _add_issue(
                    issue_counts,
                    issue_examples,
                    "replacement_kit_metadata_mismatch",
                    split=split,
                    sample_id=sample_id,
                    detail=(
                        f"metadata={replacement_kit_id!r}, "
                        f"actual={replacement_metadata['source_kit_id']!r}"
                    ),
                )
            if replacement_kit_id == source_kit_id:
                _add_issue(
                    issue_counts,
                    issue_examples,
                    "replacement_from_same_kit",
                    split=split,
                    sample_id=sample_id,
                    detail=f"kit={source_kit_id}",
                )
            swapped_category_counts[original_category] += 1

    for positive_id in positives:
        if negative_family_counts[positive_id] != 1:
            _add_issue(
                issue_counts,
                issue_examples,
                "positive_pair_count_not_one",
                split=split,
                sample_id=positive_id,
                detail=f"negative_count={negative_family_counts[positive_id]}",
            )

    report = {
        "split": split,
        "sample_count": len(records),
        "positive_count": label_counts[1],
        "negative_count": label_counts[0],
        "unique_source_kit_count": len(
            {
                str(record.get("source_kit_id", ""))
                for record in records
                if record.get("source_kit_id")
            }
        ),
        "metadata_item_count": len(item_by_id),
        "outfit_length_distribution": {
            str(length): count for length, count in sorted(outfit_length_counts.items())
        },
        "swapped_master_category_counts": dict(
            sorted(swapped_category_counts.items())
        ),
        "issue_count": sum(issue_counts.values()),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_examples": issue_examples,
    }
    report["pass"] = bool(
        report["sample_count"] > 0
        and report["positive_count"] == report["negative_count"]
        and report["issue_count"] == 0
    )
    return report


def validate_all_splits(
    records_by_split: Mapping[str, Sequence[dict]],
    metadata_by_split: Mapping[str, Sequence[dict]],
    *,
    embedding_report: Mapping[str, object],
    sampling_reports: Mapping[str, Mapping[str, object]],
    min_items: int = DEFAULT_MIN_ITEMS,
) -> dict:
    """Run split validators plus global leakage and embedding gates."""

    split_reports: dict[str, dict] = {}
    source_kits_by_split: dict[str, set[str]] = {}
    item_ids_by_split: dict[str, set[str]] = {}

    for split in SPLITS:
        records = records_by_split[split]
        metadata = metadata_by_split[split]
        split_reports[split] = validate_scorer_split(
            records,
            metadata,
            split=split,
            min_items=min_items,
        )
        source_kits_by_split[split] = {
            str(record.get("source_kit_id", ""))
            for record in records
            if int(record.get("label", -1)) == 1
        }
        item_ids_by_split[split] = {
            str(record.get("item_id", "")) for record in metadata
        }

    cross_split_kit_count = 0
    cross_split_item_count = 0
    cross_split_kit_examples: list[str] = []
    cross_split_item_examples: list[str] = []
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            shared_kits = source_kits_by_split[left] & source_kits_by_split[right]
            shared_items = item_ids_by_split[left] & item_ids_by_split[right]
            cross_split_kit_count += len(shared_kits)
            cross_split_item_count += len(shared_items)
            cross_split_kit_examples.extend(
                f"{left}<->{right}:{kit_id}"
                for kit_id in sorted(shared_kits)[
                    : max(0, MAX_EXAMPLES - len(cross_split_kit_examples))
                ]
            )
            cross_split_item_examples.extend(
                f"{left}<->{right}:{item_id}"
                for item_id in sorted(shared_items)[
                    : max(0, MAX_EXAMPLES - len(cross_split_item_examples))
                ]
            )

    embedding_splits = embedding_report.get("splits", {})
    embedding_gate = bool(
        embedding_report.get("pass")
        and all(
            embedding_splits.get(split, {}).get("pass")
            and embedding_splits.get(split, {}).get("embedding_coverage") == 1.0
            for split in SPLITS
        )
    )
    sampling_gate = all(
        bool(sampling_reports[split].get("pass")) for split in SPLITS
    )
    global_sample_ids = [
        str(record.get("sample_id", ""))
        for split in SPLITS
        for record in records_by_split[split]
    ]
    global_duplicate_sample_ids = sorted(
        sample_id
        for sample_id, count in Counter(global_sample_ids).items()
        if sample_id and count > 1
    )

    report = {
        "processing_stage": "final_scorer_dataset_validation",
        "dataset_version": DATASET_VERSION,
        "negative_version": NEGATIVE_VERSION,
        "min_items": min_items,
        "splits": split_reports,
        "embedding_validation_pass": embedding_gate,
        "negative_sampling_pass": sampling_gate,
        "source_kit_cross_split_count": cross_split_kit_count,
        "source_kit_cross_split_examples": cross_split_kit_examples[:MAX_EXAMPLES],
        "item_cross_split_count": cross_split_item_count,
        "item_cross_split_examples": cross_split_item_examples[:MAX_EXAMPLES],
        "global_duplicate_sample_id_count": len(global_duplicate_sample_ids),
        "global_duplicate_sample_id_examples": global_duplicate_sample_ids[
            :MAX_EXAMPLES
        ],
    }
    report["pass"] = bool(
        all(split_report["pass"] for split_report in split_reports.values())
        and embedding_gate
        and sampling_gate
        and report["source_kit_cross_split_count"] == 0
        and report["item_cross_split_count"] == 0
        and report["global_duplicate_sample_id_count"] == 0
    )
    report["status"] = "READY_TO_TRAIN" if report["pass"] else "BLOCKED"
    return report


def build_manifests(
    *,
    output_dir: Path,
    scorer_paths: Mapping[str, Path],
    negative_paths: Mapping[str, Path],
    metadata_paths: Mapping[str, Path],
    records_by_split: Mapping[str, Sequence[dict]],
    validation_report: Mapping[str, object],
    embedding_report_path: Path,
    embedding_report: Mapping[str, object],
    seed: int,
    git_commit: str | None = None,
) -> tuple[dict, dict]:
    """Create immutable file references and the final dataset contract."""

    split_entries: dict[str, dict] = {}
    for split in SPLITS:
        records = records_by_split[split]
        split_entries[split] = {
            "scorer_file": str(scorer_paths[split]),
            "scorer_sha256": sha256_file(scorer_paths[split]),
            "negative_file": str(negative_paths[split]),
            "negative_sha256": sha256_file(negative_paths[split]),
            "item_metadata_file": str(metadata_paths[split]),
            "item_metadata_sha256": sha256_file(metadata_paths[split]),
            "sample_count": len(records),
            "positive_count": sum(int(row["label"]) == 1 for row in records),
            "negative_count": sum(int(row["label"]) == 0 for row in records),
            "source_kit_count": len(
                {
                    str(row["source_kit_id"])
                    for row in records
                    if int(row["label"]) == 1
                }
            ),
        }

    split_manifest = {
        "manifest_version": "split-manifest-v1",
        "dataset_version": DATASET_VERSION,
        "source_dataset": "codewaly/polyvore1000",
        "split_policy": "official_train_valid_test",
        "splits": split_entries,
    }
    cache_report = embedding_report.get("cache", {})
    dataset_manifest = {
        "manifest_version": "dataset-manifest-v1",
        "dataset_version": DATASET_VERSION,
        "status": validation_report["status"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "source_dataset": "codewaly/polyvore1000",
        "data_contract_version": "data-contract-v1.0",
        "category_mapping_version": "core7-v1",
        "item_metadata_version": "core7-item-metadata-v1",
        "embedding_version": "fashionclip-512-l2-v1",
        "embedding_model": cache_report.get("model_id"),
        "embedding_dimension": cache_report.get("embedding_dim"),
        "embedding_normalization": "l2",
        "embedding_validation_report": str(embedding_report_path),
        "negative_version": NEGATIVE_VERSION,
        "negative_type": NEGATIVE_TYPE,
        "negative_seed": seed,
        "negatives_per_positive": 1,
        "minimum_outfit_items": DEFAULT_MIN_ITEMS,
        "split_manifest": str(output_dir / "split_manifest_v1.json"),
        "final_validation_report": str(output_dir / "final_validation_v1.json"),
        "splits": split_entries,
    }
    return split_manifest, dataset_manifest


def build_scorer_dataset_v1(
    *,
    data_dir: Path | str,
    output_dir: Path | str,
    embedding_report_path: Path | str,
    seed: int = DEFAULT_SEED,
    git_commit: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Generate negatives, merge pairs, validate everything, and freeze V1."""

    source_dir = Path(data_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    protected_outputs = [
        *(destination / f"negative_v1_{split}.jsonl" for split in SPLITS),
        *(destination / f"scorer_ready_v1_{split}.jsonl" for split in SPLITS),
        destination / "final_validation_v1.json",
        destination / "split_manifest_v1.json",
        destination / "dataset_manifest_v1.json",
    ]
    existing_outputs = [path for path in protected_outputs if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite versioned dataset artifacts. Existing files: "
            f"{[str(path) for path in existing_outputs]}. "
            "Use a new dataset version, or pass overwrite=True only during "
            "pre-freeze development."
        )
    embedding_path = Path(embedding_report_path)
    embedding_report = read_json(embedding_path)
    if not embedding_report.get("pass"):
        raise ValueError("Embedding validation report is not PASS")

    records_by_split: dict[str, list[dict]] = {}
    metadata_by_split: dict[str, list[dict]] = {}
    sampling_reports: dict[str, dict] = {}
    merge_reports: dict[str, dict] = {}
    scorer_paths: dict[str, Path] = {}
    negative_paths: dict[str, Path] = {}
    metadata_paths: dict[str, Path] = {}

    for split_index, split in enumerate(SPLITS):
        # A deterministic derived seed prevents identical RNG streams per split.
        split_seed = seed + split_index
        positive_path = source_dir / f"category_clean_{split}.jsonl"
        metadata_path = source_dir / f"core7_item_metadata_v1_{split}.jsonl"
        positives = read_jsonl(positive_path)
        metadata = read_jsonl(metadata_path)

        negatives, sampling_report = generate_negative_records(
            positives,
            metadata,
            split=split,
            seed=split_seed,
        )
        scorer_records, merge_report = merge_positive_negative_families(
            positives,
            negatives,
        )

        negative_path = destination / f"negative_v1_{split}.jsonl"
        scorer_path = destination / f"scorer_ready_v1_{split}.jsonl"
        sampling_report_path = (
            destination / f"negative_sampling_v1_{split}_report.json"
        )
        write_jsonl(negatives, negative_path)
        write_jsonl(scorer_records, scorer_path)
        sampling_report["output_path"] = str(negative_path)
        sampling_report["merge"] = merge_report
        write_json(sampling_report, sampling_report_path)

        records_by_split[split] = scorer_records
        metadata_by_split[split] = metadata
        sampling_reports[split] = sampling_report
        merge_reports[split] = merge_report
        scorer_paths[split] = scorer_path
        negative_paths[split] = negative_path
        metadata_paths[split] = metadata_path

    final_report = validate_all_splits(
        records_by_split,
        metadata_by_split,
        embedding_report=embedding_report,
        sampling_reports=sampling_reports,
    )
    final_report["merge_reports"] = merge_reports
    final_report_path = destination / "final_validation_v1.json"
    write_json(final_report, final_report_path)

    split_manifest, dataset_manifest = build_manifests(
        output_dir=destination,
        scorer_paths=scorer_paths,
        negative_paths=negative_paths,
        metadata_paths=metadata_paths,
        records_by_split=records_by_split,
        validation_report=final_report,
        embedding_report_path=embedding_path,
        embedding_report=embedding_report,
        seed=seed,
        git_commit=git_commit,
    )
    split_manifest_path = destination / "split_manifest_v1.json"
    dataset_manifest_path = destination / "dataset_manifest_v1.json"
    write_json(split_manifest, split_manifest_path)
    write_json(dataset_manifest, dataset_manifest_path)

    return {
        "status": final_report["status"],
        "dataset_version": DATASET_VERSION,
        "negative_version": NEGATIVE_VERSION,
        "seed": seed,
        "sampling_reports": sampling_reports,
        "final_validation": final_report,
        "split_manifest_path": str(split_manifest_path),
        "dataset_manifest_path": str(dataset_manifest_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate the Core-7 scorer-ready dataset V1"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = build_scorer_dataset_v1(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        embedding_report_path=args.embedding_report,
        seed=args.seed,
        git_commit=args.git_commit,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "READY_TO_TRAIN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
