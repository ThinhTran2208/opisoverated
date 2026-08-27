# -*- coding: utf-8 -*-
"""Prepare Core-7 V2 positives without mutating the frozen V1 mapping.

V2 is expressed as a small, versioned override file on top of the immutable
``category_mapping_core7_v1.json``. The resolved mapping is validated before
any artifact is written. Output JSONL schema remains item-metadata schema V1;
``category_mapping_version`` records the independent Core-7 mapping version.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from . import prepare_core7_dataset as base


DEFAULT_MAPPING_PATH = Path("configs/category_mapping_core7_v2.json")
EXPECTED_MAPPING_VERSION = "core7-v2"


def load_category_mapping_v2(path: Path | str = DEFAULT_MAPPING_PATH) -> Tuple[dict, Dict[str, str]]:
    """Resolve and validate a Core-7 V2 base+overrides mapping artifact."""

    mapping_path = Path(path)
    with mapping_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Core-7 V2 mapping must be a JSON object")
    if payload.get("mapping_version") != EXPECTED_MAPPING_VERSION:
        raise ValueError(
            f"Expected mapping_version={EXPECTED_MAPPING_VERSION}, "
            f"got {payload.get('mapping_version')!r}"
        )
    if payload.get("status") != "frozen":
        raise ValueError("Core-7 V2 mapping must be frozen before full generation")

    base_name = payload.get("base_mapping")
    if not isinstance(base_name, str) or not base_name.strip():
        raise ValueError("Core-7 V2 mapping must declare base_mapping")
    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = mapping_path.parent / base_path
    base_metadata, resolved = base.load_category_mapping(base_path)
    if base_metadata.get("mapping_version") != "core7-v1":
        raise ValueError("Core-7 V2 must inherit from frozen core7-v1")
    if base_metadata.get("status") != "frozen":
        raise ValueError("Core-7 V1 base mapping is not frozen")

    raw_overrides = payload.get("overrides", {})
    if not isinstance(raw_overrides, Mapping):
        raise ValueError("Core-7 V2 overrides must be a JSON object")
    resolved = dict(resolved)
    for raw_master, raw_decision in raw_overrides.items():
        master = str(raw_master).strip()
        decision = str(raw_decision).strip().upper()
        if master not in resolved:
            raise ValueError(f"Core-7 V2 override references unknown category: {master}")
        if decision not in base.ALLOWED_MAPPING_VALUES:
            raise ValueError(f"Invalid Core-7 V2 decision for {master}: {decision}")
        resolved[master] = decision

    declared_counts = payload.get("decision_counts")
    actual_counts = dict(Counter(resolved.values()))
    if declared_counts is not None:
        normalized_counts = {
            str(key).strip().upper(): int(value)
            for key, value in declared_counts.items()
        }
        if normalized_counts != actual_counts:
            raise ValueError(
                "Core-7 V2 decision_counts does not match resolved mapping: "
                f"declared={normalized_counts}, actual={actual_counts}"
            )

    metadata = {key: value for key, value in payload.items() if key != "overrides"}
    metadata["base_mapping_version"] = base_metadata["mapping_version"]
    metadata["resolved_mapping_count"] = len(resolved)
    return metadata, resolved


def prepare_clean_positive_split_v2(
    *,
    split: str,
    output_path: Path | str,
    item_metadata_output_path: Path | str | None = None,
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
    dataset_name: str = base.DATASET_NAME,
    min_items: int = base.DEFAULT_MIN_OUTFIT_ITEMS,
    debug_limit: Optional[int] = None,
) -> dict:
    """Run Core-7 V2 category cleaning while preserving V1 artifacts."""

    mapping_metadata, mapping = load_category_mapping_v2(mapping_path)
    mapping_version = str(mapping_metadata["mapping_version"])
    metadata_output_path = (
        Path(item_metadata_output_path)
        if item_metadata_output_path is not None
        else base.default_item_metadata_output_path(output_path, split=split)
    )

    print("=" * 72)
    print(f"CORE-7 V2 CLEAN POSITIVES - split={split}")
    print("=" * 72)
    print("Mapping version:", mapping_version)
    print("Mapping status :", mapping_metadata["status"])
    print("Base mapping   :", mapping_metadata["base_mapping_version"])
    print("Item metadata  :", base.CORE7_ITEM_METADATA_VERSION)
    print("Minimum items  :", min_items)
    print()

    items_ds, kits_ds = base.load_required_datasets(dataset_name=dataset_name, split=split)
    raw_kit_to_items, _, _ = base.build_item_indexes(items_ds)
    filtered_kit_to_items, filter_report = base.filter_items_by_core_category(
        raw_kit_to_items, mapping
    )
    samples, outfit_report = base.build_clean_positive_samples(
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
    validation_report = base.validate_clean_positive_samples(
        samples, item_to_coarse, min_items=min_items
    )
    item_metadata = base.build_core7_item_metadata(
        samples,
        filtered_kit_to_items,
        split=split,
        mapping_version=mapping_version,
    )
    item_metadata_validation = base.validate_core7_item_metadata(
        item_metadata,
        samples,
        split=split,
        mapping_version=mapping_version,
    )

    written = base.write_jsonl(samples, output_path)
    metadata_written = base.write_jsonl(item_metadata, metadata_output_path)
    if written != len(samples) or metadata_written != len(item_metadata):
        raise RuntimeError("Written Core-7 V2 record counts do not match in-memory counts")

    report = {
        "processing_stage": "category_filtered_positives",
        "ready_to_train": False,
        "dataset_name": dataset_name,
        "split": split,
        "mapping_version": mapping_version,
        "mapping_status": mapping_metadata["status"],
        "base_mapping_version": mapping_metadata["base_mapping_version"],
        "item_metadata_version": base.CORE7_ITEM_METADATA_VERSION,
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
    print("Metadata items   :", f"{len(item_metadata):,}")
    print("Validation       : PASS")
    print("Positive output  :", output_path)
    print("Metadata output  :", metadata_output_path)
    print()
    return report
