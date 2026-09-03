# -*- coding: utf-8 -*-
"""Optional item metadata adapters for Recommendation V1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.scorer.dataset import CATEGORY_TO_ID


IMAGE_REFERENCE_KEYS = ("image_reference", "image_url", "image_path", "image")


class ItemMetadataIndex:
    """Small, permissive metadata index with strict duplicate consistency."""

    def __init__(self, records: Iterable[Mapping[str, object]] = ()) -> None:
        self._records: dict[str, dict[str, object]] = {}
        for raw in records:
            self.add(raw)

    def __len__(self) -> int:
        return len(self._records)

    def add(self, raw: Mapping[str, object]) -> None:
        row = dict(raw)
        item_id = str(row.get("item_id", "")).strip()
        if not item_id:
            raise ValueError("Metadata item_id must be non-empty")
        coarse = row.get("coarse_category")
        if coarse is not None:
            normalized = str(coarse).strip().upper()
            if normalized not in CATEGORY_TO_ID:
                raise ValueError(f"Unknown Core-7 category for {item_id}: {coarse!r}")
            row["coarse_category"] = normalized
            supplied_id = row.get("coarse_category_id")
            expected_id = CATEGORY_TO_ID[normalized]
            if supplied_id is not None and int(supplied_id) != expected_id:
                raise ValueError(f"Category name/ID mismatch for {item_id}")
            row["coarse_category_id"] = expected_id
        elif row.get("coarse_category_id") is not None:
            category_id = int(row["coarse_category_id"])
            reverse = {value: key for key, value in CATEGORY_TO_ID.items()}
            if category_id not in reverse:
                raise ValueError(f"Unknown Core-7 category ID for {item_id}: {category_id}")
            row["coarse_category_id"] = category_id
            row["coarse_category"] = reverse[category_id]
        row["item_id"] = item_id

        previous = self._records.get(item_id)
        if previous is not None:
            for key in ("coarse_category", "coarse_category_id"):
                if previous.get(key) is not None and row.get(key) is not None:
                    if previous[key] != row[key]:
                        raise ValueError(f"Conflicting {key} for item {item_id}")
            merged = dict(previous)
            merged.update({key: value for key, value in row.items() if value is not None})
            row = merged
        self._records[item_id] = row

    def get(self, item_id: str) -> dict[str, object] | None:
        row = self._records.get(str(item_id))
        return None if row is None else dict(row)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    def category_id(self, item_id: str) -> int | None:
        row = self._records.get(str(item_id))
        if row is None or row.get("coarse_category_id") is None:
            return None
        return int(row["coarse_category_id"])

    def master_category(self, item_id: str) -> str | None:
        row = self._records.get(str(item_id))
        if row is None or not str(row.get("master_category", "")).strip():
            return None
        return str(row["master_category"])

    def coarse_category(self, item_id: str) -> str | None:
        row = self._records.get(str(item_id))
        if row is None or not str(row.get("coarse_category", "")).strip():
            return None
        return str(row["coarse_category"])

    def image_reference(self, item_id: str) -> str | None:
        row = self._records.get(str(item_id))
        if row is None:
            return None
        for key in IMAGE_REFERENCE_KEYS:
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return None

    @classmethod
    def from_jsonl(cls, paths: Path | str | Sequence[Path | str]) -> "ItemMetadataIndex":
        sources = [paths] if isinstance(paths, (str, Path)) else list(paths)
        records = []
        for source in sources:
            path = Path(source)
            with path.open("r", encoding="utf-8") as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"Expected object at {path}:{line_number}")
                    records.append(row)
        return cls(records)


def load_core7_master_mapping(
    v1_mapping_path: Path | str,
    v2_mapping_path: Path | str,
) -> dict[str, str]:
    """Compose the frozen V1 base map with the explicit V2 overrides."""

    v1 = json.loads(Path(v1_mapping_path).read_text(encoding="utf-8"))
    v2 = json.loads(Path(v2_mapping_path).read_text(encoding="utf-8"))
    mapping = dict(v1.get("mapping", {}))
    mapping.update(v2.get("overrides", {}))
    return {str(key): str(value).upper() for key, value in mapping.items()}


def metadata_from_compatibility_jsonl(
    paths: Sequence[Path | str],
    *,
    master_to_core7: Mapping[str, str],
) -> ItemMetadataIndex:
    """Recover partial category metadata from raw one-item-swap rows.

    Raw compatibility files contain category evidence only for the original and
    replacement items in negative rows.  They do not provide images or complete
    metadata for every outfit item, so this adapter intentionally returns a
    partial index and drops categories outside frozen Core-7 V2.
    """

    index = ItemMetadataIndex()
    for source in paths:
        path = Path(source)
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                if row.get("label") != 0:
                    continue
                master_category = str(row.get("swap_category", "")).strip()
                coarse = master_to_core7.get(master_category, "DROP")
                if coarse == "DROP":
                    continue
                if coarse not in CATEGORY_TO_ID:
                    raise ValueError(
                        f"Mapping produced unknown Core-7 category {coarse!r}"
                    )
                for key in ("original_item_id", "replacement_item_id"):
                    item_id = str(row.get(key, "")).strip()
                    if item_id:
                        index.add(
                            {
                                "item_id": item_id,
                                "master_category": master_category,
                                "coarse_category": coarse,
                                "metadata_source": str(path),
                            }
                        )
    return index
