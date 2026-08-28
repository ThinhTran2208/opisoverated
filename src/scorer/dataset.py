# -*- coding: utf-8 -*-
"""Scorer V1 dataset, padding, and pair-mask utilities.

This module is the shared S1 data interface for ``type_aware_pairwise_v1``.
It consumes the frozen Core-7 V2 scorer-ready JSONL, split item metadata, and
precomputed FashionCLIP cache. It does not read images and does not expose
negative-generation metadata as neural input features.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # Keep schema/path helpers importable in lightweight CI.
    torch = None


SCORER_VERSION = "type_aware_pairwise_v1"
DATASET_VERSION = "polyvore1000-core7-compat-v2"
CATEGORY_MAPPING_VERSION = "core7-v2"
ITEM_METADATA_VERSION = "core7-item-metadata-v1"
EMBEDDING_VERSION = "fashionclip-512-l2-v1"
EMBEDDING_DIM = 512
MIN_ITEMS = 3
MAX_ITEMS = 8
SPLITS = ("train", "valid", "test")

CATEGORY_TO_ID = {
    "TOP": 1,
    "BOTTOM": 2,
    "DRESS": 3,
    "OUTERWEAR": 4,
    "SHOES": 5,
    "BAG": 6,
    "HAT": 7,
}
PAD_CATEGORY_ID = 0
CATEGORY_VOCAB_SIZE = 8


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for scorer Dataset/DataLoader operations. "
            "Install torch in the training environment."
        )


def read_jsonl(path: Path | str) -> list[dict]:
    """Read UTF-8 JSONL and reject malformed or non-object rows."""

    source = Path(path)
    rows: list[dict] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {source}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {source}:{line_number}")
            rows.append(row)
    return rows


def scorer_split_path(scorer_ready_dir: Path | str, split: str) -> Path:
    """Return the canonical frozen V2 scorer JSONL path for ``split``."""

    _validate_split(split)
    return Path(scorer_ready_dir) / f"scorer_ready_v2_{split}.jsonl"


def metadata_split_path(core7_dir: Path | str, split: str) -> Path:
    """Return the canonical Core-7 item metadata path for ``split``."""

    _validate_split(split)
    return Path(core7_dir) / f"core7_item_metadata_v1_{split}.jsonl"


def _validate_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")


def build_metadata_index(records: Sequence[Mapping[str, object]]) -> dict[str, dict]:
    """Build a strict ``item_id -> metadata`` lookup for one split."""

    index: dict[str, dict] = {}
    for row_number, raw in enumerate(records, start=1):
        row = dict(raw)
        item_id = str(row.get("item_id", "")).strip()
        if not item_id:
            raise ValueError(f"Metadata row {row_number} has invalid item_id")
        if item_id in index:
            raise ValueError(f"Duplicate metadata item_id: {item_id}")
        if row.get("category_mapping_version") != CATEGORY_MAPPING_VERSION:
            raise ValueError(
                f"Metadata item {item_id} category mapping mismatch: "
                f"{row.get('category_mapping_version')!r}"
            )
        if row.get("item_metadata_version") != ITEM_METADATA_VERSION:
            raise ValueError(
                f"Metadata item {item_id} version mismatch: "
                f"{row.get('item_metadata_version')!r}"
            )
        coarse = str(row.get("coarse_category", "")).strip().upper()
        if coarse not in CATEGORY_TO_ID:
            raise ValueError(f"Unknown coarse_category for item {item_id}: {coarse!r}")
        row["item_id"] = item_id
        row["coarse_category"] = coarse
        index[item_id] = row
    return index


def paired_family_indices(
    records: Sequence[Mapping[str, object]],
    *,
    max_families: int | None = None,
) -> list[tuple[int, int]]:
    """Return complete ``(positive_index, negative_index)`` families.

    The frozen V2 protocol has exactly one paired negative for every positive.
    The function validates that relationship without relying on row adjacency.
    """

    if max_families is not None and max_families < 1:
        raise ValueError("max_families must be >= 1 when provided")

    seen_sample_ids: set[str] = set()
    positives: dict[str, int] = {}
    positive_order: list[str] = []
    negatives_by_positive: dict[str, list[int]] = defaultdict(list)

    for index, row in enumerate(records):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"Record {index} has invalid sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)

        label = row.get("label")
        pair_id = row.get("paired_positive_sample_id")
        if label == 1:
            if pair_id not in (None, ""):
                raise ValueError(
                    f"Positive {sample_id} must have null paired_positive_sample_id"
                )
            positives[sample_id] = index
            positive_order.append(sample_id)
        elif label == 0:
            normalized_pair_id = str(pair_id or "").strip()
            if not normalized_pair_id:
                raise ValueError(f"Negative {sample_id} is missing paired positive ID")
            negatives_by_positive[normalized_pair_id].append(index)
        else:
            raise ValueError(f"Record {sample_id} has invalid label: {label!r}")

    orphan_ids = sorted(set(negatives_by_positive) - set(positives))
    if orphan_ids:
        raise ValueError(f"Negatives reference missing positives: {orphan_ids[:10]}")

    families: list[tuple[int, int]] = []
    for positive_id in positive_order:
        negative_indices = negatives_by_positive.get(positive_id, [])
        if len(negative_indices) != 1:
            raise ValueError(
                f"Positive {positive_id} must have exactly one negative, "
                f"found {len(negative_indices)}"
            )
        families.append((positives[positive_id], negative_indices[0]))

    if len(families) != len(positives) or len(families) != sum(
        len(rows) for rows in negatives_by_positive.values()
    ):
        raise ValueError("Paired family coverage is incomplete")

    return families if max_families is None else families[:max_families]


def flatten_family_indices(families: Sequence[tuple[int, int]]) -> list[int]:
    """Flatten complete pair families for ``torch.utils.data.Subset``."""

    return [index for family in families for index in family]


def build_pair_index_tuples(item_count: int) -> list[tuple[int, int]]:
    """Return all unordered pair positions ``i < j`` for ``item_count`` items."""

    if item_count < 0:
        raise ValueError("item_count must be >= 0")
    return [(i, j) for i in range(item_count) for j in range(i + 1, item_count)]


def build_pair_mask(item_mask):
    """Build BoolTensor ``[B, L, L]`` for valid real-item pairs with ``i < j``."""

    require_torch()
    if not isinstance(item_mask, torch.Tensor):
        raise TypeError("item_mask must be a torch.Tensor")
    if item_mask.ndim != 2:
        raise ValueError(f"item_mask must have shape [B, L], got {tuple(item_mask.shape)}")
    if item_mask.dtype != torch.bool:
        raise ValueError("item_mask must have dtype torch.bool")

    _, length = item_mask.shape
    upper = torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=item_mask.device),
        diagonal=1,
    )
    return item_mask.unsqueeze(2) & item_mask.unsqueeze(1) & upper.unsqueeze(0)


def load_embedding_cache(path: Path | str) -> Mapping[str, object]:
    """Load the team-owned FashionCLIP cache on CPU."""

    require_torch()
    source = Path(path)
    try:
        cache = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # Older supported PyTorch versions.
        cache = torch.load(source, map_location="cpu")
    if not isinstance(cache, Mapping):
        raise ValueError("Embedding cache must contain a mapping/dictionary")
    return cache


def inspect_embedding_cache_for_scorer(
    cache: Mapping[str, object],
) -> tuple[list[str], object]:
    """Validate the scorer-facing cache shape and return IDs + matrix.

    Full provenance/norm validation belongs to NB3/NB4. S1 deliberately checks
    only the interface it consumes and never renormalizes embeddings.
    """

    require_torch()
    required = {"item_ids", "embeddings"}
    missing = sorted(required - set(cache))
    if missing:
        raise ValueError(f"Embedding cache missing keys: {missing}")

    item_ids = [str(item_id) for item_id in cache["item_ids"]]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Embedding cache contains duplicate item_ids")

    embeddings = cache["embeddings"]
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("cache['embeddings'] must be a torch.Tensor")
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected embeddings [N, {EMBEDDING_DIM}], got {tuple(embeddings.shape)}"
        )
    if embeddings.shape[0] != len(item_ids):
        raise ValueError("Embedding rows do not match item_ids length")
    return item_ids, embeddings


class EmbeddingStore:
    """Load the FashionCLIP cache once and share it across split datasets."""

    def __init__(self, embedding_cache_path: Path | str) -> None:
        require_torch()
        self.embedding_cache_path = Path(embedding_cache_path)
        cache = load_embedding_cache(self.embedding_cache_path)
        item_ids, embeddings = inspect_embedding_cache_for_scorer(cache)
        self.embedding_matrix = embeddings
        self.embedding_row_by_item = {
            item_id: row for row, item_id in enumerate(item_ids)
        }


class ScorerDataset:
    """Map frozen scorer-ready records to variable-length tensor samples."""

    def __init__(
        self,
        samples_path: Path | str,
        metadata_path: Path | str,
        embedding_cache_path: Path | str | None = None,
        *,
        embedding_store: EmbeddingStore | None = None,
        min_items: int = MIN_ITEMS,
        max_items: int = MAX_ITEMS,
    ) -> None:
        require_torch()
        if min_items < 1 or max_items < min_items:
            raise ValueError("Require 1 <= min_items <= max_items")
        if embedding_store is None and embedding_cache_path is None:
            raise ValueError("Provide embedding_cache_path or embedding_store")

        self.samples_path = Path(samples_path)
        self.metadata_path = Path(metadata_path)
        self.min_items = min_items
        self.max_items = max_items

        self.records = read_jsonl(self.samples_path)
        self.metadata_by_item = build_metadata_index(read_jsonl(self.metadata_path))
        self.pair_families = paired_family_indices(self.records)

        self.embedding_store = embedding_store or EmbeddingStore(embedding_cache_path)
        self.embedding_cache_path = self.embedding_store.embedding_cache_path
        self.embedding_matrix = self.embedding_store.embedding_matrix
        self.embedding_row_by_item = self.embedding_store.embedding_row_by_item

        self._validate_records()

    def _validate_records(self) -> None:
        for row_number, record in enumerate(self.records, start=1):
            sample_id = str(record.get("sample_id", "")).strip()
            items = record.get("items")
            if not isinstance(items, list):
                raise ValueError(f"Sample {sample_id or row_number} has invalid items")
            if not self.min_items <= len(items) <= self.max_items:
                raise ValueError(
                    f"Sample {sample_id} length={len(items)} outside "
                    f"[{self.min_items}, {self.max_items}]"
                )
            if len(items) != len(set(str(item) for item in items)):
                raise ValueError(f"Sample {sample_id} contains duplicate item IDs")
            if record.get("label") not in (0, 1):
                raise ValueError(f"Sample {sample_id} has invalid label")

            for raw_item_id in items:
                item_id = str(raw_item_id)
                if item_id not in self.metadata_by_item:
                    raise ValueError(f"Sample {sample_id} missing metadata for {item_id}")
                if item_id not in self.embedding_row_by_item:
                    raise ValueError(f"Sample {sample_id} missing embedding for {item_id}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        item_ids = [str(item_id) for item_id in record["items"]]
        embedding_rows = [self.embedding_row_by_item[item_id] for item_id in item_ids]
        embeddings = self.embedding_matrix[embedding_rows].float()

        category_ids = torch.tensor(
            [
                CATEGORY_TO_ID[self.metadata_by_item[item_id]["coarse_category"]]
                for item_id in item_ids
            ],
            dtype=torch.long,
        )

        return {
            "sample_id": str(record["sample_id"]),
            "source_kit_id": str(record.get("source_kit_id", "")),
            "paired_positive_sample_id": record.get("paired_positive_sample_id"),
            "item_ids": item_ids,
            "item_embeddings": embeddings,
            "coarse_category_ids": category_ids,
            "label": float(record["label"]),
            "negative_metadata": record.get("negative_metadata"),
        }


def collate_scorer_batch(
    samples: Sequence[Mapping[str, object]],
    *,
    max_items: int = MAX_ITEMS,
) -> dict:
    """Pad variable-length samples to the locked Scorer V1 batch contract."""

    require_torch()
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    batch_size = len(samples)
    embeddings = torch.zeros(
        (batch_size, max_items, EMBEDDING_DIM), dtype=torch.float32
    )
    category_ids = torch.zeros((batch_size, max_items), dtype=torch.long)
    item_mask = torch.zeros((batch_size, max_items), dtype=torch.bool)
    labels = torch.empty(batch_size, dtype=torch.float32)

    for batch_index, sample in enumerate(samples):
        sample_embeddings = sample["item_embeddings"]
        sample_categories = sample["coarse_category_ids"]
        if not isinstance(sample_embeddings, torch.Tensor) or sample_embeddings.ndim != 2:
            raise ValueError("item_embeddings must be rank-2 tensors")
        if sample_embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Expected embedding dim {EMBEDDING_DIM}")
        if not isinstance(sample_categories, torch.Tensor) or sample_categories.ndim != 1:
            raise ValueError("coarse_category_ids must be rank-1 tensors")

        item_count = int(sample_embeddings.shape[0])
        if item_count < MIN_ITEMS or item_count > max_items:
            raise ValueError(
                f"Sample item count {item_count} outside [{MIN_ITEMS}, {max_items}]"
            )
        if sample_categories.shape[0] != item_count:
            raise ValueError("Category count does not match embedding row count")

        embeddings[batch_index, :item_count] = sample_embeddings.float()
        category_ids[batch_index, :item_count] = sample_categories.long()
        item_mask[batch_index, :item_count] = True
        labels[batch_index] = float(sample["label"])

    return {
        "item_embeddings": embeddings,
        "coarse_category_ids": category_ids,
        "item_mask": item_mask,
        "pair_mask": build_pair_mask(item_mask),
        "labels": labels,
        "sample_ids": [str(sample["sample_id"]) for sample in samples],
        "source_kit_ids": [str(sample.get("source_kit_id", "")) for sample in samples],
        "paired_positive_sample_ids": [
            sample.get("paired_positive_sample_id") for sample in samples
        ],
        "item_ids": [list(sample["item_ids"]) for sample in samples],
        "negative_metadata": [sample.get("negative_metadata") for sample in samples],
    }


def build_dataset_from_runtime(
    runtime_paths,
    split: str,
    *,
    embedding_store: EmbeddingStore | None = None,
) -> ScorerDataset:
    """Construct a split dataset from ``src.data.runtime_paths.RuntimePaths``."""

    _validate_split(split)
    return ScorerDataset(
        scorer_split_path(runtime_paths.scorer_ready_dir, split),
        metadata_split_path(runtime_paths.core7_dir, split),
        runtime_paths.embedding_cache if embedding_store is None else None,
        embedding_store=embedding_store,
    )


def build_datasets_from_runtime(
    runtime_paths,
    *,
    splits: Sequence[str] = SPLITS,
) -> tuple[dict[str, ScorerDataset], EmbeddingStore]:
    """Build multiple splits while loading the embedding cache exactly once."""

    store = EmbeddingStore(runtime_paths.embedding_cache)
    datasets = {
        split: build_dataset_from_runtime(
            runtime_paths, split, embedding_store=store
        )
        for split in splits
    }
    return datasets, store
