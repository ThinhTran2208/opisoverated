# -*- coding: utf-8 -*-
"""Sharded FashionCLIP catalog used by Recommendation V1.

The catalog deliberately searches one shard at a time.  Adding the remaining
shards therefore changes data volume, not the Recommendation API or storage
format, and does not require concatenating the full catalog in memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    import numpy as np
except ModuleNotFoundError:  # Keep package importable in lightweight CI.
    np = None


EMBEDDING_DIM = 512
DEFAULT_SHARD_GLOB = "embeddings_shard_*.npz"


def require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for the sharded recommendation catalog")


@dataclass(frozen=True)
class SearchHit:
    item_id: str
    similarity: float


@dataclass(frozen=True)
class CatalogStatus:
    loaded_items: int
    loaded_shards: int
    expected_items: int | None
    expected_shards: int | None
    is_complete: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "loaded_items": self.loaded_items,
            "loaded_shards": self.loaded_shards,
            "expected_items": self.expected_items,
            "expected_shards": self.expected_shards,
            "is_complete": self.is_complete,
        }


class NpzShardCatalog:
    """Exact cosine search over independently loadable NPZ shards.

    Each shard must contain ``item_ids`` with shape ``[N]`` and ``embeddings``
    with shape ``[N, 512]``.  Embeddings stay float16 on disk and are converted
    shard-by-shard to float32 only while searching or selecting requested rows.
    """

    def __init__(
        self,
        catalog_dir: Path | str,
        *,
        manifest_path: Path | str | None = None,
        shard_glob: str = DEFAULT_SHARD_GLOB,
        validate_norms: bool = True,
        norm_tolerance: float = 0.02,
    ) -> None:
        require_numpy()
        self.catalog_dir = Path(catalog_dir).expanduser().resolve()
        if not self.catalog_dir.is_dir():
            raise FileNotFoundError(f"Catalog directory not found: {self.catalog_dir}")
        if norm_tolerance <= 0:
            raise ValueError("norm_tolerance must be positive")
        self.norm_tolerance = float(norm_tolerance)
        self.manifest_path = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else self.catalog_dir / "cache_manifest.json"
        )
        self.manifest = self._load_manifest()
        self.shard_paths = self._resolve_shards(shard_glob)
        self._locator_by_item: dict[str, tuple[Path, int]] = {}
        self._item_ids: list[str] = []
        loaded_items = 0

        for path in self.shard_paths:
            item_ids, embeddings = self._load_shard(path)
            if validate_norms:
                self._validate_norms(path, embeddings)
            for row, item_id in enumerate(item_ids):
                normalized = str(item_id)
                if not normalized:
                    raise ValueError(f"Empty item_id in {path} at row {row}")
                if normalized in self._locator_by_item:
                    raise ValueError(f"Duplicate catalog item_id: {normalized}")
                self._locator_by_item[normalized] = (path, row)
                self._item_ids.append(normalized)
            loaded_items += len(item_ids)

        expected_items = self._optional_positive_int("total_items")
        expected_shards = self._optional_positive_int("total_shards")
        complete = None
        if expected_items is not None or expected_shards is not None:
            complete = (
                (expected_items is None or loaded_items == expected_items)
                and (expected_shards is None or len(self.shard_paths) == expected_shards)
            )
        self.status = CatalogStatus(
            loaded_items=loaded_items,
            loaded_shards=len(self.shard_paths),
            expected_items=expected_items,
            expected_shards=expected_shards,
            is_complete=complete,
        )

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(self._item_ids)

    def __len__(self) -> int:
        return len(self._item_ids)

    def __contains__(self, item_id: object) -> bool:
        return str(item_id) in self._locator_by_item

    def _load_manifest(self) -> dict[str, object]:
        if not self.manifest_path.is_file():
            return {}
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Catalog manifest must be a JSON object")
        expected_dim = payload.get("expected_dim")
        if expected_dim is not None and int(expected_dim) != EMBEDDING_DIM:
            raise ValueError(
                f"Catalog manifest expected_dim must be {EMBEDDING_DIM}, got {expected_dim}"
            )
        if payload.get("normalized") is False:
            raise ValueError("Recommendation V1 requires L2-normalized embeddings")
        return payload

    def _optional_positive_int(self, key: str) -> int | None:
        value = self.manifest.get(key)
        if value is None:
            return None
        parsed = int(value)
        if parsed < 1:
            raise ValueError(f"Manifest {key} must be positive")
        return parsed

    def _resolve_shards(self, shard_glob: str) -> tuple[Path, ...]:
        completed = self.manifest.get("completed_shards")
        if completed is not None:
            if not isinstance(completed, list) or not all(
                isinstance(value, str) and value for value in completed
            ):
                raise ValueError("Manifest completed_shards must be a list of filenames")
            paths = tuple(self.catalog_dir / name for name in completed)
        else:
            paths = tuple(sorted(self.catalog_dir.glob(shard_glob)))
        if not paths:
            raise FileNotFoundError(
                f"No embedding shards found in {self.catalog_dir} ({shard_glob})"
            )
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Manifest references missing shards: {missing}")
        return paths

    @staticmethod
    def _load_shard(path: Path):
        with np.load(path, allow_pickle=False) as payload:
            missing = {"item_ids", "embeddings"} - set(payload.files)
            if missing:
                raise ValueError(f"Shard {path} missing arrays: {sorted(missing)}")
            item_ids = np.asarray(payload["item_ids"]).astype(str)
            embeddings = np.asarray(payload["embeddings"])
        if item_ids.ndim != 1:
            raise ValueError(f"Shard {path} item_ids must have shape [N]")
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Shard {path} embeddings must have shape [N, {EMBEDDING_DIM}]"
            )
        if embeddings.shape[0] != item_ids.shape[0]:
            raise ValueError(f"Shard {path} item_ids/embeddings row mismatch")
        if not np.issubdtype(embeddings.dtype, np.floating):
            raise ValueError(f"Shard {path} embeddings must be floating point")
        if not bool(np.isfinite(embeddings).all()):
            raise ValueError(f"Shard {path} contains NaN/Inf")
        return item_ids, embeddings

    def _validate_norms(self, path: Path, embeddings) -> None:
        norms = np.linalg.norm(embeddings.astype(np.float32, copy=False), axis=1)
        if not bool(np.all(np.abs(norms - 1.0) <= self.norm_tolerance)):
            index = int(np.flatnonzero(np.abs(norms - 1.0) > self.norm_tolerance)[0])
            raise ValueError(
                f"Shard {path} row {index} is not L2-normalized: norm={norms[index]}"
            )

    @staticmethod
    def _normalize_query(query):
        vector = np.asarray(query, dtype=np.float32)
        if vector.shape != (EMBEDDING_DIM,):
            raise ValueError(f"query must have shape [{EMBEDDING_DIM}]")
        if not bool(np.isfinite(vector).all()):
            raise ValueError("query contains NaN/Inf")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("query must have non-zero norm")
        return vector / norm

    def search(
        self,
        query,
        *,
        k: int,
        exclude_item_ids: Iterable[str] = (),
    ) -> list[SearchHit]:
        """Return deterministic global Top-K cosine hits across all shards."""

        if k < 1:
            raise ValueError("k must be >= 1")
        vector = self._normalize_query(query)
        excluded = {str(value) for value in exclude_item_ids}
        shard_winners: list[SearchHit] = []

        for path in self.shard_paths:
            item_ids, embeddings = self._load_shard(path)
            similarities = embeddings.astype(np.float32, copy=False) @ vector
            eligible = np.fromiter(
                (str(item_id) not in excluded for item_id in item_ids),
                dtype=bool,
                count=len(item_ids),
            )
            indices = np.flatnonzero(eligible)
            if indices.size == 0:
                continue
            # A stable full shard sort is cheap for the current 5k-row shards and
            # gives deterministic item_id tie-breaking.
            order = sorted(
                indices.tolist(),
                key=lambda row: (-float(similarities[row]), str(item_ids[row])),
            )[:k]
            shard_winners.extend(
                SearchHit(str(item_ids[row]), float(similarities[row])) for row in order
            )

        return sorted(
            shard_winners,
            key=lambda hit: (-hit.similarity, hit.item_id),
        )[: min(k, len(shard_winners))]

    def get_embeddings(self, item_ids: Sequence[str]):
        """Fetch rows in caller order while loading each referenced shard once."""

        requested = [str(value) for value in item_ids]
        missing = [item_id for item_id in requested if item_id not in self._locator_by_item]
        if missing:
            raise KeyError(f"Catalog items not found: {missing[:10]}")
        result = np.empty((len(requested), EMBEDDING_DIM), dtype=np.float32)
        rows_by_path: dict[Path, list[tuple[int, int]]] = {}
        for output_row, item_id in enumerate(requested):
            path, shard_row = self._locator_by_item[item_id]
            rows_by_path.setdefault(path, []).append((output_row, shard_row))
        for path, selections in rows_by_path.items():
            _, embeddings = self._load_shard(path)
            for output_row, shard_row in selections:
                result[output_row] = embeddings[shard_row]
        return result

