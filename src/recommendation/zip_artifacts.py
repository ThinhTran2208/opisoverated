# -*- coding: utf-8 -*-
"""Load Recommendation V1 artifacts directly from the ML_Final ZIP."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # Keep imports lightweight in portability CI.
    torch = None

from .catalog import CatalogStatus, SearchHit
from .metadata import ItemMetadataIndex
from .reranker import FROZEN_V5_SHA256, FrozenScorerReranker


ML_ROOT = "ML_Final"
EMBEDDING_ENTRY = f"{ML_ROOT}/fashionclip_item_embeddings.pt"
EMBEDDING_MANIFEST_ENTRY = f"{ML_ROOT}/embedding_manifest_v1.json"
FROZEN_V5_ENTRY = (
    f"{ML_ROOT}/scorer_runs/type_aware_pairwise_v1/"
    "final_val_auc_v5_seed42/best.pt"
)
METADATA_ENTRY_TEMPLATE = (
    f"{ML_ROOT}/polyvore_core7_v2/core7_drop_v2/"
    "core7_item_metadata_v1_{split}.jsonl"
)
SCORER_READY_ENTRY_TEMPLATE = (
    f"{ML_ROOT}/polyvore_core7_v2/scorer_ready_v2/"
    "scorer_ready_v2_{split}.jsonl"
)
SPLITS = ("train", "valid", "test")


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to load ML_Final ZIP artifacts")


class TensorEmbeddingCatalog:
    """In-memory exact cosine catalog backed by the full ZIP tensor."""

    def __init__(
        self,
        item_ids: Sequence[str],
        embeddings,
        *,
        expected_count: int | None = None,
        norm_tolerance: float = 0.02,
    ) -> None:
        require_torch()
        normalized_ids = [str(value) for value in item_ids]
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("Embedding catalog contains duplicate item_ids")
        matrix = torch.as_tensor(embeddings, dtype=torch.float32, device="cpu")
        if matrix.ndim != 2 or tuple(matrix.shape[1:]) != (512,):
            raise ValueError("Embedding catalog must have shape [N, 512]")
        if matrix.shape[0] != len(normalized_ids):
            raise ValueError("Embedding rows do not match item_ids")
        if expected_count is not None and len(normalized_ids) != int(expected_count):
            raise ValueError(
                f"Expected {expected_count} embeddings, found {len(normalized_ids)}"
            )
        for start in range(0, len(normalized_ids), 8192):
            chunk = matrix[start : start + 8192]
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError("Embedding catalog contains NaN/Inf")
            norms = torch.linalg.vector_norm(chunk, dim=1)
            if bool(torch.any(torch.abs(norms - 1.0) > norm_tolerance)):
                raise ValueError("Embedding catalog contains non-normalized rows")

        self.item_ids = tuple(normalized_ids)
        self.embedding_matrix = matrix.contiguous()
        self.embedding_row_by_item = {
            item_id: row for row, item_id in enumerate(self.item_ids)
        }
        self.status = CatalogStatus(
            loaded_items=len(self.item_ids),
            loaded_shards=1,
            expected_items=expected_count,
            expected_shards=1,
            is_complete=(
                None if expected_count is None else len(self.item_ids) == expected_count
            ),
        )

    def __len__(self) -> int:
        return len(self.item_ids)

    def __contains__(self, item_id: object) -> bool:
        return str(item_id) in self.embedding_row_by_item

    @staticmethod
    def _query_tensor(query):
        vector = torch.as_tensor(query, dtype=torch.float32, device="cpu")
        if vector.shape != (512,):
            raise ValueError("query must have shape [512]")
        if not bool(torch.isfinite(vector).all()):
            raise ValueError("query contains NaN/Inf")
        norm = torch.linalg.vector_norm(vector)
        if float(norm) <= 1e-12:
            raise ValueError("query must have non-zero norm")
        return vector / norm

    def search(self, query, *, k: int, exclude_item_ids=()) -> list[SearchHit]:
        if k < 1:
            raise ValueError("k must be >= 1")
        vector = self._query_tensor(query)
        scores = torch.mv(self.embedding_matrix, vector)
        excluded_rows = [
            self.embedding_row_by_item[item_id]
            for item_id in {str(value) for value in exclude_item_ids}
            if item_id in self.embedding_row_by_item
        ]
        if excluded_rows:
            scores[torch.tensor(excluded_rows, dtype=torch.long)] = -torch.inf
        available = len(self.item_ids) - len(excluded_rows)
        count = min(int(k), available)
        if count <= 0:
            return []
        values, indices = torch.topk(scores, count, largest=True, sorted=True)
        hits = [
            SearchHit(self.item_ids[int(row)], float(score))
            for score, row in zip(values.tolist(), indices.tolist())
        ]
        return sorted(hits, key=lambda hit: (-hit.similarity, hit.item_id))

    def get_embeddings(self, item_ids: Sequence[str]):
        requested = [str(value) for value in item_ids]
        missing = [
            item_id for item_id in requested if item_id not in self.embedding_row_by_item
        ]
        if missing:
            raise KeyError(f"Embedding items not found: {missing[:10]}")
        rows = torch.tensor(
            [self.embedding_row_by_item[item_id] for item_id in requested],
            dtype=torch.long,
        )
        return self.embedding_matrix.index_select(0, rows)


class MLFinalZipBundle:
    """Read only the requested entries from one immutable artifact ZIP."""

    def __init__(self, archive_path: Path | str) -> None:
        self.archive_path = Path(archive_path).expanduser().resolve()
        if not self.archive_path.is_file():
            raise FileNotFoundError(self.archive_path)
        with zipfile.ZipFile(self.archive_path, "r") as archive:
            names = set(archive.namelist())
        required = {
            EMBEDDING_ENTRY,
            EMBEDDING_MANIFEST_ENTRY,
            FROZEN_V5_ENTRY,
            *(METADATA_ENTRY_TEMPLATE.format(split=split) for split in SPLITS),
            *(SCORER_READY_ENTRY_TEMPLATE.format(split=split) for split in SPLITS),
        }
        missing = sorted(required - names)
        if missing:
            raise FileNotFoundError(f"ML_Final ZIP is missing entries: {missing}")
        self._embedding_catalog: TensorEmbeddingCatalog | None = None
        self._metadata_index: ItemMetadataIndex | None = None

    def read_bytes(self, entry_name: str) -> bytes:
        with zipfile.ZipFile(self.archive_path, "r") as archive:
            return archive.read(entry_name)

    def read_json(self, entry_name: str) -> dict[str, object]:
        payload = json.loads(self.read_bytes(entry_name).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {entry_name}")
        return payload

    def read_jsonl(self, entry_name: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        with zipfile.ZipFile(self.archive_path, "r") as archive:
            with archive.open(entry_name, "r") as stream:
                for line_number, raw in enumerate(stream, start=1):
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"Expected JSON object in {entry_name}:{line_number}"
                        )
                    rows.append(row)
        return rows

    @property
    def embedding_manifest(self) -> dict[str, object]:
        return self.read_json(EMBEDDING_MANIFEST_ENTRY)

    def load_embedding_catalog(self) -> TensorEmbeddingCatalog:
        if self._embedding_catalog is not None:
            return self._embedding_catalog
        require_torch()
        manifest = self.embedding_manifest
        if manifest.get("embedding_version") != "fashionclip-512-l2-v1":
            raise ValueError("Unsupported embedding version")
        if int(manifest.get("embedding_dimension", 0)) != 512:
            raise ValueError("Embedding manifest dimension must be 512")
        if manifest.get("normalization") != "l2":
            raise ValueError("Embedding manifest must declare L2 normalization")
        raw = self.read_bytes(EMBEDDING_ENTRY)
        actual_sha = hashlib.sha256(raw).hexdigest()
        expected_sha = str(manifest.get("cache_sha256", ""))
        if actual_sha != expected_sha:
            raise ValueError(
                f"Embedding cache SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        cache = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if not isinstance(cache, Mapping):
            raise ValueError("Embedding cache must contain a mapping")
        if cache.get("normalized") is not True:
            raise ValueError("Embedding cache must be marked normalized")
        self._embedding_catalog = TensorEmbeddingCatalog(
            cache["item_ids"],
            cache["embeddings"],
            expected_count=int(manifest["item_count"]),
        )
        return self._embedding_catalog

    def load_metadata_index(
        self,
        splits: Sequence[str] = SPLITS,
    ) -> ItemMetadataIndex:
        if tuple(splits) == SPLITS and self._metadata_index is not None:
            return self._metadata_index
        records = []
        for split in splits:
            if split not in SPLITS:
                raise ValueError(f"Unknown split: {split}")
            records.extend(
                self.read_jsonl(METADATA_ENTRY_TEMPLATE.format(split=split))
            )
        index = ItemMetadataIndex(records)
        if tuple(splits) == SPLITS:
            self._metadata_index = index
        return index

    def load_scorer_ready(self, split: str) -> list[dict[str, object]]:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        return self.read_jsonl(SCORER_READY_ENTRY_TEMPLATE.format(split=split))

    def load_frozen_v5_reranker(
        self,
        *,
        device: str | object = "cpu",
        batch_size: int = 256,
    ) -> FrozenScorerReranker:
        return FrozenScorerReranker.load_v5_bytes(
            self.read_bytes(FROZEN_V5_ENTRY),
            device=device,
            batch_size=batch_size,
            expected_sha256=FROZEN_V5_SHA256,
        )

    def validate_image_catalog(self, image_resolver) -> dict[str, object]:
        catalog = self.load_embedding_catalog()
        embedding_ids = set(catalog.item_ids)
        image_ids = set(image_resolver.item_ids)
        missing_images = sorted(embedding_ids - image_ids)
        missing_embeddings = sorted(image_ids - embedding_ids)
        if missing_images or missing_embeddings:
            raise ValueError(
                "Embedding/image item_id mismatch: "
                f"missing_images={missing_images[:10]}, "
                f"missing_embeddings={missing_embeddings[:10]}"
            )
        return {
            "embedding_count": len(embedding_ids),
            "image_count": len(image_ids),
            "mapping_exact": True,
        }

