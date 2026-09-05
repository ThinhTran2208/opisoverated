# -*- coding: utf-8 -*-
"""Load Recommendation V2 artifacts directly from an ML_Final directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # Keep imports lightweight in portability CI.
    torch = None

from .metadata import ItemMetadataIndex
from .reranker import FROZEN_V5_SHA256, FrozenScorerReranker
from .zip_artifacts import TensorEmbeddingCatalog


EMBEDDING_REL = Path("fashionclip_item_embeddings.pt")
EMBEDDING_MANIFEST_REL = Path("embedding_manifest_v1.json")
FROZEN_V5_REL = Path(
    "scorer_runs/type_aware_pairwise_v1/final_val_auc_v5_seed42/best.pt"
)
METADATA_REL_TEMPLATE = Path(
    "polyvore_core7_v2/core7_drop_v2/core7_item_metadata_v1_{split}.jsonl"
)
SCORER_READY_REL_TEMPLATE = Path(
    "polyvore_core7_v2/scorer_ready_v2/scorer_ready_v2_{split}.jsonl"
)
SPLITS = ("train", "valid", "test")


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to load ML_Final directory artifacts")


class MLFinalDirectoryBundle:
    """Portable loader for the canonical ML_Final directory layout."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"ML_Final directory not found: {self.root}")
        required = [
            EMBEDDING_REL,
            EMBEDDING_MANIFEST_REL,
            FROZEN_V5_REL,
            *(Path(str(METADATA_REL_TEMPLATE).format(split=split)) for split in SPLITS),
            *(Path(str(SCORER_READY_REL_TEMPLATE).format(split=split)) for split in SPLITS),
        ]
        missing = [str(self.root / rel) for rel in required if not (self.root / rel).is_file()]
        if missing:
            raise FileNotFoundError(f"ML_Final directory is missing files: {missing[:10]}")
        self._embedding_catalog: TensorEmbeddingCatalog | None = None
        self._metadata_index: ItemMetadataIndex | None = None

    def _path(self, rel: Path) -> Path:
        return self.root / rel

    @property
    def embedding_manifest(self) -> dict[str, object]:
        payload = json.loads(self._path(EMBEDDING_MANIFEST_REL).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Embedding manifest must be a JSON object")
        return payload

    def _read_jsonl(self, rel: Path) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        with self._path(rel).open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, start=1):
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {rel}:{line_number}")
                rows.append(row)
        return rows

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
        embedding_path = self._path(EMBEDDING_REL)
        actual_sha = hashlib.sha256(embedding_path.read_bytes()).hexdigest()
        expected_sha = str(manifest.get("cache_sha256", ""))
        if actual_sha != expected_sha:
            raise ValueError(
                f"Embedding cache SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        cache = torch.load(embedding_path, map_location="cpu", weights_only=True)
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

    def load_metadata_index(self, splits: Sequence[str] = SPLITS) -> ItemMetadataIndex:
        if tuple(splits) == SPLITS and self._metadata_index is not None:
            return self._metadata_index
        records: list[dict[str, object]] = []
        for split in splits:
            if split not in SPLITS:
                raise ValueError(f"Unknown split: {split}")
            rel = Path(str(METADATA_REL_TEMPLATE).format(split=split))
            records.extend(self._read_jsonl(rel))
        index = ItemMetadataIndex(records)
        if tuple(splits) == SPLITS:
            self._metadata_index = index
        return index

    def load_scorer_ready(self, split: str) -> list[dict[str, object]]:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        rel = Path(str(SCORER_READY_REL_TEMPLATE).format(split=split))
        return self._read_jsonl(rel)

    def load_frozen_v5_reranker(
        self,
        *,
        device: str | object = "cpu",
        batch_size: int = 256,
    ) -> FrozenScorerReranker:
        return FrozenScorerReranker.load_v5_bytes(
            self._path(FROZEN_V5_REL).read_bytes(),
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
        return {
            "embedding_count": len(embedding_ids),
            "image_count": len(image_ids),
            "mapping_exact": not missing_images and not missing_embeddings,
            "missing_images_count": len(missing_images),
            "missing_embeddings_count": len(missing_embeddings),
            "missing_images_preview": missing_images[:10],
            "missing_embeddings_preview": missing_embeddings[:10],
        }
