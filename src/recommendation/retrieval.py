# -*- coding: utf-8 -*-
"""Hybrid candidate retrieval for Recommendation V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import numpy as np
except ModuleNotFoundError:  # Keep package importable in lightweight CI.
    np = None

from .catalog import EMBEDDING_DIM, SearchHit, require_numpy
from .metadata import ItemMetadataIndex


@dataclass(frozen=True)
class HybridCandidate:
    item_id: str
    problematic_similarity: float | None
    context_similarity: float | None
    problematic_rank: int | None
    context_rank: int | None
    category_id: int
    master_category: str
    coarse_category: str


@dataclass(frozen=True)
class RetrievalResult:
    candidates: tuple[HybridCandidate, ...]
    problematic_hits: tuple[SearchHit, ...]
    context_hits: tuple[SearchHit, ...]
    problematic_hit_count: int
    context_hit_count: int
    union_count: int
    master_category_filtered_count: int
    missing_metadata_count: int
    missing_image_count: int
    missing_embedding_count: int


class HybridRetriever:
    """Union two cosine channels, de-duplicate, then apply available metadata."""

    def __init__(
        self,
        catalog,
        *,
        metadata: ItemMetadataIndex | None = None,
        image_resolver=None,
        top_k_problematic: int = 200,
        top_k_context: int = 200,
    ) -> None:
        if top_k_problematic < 1 or top_k_context < 1:
            raise ValueError("Both retrieval Top-K values must be >= 1")
        self.catalog = catalog
        self.metadata = metadata if metadata is not None else ItemMetadataIndex()
        self.image_resolver = image_resolver
        self.top_k_problematic = int(top_k_problematic)
        self.top_k_context = int(top_k_context)

    @staticmethod
    def _context_query(outfit_embeddings, problematic_index: int):
        require_numpy()
        matrix = np.asarray(outfit_embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"outfit_embeddings must have shape [N, {EMBEDDING_DIM}]")
        if not 0 <= problematic_index < matrix.shape[0]:
            raise ValueError("problematic_index is outside the outfit")
        kept = np.delete(matrix, problematic_index, axis=0)
        if kept.shape[0] < 1:
            raise ValueError("Outfit context must contain at least one retained item")
        context = kept.mean(axis=0)
        norm = float(np.linalg.norm(context))
        if norm <= 1e-12:
            raise ValueError("Outfit context centroid has zero norm")
        return context / norm

    @staticmethod
    def _by_item(hits: Sequence[SearchHit]) -> dict[str, tuple[int, float]]:
        return {
            hit.item_id: (rank, float(hit.similarity))
            for rank, hit in enumerate(hits, start=1)
        }

    def retrieve(
        self,
        *,
        outfit_item_ids: Sequence[str],
        outfit_embeddings,
        problematic_index: int,
        problematic_category_id: int,
        problematic_master_category: str | None = None,
    ) -> RetrievalResult:
        require_numpy()
        matrix = np.asarray(outfit_embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(outfit_item_ids):
            raise ValueError("outfit_item_ids must align with outfit_embeddings")
        if not 1 <= int(problematic_category_id) <= 7:
            raise ValueError("problematic_category_id must be a Core-7 ID in [1, 7]")
        target_item_id = str(outfit_item_ids[problematic_index])
        target_master = (
            str(problematic_master_category).strip()
            if problematic_master_category is not None
            else self.metadata.master_category(target_item_id)
        )
        if not target_master:
            raise ValueError(
                f"Missing master_category for problematic item {target_item_id}"
            )
        metadata_category_id = self.metadata.category_id(target_item_id)
        if (
            metadata_category_id is not None
            and metadata_category_id != int(problematic_category_id)
        ):
            raise ValueError("Problematic item category does not match metadata")
        excluded = {str(value) for value in outfit_item_ids}
        problematic_hits = self.catalog.search(
            matrix[problematic_index],
            k=self.top_k_problematic,
            exclude_item_ids=excluded,
        )
        context_hits = self.catalog.search(
            self._context_query(matrix, problematic_index),
            k=self.top_k_context,
            exclude_item_ids=excluded,
        )
        problematic_by_item = self._by_item(problematic_hits)
        context_by_item = self._by_item(context_hits)
        union_ids = set(problematic_by_item) | set(context_by_item)
        master_filtered_count = 0
        missing_metadata_count = 0
        missing_image_count = 0
        missing_embedding_count = 0
        candidates: list[HybridCandidate] = []

        for item_id in union_ids:
            candidate_category = self.metadata.category_id(item_id)
            candidate_master = self.metadata.master_category(item_id)
            candidate_coarse = self.metadata.coarse_category(item_id)
            if (
                candidate_category is None
                or candidate_master is None
                or candidate_coarse is None
            ):
                missing_metadata_count += 1
                continue
            if candidate_master != target_master:
                master_filtered_count += 1
                continue
            if item_id not in self.catalog:
                missing_embedding_count += 1
                continue
            if self.image_resolver is not None and item_id not in self.image_resolver:
                missing_image_count += 1
                continue
            p = problematic_by_item.get(item_id)
            c = context_by_item.get(item_id)
            candidates.append(
                HybridCandidate(
                    item_id=item_id,
                    problematic_similarity=None if p is None else p[1],
                    context_similarity=None if c is None else c[1],
                    problematic_rank=None if p is None else p[0],
                    context_rank=None if c is None else c[0],
                    category_id=candidate_category,
                    master_category=candidate_master,
                    coarse_category=candidate_coarse,
                )
            )

        # Retrieval order is only a deterministic handoff order.  The frozen
        # scorer, not this heuristic, owns final recommendation ordering.
        candidates.sort(
            key=lambda row: (
                min(
                    row.problematic_rank or self.top_k_problematic + 1,
                    row.context_rank or self.top_k_context + 1,
                ),
                row.item_id,
            )
        )
        return RetrievalResult(
            candidates=tuple(candidates),
            problematic_hits=tuple(problematic_hits),
            context_hits=tuple(context_hits),
            problematic_hit_count=len(problematic_hits),
            context_hit_count=len(context_hits),
            union_count=len(union_ids),
            master_category_filtered_count=master_filtered_count,
            missing_metadata_count=missing_metadata_count,
            missing_image_count=missing_image_count,
            missing_embedding_count=missing_embedding_count,
        )
