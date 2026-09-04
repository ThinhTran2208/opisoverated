# -*- coding: utf-8 -*-
"""Standalone Recommendation V2 orchestration.

V2 changes candidate retrieval to category-aware search before cosine Top-K.
It supports both legacy ZIP-direct artifacts and portable directory artifacts.
Frozen scorer reranking and LOO integration remain unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import torch
except ModuleNotFoundError:  # Keep package importable in lightweight CI.
    torch = None

from src.diagnosis.loo import diagnose_outfit

from .metadata import ItemMetadataIndex
from .retrieval import HybridRetriever, RetrievalResult
from .reranker import FrozenScorerReranker, RerankedCandidate
from .trace import CandidateTraceWriter, candidate_trace_record


RECOMMENDATION_VERSION = "category-aware-hybrid-v2"


@dataclass(frozen=True)
class RecommendationItem:
    item_id: str
    rank: int
    image_url: str
    master_category: str
    coarse_category: str

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "rank": self.rank,
            "image_url": self.image_url,
            "master_category": self.master_category,
            "coarse_category": self.coarse_category,
        }


@dataclass(frozen=True)
class RecommendationResult:
    items: tuple[RecommendationItem, ...]
    internal_metadata: dict[str, object]

    def to_public_dict(self) -> dict[str, object]:
        return {
            "status": "ok",
            "recommendation_version": RECOMMENDATION_VERSION,
            "items": [item.to_dict() for item in self.items],
        }


class RecommendationPipeline:
    def __init__(
        self,
        *,
        catalog,
        reranker: FrozenScorerReranker,
        metadata: ItemMetadataIndex,
        image_resolver,
        image_url_base: str = "/recommendation/images",
        top_k_problematic: int = 200,
        top_k_context: int = 200,
        final_k: int = 3,
        trace_writer: CandidateTraceWriter | None = None,
    ) -> None:
        if final_k < 1:
            raise ValueError("final_k must be >= 1")
        self.catalog = catalog
        self.reranker = reranker
        self.metadata = metadata
        self.image_resolver = image_resolver
        self.image_url_base = str(image_url_base).rstrip("/")
        self.retriever = HybridRetriever(
            catalog,
            metadata=self.metadata,
            image_resolver=image_resolver,
            top_k_problematic=top_k_problematic,
            top_k_context=top_k_context,
        )
        self.final_k = int(final_k)
        self.trace_writer = trace_writer

    @staticmethod
    def _load_config(config_path: Path | str) -> dict[str, object]:
        source = Path(config_path).resolve()
        config = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Recommendation config must be a JSON object")
        if config.get("recommendation_version") != RECOMMENDATION_VERSION:
            raise ValueError("Unsupported recommendation_version")
        if config.get("retrieval_scope") != "exact_master_category_before_cosine":
            raise ValueError("Recommendation V2 requires category-aware retrieval")
        if config.get("runtime_fallback") != "core7_when_master_category_unavailable":
            raise ValueError("Recommendation V2 runtime fallback contract mismatch")
        if config.get("scorer_version") != "type_aware_pairwise_v1":
            raise ValueError("Recommendation V2 requires the frozen V5 scorer contract")
        if config.get("embedding_version") != "fashionclip-512-l2-v1":
            raise ValueError("Recommendation V2 requires FashionCLIP 512-D embeddings")
        if config.get("category_mapping_version") != "core7-v2":
            raise ValueError("Recommendation V2 requires Core-7 V2 categories")
        return config

    @classmethod
    def _build(
        cls,
        *,
        config: dict[str, object],
        bundle,
        catalog,
        metadata,
        image_resolver,
        device: str | object,
    ) -> "RecommendationPipeline":
        expected_embeddings = int(config.get("expected_embedding_count", 142_480))
        if len(catalog) != expected_embeddings:
            raise ValueError(
                f"Expected {expected_embeddings} catalog embeddings, found {len(catalog)}"
            )
        image_validation = bundle.validate_image_catalog(image_resolver)
        reranker = bundle.load_frozen_v5_reranker(
            device=device,
            batch_size=int(config.get("rerank_batch_size", 256)),
        )
        pipeline = cls(
            catalog=catalog,
            reranker=reranker,
            metadata=metadata,
            image_resolver=image_resolver,
            image_url_base=str(config.get("image_url_base", "/recommendation/images")),
            top_k_problematic=int(config.get("top_k_problematic", 200)),
            top_k_context=int(config.get("top_k_context", 200)),
            final_k=int(config.get("final_k", 3)),
            trace_writer=CandidateTraceWriter(),
        )
        pipeline.artifact_bundle = bundle
        pipeline.image_validation = image_validation
        return pipeline

    @classmethod
    def load_from_archives(
        cls,
        config_path: Path | str,
        *,
        ml_zip_path: Path | str,
        image_zip_paths: Sequence[Path | str],
        device: str | object = "cpu",
    ) -> "RecommendationPipeline":
        from .reranker import FROZEN_V5_SHA256
        from .zip_artifacts import FROZEN_V5_ENTRY, MLFinalZipBundle
        from .zip_images import ZipImageResolver

        config = cls._load_config(config_path)
        if config.get("checkpoint_entry") != FROZEN_V5_ENTRY:
            raise ValueError("Recommendation config does not target frozen V5 in ZIP")
        if config.get("checkpoint_sha256") != FROZEN_V5_SHA256:
            raise ValueError("Recommendation config frozen V5 SHA-256 mismatch")

        bundle = MLFinalZipBundle(ml_zip_path)
        catalog = bundle.load_embedding_catalog()
        metadata = bundle.load_metadata_index()
        image_resolver = ZipImageResolver(
            image_zip_paths,
            expected_count=int(config.get("expected_image_count", 142_480)),
        )
        return cls._build(
            config=config,
            bundle=bundle,
            catalog=catalog,
            metadata=metadata,
            image_resolver=image_resolver,
            device=device,
        )

    @classmethod
    def load_from_directories(
        cls,
        config_path: Path | str,
        *,
        artifact_root: Path | str,
        image_root: Path | str,
        device: str | object = "cpu",
    ) -> "RecommendationPipeline":
        from .directory_artifacts import MLFinalDirectoryBundle
        from .directory_images import DirectoryImageResolver
        from .reranker import FROZEN_V5_SHA256

        config = cls._load_config(config_path)
        if config.get("checkpoint_sha256") != FROZEN_V5_SHA256:
            raise ValueError("Recommendation config frozen V5 SHA-256 mismatch")

        bundle = MLFinalDirectoryBundle(artifact_root)
        catalog = bundle.load_embedding_catalog()
        metadata = bundle.load_metadata_index()
        # Important for very large mounted Drive folders: do not enumerate all
        # image files. Declare the frozen embedding item inventory and resolve
        # concrete image files lazily only when evaluation/runtime reads them.
        image_resolver = DirectoryImageResolver(
            image_root,
            expected_count=int(config.get("expected_image_count", 142_480)),
            known_item_ids=catalog.item_ids,
        )
        return cls._build(
            config=config,
            bundle=bundle,
            catalog=catalog,
            metadata=metadata,
            image_resolver=image_resolver,
            device=device,
        )

    def recommend(
        self,
        *,
        outfit_item_ids: Sequence[str],
        outfit_embeddings,
        outfit_category_ids: Sequence[int],
        problematic_index: int,
        loo_result: dict[str, object] | None = None,
        query_id: str | None = None,
        source_split: str | None = None,
        ground_truth_item_id: str | None = None,
    ) -> RecommendationResult:
        if len(outfit_item_ids) != len(outfit_category_ids):
            raise ValueError("outfit item IDs and category IDs must align")
        if not 0 <= problematic_index < len(outfit_item_ids):
            raise ValueError("problematic_index is outside the outfit")
        retrieval, reranked = self.rank_candidates(
            outfit_item_ids=outfit_item_ids,
            outfit_embeddings=outfit_embeddings,
            outfit_category_ids=outfit_category_ids,
            problematic_index=problematic_index,
        )
        selected = reranked[: self.final_k]
        if len(selected) < self.final_k:
            if self.trace_writer is not None:
                self.trace_writer.append(
                    candidate_trace_record(
                        query_id=query_id or "runtime",
                        source_split=source_split,
                        problematic_item_index=problematic_index,
                        problematic_item_id=str(outfit_item_ids[problematic_index]),
                        ground_truth_item_id=ground_truth_item_id,
                        replacement_item_id=str(outfit_item_ids[problematic_index]),
                        item_ids=[row.item_id for row in retrieval.problematic_hits],
                        context_ids=[row.item_id for row in retrieval.context_hits],
                        hybrid_ids=[row.item_id for row in retrieval.candidates],
                        final_ids=[row.item_id for row in selected],
                        candidate_counts={
                            "category_pool": retrieval.category_pool_count,
                            "item_retrieval": retrieval.problematic_hit_count,
                            "context_retrieval": retrieval.context_hit_count,
                            "hybrid_top200": min(200, len(retrieval.candidates)),
                            "reranked": len(reranked),
                            "final": len(selected),
                        },
                        excluded_counts={
                            "master_category": retrieval.master_category_filtered_count,
                            "missing_metadata": retrieval.missing_metadata_count,
                            "missing_image": retrieval.missing_image_count,
                            "missing_embedding": retrieval.missing_embedding_count,
                        },
                        failure_reason="fewer_than_three_final_candidates",
                    )
                )
            raise RuntimeError(
                f"Only {len(selected)} eligible candidates remain; "
                f"Recommendation V2 requires Top-{self.final_k}"
            )

        items = []
        for rank, candidate in enumerate(selected, start=1):
            master_category = self.metadata.master_category(candidate.item_id)
            coarse_category = self.metadata.coarse_category(candidate.item_id)
            if master_category is None or coarse_category is None:
                raise RuntimeError("Selected candidate is missing category metadata")
            items.append(
                RecommendationItem(
                    item_id=candidate.item_id,
                    rank=rank,
                    image_url=self.image_resolver.image_url(
                        candidate.item_id,
                        base_path=self.image_url_base,
                    ),
                    master_category=master_category,
                    coarse_category=coarse_category,
                )
            )

        if self.trace_writer is not None:
            self.trace_writer.append(
                candidate_trace_record(
                    query_id=query_id or "runtime",
                    source_split=source_split,
                    problematic_item_index=problematic_index,
                    problematic_item_id=str(outfit_item_ids[problematic_index]),
                    ground_truth_item_id=ground_truth_item_id,
                    replacement_item_id=str(outfit_item_ids[problematic_index]),
                    item_ids=[row.item_id for row in retrieval.problematic_hits],
                    context_ids=[row.item_id for row in retrieval.context_hits],
                    hybrid_ids=[row.item_id for row in retrieval.candidates],
                    final_ids=[row.item_id for row in selected],
                    candidate_counts={
                        "category_pool": retrieval.category_pool_count,
                        "item_retrieval": retrieval.problematic_hit_count,
                        "context_retrieval": retrieval.context_hit_count,
                        "hybrid_top200": min(200, len(retrieval.candidates)),
                        "reranked": len(reranked),
                        "final": len(selected),
                    },
                    excluded_counts={
                        "master_category": retrieval.master_category_filtered_count,
                        "missing_metadata": retrieval.missing_metadata_count,
                        "missing_image": retrieval.missing_image_count,
                        "missing_embedding": retrieval.missing_embedding_count,
                    },
                )
            )
        return RecommendationResult(
            items=tuple(items),
            internal_metadata=self._internal_metadata(
                retrieval,
                reranked,
                problematic_index=problematic_index,
                loo_result=loo_result,
            ),
        )

    def rank_candidates(
        self,
        *,
        outfit_item_ids: Sequence[str],
        outfit_embeddings,
        outfit_category_ids: Sequence[int],
        problematic_index: int,
    ) -> tuple[RetrievalResult, list[RerankedCandidate]]:
        if len(outfit_item_ids) != len(outfit_category_ids):
            raise ValueError("outfit item IDs and category IDs must align")
        if not 0 <= problematic_index < len(outfit_item_ids):
            raise ValueError("problematic_index is outside the outfit")
        target_category = int(outfit_category_ids[problematic_index])
        target_master = self.metadata.master_category(str(outfit_item_ids[problematic_index]))
        retrieval = self.retriever.retrieve(
            outfit_item_ids=outfit_item_ids,
            outfit_embeddings=outfit_embeddings,
            problematic_index=problematic_index,
            problematic_category_id=target_category,
            problematic_master_category=target_master,
        )
        candidate_ids = [candidate.item_id for candidate in retrieval.candidates]
        candidate_embeddings = self.catalog.get_embeddings(candidate_ids)
        reranked = self.reranker.rerank(
            outfit_embeddings=outfit_embeddings,
            outfit_category_ids=outfit_category_ids,
            problematic_index=problematic_index,
            candidate_item_ids=candidate_ids,
            candidate_embeddings=candidate_embeddings,
            candidate_category_ids=[candidate.category_id for candidate in retrieval.candidates],
        )
        return retrieval, reranked

    def recommend_with_loo(
        self,
        *,
        outfit_item_ids: Sequence[str],
        outfit_embeddings,
        outfit_category_ids: Sequence[int],
    ) -> RecommendationResult:
        if torch is None:
            raise RuntimeError("PyTorch is required for LOO recommendation")
        diagnosis = diagnose_outfit(
            self.reranker.scorer,
            torch.as_tensor(outfit_embeddings, dtype=torch.float32),
            torch.as_tensor(outfit_category_ids, dtype=torch.long),
            item_ids=outfit_item_ids,
        )
        return self.recommend(
            outfit_item_ids=outfit_item_ids,
            outfit_embeddings=outfit_embeddings,
            outfit_category_ids=outfit_category_ids,
            problematic_index=int(diagnosis["problematic_item_index"]),
            loo_result=diagnosis,
        )

    def _internal_metadata(
        self,
        retrieval: RetrievalResult,
        reranked: Sequence[RerankedCandidate],
        *,
        problematic_index: int,
        loo_result: dict[str, object] | None,
    ) -> dict[str, object]:
        return {
            "problematic_item_index": problematic_index,
            "loo_protocol_version": (
                None if loo_result is None else loo_result.get("protocol_version")
            ),
            "retrieval": {
                "retrieval_scope": retrieval.retrieval_scope,
                "used_core7_fallback": retrieval.used_core7_fallback,
                "target_master_category": retrieval.target_master_category,
                "target_category_id": retrieval.target_category_id,
                "category_pool_count": retrieval.category_pool_count,
                "problematic_hit_count": retrieval.problematic_hit_count,
                "context_hit_count": retrieval.context_hit_count,
                "union_count": retrieval.union_count,
                "master_category_filtered_count": retrieval.master_category_filtered_count,
                "missing_metadata_count": retrieval.missing_metadata_count,
                "missing_image_count": retrieval.missing_image_count,
                "missing_embedding_count": retrieval.missing_embedding_count,
            },
            "catalog": self.catalog.status.to_dict() if hasattr(self.catalog, "status") else None,
            "reranked_candidates": [
                {
                    "item_id": row.item_id,
                    "compatibility_logit": row.compatibility_logit,
                    "improvement_logit": row.improvement_logit,
                    "category_id_used": row.category_id_used,
                    "used_category_fallback": row.used_category_fallback,
                }
                for row in reranked
            ],
            "retrieval_candidates": [
                {
                    "item_id": row.item_id,
                    "problematic_rank": row.problematic_rank,
                    "context_rank": row.context_rank,
                    "master_category": row.master_category,
                    "coarse_category": row.coarse_category,
                }
                for row in retrieval.candidates
            ],
        }
