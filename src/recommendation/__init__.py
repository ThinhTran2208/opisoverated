"""Recommendation V1: standalone hybrid retrieval + frozen scorer reranking."""

from .catalog import CatalogStatus, NpzShardCatalog, SearchHit
from .evaluation import EVALUATION_PROTOCOL, Evaluation3Evaluator
from .metadata import (
    ItemMetadataIndex,
    load_core7_master_mapping,
    metadata_from_compatibility_jsonl,
)
from .pipeline import (
    RECOMMENDATION_VERSION,
    RecommendationItem,
    RecommendationPipeline,
    RecommendationResult,
)
from .reranker import FROZEN_V5_SHA256, FrozenScorerReranker
from .retrieval import HybridCandidate, HybridRetriever, RetrievalResult
from .trace import CandidateTraceWriter, candidate_trace_record
from .zip_artifacts import MLFinalZipBundle, TensorEmbeddingCatalog
from .zip_images import ZipImageRef, ZipImageResolver

__all__ = [
    "CatalogStatus",
    "EVALUATION_PROTOCOL",
    "Evaluation3Evaluator",
    "FROZEN_V5_SHA256",
    "FrozenScorerReranker",
    "HybridCandidate",
    "HybridRetriever",
    "ItemMetadataIndex",
    "NpzShardCatalog",
    "MLFinalZipBundle",
    "RECOMMENDATION_VERSION",
    "RecommendationItem",
    "RecommendationPipeline",
    "RecommendationResult",
    "RetrievalResult",
    "CandidateTraceWriter",
    "candidate_trace_record",
    "SearchHit",
    "TensorEmbeddingCatalog",
    "ZipImageRef",
    "ZipImageResolver",
    "load_core7_master_mapping",
    "metadata_from_compatibility_jsonl",
]
