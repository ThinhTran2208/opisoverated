# -*- coding: utf-8 -*-

import unittest
import tempfile
import json
from pathlib import Path

from src.recommendation.evaluation import Evaluation3Evaluator
from src.recommendation.catalog import SearchHit
from src.recommendation.trace import CandidateTraceWriter
from src.recommendation.metadata import ItemMetadataIndex
from src.recommendation.reranker import RerankedCandidate
from src.recommendation.retrieval import HybridCandidate, RetrievalResult


class _EverythingCatalog:
    def __contains__(self, item_id):
        return True

    def get_embeddings(self, item_ids):
        return [[0.0] * 512 for _ in item_ids]


class _EverythingImageResolver:
    def __contains__(self, item_id):
        return True

    def read_bytes(self, item_id):
        return b"image"


class _EvaluationPipeline:
    def __init__(self):
        self.catalog = _EverythingCatalog()
        self.image_resolver = _EverythingImageResolver()
        self.metadata = ItemMetadataIndex(
            [
                {"item_id": item_id, "master_category": "Tops", "coarse_category": "TOP"}
                for item_id in ("query-a", "query-b", "gt-a", "gt-b", "other")
            ]
        )

    def rank_candidates(self, *, outfit_item_ids, **kwargs):
        if outfit_item_ids[0] == "query-a":
            candidates = (
                HybridCandidate("gt-a", 0.8, None, 40, None, 1, "Tops", "TOP"),
                HybridCandidate("other", 0.7, None, 50, None, 1, "Tops", "TOP"),
            )
            reranked = [
                RerankedCandidate("other", 2.0, 1.0, 1, False),
                RerankedCandidate("gt-a", 1.0, 0.0, 1, False),
            ]
        else:
            candidates = (
                HybridCandidate("other", 0.7, None, 10, None, 1, "Tops", "TOP"),
            )
            reranked = [RerankedCandidate("other", 1.0, 0.0, 1, False)]
        retrieval = RetrievalResult(
            candidates=candidates,
            problematic_hits=tuple(SearchHit(row.item_id, 1.0) for row in candidates),
            context_hits=tuple(reversed([SearchHit(row.item_id, 1.0) for row in candidates])),
            problematic_hit_count=len(candidates),
            context_hit_count=len(candidates),
            union_count=len(candidates),
            master_category_filtered_count=0,
            missing_metadata_count=0,
            missing_image_count=0,
            missing_embedding_count=0,
        )
        return retrieval, reranked


class Evaluation3Tests(unittest.TestCase):
    def test_metrics_use_original_item_as_single_ground_truth(self):
        records = [
            {
                "sample_id": "a",
                "label": 0,
                "items": ["query-a"],
                "negative_metadata": {
                    "swapped_item_index": 0,
                    "original_item_id": "gt-a",
                    "replacement_item_id": "query-a",
                },
            },
            {
                "sample_id": "b",
                "label": 0,
                "items": ["query-b"],
                "negative_metadata": {
                    "swapped_item_index": 0,
                    "original_item_id": "gt-b",
                    "replacement_item_id": "query-b",
                },
            },
        ]
        report = Evaluation3Evaluator(_EvaluationPipeline()).evaluate(records)
        self.assertEqual(report["valid_queries"], 2)
        for stage in ("item_only", "context_only", "hybrid"):
            self.assertEqual(report["retrieval"][stage]["recall_at_50"], 0.5)
            self.assertEqual(report["retrieval"][stage]["recall_at_100"], 0.5)
            self.assertEqual(report["retrieval"][stage]["recall_at_200"], 0.5)
        self.assertEqual(report["reranking"]["hit_at_1"], 0.0)
        self.assertEqual(report["reranking"]["hit_at_3"], 0.5)
        self.assertEqual(report["reranking"]["mrr"], 0.25)

    def test_trace_order_schema_and_excluded_query(self):
        records = [{
            "sample_id": "a", "label": 0, "items": ["query-a"],
            "negative_metadata": {"swapped_item_index": 0,
                                  "original_item_id": "gt-a",
                                  "replacement_item_id": "query-a"},
        }, {
            "sample_id": "bad", "label": 0, "items": ["query-b"],
            "negative_metadata": {"swapped_item_index": 4,
                                  "original_item_id": "gt-b",
                                  "replacement_item_id": "query-b"},
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            report = Evaluation3Evaluator(_EvaluationPipeline()).evaluate(
                records, trace_writer=CandidateTraceWriter(path))
            traces = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(traces), 2)
        self.assertEqual(traces[0]["item_retrieval_top200"], ["gt-a", "other"])
        self.assertEqual(traces[0]["context_retrieval_top200"], ["other", "gt-a"])
        self.assertEqual(traces[0]["hybrid_candidates_top200"], ["gt-a", "other"])
        self.assertEqual(traces[0]["final_top3"], ["other", "gt-a"])
        self.assertTrue({"candidate_counts", "excluded_counts"} <= set(traces[0]))
        self.assertEqual(traces[1]["failure_reason"], "invalid_negative_metadata")
        self.assertEqual(report["excluded_queries"], 1)
        self.assertEqual(report["excluded"]["invalid_negative_metadata"], 1)
