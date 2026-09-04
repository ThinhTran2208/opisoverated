# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from src.recommendation.catalog import CatalogStatus, NpzShardCatalog, np
from src.recommendation.metadata import ItemMetadataIndex
from src.recommendation.pipeline import RecommendationPipeline
from src.recommendation.reranker import FrozenScorerReranker, RerankedCandidate, torch
from src.recommendation.retrieval import HybridRetriever


def _unit(index):
    vector = [0.0] * 512
    vector[index] = 1.0
    return vector


class _FakeCatalog:
    def __init__(self):
        self.status = CatalogStatus(8, 1, 8, 1, True)
        self.vectors = {
            "same": _unit(10),
            "p-only": _unit(11),
            "c-only": _unit(12),
            "wrong": _unit(10),
            "outfit-a": _unit(10),
            "outfit-b": _unit(11),
            "outfit-c": _unit(12),
        }

    def get_embeddings(self, item_ids):
        rows = [self.vectors.get(item_id, _unit(0)) for item_id in item_ids]
        if np is None:
            return rows
        return np.asarray(rows, dtype=np.float32)

    def __contains__(self, item_id):
        return item_id in self.vectors


class _FakeImageResolver:
    def __contains__(self, item_id):
        return item_id != "missing-image"

    def image_url(self, item_id, *, base_path="/recommendation/images"):
        return f"{base_path}/{item_id}"


class _FakeReranker:
    scorer = object()

    def rerank(self, **kwargs):
        # Deliberately reverse retrieval order to prove scorer owns final order.
        rows = []
        for rank, item_id in enumerate(reversed(kwargs["candidate_item_ids"])):
            rows.append(
                RerankedCandidate(
                    item_id=item_id,
                    compatibility_logit=10.0 - rank,
                    improvement_logit=1.0 - rank,
                    category_id_used=1,
                    used_category_fallback=False,
                )
            )
        return rows


@unittest.skipIf(np is None, "NumPy is not installed in lightweight portability CI")
class NpzShardCatalogTests(unittest.TestCase):
    def test_streaming_search_and_incomplete_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors_a = np.asarray([_unit(0), _unit(1)], dtype=np.float16)
            vectors_b = np.asarray([_unit(0), _unit(2)], dtype=np.float16)
            np.savez(root / "embeddings_shard_0000.npz", item_ids=["b", "c"], embeddings=vectors_a)
            np.savez(root / "embeddings_shard_0001.npz", item_ids=["a", "d"], embeddings=vectors_b)
            (root / "cache_manifest.json").write_text(
                json.dumps(
                    {
                        "expected_dim": 512,
                        "normalized": True,
                        "total_items": 6,
                        "total_shards": 3,
                        "completed_shards": [
                            "embeddings_shard_0000.npz",
                            "embeddings_shard_0001.npz",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            catalog = NpzShardCatalog(root)
            self.assertEqual(catalog.status.loaded_items, 4)
            self.assertFalse(catalog.status.is_complete)
            hits = catalog.search(_unit(0), k=2)
            self.assertEqual([hit.item_id for hit in hits], ["a", "b"])
            selected = catalog.get_embeddings(["d", "c"])
            self.assertEqual(selected.shape, (2, 512))
            self.assertEqual(float(selected[0, 2]), 1.0)


@unittest.skipIf(np is None, "NumPy is not installed in lightweight portability CI")
class HybridRetrievalTests(unittest.TestCase):
    def test_category_filter_happens_before_both_cosine_searches(self):
        catalog = _FakeCatalog()
        metadata = ItemMetadataIndex(
            [
                {"item_id": "outfit-a", "master_category": "T-Shirts", "coarse_category": "TOP"},
                {"item_id": "outfit-b", "master_category": "Jeans", "coarse_category": "BOTTOM"},
                {"item_id": "outfit-c", "master_category": "Shoes", "coarse_category": "SHOES"},
                {"item_id": "same", "master_category": "T-Shirts", "coarse_category": "TOP"},
                {"item_id": "wrong", "master_category": "Sweaters", "coarse_category": "TOP"},
                {"item_id": "p-only", "master_category": "T-Shirts", "coarse_category": "TOP"},
                {"item_id": "c-only", "master_category": "T-Shirts", "coarse_category": "TOP"},
            ]
        )
        result = HybridRetriever(
            catalog,
            metadata=metadata,
            image_resolver=_FakeImageResolver(),
            top_k_problematic=2,
            top_k_context=2,
        ).retrieve(
            outfit_item_ids=["outfit-a", "outfit-b", "outfit-c"],
            outfit_embeddings=np.asarray([_unit(10), _unit(11), _unit(12)]),
            problematic_index=0,
            problematic_category_id=1,
        )
        self.assertEqual(result.category_pool_count, 3)
        self.assertEqual(result.master_category_filtered_count, 0)
        self.assertNotIn("wrong", [row.item_id for row in result.problematic_hits])
        self.assertNotIn("wrong", [row.item_id for row in result.context_hits])
        self.assertTrue(
            all(candidate.master_category == "T-Shirts" for candidate in result.candidates)
        )
        self.assertEqual([row.item_id for row in result.problematic_hits][0], "same")
        self.assertEqual(len(result.problematic_hits), 2)
        self.assertEqual(len(result.context_hits), 2)

    def test_public_boundary_contains_no_scores(self):
        catalog = _FakeCatalog()
        metadata = ItemMetadataIndex(
            [
                {"item_id": "outfit-a", "master_category": "T-Shirts", "coarse_category": "TOP"},
                {"item_id": "outfit-b", "master_category": "Jeans", "coarse_category": "BOTTOM"},
                {"item_id": "outfit-c", "master_category": "Shoes", "coarse_category": "SHOES"},
                {"item_id": "same", "master_category": "T-Shirts", "coarse_category": "TOP"},
                {"item_id": "wrong", "master_category": "Sweaters", "coarse_category": "TOP"},
                {"item_id": "p-only", "master_category": "T-Shirts", "coarse_category": "TOP"},
                {"item_id": "c-only", "master_category": "T-Shirts", "coarse_category": "TOP"},
            ]
        )
        result = RecommendationPipeline(
            catalog=catalog,
            reranker=_FakeReranker(),
            metadata=metadata,
            image_resolver=_FakeImageResolver(),
            top_k_problematic=3,
            top_k_context=3,
            final_k=3,
        ).recommend(
            outfit_item_ids=["outfit-a", "outfit-b", "outfit-c"],
            outfit_embeddings=np.asarray([_unit(10), _unit(11), _unit(12)]),
            outfit_category_ids=[1, 2, 5],
            problematic_index=0,
        )
        public = result.to_public_dict()
        self.assertEqual(public["recommendation_version"], "category-aware-hybrid-v2")
        self.assertEqual(len(public["items"]), 3)
        self.assertTrue(
            all(
                set(item)
                == {
                    "item_id",
                    "rank",
                    "image_url",
                    "master_category",
                    "coarse_category",
                }
                for item in public["items"]
            )
        )
        self.assertEqual([item["rank"] for item in public["items"]], [1, 2, 3])
        self.assertNotIn("score", json.dumps(public).lower())
        self.assertNotIn("logit", json.dumps(public).lower())
        self.assertNotIn("uplift", json.dumps(public).lower())
        self.assertIn("compatibility_logit", result.internal_metadata["reranked_candidates"][0])
        self.assertEqual(
            result.internal_metadata["retrieval"]["retrieval_scope"],
            "exact_master_category_before_cosine",
        )


class MetadataTests(unittest.TestCase):
    def test_metadata_rejects_category_mismatch(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ItemMetadataIndex(
                [{"item_id": "x", "coarse_category": "TOP", "coarse_category_id": 2}]
            )

    def test_image_reference_uses_supported_metadata_keys(self):
        metadata = ItemMetadataIndex(
            [{"item_id": "x", "coarse_category": "TOP", "image_url": "https://img/x.jpg"}]
        )
        self.assertEqual(metadata.image_reference("x"), "https://img/x.jpg")
        self.assertIsNone(metadata.image_reference("missing"))


@unittest.skipIf(torch is None, "PyTorch is not installed in lightweight portability CI")
class FrozenScorerRerankerTests(unittest.TestCase):
    def test_batching_sort_and_unknown_category_fallback(self):
        class FakeScorer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, item_embeddings, coarse_category_ids, item_mask):
                logits = item_embeddings[:, :, 0].sum(dim=1)
                logits = logits + coarse_category_ids.float().sum(dim=1) * 0.01
                return {"compatibility_logit": logits + self.anchor}

        scorer = FakeScorer().train()
        reranker = FrozenScorerReranker(scorer, batch_size=2)
        outfit = torch.stack(
            [
                torch.nn.functional.one_hot(torch.tensor(index), 512).float()
                for index in (10, 11, 12)
            ]
        )
        candidates = torch.zeros((3, 512), dtype=torch.float32)
        candidates[:, 0] = torch.tensor([0.2, 0.9, 0.5])
        candidates[:, 1] = torch.sqrt(1.0 - candidates[:, 0].square())
        result = reranker.rerank(
            outfit_embeddings=outfit,
            outfit_category_ids=[1, 2, 5],
            problematic_index=0,
            candidate_item_ids=["low", "high", "middle"],
            candidate_embeddings=candidates,
            candidate_category_ids=[None, 1, 1],
        )
        self.assertFalse(scorer.training)
        self.assertFalse(next(scorer.parameters()).requires_grad)
        self.assertEqual([row.item_id for row in result], ["high", "middle", "low"])
        low = next(row for row in result if row.item_id == "low")
        self.assertTrue(low.used_category_fallback)
        self.assertEqual(low.category_id_used, 1)
