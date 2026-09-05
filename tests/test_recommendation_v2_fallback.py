# -*- coding: utf-8 -*-

import unittest

from src.recommendation.catalog import CatalogStatus, np
from src.recommendation.metadata import ItemMetadataIndex
from src.recommendation.retrieval import HybridRetriever


def _unit(index):
    vector = [0.0] * 512
    vector[index] = 1.0
    return vector


class _Catalog:
    def __init__(self):
        self.status = CatalogStatus(4, 1, 4, 1, True)
        self.rows = {
            "top-a": _unit(0),
            "top-b": _unit(1),
            "bottom-a": _unit(2),
            "bottom-b": _unit(3),
        }

    def __contains__(self, item_id):
        return str(item_id) in self.rows

    def get_embeddings(self, item_ids):
        values = [self.rows[str(item_id)] for item_id in item_ids]
        if np is None:
            return values
        return np.asarray(values, dtype=np.float32)


class _Images:
    def __contains__(self, item_id):
        return True


@unittest.skipIf(np is None, "NumPy is required")
class RecommendationV2FallbackTests(unittest.TestCase):
    def setUp(self):
        self.catalog = _Catalog()
        self.metadata = ItemMetadataIndex(
            [
                {
                    "item_id": "problem",
                    "coarse_category": "TOP",
                },
                {
                    "item_id": "top-a",
                    "master_category": "T-Shirts",
                    "coarse_category": "TOP",
                },
                {
                    "item_id": "top-b",
                    "master_category": "Sweaters",
                    "coarse_category": "TOP",
                },
                {
                    "item_id": "bottom-a",
                    "master_category": "Jeans",
                    "coarse_category": "BOTTOM",
                },
                {
                    "item_id": "bottom-b",
                    "master_category": "Trousers",
                    "coarse_category": "BOTTOM",
                },
            ]
        )

    def test_missing_master_category_falls_back_to_core7_pool(self):
        retriever = HybridRetriever(
            self.catalog,
            metadata=self.metadata,
            image_resolver=_Images(),
            top_k_problematic=10,
            top_k_context=10,
        )
        result = retriever.retrieve(
            outfit_item_ids=["problem", "bottom-a"],
            outfit_embeddings=np.asarray([_unit(0), _unit(2)], dtype=np.float32),
            problematic_index=0,
            problematic_category_id=1,
            problematic_master_category=None,
        )

        self.assertTrue(result.used_core7_fallback)
        self.assertEqual(result.retrieval_scope, "core7_fallback")
        self.assertIsNone(result.target_master_category)
        self.assertEqual(result.target_category_id, 1)
        self.assertEqual(
            {candidate.item_id for candidate in result.candidates},
            {"top-a", "top-b"},
        )
        self.assertTrue(
            all(candidate.coarse_category == "TOP" for candidate in result.candidates)
        )

    def test_exact_master_category_still_has_priority(self):
        retriever = HybridRetriever(
            self.catalog,
            metadata=self.metadata,
            image_resolver=_Images(),
            top_k_problematic=10,
            top_k_context=10,
        )
        result = retriever.retrieve(
            outfit_item_ids=["problem", "bottom-a"],
            outfit_embeddings=np.asarray([_unit(0), _unit(2)], dtype=np.float32),
            problematic_index=0,
            problematic_category_id=1,
            problematic_master_category="T-Shirts",
        )

        self.assertFalse(result.used_core7_fallback)
        self.assertEqual(result.retrieval_scope, "exact_master_category")
        self.assertEqual(result.target_master_category, "T-Shirts")
        self.assertEqual(
            [candidate.item_id for candidate in result.candidates],
            ["top-a"],
        )


if __name__ == "__main__":
    unittest.main()
