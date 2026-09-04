# -*- coding: utf-8 -*-
"""Host ZIP integration test; skipped in portable CI without artifacts."""

import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.recommendation.demo import _fixture_result
from src.recommendation.pipeline import RecommendationPipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
ML_ZIP = Path(r"D:\BKU\VSC\ML_Final-20260903T034319Z-1-001.zip")
IMAGE_ZIPS = [
    Path(r"D:\BKU\VSC\images-20260903T034922Z-1-001.zip"),
    Path(r"D:\BKU\VSC\images-20260903T034922Z-1-002.zip"),
    Path(r"D:\BKU\VSC\images-20260903T034922Z-1-003.zip"),
]


@unittest.skipUnless(
    torch is not None and ML_ZIP.is_file() and all(path.is_file() for path in IMAGE_ZIPS),
    "Recommendation external artifacts or runtime dependencies are unavailable",
)
class RecommendationHostDataSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = RecommendationPipeline.load_from_archives(
            REPO_ROOT / "configs" / "recommendation_category_aware_v2.json",
            ml_zip_path=ML_ZIP,
            image_zip_paths=IMAGE_ZIPS,
            device="cpu",
        )

    def test_full_zip_mapping_and_frozen_v5_flow(self):
        self.assertEqual(len(self.pipeline.catalog), 142480)
        self.assertEqual(len(self.pipeline.image_resolver), 142480)
        self.assertTrue(self.pipeline.image_validation["mapping_exact"])
        first = self.pipeline.image_resolver.first_ref
        self.assertEqual(first.item_id, "100002074_1")
        self.assertTrue(self.pipeline.image_resolver.read_bytes(first.item_id).startswith(b"\xff\xd8\xff"))
        records = self.pipeline.artifact_bundle.load_scorer_ready("test")
        _, result = _fixture_result(self.pipeline, records)
        public = result.to_public_dict()
        self.assertEqual(public["recommendation_version"], "category-aware-hybrid-v2")
        self.assertEqual(len(public["items"]), 3)
        self.assertNotIn("score", str(public).lower())
        self.assertNotIn("logit", str(public).lower())

    def test_all_packaged_metadata_and_scorer_ready_splits_load(self):
        bundle = self.pipeline.artifact_bundle
        self.assertEqual(int(bundle.embedding_manifest["item_count"]), 142480)
        expected_metadata = {"train": 64032, "valid": 4620, "test": 9530}
        expected_scorer_ready = {"train": 30918, "valid": 2284, "test": 4654}
        for split in ("train", "valid", "test"):
            self.assertEqual(
                len(bundle.load_metadata_index((split,))),
                expected_metadata[split],
            )
            self.assertEqual(
                len(bundle.load_scorer_ready(split)),
                expected_scorer_ready[split],
            )
