# -*- coding: utf-8 -*-

import tempfile
import unittest
import zipfile
from pathlib import Path

from src.recommendation.zip_images import ZipImageResolver

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except ModuleNotFoundError:
    FastAPI = None
    TestClient = None

from src.recommendation.http_api import create_image_router


JPEG = b"\xff\xd8\xff\xe0fake-jpeg"


class ZipImageResolverTests(unittest.TestCase):
    def _archive(self, root: Path, name: str, entries: dict[str, bytes]) -> Path:
        path = root / name
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for entry_name, payload in entries.items():
                archive.writestr(entry_name, payload)
        return path

    def test_indexes_once_resolves_and_reads_one_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._archive(
                root,
                "images-001.zip",
                {"images/b.jpg": JPEG, "images/a.jpg": JPEG},
            )
            second = self._archive(
                root,
                "images-002.zip",
                {"images/c.jpg": JPEG},
            )
            resolver = ZipImageResolver([first, second], expected_count=3)
            self.assertEqual(len(resolver), 3)
            self.assertEqual(resolver.first_ref.item_id, "a")
            self.assertEqual(resolver.resolve("c").archive_path, second.resolve())
            self.assertEqual(resolver.read_bytes("b"), JPEG)
            self.assertEqual(
                resolver.image_url("c"),
                "/recommendation/images/c",
            )

    def test_duplicate_item_id_across_archives_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._archive(root, "a.zip", {"images/x.jpg": JPEG})
            second = self._archive(root, "b.zip", {"other/x.jpeg": JPEG})
            with self.assertRaisesRegex(ValueError, "Duplicate image item_id"):
                ZipImageResolver([first, second], expected_count=None)

    @unittest.skipUnless(FastAPI is not None, "FastAPI is not installed")
    def test_fastapi_endpoint_streams_selected_zip_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root, "images.zip", {"images/x.jpg": JPEG})
            resolver = ZipImageResolver([archive], expected_count=1)
            app = FastAPI()
            app.include_router(create_image_router(resolver))
            response = TestClient(app).get("/recommendation/images/x")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, JPEG)
            self.assertEqual(response.headers["content-type"], "image/jpeg")
            self.assertEqual(
                TestClient(app).get("/recommendation/images/missing").status_code,
                404,
            )
