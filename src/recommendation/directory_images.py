# -*- coding: utf-8 -*-
"""Portable indexed access to item images stored in a normal directory."""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import quote


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class DirectoryImageRef:
    item_id: str
    path: Path
    byte_size: int


class DirectoryImageResolver:
    """Build a deterministic ``item_id -> image path`` index once.

    The resolver is filesystem-agnostic: the root may live on a local disk,
    Google Drive mounted by Colab, a server volume, or a Docker bind mount.
    Image filenames must use ``<item_id>.jpg`` or ``<item_id>.jpeg``.
    """

    def __init__(
        self,
        image_root: Path | str,
        *,
        expected_count: int | None = 142_480,
    ) -> None:
        self.image_root = Path(image_root).expanduser().resolve()
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")

        self._by_item: dict[str, DirectoryImageRef] = {}
        self._ordered_refs: list[DirectoryImageRef] = []

        with os.scandir(self.image_root) as entries:
            rows = sorted(
                (entry for entry in entries if entry.is_file()),
                key=lambda entry: entry.name,
            )
        for entry in rows:
            path = Path(entry.path)
            if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            item_id = path.stem.strip()
            if not item_id:
                raise ValueError(f"Invalid image filename: {path.name!r}")
            if item_id in self._by_item:
                previous = self._by_item[item_id]
                raise ValueError(
                    f"Duplicate image item_id {item_id}: {previous.path} and {path}"
                )
            ref = DirectoryImageRef(
                item_id=item_id,
                path=path,
                byte_size=int(entry.stat().st_size),
            )
            self._by_item[item_id] = ref
            self._ordered_refs.append(ref)

        if expected_count is not None and len(self._by_item) != int(expected_count):
            raise ValueError(
                f"Expected {expected_count} unique images, found {len(self._by_item)} "
                f"under {self.image_root}"
            )

    def __len__(self) -> int:
        return len(self._by_item)

    def __contains__(self, item_id: object) -> bool:
        return str(item_id) in self._by_item

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(self._by_item)

    @property
    def first_ref(self) -> DirectoryImageRef:
        if not self._ordered_refs:
            raise RuntimeError("Image directory index is empty")
        return self._ordered_refs[0]

    def resolve(self, item_id: str) -> DirectoryImageRef:
        normalized = str(item_id)
        try:
            return self._by_item[normalized]
        except KeyError as error:
            raise KeyError(f"No image found for item_id={normalized!r}") from error

    def read_bytes(self, item_id: str) -> bytes:
        ref = self.resolve(item_id)
        payload = ref.path.read_bytes()
        if not payload:
            raise ValueError(f"Image file is empty: {ref.path}")
        return payload

    def validate_readable(self, item_ids: Sequence[str]) -> dict[str, str]:
        failures: dict[str, str] = {}
        for item_id in dict.fromkeys(str(value) for value in item_ids):
            try:
                payload = self.read_bytes(item_id)
                if not payload:
                    raise ValueError("empty image")
            except Exception as error:
                failures[item_id] = type(error).__name__
        return failures

    def media_type(self, item_id: str) -> str:
        ref = self.resolve(item_id)
        return mimetypes.guess_type(ref.path.name)[0] or "image/jpeg"

    def image_url(
        self,
        item_id: str,
        *,
        base_path: str = "/recommendation/images",
    ) -> str:
        self.resolve(item_id)
        return f"{base_path.rstrip('/')}/{quote(str(item_id), safe='')}"

    def write_selected_image(self, item_id: str, destination: Path | str) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.read_bytes(item_id))
        return target
