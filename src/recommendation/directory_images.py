# -*- coding: utf-8 -*-
"""Portable lazy access to item images stored in a normal directory."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import quote


SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg")


@dataclass(frozen=True)
class DirectoryImageRef:
    item_id: str
    path: Path
    byte_size: int


class DirectoryImageResolver:
    """Resolve ``item_id -> image path`` without enumerating a huge directory.

    Large Google Drive folders can time out when ``glob``/``scandir`` attempts to
    list every image. Directory mode therefore supports a declared item inventory
    (normally the frozen embedding catalog item IDs) and resolves actual files
    lazily only when image bytes are needed.

    This remains filesystem-agnostic: ``image_root`` may live on a local disk,
    mounted Google Drive, server volume, or Docker bind mount.
    """

    def __init__(
        self,
        image_root: Path | str,
        *,
        expected_count: int | None = 142_480,
        known_item_ids: Sequence[str] | None = None,
    ) -> None:
        self.image_root = Path(image_root).expanduser().resolve()
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")

        self._known_item_ids = (
            tuple(str(value) for value in known_item_ids)
            if known_item_ids is not None
            else None
        )
        self._known_item_set = (
            set(self._known_item_ids) if self._known_item_ids is not None else None
        )
        if self._known_item_ids is not None and len(self._known_item_ids) != len(self._known_item_set):
            raise ValueError("known_item_ids contains duplicates")
        if (
            expected_count is not None
            and self._known_item_ids is not None
            and len(self._known_item_ids) != int(expected_count)
        ):
            raise ValueError(
                f"Expected {expected_count} declared image item IDs, "
                f"found {len(self._known_item_ids)}"
            )

        self.expected_count = expected_count
        self.inventory_is_declared = self._known_item_ids is not None
        self._resolved: dict[str, DirectoryImageRef] = {}
        self._missing: set[str] = set()

    def __len__(self) -> int:
        if self._known_item_ids is None:
            raise TypeError(
                "Directory image count is unknown without a declared item inventory"
            )
        return len(self._known_item_ids)

    def __contains__(self, item_id: object) -> bool:
        normalized = str(item_id)
        # In declared-inventory mode, membership is intentionally metadata-only
        # so retrieval does not issue hundreds of thousands of Drive stat calls.
        if self._known_item_set is not None:
            return normalized in self._known_item_set
        try:
            self.resolve(normalized)
            return True
        except KeyError:
            return False

    @property
    def item_ids(self) -> tuple[str, ...]:
        if self._known_item_ids is None:
            raise RuntimeError(
                "Directory image inventory is lazy; provide known_item_ids to expose item_ids"
            )
        return self._known_item_ids

    @property
    def first_ref(self) -> DirectoryImageRef:
        if not self._known_item_ids:
            raise RuntimeError("No declared image item IDs are available")
        return self.resolve(self._known_item_ids[0])

    def _candidate_paths(self, item_id: str):
        for suffix in SUPPORTED_IMAGE_SUFFIXES:
            yield self.image_root / f"{item_id}{suffix}"

    def resolve(self, item_id: str) -> DirectoryImageRef:
        normalized = str(item_id)
        cached = self._resolved.get(normalized)
        if cached is not None:
            return cached
        if normalized in self._missing:
            raise KeyError(f"No image found for item_id={normalized!r}")
        if self._known_item_set is not None and normalized not in self._known_item_set:
            raise KeyError(f"Unknown image item_id={normalized!r}")

        for path in self._candidate_paths(normalized):
            try:
                if path.is_file():
                    ref = DirectoryImageRef(
                        item_id=normalized,
                        path=path,
                        byte_size=int(path.stat().st_size),
                    )
                    self._resolved[normalized] = ref
                    return ref
            except OSError:
                continue
        self._missing.add(normalized)
        raise KeyError(f"No image found for item_id={normalized!r}")

    def read_bytes(self, item_id: str) -> bytes:
        ref = self.resolve(item_id)
        payload = ref.path.read_bytes()
        if not payload:
            raise ValueError(f"Image file is empty: {ref.path}")
        return payload

    def validate_readable(self, item_ids: Sequence[str]) -> dict[str, str]:
        """Check only explicitly requested images; never enumerate the directory."""

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
        # Public image URLs are generated from the stable item ID. Actual file
        # existence is checked when bytes are served/read, not by bulk scanning.
        if self._known_item_set is not None and str(item_id) not in self._known_item_set:
            raise KeyError(f"Unknown image item_id={str(item_id)!r}")
        return f"{base_path.rstrip('/')}/{quote(str(item_id), safe='')}"

    def write_selected_image(self, item_id: str, destination: Path | str) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.read_bytes(item_id))
        return target
