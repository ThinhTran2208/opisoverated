# -*- coding: utf-8 -*-
"""Direct, indexed access to item images stored across ZIP archives."""

from __future__ import annotations

import mimetypes
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence
from urllib.parse import quote


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class ZipImageRef:
    item_id: str
    archive_path: Path
    internal_path: str
    byte_size: int


class ZipImageResolver:
    """Cache ``item_id -> archive entry`` without extracting image archives."""

    def __init__(
        self,
        archive_paths: Sequence[Path | str],
        *,
        expected_count: int | None = 142_480,
    ) -> None:
        paths = tuple(Path(path).expanduser().resolve() for path in archive_paths)
        if not paths:
            raise ValueError("At least one image ZIP is required")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Image ZIPs not found: {missing}")

        self.archive_paths = paths
        self._by_item: dict[str, ZipImageRef] = {}
        self._ordered_refs: list[ZipImageRef] = []
        for archive_path in paths:
            with zipfile.ZipFile(archive_path, "r") as archive:
                entries = sorted(
                    (
                        entry
                        for entry in archive.infolist()
                        if not entry.is_dir()
                        and PurePosixPath(entry.filename).suffix.lower()
                        in SUPPORTED_IMAGE_SUFFIXES
                    ),
                    key=lambda entry: entry.filename,
                )
                for entry in entries:
                    item_id = PurePosixPath(entry.filename).stem.strip()
                    if not item_id:
                        raise ValueError(
                            f"Invalid image entry in {archive_path}: {entry.filename!r}"
                        )
                    if item_id in self._by_item:
                        previous = self._by_item[item_id]
                        raise ValueError(
                            "Duplicate image item_id "
                            f"{item_id}: {previous.archive_path}!{previous.internal_path} "
                            f"and {archive_path}!{entry.filename}"
                        )
                    ref = ZipImageRef(
                        item_id=item_id,
                        archive_path=archive_path,
                        internal_path=entry.filename,
                        byte_size=int(entry.file_size),
                    )
                    self._by_item[item_id] = ref
                    self._ordered_refs.append(ref)

        if expected_count is not None and len(self._by_item) != int(expected_count):
            raise ValueError(
                f"Expected {expected_count} unique images, found {len(self._by_item)}"
            )

    def __len__(self) -> int:
        return len(self._by_item)

    def __contains__(self, item_id: object) -> bool:
        return str(item_id) in self._by_item

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(self._by_item)

    @property
    def first_ref(self) -> ZipImageRef:
        if not self._ordered_refs:
            raise RuntimeError("Image ZIP index is empty")
        return self._ordered_refs[0]

    def resolve(self, item_id: str) -> ZipImageRef:
        normalized = str(item_id)
        try:
            return self._by_item[normalized]
        except KeyError as error:
            raise KeyError(f"No image found for item_id={normalized!r}") from error

    def read_bytes(self, item_id: str) -> bytes:
        """Read one compressed entry; never extract the containing archive."""

        ref = self.resolve(item_id)
        with zipfile.ZipFile(ref.archive_path, "r") as archive:
            payload = archive.read(ref.internal_path)
        if not payload:
            raise ValueError(f"Image entry is empty: {ref.internal_path}")
        return payload

    def media_type(self, item_id: str) -> str:
        ref = self.resolve(item_id)
        return mimetypes.guess_type(ref.internal_path)[0] or "image/jpeg"

    def image_url(
        self,
        item_id: str,
        *,
        base_path: str = "/recommendation/images",
    ) -> str:
        self.resolve(item_id)
        return f"{base_path.rstrip('/')}/{quote(str(item_id), safe='')}"

    def write_selected_image(self, item_id: str, destination: Path | str) -> Path:
        """Materialize exactly one requested image for demos or request-scoped use."""

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.read_bytes(item_id))
        return target

