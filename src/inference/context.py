# -*- coding: utf-8 -*-
"""Shared lifecycle object between detection, scorer/LOO, and VLM adapters."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping


def _leading_length(value: object, name: str) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) < 1:
            raise ValueError(f"{name} must have a leading item dimension")
        return int(shape[0])
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must expose a leading item dimension") from error


@dataclass
class InferenceContext:
    """One detected outfit plus the image references needed downstream.

    Embeddings and categories live separately from garment metadata so the scorer
    consumes tensors directly while the VLM receives structured garment metadata,
    the original outfit image, and crop references. ``close`` owns the temporary
    image lifecycle.
    """

    garments: list[Mapping[str, object]]
    embeddings: object
    categories: object
    crop_image_refs: list[str | Path] = field(default_factory=list)
    original_image: object | None = None
    original_image_ref: str | Path | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: dict[str, object] = field(default_factory=dict)
    cleanup: Callable[[], None] | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.garments = [dict(garment) for garment in self.garments]
        self.crop_image_refs = list(self.crop_image_refs)
        self.metadata = dict(self.metadata)
        self.request_id = str(self.request_id).strip()
        if not self.request_id:
            raise ValueError("request_id must be non-empty")

        item_count = len(self.garments)
        if _leading_length(self.embeddings, "embeddings") != item_count:
            raise ValueError("embeddings item count does not match garments")
        if _leading_length(self.categories, "categories") != item_count:
            raise ValueError("categories item count does not match garments")
        if self.crop_image_refs and len(self.crop_image_refs) != item_count:
            raise ValueError("crop_image_refs must be empty or contain one ref per garment")

    def __len__(self) -> int:
        return len(self.garments)

    @property
    def item_ids(self) -> list[str]:
        return [str(garment.get("item_id", f"garment-{index}")) for index, garment in enumerate(self.garments)]

    @property
    def coarse_categories(self) -> list[str]:
        categories: list[str] = []
        for index, garment in enumerate(self.garments):
            value = garment.get("coarse_category")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"garments[{index}] is missing coarse_category")
            categories.append(value.strip().upper())
        return categories

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.cleanup is not None:
            self.cleanup()

    def __enter__(self) -> "InferenceContext":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
