# -*- coding: utf-8 -*-
"""Small adapter protocols isolating unfinished image/VLM components.

Detection + crop + FashionCLIP + runtime category resolution can implement
``GarmentPreprocessor`` without changing the stable scorer/diagnosis pipeline.
Likewise the explanation/VLM layer consumes structured evidence only.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class GarmentPreprocessor(Protocol):
    """Convert one user image into deploy-facing precomputed garment records."""

    def prepare(self, image: object) -> Sequence[Mapping[str, object]]:
        ...


@runtime_checkable
class ExplanationProvider(Protocol):
    """Generate explanation text/metadata from structured ML evidence."""

    def explain(self, evidence: Mapping[str, object]) -> object:
        ...
