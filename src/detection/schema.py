# -*- coding: utf-8 -*-
"""Lightweight schemas for garment detection and Core-7 classification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .config import EXPECTED_EMBEDDING_DIM


def _validate_box(box: Sequence[float], *, name: str) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"{name} must contain four values")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite values")
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{name} must have positive area")
    return values


@dataclass(frozen=True)
class DetectionCandidate:
    detection_index: int
    box_xyxy: tuple[float, float, float, float]
    detector_label: str
    detector_confidence: float | None
    detector_class_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "box_xyxy", _validate_box(self.box_xyxy, name="box_xyxy"))
        if self.detection_index < 0:
            raise ValueError("detection_index must be >= 0")
        if not self.detector_label.strip():
            raise ValueError("detector_label must be non-empty")
        if self.detector_confidence is not None:
            confidence = float(self.detector_confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("detector_confidence must be null or in [0, 1]")


@dataclass(frozen=True)
class CategoryPrediction:
    coarse_category: str
    coarse_category_id: int
    similarity: float
    margin: float
    similarities: Mapping[str, float]
    source: str

    def __post_init__(self) -> None:
        if not self.coarse_category.strip():
            raise ValueError("coarse_category must be non-empty")
        if self.coarse_category_id < 1:
            raise ValueError("coarse_category_id must be >= 1")
        if not math.isfinite(float(self.similarity)):
            raise ValueError("similarity must be finite")
        if not math.isfinite(float(self.margin)) or self.margin < 0.0:
            raise ValueError("margin must be finite and >= 0")


@dataclass
class DetectedGarment:
    candidate: DetectionCandidate
    crop_box_xyxy: tuple[int, int, int, int]
    category: CategoryPrediction
    embedding: object = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.crop_box_xyxy) != 4:
            raise ValueError("crop_box_xyxy must contain four values")
        shape = getattr(self.embedding, "shape", None)
        if shape is not None:
            if len(shape) != 1 or int(shape[0]) != EXPECTED_EMBEDDING_DIM:
                raise ValueError(
                    f"garment embedding must have shape [{EXPECTED_EMBEDDING_DIM}], "
                    f"got {tuple(shape)}"
                )
        elif isinstance(self.embedding, Sequence) and not isinstance(self.embedding, (str, bytes)):
            if len(self.embedding) != EXPECTED_EMBEDDING_DIM:
                raise ValueError(
                    f"garment embedding must contain {EXPECTED_EMBEDDING_DIM} values"
                )

    def metadata_dict(self) -> dict:
        return {
            "detection_index": self.candidate.detection_index,
            "box_xyxy": list(self.candidate.box_xyxy),
            "crop_box_xyxy": list(self.crop_box_xyxy),
            "detector_label": self.candidate.detector_label,
            "detector_class_id": self.candidate.detector_class_id,
            "detector_confidence": self.candidate.detector_confidence,
            "coarse_category": self.category.coarse_category,
            "coarse_category_id": self.category.coarse_category_id,
            "coarse_category_source": self.category.source,
            "category_similarity": self.category.similarity,
            "category_margin": self.category.margin,
            "category_similarities": dict(self.category.similarities),
            "embedding_dimension": EXPECTED_EMBEDDING_DIM,
            "embedding_normalization": "l2",
        }


@dataclass
class DetectionResult:
    detection_version: str
    detector_repo_id: str
    fashionclip_model_id: str
    category_classifier_version: str
    category_mapping_version: str
    image_width: int
    image_height: int
    garments: list[DetectedGarment]
    rejected_detections: list[dict]
    detector_runtime_ms: float | None = None

    def metadata_dict(self) -> dict:
        return {
            "detection_version": self.detection_version,
            "detector": {
                "repo_id": self.detector_repo_id,
                "runtime_ms": self.detector_runtime_ms,
            },
            "fashionclip": {
                "model_id": self.fashionclip_model_id,
                "embedding_dimension": EXPECTED_EMBEDDING_DIM,
                "normalization": "l2",
            },
            "taxonomy": {
                "category_mapping_version": self.category_mapping_version,
                "category_classifier_version": self.category_classifier_version,
                "master_category": None,
                "master_category_semantics": "not_inferred_for_user_images",
            },
            "image": {"width": self.image_width, "height": self.image_height},
            "garment_count": len(self.garments),
            "garments": [garment.metadata_dict() for garment in self.garments],
            "rejected_detections": list(self.rejected_detections),
        }
