# -*- coding: utf-8 -*-
"""Configuration and canonical taxonomy for garment detection V1.

The module intentionally uses only the Python standard library so repository
unit tests can import it without installing GPU/runtime dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DETECTION_VERSION = "rfdetr-fashionclip-core7-v1"
CATEGORY_CLASSIFIER_VERSION = "fashionclip-zero-shot-core7-v1"
DEFAULT_RFDETR_REPO_ID = "resoa/garment-detector-seg"
DEFAULT_FASHIONCLIP_MODEL_ID = "patrickjohncyh/fashion-clip"
DEFAULT_CATEGORY_MAPPING_VERSION = "core7-v2"
EXPECTED_EMBEDDING_DIM = 512

CORE7_CATEGORY_TO_ID = {
    "TOP": 1,
    "BOTTOM": 2,
    "DRESS": 3,
    "OUTERWEAR": 4,
    "SHOES": 5,
    "BAG": 6,
    "HAT": 7,
}
CORE7_CATEGORIES = tuple(CORE7_CATEGORY_TO_ID)


@dataclass(frozen=True)
class DetectionConfig:
    detection_version: str
    detector_repo_id: str
    detector_threshold: float
    supported_detector_labels: tuple[str, ...]
    fashionclip_model_id: str
    embedding_dim: int
    category_mapping_version: str
    category_classifier_version: str
    category_prompts: Mapping[str, tuple[str, ...]]
    min_category_similarity: float | None
    min_category_margin: float
    crop_padding_ratio: float
    min_crop_side_px: int
    scorer_min_items: int
    scorer_max_items: int
    scorer_category_ids: Mapping[str, int]
    scorer_padding_category_id: int


def _as_prompt_mapping(value: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise ValueError("category_prompts must be a JSON object")
    parsed: dict[str, tuple[str, ...]] = {}
    for raw_category, raw_prompts in value.items():
        category = str(raw_category).strip().upper()
        if not isinstance(raw_prompts, list) or not raw_prompts:
            raise ValueError(f"category_prompts[{category!r}] must be a non-empty list")
        prompts = tuple(str(prompt).strip() for prompt in raw_prompts)
        if any(not prompt for prompt in prompts):
            raise ValueError(f"category_prompts[{category!r}] contains an empty prompt")
        parsed[category] = prompts
    if set(parsed) != set(CORE7_CATEGORIES):
        raise ValueError(
            "category_prompts must define exactly the seven canonical categories: "
            f"{CORE7_CATEGORIES}"
        )
    return parsed


def validate_detection_config(config: DetectionConfig) -> DetectionConfig:
    if config.detection_version != DETECTION_VERSION:
        raise ValueError(
            f"detection_version must be {DETECTION_VERSION!r}, "
            f"got {config.detection_version!r}"
        )
    if config.category_classifier_version != CATEGORY_CLASSIFIER_VERSION:
        raise ValueError(
            "category_classifier_version mismatch: "
            f"{config.category_classifier_version!r}"
        )
    if not config.detector_repo_id.strip():
        raise ValueError("detector_repo_id must be non-empty")
    if not 0.0 < config.detector_threshold <= 1.0:
        raise ValueError("detector_threshold must be in (0, 1]")
    if not config.supported_detector_labels:
        raise ValueError("supported_detector_labels must not be empty")
    if len(set(config.supported_detector_labels)) != len(config.supported_detector_labels):
        raise ValueError("supported_detector_labels contains duplicates")
    if config.fashionclip_model_id != DEFAULT_FASHIONCLIP_MODEL_ID:
        raise ValueError(
            "Detection V1 must reuse the frozen scorer FashionCLIP model: "
            f"{DEFAULT_FASHIONCLIP_MODEL_ID!r}"
        )
    if config.embedding_dim != EXPECTED_EMBEDDING_DIM:
        raise ValueError(f"embedding_dim must be {EXPECTED_EMBEDDING_DIM}")
    if config.category_mapping_version != DEFAULT_CATEGORY_MAPPING_VERSION:
        raise ValueError(
            "category_mapping_version must match the canonical scorer taxonomy: "
            f"{DEFAULT_CATEGORY_MAPPING_VERSION!r}"
        )
    if set(config.category_prompts) != set(CORE7_CATEGORIES):
        raise ValueError("category_prompts does not match canonical Core-7 categories")
    if config.min_category_similarity is not None and not -1.0 <= config.min_category_similarity <= 1.0:
        raise ValueError("min_category_similarity must be null or in [-1, 1]")
    if config.min_category_margin < 0.0:
        raise ValueError("min_category_margin must be >= 0")
    if not 0.0 <= config.crop_padding_ratio <= 0.5:
        raise ValueError("crop_padding_ratio must be in [0, 0.5]")
    if config.min_crop_side_px < 1:
        raise ValueError("min_crop_side_px must be >= 1")
    if config.scorer_min_items < 1:
        raise ValueError("scorer_min_items must be >= 1")
    if config.scorer_max_items < config.scorer_min_items:
        raise ValueError("scorer_max_items must be >= scorer_min_items")
    if dict(config.scorer_category_ids) != CORE7_CATEGORY_TO_ID:
        raise ValueError("scorer_handoff.category_ids must exactly match canonical Core-7 IDs")
    if config.scorer_padding_category_id != 0:
        raise ValueError("scorer_handoff.padding_category_id must be 0")
    return config


def load_detection_config(path: Path | str) -> DetectionConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {source}")

    detector = payload.get("detector")
    encoder = payload.get("fashionclip")
    classifier = payload.get("coarse_category_classifier")
    crop = payload.get("crop")
    scorer = payload.get("scorer_handoff")
    for name, section in (
        ("detector", detector),
        ("fashionclip", encoder),
        ("coarse_category_classifier", classifier),
        ("crop", crop),
        ("scorer_handoff", scorer),
    ):
        if not isinstance(section, dict):
            raise ValueError(f"Missing or invalid config section: {name}")

    labels = detector.get("supported_labels")
    if not isinstance(labels, list):
        raise ValueError("detector.supported_labels must be a list")

    config = DetectionConfig(
        detection_version=str(payload.get("detection_version", "")),
        detector_repo_id=str(detector.get("repo_id", "")),
        detector_threshold=float(detector.get("threshold", 0.35)),
        supported_detector_labels=tuple(str(label).strip().lower() for label in labels),
        fashionclip_model_id=str(encoder.get("model_id", "")),
        embedding_dim=int(encoder.get("embedding_dim", 0)),
        category_mapping_version=str(classifier.get("category_mapping_version", "")),
        category_classifier_version=str(classifier.get("version", "")),
        category_prompts=_as_prompt_mapping(classifier.get("prompts")),
        min_category_similarity=(
            None
            if classifier.get("min_similarity") is None
            else float(classifier["min_similarity"])
        ),
        min_category_margin=float(classifier.get("min_margin", 0.0)),
        crop_padding_ratio=float(crop.get("padding_ratio", 0.03)),
        min_crop_side_px=int(crop.get("min_side_px", 24)),
        scorer_min_items=int(scorer.get("min_items", 3)),
        scorer_max_items=int(scorer.get("max_items", 8)),
        scorer_category_ids={
            str(key).strip().upper(): int(value)
            for key, value in dict(scorer.get("category_ids", {})).items()
        },
        scorer_padding_category_id=int(scorer.get("padding_category_id", -1)),
    )
    return validate_detection_config(config)
