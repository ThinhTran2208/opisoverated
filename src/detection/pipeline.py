# -*- coding: utf-8 -*-
"""End-to-end garment detection -> FashionCLIP -> Core-7 scorer handoff."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from .config import DetectionConfig
from .fashionclip import FashionCLIPCore7Encoder
from .rfdetr import RFDETRFashionpediaDetector
from .schema import DetectedGarment, DetectionCandidate, DetectionResult


def expand_and_clamp_xyxy(
    box_xyxy: Sequence[float],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive")
    if len(box_xyxy) != 4:
        raise ValueError("box_xyxy must have four values")
    x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise ValueError("box_xyxy must contain finite values")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box_xyxy must have positive area")

    pad_x = (x1 - x0) * padding_ratio
    pad_y = (y1 - y0) * padding_ratio
    left = max(0, min(image_width - 1, math.floor(x0 - pad_x)))
    top = max(0, min(image_height - 1, math.floor(y0 - pad_y)))
    right = max(left + 1, min(image_width, math.ceil(x1 + pad_x)))
    bottom = max(top + 1, min(image_height, math.ceil(y1 + pad_y)))
    return int(left), int(top), int(right), int(bottom)


def build_scorer_batch_lists(
    result: DetectionResult,
    *,
    min_items: int,
    max_items: int,
) -> dict:
    """Build a dependency-free representation of the scorer handoff."""

    item_count = len(result.garments)
    if item_count < min_items:
        raise ValueError(
            f"Detected outfit has {item_count} accepted garments; scorer requires >= {min_items}"
        )
    if item_count > max_items:
        raise ValueError(
            f"Detected outfit has {item_count} accepted garments; scorer supports <= {max_items}. "
            "Detection V1 does not silently truncate garments."
        )
    return {
        "item_embeddings": [garment.embedding for garment in result.garments],
        "coarse_category_ids": [garment.category.coarse_category_id for garment in result.garments],
        "item_mask": [True] * item_count,
    }


def build_scorer_batch_torch(
    result: DetectionResult,
    *,
    min_items: int,
    max_items: int,
) -> dict:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required to build tensor scorer inputs") from error

    raw = build_scorer_batch_lists(result, min_items=min_items, max_items=max_items)
    embeddings = []
    for embedding in raw["item_embeddings"]:
        if isinstance(embedding, torch.Tensor):
            embeddings.append(embedding.float())
        else:
            embeddings.append(torch.tensor(embedding, dtype=torch.float32))
    return {
        "item_embeddings": torch.stack(embeddings, dim=0).unsqueeze(0),
        "coarse_category_ids": torch.tensor(
            [raw["coarse_category_ids"]], dtype=torch.long
        ),
        "item_mask": torch.tensor([raw["item_mask"]], dtype=torch.bool),
    }


class DetectionPipeline:
    def __init__(
        self,
        config: DetectionConfig,
        *,
        detector: RFDETRFashionpediaDetector | None = None,
        encoder: FashionCLIPCore7Encoder | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.detector = detector or RFDETRFashionpediaDetector(config, device=device)
        self.encoder = encoder or FashionCLIPCore7Encoder(config, device=device)

    def run(self, image_source: Path | str | object) -> tuple[DetectionResult, object]:
        try:
            from PIL import Image
        except ModuleNotFoundError as error:
            raise RuntimeError("Pillow is required to run garment detection") from error

        if isinstance(image_source, (str, Path)):
            image = Image.open(image_source).convert("RGB")
            detector_source = str(image_source)
        else:
            image = image_source.convert("RGB") if hasattr(image_source, "convert") else image_source
            detector_source = image

        candidates, rejected, runtime_ms = self.detector.detect(detector_source)
        crop_records: list[tuple[DetectionCandidate, tuple[int, int, int, int], object]] = []
        for candidate in candidates:
            crop_box = expand_and_clamp_xyxy(
                candidate.box_xyxy,
                image_width=image.width,
                image_height=image.height,
                padding_ratio=self.config.crop_padding_ratio,
            )
            left, top, right, bottom = crop_box
            if min(right - left, bottom - top) < self.config.min_crop_side_px:
                rejected.append(
                    {
                        "detection_index": candidate.detection_index,
                        "reason": "crop_too_small",
                        "detector_label": candidate.detector_label,
                        "crop_box_xyxy": list(crop_box),
                    }
                )
                continue
            crop_records.append((candidate, crop_box, image.crop(crop_box)))

        embeddings, decisions = self.encoder.encode_and_classify(
            [record[2] for record in crop_records]
        )
        garments: list[DetectedGarment] = []
        for (candidate, crop_box, _), embedding, decision in zip(
            crop_records, embeddings, decisions
        ):
            if not decision.accepted:
                rejected.append(
                    {
                        "detection_index": candidate.detection_index,
                        "reason": decision.rejection_reason,
                        "detector_label": candidate.detector_label,
                        "coarse_category_candidate": decision.coarse_category,
                        "category_similarity": decision.similarity,
                        "category_margin": decision.margin,
                    }
                )
                continue
            garments.append(
                DetectedGarment(
                    candidate=candidate,
                    crop_box_xyxy=crop_box,
                    category=self.encoder.to_category_prediction(decision),
                    embedding=embedding,
                )
            )

        result = DetectionResult(
            detection_version=self.config.detection_version,
            detector_repo_id=self.config.detector_repo_id,
            fashionclip_model_id=self.config.fashionclip_model_id,
            category_classifier_version=self.config.category_classifier_version,
            category_mapping_version=self.config.category_mapping_version,
            image_width=image.width,
            image_height=image.height,
            garments=garments,
            rejected_detections=rejected,
            detector_runtime_ms=runtime_ms,
        )
        return result, image


def save_detection_result(
    result: DetectionResult,
    image: object,
    output_dir: Path | str,
    *,
    scorer_min_items: int,
    scorer_max_items: int,
) -> dict:
    destination = Path(output_dir)
    crops_dir = destination / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    crop_paths: list[str] = []
    for index, garment in enumerate(result.garments):
        crop = image.crop(garment.crop_box_xyxy)
        name = f"garment_{index:02d}_{garment.category.coarse_category.lower()}.jpg"
        path = crops_dir / name
        crop.save(path, quality=95)
        crop_paths.append(str(path))

    metadata = result.metadata_dict()
    for garment_metadata, crop_path in zip(metadata["garments"], crop_paths):
        garment_metadata["crop_path"] = crop_path
    metadata_path = destination / "detection_result.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    scorer_path = None
    scorer_error = None
    try:
        scorer_batch = build_scorer_batch_torch(
            result,
            min_items=scorer_min_items,
            max_items=scorer_max_items,
        )
        import torch

        scorer_path = destination / "scorer_inputs.pt"
        torch.save(scorer_batch, scorer_path)
    except ValueError as error:
        scorer_error = str(error)

    return {
        "metadata_path": str(metadata_path),
        "crop_paths": crop_paths,
        "scorer_inputs_path": None if scorer_path is None else str(scorer_path),
        "scorer_handoff_error": scorer_error,
    }
