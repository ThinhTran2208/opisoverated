# -*- coding: utf-8 -*-
"""Adapters joining merged detection/VLM modules to production inference."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable

from .context import InferenceContext


@runtime_checkable
class GarmentPreprocessor(Protocol):
    """Convert one user image into a managed ``InferenceContext``."""

    def prepare(self, image: object) -> InferenceContext:
        ...


@runtime_checkable
class ExplanationProvider(Protocol):
    """Explain the authoritative LOO result using garment crops as visual evidence."""

    def explain(
        self,
        loo_result: Mapping[str, object],
        garments: Sequence[Mapping[str, object]],
        crop_image_refs: Sequence[str | Path],
        *,
        sample_id: str,
    ) -> object:
        ...


class DetectionAdapter:
    """RF-DETR + FashionCLIP adapter producing the canonical inference context."""

    def __init__(self, detection_pipeline) -> None:
        self.pipeline = detection_pipeline

    @classmethod
    def from_config(
        cls,
        config_path: Path | str,
        *,
        device: str | None = None,
    ) -> "DetectionAdapter":
        # Lazy import keeps scorer-only portability CI independent of the heavy
        # RF-DETR/FashionCLIP runtime dependencies.
        from src.detection import DetectionPipeline, load_detection_config

        config = load_detection_config(config_path)
        return cls(DetectionPipeline(config, device=device))

    def prepare(self, image: object) -> InferenceContext:
        result, original_image = self.pipeline.run(image)
        temporary_dir = tempfile.TemporaryDirectory(prefix="outfit-inference-")
        try:
            crop_refs: list[Path] = []
            garments: list[dict[str, object]] = []
            embeddings = []
            category_ids = []

            for index, garment in enumerate(result.garments):
                item_id = f"garment-{index}"
                crop_path = Path(temporary_dir.name) / f"{item_id}.png"
                original_image.crop(garment.crop_box_xyxy).save(crop_path, format="PNG")
                crop_refs.append(crop_path)

                garments.append(
                    {
                        "item_id": item_id,
                        "coarse_category": garment.category.coarse_category,
                        "coarse_category_id": garment.category.coarse_category_id,
                        "detection_label": garment.candidate.detector_label,
                        "detection_confidence": garment.candidate.detector_confidence,
                        "bbox": list(garment.candidate.box_xyxy),
                        "crop_bbox": list(garment.crop_box_xyxy),
                        "category_similarity": garment.category.similarity,
                        "category_margin": garment.category.margin,
                    }
                )
                embeddings.append(garment.embedding)
                category_ids.append(garment.category.coarse_category_id)

            try:
                import torch
            except ModuleNotFoundError:  # pragma: no cover - detection already requires torch.
                stacked_embeddings = embeddings
                stacked_categories = category_ids
            else:
                stacked_embeddings = (
                    torch.stack([torch.as_tensor(value).float() for value in embeddings], dim=0)
                    if embeddings
                    else torch.empty((0, 512), dtype=torch.float32)
                )
                stacked_categories = torch.tensor(category_ids, dtype=torch.long)

            return InferenceContext(
                garments=garments,
                embeddings=stacked_embeddings,
                categories=stacked_categories,
                crop_image_refs=crop_refs,
                original_image=original_image,
                metadata={
                    "detection": result.metadata_dict(),
                },
                cleanup=temporary_dir.cleanup,
            )
        except Exception:
            temporary_dir.cleanup()
            raise


class VLMAdapter:
    """Adapter from raw LOO evidence + garment crops to VLM Explanation V1."""

    def __init__(self, explanation_pipeline) -> None:
        self.pipeline = explanation_pipeline

    @classmethod
    def from_config(cls, config_path: Path | str) -> "VLMAdapter":
        # Lazy import is intentional: the canonical Qwen runtime currently uses
        # a different Transformers major range from RF-DETR.
        from src.vlm import VLMExplanationPipeline, load_vlm_config
        from src.vlm.qwen_backend import Qwen3VLBackend

        config = load_vlm_config(config_path)
        backend = Qwen3VLBackend.from_config(config)
        return cls(VLMExplanationPipeline(backend, config))

    def explain(
        self,
        loo_result: Mapping[str, object],
        garments: Sequence[Mapping[str, object]],
        crop_image_refs: Sequence[str | Path],
        *,
        sample_id: str,
    ) -> object:
        if len(crop_image_refs) != len(garments):
            raise ValueError("VLMAdapter requires exactly one crop ref per garment")

        from src.vlm import build_vlm_evidence

        item_ids: list[str] = []
        coarse_categories: list[str] = []
        for index, garment in enumerate(garments):
            item_id = str(garment.get("item_id", f"garment-{index}"))
            category = garment.get("coarse_category")
            if not isinstance(category, str) or not category.strip():
                raise ValueError(f"garments[{index}] is missing coarse_category")
            item_ids.append(item_id)
            coarse_categories.append(category.strip().upper())

        evidence = build_vlm_evidence(
            loo_result,
            sample_id=sample_id,
            item_ids=item_ids,
            coarse_categories=coarse_categories,
        )
        return self.pipeline.explain(evidence, crop_image_refs)
