# -*- coding: utf-8 -*-
"""Frozen FashionCLIP image encoder and direct Core-7 zero-shot classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .config import (
    CATEGORY_CLASSIFIER_VERSION,
    CORE7_CATEGORY_TO_ID,
    DetectionConfig,
    EXPECTED_EMBEDDING_DIM,
)
from .schema import CategoryPrediction


@dataclass(frozen=True)
class PredictionDecision:
    accepted: bool
    coarse_category: str
    coarse_category_id: int
    similarity: float
    margin: float
    similarities: Mapping[str, float]
    rejection_reason: str | None = None


def select_core7_prediction(
    scores: Mapping[str, float],
    *,
    min_similarity: float | None = None,
    min_margin: float = 0.0,
) -> PredictionDecision:
    """Select the best Core-7 category from cosine similarities."""

    normalized = {str(key).strip().upper(): float(value) for key, value in scores.items()}
    if set(normalized) != set(CORE7_CATEGORY_TO_ID):
        raise ValueError("scores must contain exactly the seven canonical Core-7 categories")
    ranked = sorted(normalized.items(), key=lambda pair: pair[1], reverse=True)
    best_category, best_score = ranked[0]
    second_score = ranked[1][1]
    margin = best_score - second_score

    reason = None
    if min_similarity is not None and best_score < min_similarity:
        reason = "category_similarity_below_threshold"
    elif margin < min_margin:
        reason = "category_margin_below_threshold"

    return PredictionDecision(
        accepted=reason is None,
        coarse_category=best_category,
        coarse_category_id=CORE7_CATEGORY_TO_ID[best_category],
        similarity=best_score,
        margin=margin,
        similarities=normalized,
        rejection_reason=reason,
    )


def _extract_feature_tensor(output: object, *, feature_name: str):
    """Return the projected feature tensor from Transformers v4 or v5 helpers.

    Transformers v4 ``CLIPModel.get_*_features`` returned a tensor directly.
    Transformers v5 returns ``BaseModelOutputWithPooling`` and stores the same
    projected embedding in ``pooler_output``. Supporting both shapes keeps the
    detection runtime compatible with the RF-DETR-required Transformers v5
    range without breaking older local environments.
    """

    if hasattr(output, "pooler_output"):
        tensor = getattr(output, "pooler_output")
        if tensor is None:
            raise ValueError(f"{feature_name} output has pooler_output=None")
        return tensor

    # Transformers v4 behavior: the helper returned the tensor itself.
    if hasattr(output, "float") and hasattr(output, "shape"):
        return output

    raise TypeError(
        f"Unsupported {feature_name} output type: {type(output).__name__}. "
        "Expected a tensor or an object with pooler_output."
    )


class FashionCLIPCore7Encoder:
    """Encode garment crops once, then reuse the same 512-d vector for scorer + category."""

    def __init__(self, config: DetectionConfig, *, device: str | None = None) -> None:
        self.config = config
        self.device = device
        self._model = None
        self._processor = None
        self._torch = None
        self._category_names: list[str] | None = None
        self._category_prototypes = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "FashionCLIP runtime dependencies are missing. Install "
                "requirements-detection.txt before running detection."
            ) from error

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = CLIPModel.from_pretrained(self.config.fashionclip_model_id)
        processor = CLIPProcessor.from_pretrained(self.config.fashionclip_model_id)
        model.eval().to(device)

        self.device = device
        self._torch = torch
        self._model = model
        self._processor = processor
        self._build_category_prototypes()

    def _normalize(self, tensor):
        torch = self._torch
        norms = torch.linalg.vector_norm(tensor, dim=-1, keepdim=True).clamp_min(1e-12)
        return tensor / norms

    def _build_category_prototypes(self) -> None:
        torch = self._torch
        model = self._model
        processor = self._processor
        category_names: list[str] = []
        prototypes = []

        with torch.inference_mode():
            for category in CORE7_CATEGORY_TO_ID:
                prompts = list(self.config.category_prompts[category])
                inputs = processor(text=prompts, return_tensors="pt", padding=True)
                text_kwargs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                    if key in {"input_ids", "attention_mask"}
                }
                output = model.get_text_features(**text_kwargs)
                features = _extract_feature_tensor(output, feature_name="text features")
                features = self._normalize(features.float())
                prototype = self._normalize(features.mean(dim=0, keepdim=True))[0]
                category_names.append(category)
                prototypes.append(prototype)

        self._category_names = category_names
        self._category_prototypes = torch.stack(prototypes, dim=0)

    def encode_and_classify(self, crops: Sequence[object]) -> tuple[list[object], list[PredictionDecision]]:
        if not crops:
            return [], []
        self._load()
        torch = self._torch
        model = self._model
        processor = self._processor

        inputs = processor(images=list(crops), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            output = model.get_image_features(pixel_values=pixel_values)
        features = _extract_feature_tensor(output, feature_name="image features")
        features = self._normalize(features.float())
        if features.ndim != 2 or int(features.shape[1]) != EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                "FashionCLIP projected image embedding contract violated: "
                f"expected [N, {EXPECTED_EMBEDDING_DIM}], got {tuple(features.shape)}"
            )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("FashionCLIP produced NaN/Inf embeddings")

        similarity_matrix = features @ self._category_prototypes.T
        decisions: list[PredictionDecision] = []
        for row in similarity_matrix.detach().cpu().tolist():
            scores = dict(zip(self._category_names, (float(value) for value in row)))
            decisions.append(
                select_core7_prediction(
                    scores,
                    min_similarity=self.config.min_category_similarity,
                    min_margin=self.config.min_category_margin,
                )
            )

        cpu_features = [row.detach().cpu() for row in features]
        return cpu_features, decisions

    @staticmethod
    def to_category_prediction(decision: PredictionDecision) -> CategoryPrediction:
        if not decision.accepted:
            raise ValueError("Cannot convert a rejected category decision")
        return CategoryPrediction(
            coarse_category=decision.coarse_category,
            coarse_category_id=decision.coarse_category_id,
            similarity=decision.similarity,
            margin=decision.margin,
            similarities=decision.similarities,
            source=CATEGORY_CLASSIFIER_VERSION,
        )
