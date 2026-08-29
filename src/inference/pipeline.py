# -*- coding: utf-8 -*-
"""Stable production inference boundary for scorer + calibration + LOO.

The deploy-facing core accepts precomputed garment records today, while image
preprocessing (detection/cropping/FashionCLIP/category resolution) and VLM
explanation remain plug-in adapters.  This prevents unfinished components from
changing the frozen scorer/calibration contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # Keep schema/manifest importable in lightweight CI.
    torch = None

from src.calibration import load_calibrator
from src.diagnosis.loo import diagnose_outfit
from src.scorer.checkpoint import load_checkpoint, sha256_file
from src.scorer.model import TypeAwarePairwiseScorer


PIPELINE_VERSION = "outfit-production-inference-v1"
MIN_ITEMS = 3
MAX_ITEMS = 8
EMBEDDING_DIM = 512
CATEGORY_MIN_ID = 1
CATEGORY_MAX_ID = 7


class InferenceInputError(ValueError):
    """Expected user/runtime input error with a stable machine-readable code."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, object] | None = None):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for production scorer inference")


def _clean_item_for_output(item: Mapping[str, object], index: int) -> dict[str, object]:
    output: dict[str, object] = {
        "item_index": index,
        "item_id": str(item.get("item_id", f"item-{index}")),
        "coarse_category_id": int(item["coarse_category_id"]),
    }
    for key in (
        "coarse_category",
        "master_category",
        "detection_label",
        "detection_confidence",
        "bbox",
    ):
        if key in item and item[key] is not None:
            output[key] = item[key]
    return output


class ProductionInferencePipeline:
    """One stable ML call for product/backend integration."""

    def __init__(
        self,
        *,
        scorer,
        calibrator,
        pipeline_version: str = PIPELINE_VERSION,
        category_mapping_version: str = "core7-v2",
        embedding_version: str = "fashionclip-512-l2-v1",
        garment_preprocessor=None,
        explanation_provider=None,
    ) -> None:
        _require_torch()
        self.scorer = scorer
        self.calibrator = calibrator
        self.pipeline_version = str(pipeline_version)
        self.category_mapping_version = str(category_mapping_version)
        self.embedding_version = str(embedding_version)
        self.garment_preprocessor = garment_preprocessor
        self.explanation_provider = explanation_provider

        if self.pipeline_version != PIPELINE_VERSION:
            raise ValueError(
                f"Expected pipeline_version={PIPELINE_VERSION!r}, got {self.pipeline_version!r}"
            )
        scorer_version = str(getattr(scorer, "scorer_version", ""))
        if not scorer_version:
            raise ValueError("scorer must expose scorer_version")
        if calibrator.scorer_version != scorer_version:
            raise ValueError(
                "Calibration/scorer version mismatch: "
                f"{calibrator.scorer_version!r} vs {scorer_version!r}"
            )
        self.scorer.eval()

    @property
    def versions(self) -> dict[str, str]:
        return {
            "pipeline_version": self.pipeline_version,
            "scorer_version": str(self.scorer.scorer_version),
            "calibration_version": str(self.calibrator.calibration_version),
            "category_mapping_version": self.category_mapping_version,
            "embedding_version": self.embedding_version,
        }

    @classmethod
    def load_from_manifest(
        cls,
        manifest_path: Path | str,
        *,
        repo_root: Path | str | None = None,
        device: str | object = "cpu",
        garment_preprocessor=None,
        explanation_provider=None,
    ) -> "ProductionInferencePipeline":
        """Load the immutable scorer + calibration bundle described by JSON."""

        _require_torch()
        manifest_source = Path(manifest_path)
        manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise ValueError("production inference manifest must be a JSON object")

        root = Path(repo_root) if repo_root is not None else manifest_source.parent.parent
        root = root.resolve()
        if manifest.get("pipeline_version") != PIPELINE_VERSION:
            raise ValueError("Unsupported pipeline_version in production manifest")

        checkpoint_path = root / str(manifest["checkpoint_path"])
        calibration_path = root / str(manifest["calibration_path"])
        expected_sha = str(manifest.get("checkpoint_sha256") or "").strip()
        if expected_sha:
            actual_sha = sha256_file(checkpoint_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    "Checkpoint SHA-256 mismatch: "
                    f"expected {expected_sha}, got {actual_sha}"
                )

        payload = load_checkpoint(checkpoint_path, map_location="cpu")
        scorer = TypeAwarePairwiseScorer.from_config(payload["config"])
        scorer.load_state_dict(payload["model_state_dict"])
        scorer.to(device).eval()
        if str(payload["scorer_version"]) != str(manifest["scorer_version"]):
            raise ValueError("Manifest/checkpoint scorer_version mismatch")

        calibrator = load_calibrator(calibration_path)
        if calibrator.scorer_version != str(payload["scorer_version"]):
            raise ValueError("Calibration artifact targets a different scorer_version")

        return cls(
            scorer=scorer,
            calibrator=calibrator,
            pipeline_version=str(manifest["pipeline_version"]),
            category_mapping_version=str(manifest["category_mapping_version"]),
            embedding_version=str(manifest["embedding_version"]),
            garment_preprocessor=garment_preprocessor,
            explanation_provider=explanation_provider,
        )

    def _validate_and_stack_items(self, items: Sequence[Mapping[str, object]]):
        if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
            raise InferenceInputError("invalid_items", "items must be a sequence of garment records")
        item_count = len(items)
        if item_count < MIN_ITEMS:
            raise InferenceInputError(
                "insufficient_garments",
                f"At least {MIN_ITEMS} garments are required",
                details={"detected_count": item_count, "minimum_required": MIN_ITEMS},
            )
        if item_count > MAX_ITEMS:
            raise InferenceInputError(
                "too_many_garments",
                f"At most {MAX_ITEMS} garments are supported; input is never silently truncated",
                details={"detected_count": item_count, "maximum_supported": MAX_ITEMS},
            )

        tensors = []
        category_ids = []
        item_ids = []
        normalized_items = []
        for index, raw_item in enumerate(items):
            if not isinstance(raw_item, Mapping):
                raise InferenceInputError(
                    "invalid_item", f"items[{index}] must be a mapping", details={"item_index": index}
                )
            if "embedding" not in raw_item:
                raise InferenceInputError(
                    "missing_embedding", f"items[{index}] is missing embedding", details={"item_index": index}
                )
            if "coarse_category_id" not in raw_item:
                raise InferenceInputError(
                    "missing_category", f"items[{index}] is missing coarse_category_id", details={"item_index": index}
                )

            vector = torch.as_tensor(raw_item["embedding"], dtype=torch.float32)
            if vector.ndim != 1 or int(vector.shape[0]) != EMBEDDING_DIM:
                raise InferenceInputError(
                    "invalid_embedding_shape",
                    f"items[{index}] embedding must have shape [{EMBEDDING_DIM}]",
                    details={"item_index": index, "shape": list(vector.shape)},
                )
            if not torch.isfinite(vector).all():
                raise InferenceInputError(
                    "non_finite_embedding", f"items[{index}] embedding contains NaN/Inf", details={"item_index": index}
                )

            raw_category_id = raw_item["coarse_category_id"]
            if isinstance(raw_category_id, bool):
                raise InferenceInputError("invalid_category", f"items[{index}] has invalid coarse_category_id")
            try:
                category_id = int(raw_category_id)
            except (TypeError, ValueError) as error:
                raise InferenceInputError(
                    "invalid_category", f"items[{index}] has invalid coarse_category_id"
                ) from error
            if not CATEGORY_MIN_ID <= category_id <= CATEGORY_MAX_ID:
                raise InferenceInputError(
                    "invalid_category",
                    f"items[{index}] coarse_category_id must be in [{CATEGORY_MIN_ID}, {CATEGORY_MAX_ID}]",
                    details={"item_index": index, "coarse_category_id": category_id},
                )

            item_id = str(raw_item.get("item_id", f"item-{index}"))
            tensors.append(vector)
            category_ids.append(category_id)
            item_ids.append(item_id)
            normalized_items.append(_clean_item_for_output(raw_item, index))

        if len(set(item_ids)) != len(item_ids):
            raise InferenceInputError("duplicate_item_id", "item_id values must be unique within an outfit")

        embeddings = torch.stack(tensors, dim=0)
        categories = torch.tensor(category_ids, dtype=torch.long)
        return embeddings, categories, item_ids, normalized_items

    def analyze_precomputed(self, items: Sequence[Mapping[str, object]]) -> dict[str, object]:
        """Run scorer -> calibration -> LOO over precomputed garment inputs."""

        embeddings, categories, item_ids, normalized_items = self._validate_and_stack_items(items)
        try:
            device = next(self.scorer.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        model_embeddings = embeddings.unsqueeze(0).to(device)
        model_categories = categories.unsqueeze(0).to(device)
        item_mask = torch.ones((1, len(items)), dtype=torch.bool, device=device)

        self.scorer.eval()
        with torch.inference_mode():
            output = self.scorer(
                item_embeddings=model_embeddings,
                coarse_category_ids=model_categories,
                item_mask=item_mask,
            )
        logit_tensor = output.get("compatibility_logit") if isinstance(output, Mapping) else None
        if not isinstance(logit_tensor, torch.Tensor) or logit_tensor.shape != (1,):
            raise RuntimeError("Scorer violated compatibility_logit [B] output contract")
        compatibility_logit = float(logit_tensor.detach().cpu()[0])
        if not math.isfinite(compatibility_logit):
            raise RuntimeError("Scorer returned non-finite compatibility_logit")

        diagnosis = diagnose_outfit(
            self.scorer,
            embeddings,
            categories,
            item_ids=item_ids,
        )
        response: dict[str, object] = {
            "status": "ok",
            "item_count": len(items),
            "items": normalized_items,
            "compatibility": {
                "compatibility_logit": compatibility_logit,
                "compatibility_score": self.calibrator.compatibility_score(compatibility_logit),
                "scorer_version": str(self.scorer.scorer_version),
                "calibration_version": str(self.calibrator.calibration_version),
            },
            "diagnosis": {
                "protocol_version": diagnosis["protocol_version"],
                "problematic_item_index": diagnosis["problematic_item_index"],
                "problematic_item_id": diagnosis["problematic_item_id"],
                "ranked_item_indices": diagnosis["ranked_item_indices"],
                "deltas_without_minus_full": diagnosis["deltas_without_minus_full"],
                "uses_two_item_extrapolation": diagnosis["uses_two_item_extrapolation"],
            },
            "versions": self.versions,
        }
        if self.explanation_provider is not None:
            response["explanation"] = self.explanation_provider.explain(response)
        return response

    def analyze_precomputed_safe(self, items: Sequence[Mapping[str, object]]) -> dict[str, object]:
        try:
            return self.analyze_precomputed(items)
        except InferenceInputError as error:
            return {"status": "error", "error": error.to_dict(), "versions": self.versions}

    def analyze_image(self, image: object) -> dict[str, object]:
        if self.garment_preprocessor is None:
            raise InferenceInputError(
                "image_preprocessor_unavailable",
                "Detection/FashionCLIP runtime adapter is not configured",
            )
        items = self.garment_preprocessor.prepare(image)
        return self.analyze_precomputed(items)

    def analyze_image_safe(self, image: object) -> dict[str, object]:
        try:
            return self.analyze_image(image)
        except InferenceInputError as error:
            return {"status": "error", "error": error.to_dict(), "versions": self.versions}
