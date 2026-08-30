# -*- coding: utf-8 -*-
"""Production boundary: DetectionAdapter -> InferenceContext -> scorer/LOO -> VLMAdapter."""

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

from .context import InferenceContext


PIPELINE_VERSION = "outfit-production-inference-v1"
MIN_ITEMS = 3
MAX_ITEMS = 8
EMBEDDING_DIM = 512
EMBEDDING_NORM_TOLERANCE = 1e-3
CATEGORY_MIN_ID = 1
CATEGORY_MAX_ID = 7


class InferenceInputError(ValueError):
    """Expected user/runtime input error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
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


def _clean_item_for_output(
    item: Mapping[str, object], index: int
) -> dict[str, object]:
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
        "crop_bbox",
        "category_similarity",
        "category_margin",
    ):
        if key in item and item[key] is not None:
            output[key] = item[key]
    return output


class ProductionInferencePipeline:
    """One stable ML call for product/backend integration.

    Canonical image flow:

    DetectionAdapter -> InferenceContext -> scorer -> calibration -> LOO
                     -> VLMAdapter(raw LOO, garments, crop refs)
    """

    def __init__(
        self,
        *,
        scorer,
        calibrator,
        pipeline_version: str = PIPELINE_VERSION,
        category_mapping_version: str = "core7-v2",
        embedding_version: str = "fashionclip-512-l2-v1",
        detection_adapter=None,
        vlm_adapter=None,
        # Compatibility aliases for the first production-inference draft.
        garment_preprocessor=None,
        explanation_provider=None,
    ) -> None:
        _require_torch()
        if detection_adapter is not None and garment_preprocessor is not None:
            raise ValueError("Provide detection_adapter or garment_preprocessor, not both")
        if vlm_adapter is not None and explanation_provider is not None:
            raise ValueError("Provide vlm_adapter or explanation_provider, not both")

        self.scorer = scorer
        self.calibrator = calibrator
        self.pipeline_version = str(pipeline_version)
        self.category_mapping_version = str(category_mapping_version)
        self.embedding_version = str(embedding_version)
        self.detection_adapter = detection_adapter or garment_preprocessor
        self.vlm_adapter = vlm_adapter or explanation_provider
        # Old attributes remain readable for downstream code during migration.
        self.garment_preprocessor = self.detection_adapter
        self.explanation_provider = self.vlm_adapter

        if self.pipeline_version != PIPELINE_VERSION:
            raise ValueError(
                f"Expected pipeline_version={PIPELINE_VERSION!r}, "
                f"got {self.pipeline_version!r}"
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
        detection_adapter=None,
        vlm_adapter=None,
        garment_preprocessor=None,
        explanation_provider=None,
    ) -> "ProductionInferencePipeline":
        """Load the immutable scorer + calibration bundle described by JSON."""

        _require_torch()
        manifest_source = Path(manifest_path)
        manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise ValueError("production inference manifest must be a JSON object")

        root = (
            Path(repo_root)
            if repo_root is not None
            else manifest_source.parent.parent
        ).resolve()
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
            detection_adapter=detection_adapter,
            vlm_adapter=vlm_adapter,
            garment_preprocessor=garment_preprocessor,
            explanation_provider=explanation_provider,
        )

    def _validate_item_count(self, item_count: int) -> None:
        if item_count < MIN_ITEMS:
            raise InferenceInputError(
                "insufficient_garments",
                f"At least {MIN_ITEMS} garments are required",
                details={
                    "detected_count": item_count,
                    "minimum_required": MIN_ITEMS,
                },
            )
        if item_count > MAX_ITEMS:
            raise InferenceInputError(
                "too_many_garments",
                f"At most {MAX_ITEMS} garments are supported; input is never silently truncated",
                details={
                    "detected_count": item_count,
                    "maximum_supported": MAX_ITEMS,
                },
            )

    def _validate_embedding_matrix(self, embeddings):
        matrix = torch.as_tensor(embeddings, dtype=torch.float32)
        if matrix.ndim != 2 or int(matrix.shape[1]) != EMBEDDING_DIM:
            raise InferenceInputError(
                "invalid_embedding_shape",
                f"embeddings must have shape [N, {EMBEDDING_DIM}]",
                details={"shape": list(matrix.shape)},
            )
        if not bool(torch.isfinite(matrix).all()):
            raise InferenceInputError(
                "non_finite_embedding", "embeddings contain NaN/Inf"
            )
        norms = torch.linalg.vector_norm(matrix, dim=1)
        invalid = torch.nonzero(
            torch.abs(norms - 1.0) > EMBEDDING_NORM_TOLERANCE,
            as_tuple=False,
        ).flatten()
        if int(invalid.numel()) > 0:
            index = int(invalid[0])
            raise InferenceInputError(
                "embedding_not_l2_normalized",
                f"garment embedding at index {index} must be L2-normalized",
                details={
                    "item_index": index,
                    "embedding_norm": float(norms[index]),
                    "expected_norm": 1.0,
                    "tolerance": EMBEDDING_NORM_TOLERANCE,
                },
            )
        return matrix

    def _validate_context(self, context: InferenceContext):
        if not isinstance(context, InferenceContext):
            raise TypeError("detection_adapter.prepare() must return InferenceContext")

        item_count = len(context)
        self._validate_item_count(item_count)
        embeddings = self._validate_embedding_matrix(context.embeddings)
        categories = torch.as_tensor(context.categories)
        if categories.ndim != 1 or int(categories.shape[0]) != item_count:
            raise InferenceInputError(
                "invalid_category_shape",
                "categories must have shape [N]",
                details={"shape": list(categories.shape)},
            )
        if categories.dtype == torch.bool:
            raise InferenceInputError("invalid_category", "categories may not be boolean")
        if categories.is_floating_point() and not bool(torch.equal(categories, categories.round())):
            raise InferenceInputError("invalid_category", "categories must contain integer IDs")
        categories = categories.long()
        if bool(torch.any(categories < CATEGORY_MIN_ID)) or bool(
            torch.any(categories > CATEGORY_MAX_ID)
        ):
            raise InferenceInputError(
                "invalid_category",
                f"coarse category IDs must be in [{CATEGORY_MIN_ID}, {CATEGORY_MAX_ID}]",
            )

        item_ids: list[str] = []
        normalized_items: list[dict[str, object]] = []
        for index, raw_item in enumerate(context.garments):
            if not isinstance(raw_item, Mapping):
                raise InferenceInputError(
                    "invalid_item", f"garments[{index}] must be a mapping"
                )
            if "coarse_category_id" not in raw_item:
                raise InferenceInputError(
                    "missing_category",
                    f"garments[{index}] is missing coarse_category_id",
                )
            raw_category_id = raw_item["coarse_category_id"]
            if isinstance(raw_category_id, bool):
                raise InferenceInputError(
                    "invalid_category", f"garments[{index}] has invalid coarse_category_id"
                )
            try:
                category_id = int(raw_category_id)
            except (TypeError, ValueError) as error:
                raise InferenceInputError(
                    "invalid_category", f"garments[{index}] has invalid coarse_category_id"
                ) from error
            if category_id != int(categories[index]):
                raise InferenceInputError(
                    "category_context_mismatch",
                    f"garments[{index}] category does not match InferenceContext.categories",
                )
            item_id = str(raw_item.get("item_id", f"item-{index}"))
            item_ids.append(item_id)
            normalized_items.append(_clean_item_for_output(raw_item, index))

        if len(set(item_ids)) != len(item_ids):
            raise InferenceInputError(
                "duplicate_item_id", "item_id values must be unique within an outfit"
            )
        return embeddings, categories, item_ids, normalized_items

    def _context_from_precomputed(
        self, items: Sequence[Mapping[str, object]]
    ) -> InferenceContext:
        if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
            raise InferenceInputError(
                "invalid_items", "items must be a sequence of garment records"
            )
        self._validate_item_count(len(items))

        garments: list[dict[str, object]] = []
        embeddings = []
        category_ids = []
        for index, raw_item in enumerate(items):
            if not isinstance(raw_item, Mapping):
                raise InferenceInputError(
                    "invalid_item",
                    f"items[{index}] must be a mapping",
                    details={"item_index": index},
                )
            if "embedding" not in raw_item:
                raise InferenceInputError(
                    "missing_embedding",
                    f"items[{index}] is missing embedding",
                    details={"item_index": index},
                )
            if "coarse_category_id" not in raw_item:
                raise InferenceInputError(
                    "missing_category",
                    f"items[{index}] is missing coarse_category_id",
                    details={"item_index": index},
                )
            embeddings.append(raw_item["embedding"])
            category_ids.append(raw_item["coarse_category_id"])
            garments.append(
                {key: value for key, value in raw_item.items() if key != "embedding"}
            )

        try:
            matrix = torch.stack(
                [torch.as_tensor(value, dtype=torch.float32) for value in embeddings],
                dim=0,
            )
        except RuntimeError as error:
            raise InferenceInputError(
                "invalid_embedding_shape",
                f"all item embeddings must have shape [{EMBEDDING_DIM}]",
            ) from error
        return InferenceContext(
            garments=garments,
            embeddings=matrix,
            categories=category_ids,
            metadata={"source": "precomputed"},
        )

    def analyze_context(
        self,
        context: InferenceContext,
        *,
        include_explanation: bool = True,
    ) -> dict[str, object]:
        """Run the stable scorer/calibration/LOO core on one inference context."""

        embeddings, categories, item_ids, normalized_items = self._validate_context(context)
        try:
            device = next(self.scorer.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        model_embeddings = embeddings.unsqueeze(0).to(device)
        model_categories = categories.unsqueeze(0).to(device)
        item_mask = torch.ones((1, len(context)), dtype=torch.bool, device=device)

        self.scorer.eval()
        with torch.inference_mode():
            output = self.scorer(
                item_embeddings=model_embeddings,
                coarse_category_ids=model_categories,
                item_mask=item_mask,
            )
        logit_tensor = (
            output.get("compatibility_logit")
            if isinstance(output, Mapping)
            else None
        )
        if not isinstance(logit_tensor, torch.Tensor) or logit_tensor.shape != (1,):
            raise RuntimeError("Scorer violated compatibility_logit [B] output contract")
        compatibility_logit = float(logit_tensor.detach().cpu()[0])
        if not math.isfinite(compatibility_logit):
            raise RuntimeError("Scorer returned non-finite compatibility_logit")

        # Keep the complete raw LOO result in memory.  The VLM adapter consumes
        # this object directly instead of reverse-engineering the public response.
        diagnosis = diagnose_outfit(
            self.scorer,
            embeddings,
            categories,
            item_ids=item_ids,
        )

        response: dict[str, object] = {
            "status": "ok",
            "request_id": context.request_id,
            "item_count": len(context),
            "items": normalized_items,
            "compatibility": {
                "compatibility_logit": compatibility_logit,
                "compatibility_score": self.calibrator.compatibility_score(
                    compatibility_logit
                ),
                "scorer_version": str(self.scorer.scorer_version),
                "calibration_version": str(self.calibrator.calibration_version),
            },
            "diagnosis": {
                "protocol_version": diagnosis["protocol_version"],
                "problematic_item_index": diagnosis["problematic_item_index"],
                "problematic_item_id": diagnosis["problematic_item_id"],
                "ranked_item_indices": diagnosis["ranked_item_indices"],
                "deltas_without_minus_full": diagnosis["deltas_without_minus_full"],
                "uses_two_item_extrapolation": diagnosis[
                    "uses_two_item_extrapolation"
                ],
            },
            "versions": self.versions,
        }
        if context.metadata:
            response["preprocessing"] = dict(context.metadata)

        if include_explanation and self.vlm_adapter is not None:
            if len(context.crop_image_refs) != len(context):
                raise RuntimeError(
                    "VLM explanation requires one crop image reference per garment"
                )
            response["explanation"] = self.vlm_adapter.explain(
                diagnosis,
                context.garments,
                context.crop_image_refs,
                sample_id=context.request_id,
            )
        return response

    def analyze_precomputed(
        self, items: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        """Compatibility endpoint for already-computed FashionCLIP records."""

        context = self._context_from_precomputed(items)
        return self.analyze_context(context, include_explanation=False)

    def analyze_precomputed_safe(
        self, items: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        try:
            return self.analyze_precomputed(items)
        except InferenceInputError as error:
            return {
                "status": "error",
                "error": error.to_dict(),
                "versions": self.versions,
            }

    def analyze_image(self, image: object) -> dict[str, object]:
        if self.detection_adapter is None:
            raise InferenceInputError(
                "image_preprocessor_unavailable",
                "Detection/FashionCLIP runtime adapter is not configured",
            )
        context = self.detection_adapter.prepare(image)
        if not isinstance(context, InferenceContext):
            raise TypeError("detection_adapter.prepare() must return InferenceContext")
        try:
            return self.analyze_context(context, include_explanation=True)
        finally:
            context.close()

    def analyze_image_safe(self, image: object) -> dict[str, object]:
        try:
            return self.analyze_image(image)
        except InferenceInputError as error:
            return {
                "status": "error",
                "error": error.to_dict(),
                "versions": self.versions,
            }
