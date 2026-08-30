# -*- coding: utf-8 -*-
"""RF-DETR Fashionpedia adapter.

Heavy dependencies are loaded lazily so ordinary repository unit tests do not
need ``rfdetr`` or a detector checkpoint.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Sequence

from .config import DetectionConfig
from .schema import DetectionCandidate

FASHIONPEDIA_CLASS_NAMES = (
    "shirt, blouse",
    "top, t-shirt, sweatshirt",
    "sweater",
    "cardigan",
    "jacket",
    "vest",
    "pants",
    "shorts",
    "skirt",
    "coat",
    "dress",
    "jumpsuit",
    "cape",
    "glasses",
    "hat",
    "headband, head covering, hair accessory",
    "tie",
    "glove",
    "watch",
    "belt",
    "leg warmer",
    "tights, stockings",
    "sock",
    "shoe",
    "bag, wallet",
    "scarf",
    "umbrella",
    "hood",
    "collar",
    "lapel",
    "epaulette",
    "sleeve",
    "pocket",
    "neckline",
    "buckle",
    "zipper",
    "applique",
    "bead",
    "bow",
    "flower",
    "fringe",
    "ribbon",
    "rivet",
    "ruffle",
    "sequin",
    "tassel",
)

DEFAULT_CORE7_DETECTOR_LABELS = (
    "shirt, blouse",
    "top, t-shirt, sweatshirt",
    "sweater",
    "cardigan",
    "jacket",
    "vest",
    "pants",
    "shorts",
    "skirt",
    "coat",
    "dress",
    "jumpsuit",
    "cape",
    "hat",
    "shoe",
    "bag, wallet",
)


def normalize_detector_label(value: object) -> str:
    if hasattr(value, "item") and not isinstance(value, str):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        value = " ".join(str(part) for part in value)
    return " ".join(str(value).strip().lower().split())


def fashionpedia_label_for_id(class_id: int | None) -> str | None:
    if class_id is None:
        return None
    if 0 <= int(class_id) < len(FASHIONPEDIA_CLASS_NAMES):
        return FASHIONPEDIA_CLASS_NAMES[int(class_id)]
    return None


def build_detection_candidates(
    *,
    boxes: Iterable[Sequence[float]],
    confidences: Sequence[object] | None,
    class_ids: Sequence[object] | None,
    class_names: Sequence[object] | None,
    supported_labels: Sequence[str],
) -> tuple[list[DetectionCandidate], list[dict]]:
    """Normalize raw RF-DETR sequences and filter non-Core-7 object types.

    Detector labels are used only to remove Fashionpedia parts/accessories that
    cannot be scorer items. They are never mapped to a Core-7 category; the
    authoritative inference category comes from FashionCLIP zero-shot scoring.
    """

    supported = {normalize_detector_label(label) for label in supported_labels}
    accepted: list[DetectionCandidate] = []
    rejected: list[dict] = []

    for index, raw_box in enumerate(boxes):
        box = tuple(float(value) for value in raw_box)
        confidence = None
        if confidences is not None and index < len(confidences):
            raw_confidence = confidences[index]
            if raw_confidence is not None:
                confidence = float(raw_confidence)

        class_id = None
        if class_ids is not None and index < len(class_ids):
            raw_class_id = class_ids[index]
            if raw_class_id is not None:
                class_id = int(raw_class_id)

        label = ""
        if class_names is not None and index < len(class_names):
            label = normalize_detector_label(class_names[index])
        if not label:
            label = normalize_detector_label(fashionpedia_label_for_id(class_id) or "unknown")

        if label not in supported:
            rejected.append(
                {
                    "detection_index": index,
                    "reason": "unsupported_fashionpedia_object_type",
                    "detector_label": label,
                    "detector_class_id": class_id,
                    "detector_confidence": confidence,
                    "box_xyxy": list(box),
                }
            )
            continue

        accepted.append(
            DetectionCandidate(
                detection_index=index,
                box_xyxy=box,
                detector_label=label,
                detector_confidence=confidence,
                detector_class_id=class_id,
            )
        )
    return accepted, rejected


def _to_list(value: object) -> list:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)  # type: ignore[arg-type]


class RFDETRFashionpediaDetector:
    def __init__(self, config: DetectionConfig, *, device: str | None = None) -> None:
        self.config = config
        self.device = device
        self._model = None
        self._checkpoint_path: Path | None = None

    @property
    def checkpoint_path(self) -> Path | None:
        return self._checkpoint_path

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from huggingface_hub import snapshot_download
            from rfdetr import RFDETRSegSmall
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "RF-DETR runtime dependencies are missing. Install "
                "requirements-detection.txt before running detection."
            ) from error

        snapshot_dir = Path(
            snapshot_download(
                repo_id=self.config.detector_repo_id,
                allow_patterns=["*.pth"],
            )
        )
        candidates = sorted(snapshot_dir.rglob("*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"No .pth checkpoint found for {self.config.detector_repo_id}"
            )
        checkpoint = next(
            (path for path in candidates if "best_ema" in path.name.lower()),
            candidates[0],
        )
        kwargs = {"pretrain_weights": str(checkpoint)}
        if self.device:
            kwargs["device"] = self.device
        try:
            model = RFDETRSegSmall(**kwargs)
        except TypeError:
            kwargs.pop("device", None)
            model = RFDETRSegSmall(**kwargs)
        self._model = model
        self._checkpoint_path = checkpoint
        return model

    def detect(self, image_source: object) -> tuple[list[DetectionCandidate], list[dict], float]:
        model = self._load_model()
        start = time.perf_counter()
        detections = model.predict(image_source, threshold=self.config.detector_threshold)
        runtime_ms = (time.perf_counter() - start) * 1000.0

        data = getattr(detections, "data", {}) or {}
        class_names = data.get("class_name") if hasattr(data, "get") else None
        boxes = _to_list(getattr(detections, "xyxy", None))
        confidences = _to_list(getattr(detections, "confidence", None))
        class_ids = _to_list(getattr(detections, "class_id", None))
        class_names_list = _to_list(class_names)

        accepted, rejected = build_detection_candidates(
            boxes=boxes,
            confidences=confidences,
            class_ids=class_ids,
            class_names=class_names_list,
            supported_labels=self.config.supported_detector_labels,
        )
        return accepted, rejected, runtime_ms
