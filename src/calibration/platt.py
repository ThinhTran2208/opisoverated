# -*- coding: utf-8 -*-
"""Monotonic Platt calibration for scorer logits.

Calibration V1 maps the frozen scorer's raw compatibility logit ``x`` to
``sigmoid(scale * x + bias)`` with a strictly positive ``scale``.  The positive
scale preserves the scorer ranking contract: a larger raw logit can never map
to a lower calibrated compatibility score.

The module intentionally uses only the Python standard library so artifact
loading and product-score transformation remain available in lightweight
runtime/CI environments.  Fitting uses a small deterministic Adam optimizer
implemented below and does not require sklearn/scipy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence


CALIBRATION_VERSION = "platt-logistic-v1"
CALIBRATION_METHOD = "positive-scale-platt"
DEFAULT_ECE_BINS = 10


class CalibrationContractError(ValueError):
    """Raised when a calibration input or artifact violates the V1 contract."""


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise CalibrationContractError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CalibrationContractError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise CalibrationContractError(f"{name} must be finite")
    return result


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _binary_log_loss(probability: float, label: float) -> float:
    eps = 1e-15
    p = min(max(probability, eps), 1.0 - eps)
    return -(label * math.log(p) + (1.0 - label) * math.log(1.0 - p))


def calibration_metrics(
    logits: Sequence[int | float],
    labels: Sequence[int | float],
    *,
    scale: float = 1.0,
    bias: float = 0.0,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, float | int]:
    """Return NLL, Brier score and equal-width ECE for binary logits."""

    if len(logits) != len(labels):
        raise CalibrationContractError("logits and labels must have equal length")
    if not logits:
        raise CalibrationContractError("calibration metrics require at least one sample")
    if ece_bins < 2:
        raise CalibrationContractError("ece_bins must be >= 2")

    scale = _finite_float(scale, name="scale")
    bias = _finite_float(bias, name="bias")
    if scale <= 0.0:
        raise CalibrationContractError("scale must be > 0 to preserve ranking")

    probabilities: list[float] = []
    normalized_labels: list[float] = []
    for index, (raw_logit, raw_label) in enumerate(zip(logits, labels)):
        logit = _finite_float(raw_logit, name=f"logits[{index}]")
        label = _finite_float(raw_label, name=f"labels[{index}]")
        if label not in (0.0, 1.0):
            raise CalibrationContractError(f"labels[{index}] must be 0 or 1")
        probabilities.append(_sigmoid(scale * logit + bias))
        normalized_labels.append(label)

    count = len(probabilities)
    nll = sum(
        _binary_log_loss(probability, label)
        for probability, label in zip(probabilities, normalized_labels)
    ) / count
    brier = sum(
        (probability - label) ** 2
        for probability, label in zip(probabilities, normalized_labels)
    ) / count

    ece = 0.0
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        members = [
            index
            for index, probability in enumerate(probabilities)
            if probability >= lower
            and (probability < upper or (bin_index == ece_bins - 1 and probability <= 1.0))
        ]
        if not members:
            continue
        mean_probability = sum(probabilities[index] for index in members) / len(members)
        mean_label = sum(normalized_labels[index] for index in members) / len(members)
        ece += (len(members) / count) * abs(mean_probability - mean_label)

    accuracy = sum(
        (probability >= 0.5) == (label == 1.0)
        for probability, label in zip(probabilities, normalized_labels)
    ) / count

    return {
        "sample_count": count,
        "nll": nll,
        "brier": brier,
        f"ece_{ece_bins}": ece,
        "mean_probability": sum(probabilities) / count,
        "accuracy_at_0_5": accuracy,
    }


class PlattCalibrator:
    """Immutable product-score mapping loaded from a versioned JSON artifact."""

    def __init__(
        self,
        *,
        scale: float,
        bias: float,
        scorer_version: str,
        calibration_version: str = CALIBRATION_VERSION,
        method: str = CALIBRATION_METHOD,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.scale = _finite_float(scale, name="scale")
        self.bias = _finite_float(bias, name="bias")
        if self.scale <= 0.0:
            raise CalibrationContractError("scale must be > 0 to preserve scorer ranking")
        self.scorer_version = str(scorer_version).strip()
        self.calibration_version = str(calibration_version).strip()
        self.method = str(method).strip()
        if not self.scorer_version:
            raise CalibrationContractError("scorer_version is required")
        if self.calibration_version != CALIBRATION_VERSION:
            raise CalibrationContractError(
                f"Expected calibration_version={CALIBRATION_VERSION!r}, "
                f"got {self.calibration_version!r}"
            )
        if self.method != CALIBRATION_METHOD:
            raise CalibrationContractError(
                f"Expected method={CALIBRATION_METHOD!r}, got {self.method!r}"
            )
        self.metadata = dict(metadata or {})

    def probability(self, compatibility_logit: int | float) -> float:
        """Map one raw scorer logit to calibrated compatibility in [0, 1]."""

        logit = _finite_float(compatibility_logit, name="compatibility_logit")
        return _sigmoid(self.scale * logit + self.bias)

    def compatibility_score(self, compatibility_logit: int | float) -> int:
        """Return the user-facing integer compatibility score in [0, 100]."""

        probability = self.probability(compatibility_logit)
        return min(100, max(0, int(math.floor(probability * 100.0 + 0.5))))

    def transform_many(self, logits: Sequence[int | float]) -> list[float]:
        return [self.probability(value) for value in logits]

    def to_artifact(self) -> dict[str, object]:
        return {
            "calibration_version": self.calibration_version,
            "method": self.method,
            "scorer_version": self.scorer_version,
            "scale": self.scale,
            "bias": self.bias,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, object]) -> "PlattCalibrator":
        if not isinstance(artifact, Mapping):
            raise CalibrationContractError("calibration artifact must be a mapping")
        required = (
            "calibration_version",
            "method",
            "scorer_version",
            "scale",
            "bias",
        )
        missing = [key for key in required if key not in artifact]
        if missing:
            raise CalibrationContractError(f"calibration artifact missing keys: {missing}")
        metadata = artifact.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise CalibrationContractError("artifact metadata must be a mapping")
        return cls(
            scale=artifact["scale"],
            bias=artifact["bias"],
            scorer_version=str(artifact["scorer_version"]),
            calibration_version=str(artifact["calibration_version"]),
            method=str(artifact["method"]),
            metadata=metadata,
        )


def save_calibrator(calibrator: PlattCalibrator, path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(calibrator.to_artifact(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_calibrator(path: Path | str) -> PlattCalibrator:
    source = Path(path)
    artifact = json.loads(source.read_text(encoding="utf-8"))
    return PlattCalibrator.from_artifact(artifact)


def fit_platt_calibrator(
    logits: Sequence[int | float],
    labels: Sequence[int | float],
    *,
    scorer_version: str,
    metadata: Mapping[str, object] | None = None,
    learning_rate: float = 0.03,
    max_steps: int = 5000,
    tolerance: float = 1e-10,
) -> PlattCalibrator:
    """Fit positive-scale Platt parameters with deterministic Adam.

    ``scale = exp(log_scale)`` guarantees monotonicity.  This fitter is intended
    for a frozen validation prediction set; model weights are never touched.
    """

    if len(logits) != len(labels):
        raise CalibrationContractError("logits and labels must have equal length")
    if len(logits) < 2:
        raise CalibrationContractError("calibration fitting requires at least two samples")
    if learning_rate <= 0.0:
        raise CalibrationContractError("learning_rate must be > 0")
    if max_steps < 1:
        raise CalibrationContractError("max_steps must be >= 1")

    xs = [_finite_float(value, name=f"logits[{index}]") for index, value in enumerate(logits)]
    ys: list[float] = []
    for index, value in enumerate(labels):
        label = _finite_float(value, name=f"labels[{index}]")
        if label not in (0.0, 1.0):
            raise CalibrationContractError(f"labels[{index}] must be 0 or 1")
        ys.append(label)
    if len(set(ys)) != 2:
        raise CalibrationContractError("calibration fitting requires both binary classes")

    log_scale = 0.0
    bias = 0.0
    first_moment_scale = first_moment_bias = 0.0
    second_moment_scale = second_moment_bias = 0.0
    beta1, beta2 = 0.9, 0.999
    adam_epsilon = 1e-8
    count = len(xs)
    completed_steps = 0

    for step in range(1, max_steps + 1):
        scale = math.exp(log_scale)
        grad_log_scale = 0.0
        grad_bias = 0.0
        for x_value, label in zip(xs, ys):
            probability = _sigmoid(scale * x_value + bias)
            residual = probability - label
            grad_log_scale += residual * scale * x_value
            grad_bias += residual
        grad_log_scale /= count
        grad_bias /= count

        gradient_norm = math.hypot(grad_log_scale, grad_bias)
        completed_steps = step
        if gradient_norm <= tolerance:
            break

        first_moment_scale = beta1 * first_moment_scale + (1.0 - beta1) * grad_log_scale
        first_moment_bias = beta1 * first_moment_bias + (1.0 - beta1) * grad_bias
        second_moment_scale = beta2 * second_moment_scale + (1.0 - beta2) * (grad_log_scale**2)
        second_moment_bias = beta2 * second_moment_bias + (1.0 - beta2) * (grad_bias**2)

        corrected_m_scale = first_moment_scale / (1.0 - beta1**step)
        corrected_m_bias = first_moment_bias / (1.0 - beta1**step)
        corrected_v_scale = second_moment_scale / (1.0 - beta2**step)
        corrected_v_bias = second_moment_bias / (1.0 - beta2**step)

        log_scale -= learning_rate * corrected_m_scale / (math.sqrt(corrected_v_scale) + adam_epsilon)
        bias -= learning_rate * corrected_m_bias / (math.sqrt(corrected_v_bias) + adam_epsilon)

        # Avoid overflow for pathological inputs while keeping a broad useful range.
        log_scale = min(8.0, max(-8.0, log_scale))
        bias = min(50.0, max(-50.0, bias))

    fitted_scale = math.exp(log_scale)
    fit_metrics = calibration_metrics(xs, ys, scale=fitted_scale, bias=bias)
    artifact_metadata = dict(metadata or {})
    artifact_metadata.update(
        {
            "fit_sample_count": count,
            "fit_optimizer": "deterministic-adam-v1",
            "fit_steps": completed_steps,
            "fit_metrics": fit_metrics,
        }
    )
    return PlattCalibrator(
        scale=fitted_scale,
        bias=bias,
        scorer_version=scorer_version,
        metadata=artifact_metadata,
    )
