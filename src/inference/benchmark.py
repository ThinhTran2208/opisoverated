# -*- coding: utf-8 -*-
"""GPU-aware latency benchmark harness for the production image pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

from .adapters import DetectionAdapter, RemoteVLMAdapter
from .pipeline import ProductionInferencePipeline


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    normalized = [float(value) for value in values]
    if not normalized:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(normalized),
        "mean": statistics.fmean(normalized),
        "min": min(normalized),
        "p50": _percentile(normalized, 0.50),
        "p95": _percentile(normalized, 0.95),
        "p99": _percentile(normalized, 0.99),
        "max": max(normalized),
    }


def _git_head() -> str | None:
    value = os.environ.get("GITHUB_SHA")
    if value:
        return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _hardware_metadata(torch_module) -> dict[str, object]:
    metadata: dict[str, object] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": getattr(torch_module, "__version__", None),
        "cuda_available": bool(torch_module.cuda.is_available()),
        "cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
    }
    if torch_module.cuda.is_available():
        device_index = torch_module.cuda.current_device()
        properties = torch_module.cuda.get_device_properties(device_index)
        metadata["gpu"] = {
            "index": int(device_index),
            "name": torch_module.cuda.get_device_name(device_index),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    return metadata


def _sync_cuda(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


class _TimedVLMAdapter:
    def __init__(self, delegate, torch_module) -> None:
        self.delegate = delegate
        self.torch = torch_module
        self.last_ms = 0.0

    def explain(self, loo_result, garments, crop_image_refs, *, sample_id):
        _sync_cuda(self.torch)
        start = time.perf_counter()
        try:
            return self.delegate.explain(
                loo_result,
                garments,
                crop_image_refs,
                sample_id=sample_id,
            )
        finally:
            _sync_cuda(self.torch)
            self.last_ms = (time.perf_counter() - start) * 1000.0


def _detector_runtime_ms(context) -> float | None:
    detection = context.metadata.get("detection") if isinstance(context.metadata, Mapping) else None
    if not isinstance(detection, Mapping):
        return None
    detector = detection.get("detector")
    if not isinstance(detector, Mapping):
        return None
    value = detector.get("runtime_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def run_benchmark(
    *,
    image_paths: Sequence[Path],
    manifest_path: Path,
    detection_config_path: Path,
    device: str,
    warmup_runs: int,
    measured_runs: int,
    vlm_service_url: str | None,
    vlm_timeout_seconds: float,
    require_cuda: bool,
    require_vlm: bool,
) -> dict[str, object]:
    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("Require warmup_runs >= 0 and measured_runs >= 1")
    if not image_paths:
        raise ValueError("At least one --image is required")
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    try:
        import torch
        from PIL import Image
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Benchmark requires the inference-core runtime dependencies"
        ) from error

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but CUDA is not available")
    if require_vlm and not vlm_service_url:
        raise RuntimeError("--require-vlm requires --vlm-service-url")

    detection_adapter = DetectionAdapter.from_config(
        detection_config_path,
        device=device,
    )
    timed_vlm = None
    if vlm_service_url:
        timed_vlm = _TimedVLMAdapter(
            RemoteVLMAdapter(
                vlm_service_url,
                timeout_seconds=vlm_timeout_seconds,
            ),
            torch,
        )

    repo_root = manifest_path.resolve().parent.parent
    pipeline = ProductionInferencePipeline.load_from_manifest(
        manifest_path,
        repo_root=repo_root,
        device=device,
        detection_adapter=detection_adapter,
        vlm_adapter=timed_vlm,
    )

    stage_values: dict[str, list[float]] = {
        "image_decode_ms": [],
        "detection_adapter_ms": [],
        "detector_model_ms": [],
        "post_detector_detection_ms": [],
        "scorer_calibration_loo_ms": [],
        "vlm_ms": [],
        "total_ml_ms": [],
        "end_to_end_ms": [],
    }
    records: list[dict[str, object]] = []

    def one_run(image_path: Path, *, measured: bool) -> None:
        _sync_cuda(torch)
        overall_start = time.perf_counter()
        decode_start = time.perf_counter()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        decode_ms = (time.perf_counter() - decode_start) * 1000.0

        _sync_cuda(torch)
        detection_start = time.perf_counter()
        context = detection_adapter.prepare(image)
        _sync_cuda(torch)
        detection_ms = (time.perf_counter() - detection_start) * 1000.0

        try:
            if timed_vlm is not None:
                timed_vlm.last_ms = 0.0
            _sync_cuda(torch)
            core_start = time.perf_counter()
            result = pipeline.analyze_context(
                context,
                include_explanation=timed_vlm is not None,
            )
            _sync_cuda(torch)
            core_with_vlm_ms = (time.perf_counter() - core_start) * 1000.0
            vlm_ms = timed_vlm.last_ms if timed_vlm is not None else 0.0
            core_ms = max(0.0, core_with_vlm_ms - vlm_ms)
            detector_ms = _detector_runtime_ms(context)
            total_ml_ms = detection_ms + core_with_vlm_ms
            end_to_end_ms = (time.perf_counter() - overall_start) * 1000.0

            if result.get("status") != "ok":
                raise RuntimeError(f"Benchmark inference failed: {result}")
            if require_vlm and "explanation" not in result:
                raise RuntimeError("Benchmark requires VLM explanation but none was returned")

            if measured:
                stage_values["image_decode_ms"].append(decode_ms)
                stage_values["detection_adapter_ms"].append(detection_ms)
                if detector_ms is not None:
                    stage_values["detector_model_ms"].append(detector_ms)
                    stage_values["post_detector_detection_ms"].append(
                        max(0.0, detection_ms - detector_ms)
                    )
                stage_values["scorer_calibration_loo_ms"].append(core_ms)
                if timed_vlm is not None:
                    stage_values["vlm_ms"].append(vlm_ms)
                stage_values["total_ml_ms"].append(total_ml_ms)
                stage_values["end_to_end_ms"].append(end_to_end_ms)
                records.append(
                    {
                        "image": str(image_path),
                        "item_count": int(result["item_count"]),
                        "image_decode_ms": decode_ms,
                        "detection_adapter_ms": detection_ms,
                        "detector_model_ms": detector_ms,
                        "scorer_calibration_loo_ms": core_ms,
                        "vlm_ms": vlm_ms if timed_vlm is not None else None,
                        "total_ml_ms": total_ml_ms,
                        "end_to_end_ms": end_to_end_ms,
                    }
                )
        finally:
            context.close()

    for index in range(warmup_runs):
        one_run(image_paths[index % len(image_paths)], measured=False)
    for _ in range(measured_runs):
        for path in image_paths:
            one_run(path, measured=True)

    if torch.cuda.is_available():
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    else:
        peak_allocated = None
        peak_reserved = None

    return {
        "schema_version": "production-latency-benchmark-v1",
        "git_head": _git_head(),
        "pipeline_versions": pipeline.versions,
        "device": device,
        "require_cuda": require_cuda,
        "vlm_mode": "remote" if timed_vlm is not None else "disabled",
        "warmup_runs": warmup_runs,
        "measured_runs_per_image": measured_runs,
        "image_count": len(image_paths),
        "sample_count": len(records),
        "hardware": _hardware_metadata(torch),
        "peak_cuda_memory": {
            "allocated_bytes": peak_allocated,
            "reserved_bytes": peak_reserved,
        },
        "latency_ms": {
            name: _summary(values) for name, values in stage_values.items()
        },
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True, dest="images")
    parser.add_argument(
        "--manifest",
        default="configs/production_inference_v1.json",
    )
    parser.add_argument(
        "--detection-config",
        default="configs/detection_rfdetr_fashionclip_core7_v1.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--vlm-service-url")
    parser.add_argument("--vlm-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-vlm", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_benchmark(
        image_paths=[Path(value).resolve() for value in args.images],
        manifest_path=Path(args.manifest).resolve(),
        detection_config_path=Path(args.detection_config).resolve(),
        device=args.device,
        warmup_runs=args.warmup_runs,
        measured_runs=args.runs,
        vlm_service_url=args.vlm_service_url,
        vlm_timeout_seconds=args.vlm_timeout_seconds,
        require_cuda=args.require_cuda,
        require_vlm=args.require_vlm,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["latency_ms"], indent=2))
    print(f"Saved benchmark report: {output}")


if __name__ == "__main__":
    main()
