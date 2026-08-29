# -*- coding: utf-8 -*-
"""Configuration loading for the frozen VLM explanation protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


VLM_PROTOCOL_VERSION = "vlm-explanation-v1"
CANONICAL_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
SUPPORTED_DTYPES = frozenset({"float16"})
CANONICAL_IMAGE_PIXELS = 262144
CANONICAL_MAX_NEW_TOKENS = 512


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_vlm_config(config: Mapping[str, object]) -> dict:
    """Validate and normalize the canonical Qwen3-VL Colab configuration."""

    if set(config) != {"protocol_version", "model", "vision", "generation", "output"}:
        raise ValueError("VLM config has missing or unexpected top-level fields")
    if config.get("protocol_version") != VLM_PROTOCOL_VERSION:
        raise ValueError(
            f"protocol_version must be {VLM_PROTOCOL_VERSION!r}"
        )

    model = _require_mapping(config.get("model"), "model")
    if set(model) != {"id", "dtype", "device_map", "require_cuda"}:
        raise ValueError("model has missing or unexpected fields")
    if model.get("id") != CANONICAL_MODEL_ID:
        raise ValueError(f"model.id must be {CANONICAL_MODEL_ID!r}")
    if model.get("dtype") not in SUPPORTED_DTYPES:
        raise ValueError("model.dtype must be 'float16' for the Colab T4 path")
    if model.get("device_map") != "auto":
        raise ValueError("model.device_map must be 'auto'")
    if model.get("require_cuda") is not True:
        raise ValueError("model.require_cuda must be true")

    vision = _require_mapping(config.get("vision"), "vision")
    if set(vision) != {"min_pixels", "max_pixels", "image_patch_size"}:
        raise ValueError("vision has missing or unexpected fields")
    min_pixels = _require_int(vision.get("min_pixels"), "vision.min_pixels", minimum=1)
    max_pixels = _require_int(vision.get("max_pixels"), "vision.max_pixels", minimum=1)
    if min_pixels > max_pixels:
        raise ValueError("vision.min_pixels may not exceed vision.max_pixels")
    if min_pixels % (32 * 32) or max_pixels % (32 * 32):
        raise ValueError("Qwen3-VL pixel budgets must be multiples of 32*32")
    if vision.get("image_patch_size") != 16:
        raise ValueError("Qwen3-VL qwen-vl-utils image_patch_size must be 16")
    if min_pixels != CANONICAL_IMAGE_PIXELS or max_pixels != CANONICAL_IMAGE_PIXELS:
        raise ValueError(
            f"Canonical VLM path requires exactly {CANONICAL_IMAGE_PIXELS} pixels per image"
        )

    generation = _require_mapping(config.get("generation"), "generation")
    if set(generation) != {
        "max_new_tokens",
        "do_sample",
        "num_beams",
        "repetition_penalty",
        "max_validation_retries",
    }:
        raise ValueError("generation has missing or unexpected fields")
    max_new_tokens = _require_int(
        generation.get("max_new_tokens"),
        "generation.max_new_tokens",
        minimum=64,
    )
    if max_new_tokens != CANONICAL_MAX_NEW_TOKENS:
        raise ValueError(
            f"generation.max_new_tokens must be {CANONICAL_MAX_NEW_TOKENS}"
        )
    if generation.get("do_sample") is not False:
        raise ValueError("generation.do_sample must be false for deterministic output")
    if generation.get("num_beams") != 1:
        raise ValueError("generation.num_beams must be 1")
    penalty = generation.get("repetition_penalty")
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)):
        raise ValueError("generation.repetition_penalty must be numeric")
    if float(penalty) != 1.05:
        raise ValueError("generation.repetition_penalty must be 1.05")
    retries = _require_int(
        generation.get("max_validation_retries"),
        "generation.max_validation_retries",
        minimum=0,
    )
    if retries != 1:
        raise ValueError("generation.max_validation_retries must be 1")

    output = _require_mapping(config.get("output"), "output")
    if set(output) != {"language", "include_raw_response"}:
        raise ValueError("output has missing or unexpected fields")
    if output.get("language") != "vi":
        raise ValueError("output.language must be 'vi'")
    if not isinstance(output.get("include_raw_response"), bool):
        raise ValueError("output.include_raw_response must be boolean")

    # JSON round-trip creates an ordinary detached dictionary without mutating
    # the caller's config or retaining custom Mapping implementations.
    return json.loads(json.dumps(config))


def load_vlm_config(path: str | Path) -> dict:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("VLM config must contain a JSON object")
    return validate_vlm_config(payload)
