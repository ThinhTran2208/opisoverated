# -*- coding: utf-8 -*-
"""Configuration contract for VLM explanation V2.

V2 keeps the same frozen Qwen runtime parameters as V1 while versioning the
explanation protocol independently so deploy-facing results cannot be confused
with the diagnosis-only V1 contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .config import VLM_PROTOCOL_VERSION as VLM_PROTOCOL_VERSION_V1
from .config import validate_vlm_config


VLM_PROTOCOL_VERSION_V2 = "vlm-explanation-v2"


def validate_vlm_config_v2(config: Mapping[str, object]) -> dict:
    if not isinstance(config, Mapping):
        raise ValueError("VLM V2 config must be a JSON object")
    if config.get("protocol_version") != VLM_PROTOCOL_VERSION_V2:
        raise ValueError(
            f"protocol_version must be {VLM_PROTOCOL_VERSION_V2!r}"
        )

    # Reuse the frozen V1 runtime validation without weakening or duplicating it.
    v1_view = json.loads(json.dumps(config))
    v1_view["protocol_version"] = VLM_PROTOCOL_VERSION_V1
    normalized = validate_vlm_config(v1_view)
    normalized["protocol_version"] = VLM_PROTOCOL_VERSION_V2
    return normalized


def load_vlm_config_v2(path: str | Path) -> dict:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("VLM V2 config must contain a JSON object")
    return validate_vlm_config_v2(payload)
