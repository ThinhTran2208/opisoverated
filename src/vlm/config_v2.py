# -*- coding: utf-8 -*-
"""Configuration contract for VLM explanation V2.

V2 reuses the audited V1 runtime contract where possible, but its output schema
is larger because it contains diagnosis plus three recommendation rows.  The V2
generation budget is therefore versioned independently from V1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .config import CANONICAL_MAX_NEW_TOKENS as V1_MAX_NEW_TOKENS
from .config import VLM_PROTOCOL_VERSION as VLM_PROTOCOL_VERSION_V1
from .config import validate_vlm_config


VLM_PROTOCOL_VERSION_V2 = "vlm-explanation-v2"
CANONICAL_MAX_NEW_TOKENS_V2 = 1024


def validate_vlm_config_v2(config: Mapping[str, object]) -> dict:
    if not isinstance(config, Mapping):
        raise ValueError("VLM V2 config must be a JSON object")
    if config.get("protocol_version") != VLM_PROTOCOL_VERSION_V2:
        raise ValueError(
            f"protocol_version must be {VLM_PROTOCOL_VERSION_V2!r}"
        )

    generation = config.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("generation must be a JSON object")
    if generation.get("max_new_tokens") != CANONICAL_MAX_NEW_TOKENS_V2:
        raise ValueError(
            "VLM V2 generation.max_new_tokens must be "
            f"{CANONICAL_MAX_NEW_TOKENS_V2}"
        )

    # Reuse the audited V1 validator for every shared runtime setting.  Only the
    # protocol version and generation length are translated to the V1 values for
    # validation, then restored to their V2 values in the normalized result.
    v1_view = json.loads(json.dumps(config))
    v1_view["protocol_version"] = VLM_PROTOCOL_VERSION_V1
    v1_view["generation"]["max_new_tokens"] = V1_MAX_NEW_TOKENS
    normalized = validate_vlm_config(v1_view)
    normalized["protocol_version"] = VLM_PROTOCOL_VERSION_V2
    normalized["generation"]["max_new_tokens"] = CANONICAL_MAX_NEW_TOKENS_V2
    return normalized


def load_vlm_config_v2(path: str | Path) -> dict:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("VLM V2 config must contain a JSON object")
    return validate_vlm_config_v2(payload)
