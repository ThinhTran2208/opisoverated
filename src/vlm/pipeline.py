# -*- coding: utf-8 -*-
"""Generate, parse, and hard-validate grounded VLM explanations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .config import validate_vlm_config
from .prompt import (
    EXPLANATION_SCHEMA_VERSION,
    REQUIRED_LIMITATIONS,
    append_repair_request,
    build_qwen_messages,
)
from .schema import canonical_evidence_json, validate_vlm_evidence


RUN_SCHEMA_VERSION = "vlm-run-v1"


class VLMBackend(Protocol):
    model_id: str

    def generate(
        self,
        messages: Sequence[Mapping[str, object]],
        generation: Mapping[str, object],
    ) -> str:
        """Return raw model text for one multimodal conversation."""


def extract_json_object(raw_text: str) -> dict:
    """Extract exactly one JSON object, allowing only surrounding whitespace/fence."""

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("VLM returned an empty response")
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    if start < 0:
        raise ValueError("VLM response does not contain a JSON object")
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as error:
        raise ValueError(f"VLM response is not valid JSON: {error}") from error
    trailing = text[start + end :].strip()
    if trailing:
        raise ValueError("VLM response contains text after the JSON object")
    if not isinstance(value, dict):
        raise ValueError("VLM response JSON must be an object")
    return value


def _bounded_string(value: object, name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return normalized


def _contains_recommendation_key(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if "recommend" in str(key).lower() or _contains_recommendation_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_recommendation_key(child) for child in value)
    return False


def validate_explanation(
    explanation: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict:
    """Reject malformed output or any attempt to override canonical diagnosis."""

    normalized_evidence = validate_vlm_evidence(evidence)
    if not isinstance(explanation, Mapping):
        raise TypeError("explanation must be a mapping")
    required_keys = {
        "schema_version",
        "problematic_item_index",
        "problematic_item_id",
        "headline",
        "evidence_summary",
        "visual_observations",
        "explanation",
        "uncertainty_note",
        "limitations",
    }
    if set(explanation) != required_keys:
        raise ValueError(
            f"Explanation keys must be exactly {sorted(required_keys)}"
        )
    if explanation.get("schema_version") != EXPLANATION_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {EXPLANATION_SCHEMA_VERSION!r}"
        )

    diagnosis = normalized_evidence["diagnosis"]
    if explanation.get("problematic_item_index") != diagnosis[
        "problematic_item_index"
    ]:
        raise ValueError("VLM attempted to change problematic_item_index")
    if explanation.get("problematic_item_id") != diagnosis["problematic_item_id"]:
        raise ValueError("VLM attempted to change problematic_item_id")

    normalized: dict[str, object] = {
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "problematic_item_index": diagnosis["problematic_item_index"],
        "problematic_item_id": diagnosis["problematic_item_id"],
        "headline": _bounded_string(
            explanation.get("headline"), "headline", maximum=240
        ),
        "explanation": _bounded_string(
            explanation.get("explanation"), "explanation", maximum=2000
        ),
        "uncertainty_note": _bounded_string(
            explanation.get("uncertainty_note"),
            "uncertainty_note",
            maximum=800,
        ),
    }

    evidence_summary = explanation.get("evidence_summary")
    if not isinstance(evidence_summary, list) or not 1 <= len(evidence_summary) <= 4:
        raise ValueError("evidence_summary must contain 1-4 strings")
    normalized["evidence_summary"] = [
        _bounded_string(value, f"evidence_summary[{index}]", maximum=500)
        for index, value in enumerate(evidence_summary)
    ]

    observations = explanation.get("visual_observations")
    if not isinstance(observations, list) or len(observations) > len(
        normalized_evidence["items"]
    ):
        raise ValueError("visual_observations must be a bounded list")
    normalized_observations: list[dict] = []
    item_count = len(normalized_evidence["items"])
    for row_index, row in enumerate(observations):
        if not isinstance(row, Mapping) or set(row) != {"item_indices", "observation"}:
            raise ValueError(f"visual_observations[{row_index}] has invalid schema")
        indices = row.get("item_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError("visual observation item_indices must be non-empty")
        normalized_indices: list[int] = []
        for value in indices:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("visual observation item indices must be integers")
            if not 0 <= value < item_count:
                raise ValueError("visual observation references an unknown item")
            if value not in normalized_indices:
                normalized_indices.append(value)
        normalized_observations.append(
            {
                "item_indices": normalized_indices,
                "observation": _bounded_string(
                    row.get("observation"),
                    f"visual_observations[{row_index}].observation",
                    maximum=600,
                ),
            }
        )
    normalized["visual_observations"] = normalized_observations

    limitations = explanation.get("limitations")
    if not isinstance(limitations, list) or any(
        not isinstance(value, str) or not value.strip() for value in limitations
    ):
        raise ValueError("limitations must be a list of non-empty strings")
    normalized_limitations = list(dict.fromkeys(value.strip() for value in limitations))
    missing = sorted(set(REQUIRED_LIMITATIONS) - set(normalized_limitations))
    if missing:
        raise ValueError(f"Explanation omitted required limitations: {missing}")
    normalized["limitations"] = normalized_limitations

    # Recommendation is out of scope. This recursive gate also catches hidden
    # nested recommendation payloads outside the fixed top-level schema.
    if _contains_recommendation_key(normalized):
        raise ValueError("Explanation output may not contain recommendation fields")
    return normalized


class VLMExplanationPipeline:
    """One-case inference wrapper with deterministic schema-repair retries."""

    def __init__(self, backend: VLMBackend, config: Mapping[str, object]) -> None:
        self.backend = backend
        self.config = validate_vlm_config(config)
        if backend.model_id != self.config["model"]["id"]:
            raise ValueError("Backend model_id does not match the frozen VLM config")

    def explain(
        self,
        evidence: Mapping[str, object],
        image_refs: Sequence[str | Path],
        *,
        must_exist: bool = True,
    ) -> dict:
        normalized_evidence = validate_vlm_evidence(evidence)
        vision = self.config["vision"]
        messages = build_qwen_messages(
            normalized_evidence,
            image_refs,
            min_pixels=int(vision["min_pixels"]),
            max_pixels=int(vision["max_pixels"]),
            must_exist=must_exist,
        )
        generation = dict(self.config["generation"])
        max_retries = int(generation.pop("max_validation_retries"))

        raw_response = ""
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            raw_response = self.backend.generate(messages, generation)
            try:
                parsed = extract_json_object(raw_response)
                validated = validate_explanation(parsed, normalized_evidence)
                break
            except (TypeError, ValueError) as error:
                last_error = error
                if attempt >= max_retries:
                    raise ValueError(
                        "VLM output failed validation after "
                        f"{attempt + 1} attempt(s): {error}"
                    ) from error
                messages = append_repair_request(
                    messages,
                    raw_response=raw_response,
                    validation_error=str(error),
                )
        else:  # pragma: no cover - defensive; loop always breaks or raises.
            raise RuntimeError(f"Unreachable VLM validation state: {last_error}")

        evidence_json = canonical_evidence_json(normalized_evidence)
        run = {
            "schema_version": RUN_SCHEMA_VERSION,
            "protocol_version": self.config["protocol_version"],
            "model_id": self.backend.model_id,
            "evidence_sha256": hashlib.sha256(evidence_json.encode("utf-8")).hexdigest(),
            "evidence": normalized_evidence,
            "explanation": validated,
        }
        if self.config["output"]["include_raw_response"]:
            run["raw_response"] = raw_response
        return run
