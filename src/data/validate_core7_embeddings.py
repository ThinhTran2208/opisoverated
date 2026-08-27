# -*- coding: utf-8 -*-
"""Validate Core-7 V2 positives against the frozen FashionCLIP cache.

The validation report is an evidence artifact, not a generic PASS flag. It is
bound to the exact Core-7 mapping, embedding cache, embedding manifest,
positive JSONL files, and item-metadata JSONL files through SHA-256 digests.
NB4 can therefore reject a stale report when any validated input has changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

from .prepare_core7_dataset_v2 import load_category_mapping_v2

try:
    import torch
except ModuleNotFoundError:  # Keep lightweight schema helpers importable in CI.
    torch = None


EXPECTED_MODEL_ID = "patrickjohncyh/fashion-clip"
EXPECTED_EMBEDDING_VERSION = "fashionclip-512-l2-v1"
EXPECTED_EMBEDDING_DIM = 512
EXPECTED_CATEGORY_MAPPING_VERSION = "core7-v2"
EXPECTED_ITEM_METADATA_VERSION = "core7-item-metadata-v1"
DEFAULT_MANIFEST_FILENAME = "embedding_manifest_v1.json"
DEFAULT_MIN_OUTFIT_ITEMS = 3
DEFAULT_NORM_TOLERANCE = 1e-2
MAX_EXAMPLES = 50


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_canonical_json(payload: Mapping[str, object]) -> str:
    """Hash JSON semantics independently of whitespace and key order."""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def read_json(path: Path | str) -> dict:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {source}")
    return payload


def fingerprint_category_mapping(
    mapping_path: Path | str,
    *,
    expected_mapping_version: str = EXPECTED_CATEGORY_MAPPING_VERSION,
) -> dict:
    """Fingerprint the V2 override, frozen V1 base, and resolved mapping.

    Core-7 V2 is a delta on top of V1. Hashing only the V2 JSON would miss a
    mutated base mapping, while hashing only the files would make semantic
    comparisons depend on formatting. The resolved digest closes both gaps.
    """

    source = Path(mapping_path)
    mapping_metadata, resolved = load_category_mapping_v2(source)
    payload = read_json(source)
    mapping_version = str(mapping_metadata.get("mapping_version", ""))
    if mapping_version != expected_mapping_version:
        raise ValueError(
            "Category mapping version mismatch: "
            f"expected={expected_mapping_version}, actual={mapping_version!r}"
        )
    base_name = payload.get("base_mapping")
    if not isinstance(base_name, str) or not base_name.strip():
        raise ValueError("Core-7 V2 mapping must declare base_mapping")
    base_source = Path(base_name)
    if not base_source.is_absolute():
        base_source = source.parent / base_source
    return {
        "mapping_version": mapping_version,
        "path": str(source),
        "sha256": sha256_file(source),
        "base_mapping_version": mapping_metadata["base_mapping_version"],
        "base_path": str(base_source),
        "base_sha256": sha256_file(base_source),
        "resolved_mapping_count": len(resolved),
        "resolved_mapping_sha256": _sha256_canonical_json(resolved),
    }


def read_jsonl(path: Path | str) -> list[dict]:
    """Read a JSONL file and reject malformed/non-object rows."""

    source = Path(path)
    records: list[dict] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {source}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at {source}:{line_number}"
                )
            records.append(record)
    return records


def write_jsonl(records: Iterable[dict], path: Path | str) -> int:
    """Write compact UTF-8 JSONL and return the number of rows."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
            stream.write("\n")
            count += 1
    return count


def load_embedding_cache(path: Path | str) -> Mapping[str, object]:
    """Load a team-owned PyTorch cache on CPU with the safest supported mode."""

    if torch is None:
        raise RuntimeError(
            "PyTorch is required to read the .pt embedding cache. "
            "Install torch or run this stage in Google Colab."
        )

    source = Path(path)
    try:
        cache = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        cache = torch.load(source, map_location="cpu")

    if not isinstance(cache, Mapping):
        raise ValueError("Embedding cache must contain a mapping/dictionary")
    return cache


def inspect_embedding_cache(
    cache: Mapping[str, object],
    *,
    expected_model_id: str = EXPECTED_MODEL_ID,
    expected_dim: int = EXPECTED_EMBEDDING_DIM,
    norm_tolerance: float = DEFAULT_NORM_TOLERANCE,
) -> tuple[dict, set[str]]:
    """Validate cache schema/content and return the usable item-ID set."""

    if torch is None:
        raise RuntimeError(
            "PyTorch is required to validate embedding tensors. "
            "Install torch or run this stage in Google Colab."
        )

    required_keys = {"model_id", "item_ids", "embeddings"}
    missing_keys = sorted(required_keys - set(cache))
    if missing_keys:
        raise ValueError(f"Embedding cache is missing keys: {missing_keys}")

    model_id = str(cache["model_id"])
    item_ids = [str(item_id) for item_id in cache["item_ids"]]
    embeddings = cache["embeddings"]
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("cache['embeddings'] must be a torch.Tensor")
    if embeddings.ndim != 2:
        raise ValueError(
            "cache['embeddings'] must be rank 2 [items, dimension], "
            f"got shape={tuple(embeddings.shape)}"
        )

    duplicate_ids = sorted(
        item_id for item_id, count in Counter(item_ids).items() if count > 1
    )
    row_count, embedding_dim = embeddings.shape
    row_count_matches = row_count == len(item_ids)

    if row_count_matches:
        values = embeddings.float()
        finite_mask = torch.isfinite(values).all(dim=1)
        safe_values = torch.where(torch.isfinite(values), values, 0.0)
        norms = torch.linalg.vector_norm(safe_values, dim=1)
        nonzero_mask = norms > 1e-12
        normalized_mask = torch.abs(norms - 1.0) <= norm_tolerance
        usable_mask = finite_mask & nonzero_mask & normalized_mask

        nonfinite_indices = (~finite_mask).nonzero(as_tuple=False).flatten().tolist()
        zero_indices = (~nonzero_mask).nonzero(as_tuple=False).flatten().tolist()
        bad_norm_indices = (
            finite_mask & nonzero_mask & ~normalized_mask
        ).nonzero(as_tuple=False).flatten().tolist()
        usable_item_ids = {
            item_ids[index]
            for index in usable_mask.nonzero(as_tuple=False).flatten().tolist()
        }
        finite_norms = norms[finite_mask & nonzero_mask]
        norm_min = float(finite_norms.min()) if finite_norms.numel() else None
        norm_max = float(finite_norms.max()) if finite_norms.numel() else None
        max_norm_error = (
            float(torch.abs(finite_norms - 1.0).max())
            if finite_norms.numel()
            else None
        )
    else:
        nonfinite_indices = []
        zero_indices = []
        bad_norm_indices = []
        usable_item_ids = set()
        norm_min = None
        norm_max = None
        max_norm_error = None

    def ids_at(indices: Sequence[int]) -> list[str]:
        return [item_ids[index] for index in indices[:MAX_EXAMPLES]]

    embedding_dtype = str(embeddings.dtype).replace("torch.", "")
    report = {
        "model_id": model_id,
        "expected_model_id": expected_model_id,
        "model_id_matches": model_id == expected_model_id,
        "cache_declares_normalized": bool(cache.get("normalized", False)),
        "item_id_count": len(item_ids),
        "embedding_row_count": int(row_count),
        "embedding_dim": int(embedding_dim),
        "embedding_dtype": embedding_dtype,
        "expected_embedding_dim": expected_dim,
        "row_count_matches_item_ids": row_count_matches,
        "duplicate_item_id_count": len(duplicate_ids),
        "duplicate_item_id_examples": duplicate_ids[:MAX_EXAMPLES],
        "nonfinite_row_count": len(nonfinite_indices),
        "nonfinite_item_id_examples": ids_at(nonfinite_indices),
        "zero_norm_row_count": len(zero_indices),
        "zero_norm_item_id_examples": ids_at(zero_indices),
        "bad_norm_row_count": len(bad_norm_indices),
        "bad_norm_item_id_examples": ids_at(bad_norm_indices),
        "norm_tolerance": norm_tolerance,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "max_norm_error": max_norm_error,
        "usable_item_count": len(usable_item_ids),
    }
    report["pass"] = bool(
        report["model_id_matches"]
        and report["cache_declares_normalized"]
        and report["embedding_dim"] == expected_dim
        and report["row_count_matches_item_ids"]
        and report["duplicate_item_id_count"] == 0
        and report["nonfinite_row_count"] == 0
        and report["zero_norm_row_count"] == 0
        and report["bad_norm_row_count"] == 0
    )
    return report, usable_item_ids


def inspect_embedding_manifest(
    manifest: Mapping[str, object],
    *,
    cache_report: Mapping[str, object],
    cache_sha256: str,
    expected_embedding_version: str = EXPECTED_EMBEDDING_VERSION,
) -> dict:
    """Validate the versioned embedding manifest against the actual cache."""

    required_fields = {
        "embedding_version",
        "model_name_or_version",
        "preprocessing_version",
        "embedding_dimension",
        "normalization",
        "dtype",
        "item_count",
        "cache_sha256",
    }
    missing_fields = sorted(required_fields - set(manifest))
    preprocessing_version = str(manifest.get("preprocessing_version", "")).strip()
    declared_dtype = str(manifest.get("dtype", "")).strip().lower().replace("torch.", "")
    declared_normalization = str(manifest.get("normalization", "")).strip().lower()

    checks = {
        "embedding_version_matches": (
            manifest.get("embedding_version") == expected_embedding_version
        ),
        "model_matches_cache": (
            str(manifest.get("model_name_or_version", ""))
            == str(cache_report.get("model_id", ""))
        ),
        "preprocessing_version_present": bool(preprocessing_version)
        and not preprocessing_version.upper().startswith("REPLACE_"),
        "dimension_matches_cache": (
            manifest.get("embedding_dimension") == cache_report.get("embedding_dim")
        ),
        "normalization_is_l2": declared_normalization in {"l2", "l2-normalized"},
        "dtype_matches_cache": declared_dtype
        == str(cache_report.get("embedding_dtype", "")).lower().replace("torch.", ""),
        "item_count_matches_cache": (
            manifest.get("item_count") == cache_report.get("embedding_row_count")
        ),
        "cache_sha256_matches": str(manifest.get("cache_sha256", "")) == cache_sha256,
    }
    report = {
        "embedding_version": manifest.get("embedding_version"),
        "model_name_or_version": manifest.get("model_name_or_version"),
        "preprocessing_version": manifest.get("preprocessing_version"),
        "embedding_dimension": manifest.get("embedding_dimension"),
        "normalization": manifest.get("normalization"),
        "dtype": manifest.get("dtype"),
        "item_count": manifest.get("item_count"),
        "cache_sha256": manifest.get("cache_sha256"),
        "missing_field_count": len(missing_fields),
        "missing_fields": missing_fields,
        **checks,
    }
    report["pass"] = not missing_fields and all(checks.values())
    return report


def validate_split(
    positives: Sequence[dict],
    metadata: Sequence[dict],
    usable_item_ids: set[str],
    *,
    split: str,
    expected_mapping_version: str = EXPECTED_CATEGORY_MAPPING_VERSION,
    expected_item_metadata_version: str = EXPECTED_ITEM_METADATA_VERSION,
) -> dict:
    """Check positive items, metadata contract, and cache coverage for one split."""

    sample_ids = [str(row.get("sample_id", "")) for row in positives]
    duplicate_sample_ids = sorted(
        value for value, count in Counter(sample_ids).items() if value and count > 1
    )

    required_item_ids = {
        str(item_id)
        for row in positives
        for item_id in row.get("items", [])
    }
    metadata_ids = [str(row.get("item_id", "")) for row in metadata]
    metadata_id_set = {item_id for item_id in metadata_ids if item_id}
    duplicate_metadata_ids = sorted(
        value for value, count in Counter(metadata_ids).items() if value and count > 1
    )

    missing_from_metadata = sorted(required_item_ids - metadata_id_set)
    extra_in_metadata = sorted(metadata_id_set - required_item_ids)
    missing_or_invalid_embedding = sorted(required_item_ids - usable_item_ids)
    wrong_split_metadata = [
        str(row.get("item_id", ""))
        for row in metadata
        if str(row.get("split", "")) != split
    ]
    wrong_mapping_version = [
        str(row.get("item_id", ""))
        for row in metadata
        if str(row.get("category_mapping_version", "")) != expected_mapping_version
    ]
    wrong_item_metadata_version = [
        str(row.get("item_id", ""))
        for row in metadata
        if str(row.get("item_metadata_version", "")) != expected_item_metadata_version
    ]

    report = {
        "split": split,
        "positive_sample_count": len(positives),
        "unique_required_item_count": len(required_item_ids),
        "metadata_row_count": len(metadata),
        "unique_metadata_item_count": len(metadata_id_set),
        "expected_category_mapping_version": expected_mapping_version,
        "expected_item_metadata_version": expected_item_metadata_version,
        "duplicate_sample_id_count": len(duplicate_sample_ids),
        "duplicate_sample_id_examples": duplicate_sample_ids[:MAX_EXAMPLES],
        "duplicate_metadata_item_id_count": len(duplicate_metadata_ids),
        "duplicate_metadata_item_id_examples": duplicate_metadata_ids[:MAX_EXAMPLES],
        "missing_from_metadata_count": len(missing_from_metadata),
        "missing_from_metadata_examples": missing_from_metadata[:MAX_EXAMPLES],
        "extra_in_metadata_count": len(extra_in_metadata),
        "extra_in_metadata_examples": extra_in_metadata[:MAX_EXAMPLES],
        "wrong_split_metadata_count": len(wrong_split_metadata),
        "wrong_split_metadata_examples": wrong_split_metadata[:MAX_EXAMPLES],
        "wrong_mapping_version_count": len(wrong_mapping_version),
        "wrong_mapping_version_examples": wrong_mapping_version[:MAX_EXAMPLES],
        "wrong_item_metadata_version_count": len(wrong_item_metadata_version),
        "wrong_item_metadata_version_examples": wrong_item_metadata_version[:MAX_EXAMPLES],
        "missing_or_invalid_embedding_count": len(missing_or_invalid_embedding),
        "missing_or_invalid_embedding_examples": missing_or_invalid_embedding[:MAX_EXAMPLES],
        "embedding_coverage": (
            1.0
            if not required_item_ids
            else 1.0 - len(missing_or_invalid_embedding) / len(required_item_ids)
        ),
    }
    report["pass"] = all(
        report[key] == 0
        for key in (
            "duplicate_sample_id_count",
            "duplicate_metadata_item_id_count",
            "missing_from_metadata_count",
            "extra_in_metadata_count",
            "wrong_split_metadata_count",
            "wrong_mapping_version_count",
            "wrong_item_metadata_version_count",
            "missing_or_invalid_embedding_count",
        )
    )
    return report


def _input_fingerprints(
    *,
    mapping_path: Path,
    cache_path: Path,
    manifest_path: Path,
    positives_by_split: Mapping[str, Path | str],
    metadata_by_split: Mapping[str, Path | str],
) -> dict:
    split_inputs = {}
    for split in positives_by_split:
        positive_path = Path(positives_by_split[split])
        metadata_path = Path(metadata_by_split[split])
        split_inputs[split] = {
            "positive_path": str(positive_path),
            "positive_sha256": sha256_file(positive_path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
        }
    return {
        "category_mapping": fingerprint_category_mapping(mapping_path),
        "embedding_cache": {
            "path": str(cache_path),
            "sha256": sha256_file(cache_path),
        },
        "embedding_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "splits": split_inputs,
    }


def validate_core7_embedding_coverage(
    *,
    mapping_path: Path | str,
    cache_path: Path | str,
    positives_by_split: Mapping[str, Path | str],
    metadata_by_split: Mapping[str, Path | str],
    manifest_path: Path | str | None = None,
    report_path: Path | str | None = None,
    expected_model_id: str = EXPECTED_MODEL_ID,
    expected_dim: int = EXPECTED_EMBEDDING_DIM,
    expected_mapping_version: str = EXPECTED_CATEGORY_MAPPING_VERSION,
    norm_tolerance: float = DEFAULT_NORM_TOLERANCE,
) -> dict:
    """Validate cache+manifest+coverage and bind PASS to exact input hashes."""

    split_names = list(positives_by_split)
    if set(split_names) != set(metadata_by_split):
        raise ValueError("Positive and metadata split names must match exactly")

    cache_source = Path(cache_path)
    manifest_source = (
        Path(manifest_path)
        if manifest_path is not None
        else cache_source.parent / DEFAULT_MANIFEST_FILENAME
    )
    if not cache_source.is_file():
        raise FileNotFoundError(cache_source)
    if not manifest_source.is_file():
        raise FileNotFoundError(
            f"Missing required embedding manifest: {manifest_source}"
        )

    mapping_source = Path(mapping_path)
    if not mapping_source.is_file():
        raise FileNotFoundError(mapping_source)

    inputs = _input_fingerprints(
        mapping_path=mapping_source,
        cache_path=cache_source,
        manifest_path=manifest_source,
        positives_by_split=positives_by_split,
        metadata_by_split=metadata_by_split,
    )
    cache = load_embedding_cache(cache_source)
    cache_report, usable_item_ids = inspect_embedding_cache(
        cache,
        expected_model_id=expected_model_id,
        expected_dim=expected_dim,
        norm_tolerance=norm_tolerance,
    )
    manifest = read_json(manifest_source)
    manifest_report = inspect_embedding_manifest(
        manifest,
        cache_report=cache_report,
        cache_sha256=inputs["embedding_cache"]["sha256"],
    )

    split_reports: Dict[str, dict] = {}
    for split in split_names:
        positives = read_jsonl(positives_by_split[split])
        metadata = read_jsonl(metadata_by_split[split])
        split_reports[split] = validate_split(
            positives,
            metadata,
            usable_item_ids,
            split=split,
            expected_mapping_version=expected_mapping_version,
        )

    passed = (
        cache_report["pass"]
        and manifest_report["pass"]
        and all(split_report["pass"] for split_report in split_reports.values())
    )
    report = {
        "processing_stage": "core7_embedding_validation",
        "category_mapping_version": expected_mapping_version,
        "mapping_path": str(mapping_source),
        "mapping_sha256": inputs["category_mapping"]["sha256"],
        "resolved_mapping_sha256": inputs["category_mapping"][
            "resolved_mapping_sha256"
        ],
        "embedding_version": manifest.get("embedding_version"),
        "cache_path": str(cache_source),
        "manifest_path": str(manifest_source),
        "inputs": inputs,
        "cache": cache_report,
        "manifest": manifest_report,
        "splits": split_reports,
        "pass": bool(passed),
        "reuse_category_clean_as_final": bool(passed),
        "ready_for_negative_sampling": bool(passed),
    }

    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    return report


def repair_split(
    positives: Sequence[dict],
    metadata: Sequence[dict],
    usable_item_ids: set[str],
    *,
    min_items: int = DEFAULT_MIN_OUTFIT_ITEMS,
) -> tuple[list[dict], list[dict], dict]:
    """Remove unusable items and keep aligned outfits/metadata after re-counting."""

    repaired_positives: list[dict] = []
    removed_item_references = 0
    dropped_outfits = 0

    for row in positives:
        original_items = [str(item_id) for item_id in row.get("items", [])]
        kept_items = [item_id for item_id in original_items if item_id in usable_item_ids]
        removed_item_references += len(original_items) - len(kept_items)
        if len(kept_items) < min_items:
            dropped_outfits += 1
            continue
        repaired = dict(row)
        repaired["items"] = kept_items
        repaired_positives.append(repaired)

    serialized_item_ids = {
        item_id for row in repaired_positives for item_id in row["items"]
    }
    repaired_metadata = [
        dict(row)
        for row in metadata
        if str(row.get("item_id", "")) in serialized_item_ids
    ]
    report = {
        "input_outfit_count": len(positives),
        "output_outfit_count": len(repaired_positives),
        "dropped_outfit_count": dropped_outfits,
        "removed_item_reference_count": removed_item_references,
        "output_metadata_count": len(repaired_metadata),
        "min_items": min_items,
    }
    return repaired_positives, repaired_metadata, report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Core-7 V2 positives against a FashionCLIP cache"
    )
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-model-id", default=EXPECTED_MODEL_ID)
    parser.add_argument("--expected-dim", type=int, default=EXPECTED_EMBEDDING_DIM)
    parser.add_argument(
        "--expected-mapping-version", default=EXPECTED_CATEGORY_MAPPING_VERSION
    )
    parser.add_argument("--norm-tolerance", type=float, default=DEFAULT_NORM_TOLERANCE)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    positives = {
        split: args.data_dir / f"category_clean_{split}.jsonl"
        for split in ("train", "valid", "test")
    }
    metadata = {
        split: args.data_dir / f"core7_item_metadata_v1_{split}.jsonl"
        for split in ("train", "valid", "test")
    }
    report = validate_core7_embedding_coverage(
        mapping_path=args.mapping,
        cache_path=args.cache,
        manifest_path=args.manifest,
        positives_by_split=positives,
        metadata_by_split=metadata,
        report_path=args.report,
        expected_model_id=args.expected_model_id,
        expected_dim=args.expected_dim,
        expected_mapping_version=args.expected_mapping_version,
        norm_tolerance=args.norm_tolerance,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
