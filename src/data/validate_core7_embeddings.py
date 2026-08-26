# -*- coding: utf-8 -*-
"""Validate Core-7 positives against the frozen FashionCLIP cache.

This stage answers one concrete question before negative sampling:

    Does every item serialized in ``category_clean_{split}.jsonl`` have one
    usable FashionCLIP embedding and one matching Core-7 metadata row?

When every split passes, the category-clean positives can be reused directly
as the final clean positives. No duplicate JSONL is needed. If a split fails,
``repair_split`` can remove unusable items, recompute outfit length, keep only
outfits with at least ``min_items``, and write aligned metadata.

The module has no Colab or Google Drive dependency. Paths are supplied by the
caller, so the same code works in Colab, VS Code, CI, or a backend job.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

try:
    import torch
except ModuleNotFoundError:  # Keep lightweight schema helpers importable in CI.
    torch = None


EXPECTED_MODEL_ID = "patrickjohncyh/fashion-clip"
EXPECTED_EMBEDDING_DIM = 512
DEFAULT_MIN_OUTFIT_ITEMS = 3
DEFAULT_NORM_TOLERANCE = 1e-2
MAX_EXAMPLES = 50


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
        # Compatibility with older PyTorch releases that lack weights_only.
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

    report = {
        "model_id": model_id,
        "expected_model_id": expected_model_id,
        "model_id_matches": model_id == expected_model_id,
        "cache_declares_normalized": bool(cache.get("normalized", False)),
        "item_id_count": len(item_ids),
        "embedding_row_count": int(row_count),
        "embedding_dim": int(embedding_dim),
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


def validate_split(
    positives: Sequence[dict],
    metadata: Sequence[dict],
    usable_item_ids: set[str],
    *,
    split: str,
) -> dict:
    """Check positive items, Core-7 metadata, and cache coverage for one split."""

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

    report = {
        "split": split,
        "positive_sample_count": len(positives),
        "unique_required_item_count": len(required_item_ids),
        "metadata_row_count": len(metadata),
        "unique_metadata_item_count": len(metadata_id_set),
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
        "missing_or_invalid_embedding_count": len(missing_or_invalid_embedding),
        "missing_or_invalid_embedding_examples": missing_or_invalid_embedding[
            :MAX_EXAMPLES
        ],
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
            "missing_or_invalid_embedding_count",
        )
    )
    return report


def validate_core7_embedding_coverage(
    *,
    cache_path: Path | str,
    positives_by_split: Mapping[str, Path | str],
    metadata_by_split: Mapping[str, Path | str],
    report_path: Path | str | None = None,
    expected_model_id: str = EXPECTED_MODEL_ID,
    expected_dim: int = EXPECTED_EMBEDDING_DIM,
    norm_tolerance: float = DEFAULT_NORM_TOLERANCE,
) -> dict:
    """Run cache and per-split coverage validation and optionally save a report."""

    split_names = list(positives_by_split)
    if set(split_names) != set(metadata_by_split):
        raise ValueError("Positive and metadata split names must match exactly")

    cache = load_embedding_cache(cache_path)
    cache_report, usable_item_ids = inspect_embedding_cache(
        cache,
        expected_model_id=expected_model_id,
        expected_dim=expected_dim,
        norm_tolerance=norm_tolerance,
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
        )

    passed = cache_report["pass"] and all(
        split_report["pass"] for split_report in split_reports.values()
    )
    report = {
        "processing_stage": "core7_embedding_validation",
        "cache_path": str(Path(cache_path)),
        "cache": cache_report,
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
        kept_items = [
            item_id for item_id in original_items if item_id in usable_item_ids
        ]
        removed_item_references += len(original_items) - len(kept_items)
        if len(kept_items) < min_items:
            dropped_outfits += 1
            continue
        repaired = dict(row)
        repaired["items"] = kept_items
        repaired_positives.append(repaired)

    serialized_item_ids = {
        item_id
        for row in repaired_positives
        for item_id in row["items"]
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
        description="Validate Core-7 positives against a FashionCLIP cache"
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-model-id", default=EXPECTED_MODEL_ID)
    parser.add_argument("--expected-dim", type=int, default=EXPECTED_EMBEDDING_DIM)
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
        cache_path=args.cache,
        positives_by_split=positives,
        metadata_by_split=metadata,
        report_path=args.report,
        expected_model_id=args.expected_model_id,
        expected_dim=args.expected_dim,
        norm_tolerance=args.norm_tolerance,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
