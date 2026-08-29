# -*- coding: utf-8 -*-
"""FashionCLIP cache bootstrap for the MIN2 experiment.

The frozen scorer consumes precomputed 512-d image embeddings. Large binary
caches are intentionally not committed to GitHub, so an ephemeral Colab
checkout may need to rebuild the cache before embedding validation can run.

This module keeps the cache contract used by ``validate_core7_embeddings``:

    {
        "model_id": "patrickjohncyh/fashion-clip",
        "item_ids": [...],
        "embeddings": Float16Tensor[N, 512],
        "normalized": True,
    }

Only items referenced by the experiment's clean train/valid/test positives are
encoded. If a source image cannot be decoded/preprocessed/encoded, that item is
recorded as failed and the clean positives are repaired by removing unusable
items and re-applying the MIN2 length gate. Placeholder embeddings are never
created.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from .min2_experiment import (
    MIN_SCORER_ITEMS,
    SPLITS,
    category_clean_path,
    metadata_path,
)
from .validate_core7_embeddings import (
    EXPECTED_EMBEDDING_DIM,
    EXPECTED_EMBEDDING_VERSION,
    EXPECTED_MODEL_ID,
    inspect_embedding_cache,
    inspect_embedding_manifest,
    load_embedding_cache,
    read_json,
    read_jsonl,
    repair_split,
    sha256_file,
    write_jsonl,
)

try:
    import torch
except ModuleNotFoundError:  # Keep lightweight imports usable in CI.
    torch = None


DATASET_NAME = "codewaly/polyvore1000"
ITEMS_CONFIG = "items"
DEFAULT_BATCH_SIZE = 64


def _required_item_ids_by_split(core7_dir: Path | str) -> dict[str, set[str]]:
    """Return exact item IDs serialized in MIN2 positive metadata by split."""

    required: dict[str, set[str]] = {}
    seen_global: set[str] = set()
    for split in SPLITS:
        rows = read_jsonl(metadata_path(core7_dir, split))
        ids = {str(row.get("item_id", "")).strip() for row in rows}
        ids.discard("")
        if len(ids) != len(rows):
            raise ValueError(f"split={split}: duplicate/blank metadata item_id")
        overlap = seen_global.intersection(ids)
        if overlap:
            examples = sorted(overlap)[:10]
            raise ValueError(
                f"Cross-split item IDs are not allowed; split={split}, examples={examples}"
            )
        seen_global.update(ids)
        required[split] = ids
    return required


def _all_required_ids(required_by_split: Mapping[str, set[str]]) -> set[str]:
    return set().union(*(required_by_split[split] for split in SPLITS))


def _existing_cache_status(runtime_paths, required_ids: set[str]) -> dict:
    cache_path = Path(runtime_paths.embedding_cache)
    manifest_path = Path(runtime_paths.embedding_manifest)
    if not cache_path.is_file() or not manifest_path.is_file():
        return {
            "usable": False,
            "reason": "cache_or_manifest_missing",
            "missing_required_item_ids": sorted(required_ids),
        }

    cache = load_embedding_cache(cache_path)
    cache_report, usable_ids = inspect_embedding_cache(cache)
    manifest = read_json(manifest_path)
    cache_sha = sha256_file(cache_path)
    manifest_report = inspect_embedding_manifest(
        manifest,
        cache_report=cache_report,
        cache_sha256=cache_sha,
    )
    missing = sorted(required_ids - usable_ids)
    return {
        "usable": bool(cache_report["pass"] and manifest_report["pass"] and not missing),
        "reason": "ready" if not missing else "missing_required_embeddings",
        "cache_report": cache_report,
        "manifest_report": manifest_report,
        "usable_item_ids": usable_ids,
        "known_failed_item_ids": {
            str(value) for value in cache.get("failed_item_ids", [])
        },
        "missing_required_item_ids": missing,
    }


def _repair_clean_files(runtime_paths, usable_item_ids: set[str]) -> dict:
    """Drop failed-image items and re-apply MIN2 outfit length semantics."""

    reports: dict[str, dict] = {}
    for split in SPLITS:
        positive_file = category_clean_path(runtime_paths.core7_dir, split)
        metadata_file = metadata_path(runtime_paths.core7_dir, split)
        positives = read_jsonl(positive_file)
        metadata = read_jsonl(metadata_file)
        repaired_positives, repaired_metadata, report = repair_split(
            positives,
            metadata,
            usable_item_ids,
            min_items=MIN_SCORER_ITEMS,
        )
        write_jsonl(repaired_positives, positive_file)
        write_jsonl(repaired_metadata, metadata_file)
        reports[split] = report
    return reports


def _load_fashionclip(model_id: str, device):
    try:
        import transformers
        from transformers import CLIPModel, CLIPProcessor
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "transformers is required to rebuild FashionCLIP embeddings. "
            "Install it with: pip install -U transformers"
        ) from error

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval().to(device)

    revision = getattr(model.config, "_commit_hash", None)
    if not revision:
        revision = getattr(processor, "_commit_hash", None)
    preprocessing_version = (
        f"transformers-{transformers.__version__}:CLIPProcessor@"
        f"{revision or model_id}"
    )
    return processor, model, preprocessing_version, revision


def _encode_images(model, processor, images: Sequence[object], *, device):
    if torch is None:
        raise RuntimeError("PyTorch is required to build the embedding cache")

    rgb_images = []
    for image in images:
        if image is None:
            raise ValueError("image is missing")
        convert = getattr(image, "convert", None)
        rgb_images.append(convert("RGB") if callable(convert) else image)

    inputs = processor(images=rgb_images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.inference_mode():
        features = model.get_image_features(pixel_values=pixel_values)
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("FashionCLIP image encoder did not return rank-2 features")
    features = features.float()
    if features.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            "Unexpected FashionCLIP dimension: "
            f"expected={EXPECTED_EMBEDDING_DIM}, actual={features.shape[1]}"
        )
    if not torch.isfinite(features).all():
        raise ValueError("FashionCLIP returned NaN/Inf")
    features = torch.nn.functional.normalize(features, p=2, dim=-1)
    if not torch.isfinite(features).all():
        raise ValueError("Normalized FashionCLIP embedding contains NaN/Inf")
    return features.to(dtype=torch.float16, device="cpu")


def build_min2_fashionclip_cache(
    runtime_paths,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device=None,
    overwrite: bool = False,
    model_id: str = EXPECTED_MODEL_ID,
) -> dict:
    """Build an experiment-local FashionCLIP cache from Polyvore1000 images."""

    if torch is None:
        raise RuntimeError("PyTorch is required to build FashionCLIP embeddings")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "datasets is required to rebuild FashionCLIP embeddings"
        ) from error

    cache_path = Path(runtime_paths.embedding_cache)
    manifest_path = Path(runtime_paths.embedding_manifest)
    if not overwrite and (cache_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            "Refusing to overwrite an existing embedding cache/manifest. "
            "Pass overwrite=True only for an experiment-local cache."
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    required_by_split = _required_item_ids_by_split(runtime_paths.core7_dir)
    total_required = sum(len(values) for values in required_by_split.values())
    if total_required == 0:
        raise ValueError("MIN2 clean metadata contains zero required items")

    device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"FashionCLIP device: {device}")
    print(f"Required MIN2 items: {total_required:,}")
    processor, model, preprocessing_version, model_revision = _load_fashionclip(
        model_id, device
    )

    encoded_ids: list[str] = []
    encoded_chunks: list[torch.Tensor] = []
    failed_ids: list[str] = []
    split_reports: dict[str, dict] = {}

    for split in SPLITS:
        required = required_by_split[split]
        print(f"\nEmbedding split={split}: required={len(required):,}")
        items_ds = load_dataset(DATASET_NAME, ITEMS_CONFIG, split=split)
        if "item_id" not in items_ds.column_names or "image" not in items_ds.column_names:
            raise KeyError("Polyvore1000 items split must contain item_id and image")

        source_ids = [str(value) for value in items_ds["item_id"]]
        indices = [index for index, item_id in enumerate(source_ids) if item_id in required]
        found_ids = {source_ids[index] for index in indices}
        missing_from_source = sorted(required - found_ids)
        if missing_from_source:
            raise ValueError(
                f"split={split}: {len(missing_from_source)} required item IDs are absent "
                f"from source items; examples={missing_from_source[:10]}"
            )

        subset = items_ds.select(indices).select_columns(["item_id", "image"])
        split_encoded = 0
        split_failed: list[str] = []

        for start in range(0, len(subset), batch_size):
            stop = min(start + batch_size, len(subset))
            try:
                batch = subset[start:stop]
                batch_ids = [str(value) for value in batch["item_id"]]
                embeddings = _encode_images(
                    model, processor, batch["image"], device=device
                )
                encoded_ids.extend(batch_ids)
                encoded_chunks.append(embeddings)
                split_encoded += len(batch_ids)
            except Exception as batch_error:
                # Isolate bad images instead of discarding a whole batch.
                print(
                    f"  batch {start}:{stop} failed ({type(batch_error).__name__}); "
                    "retrying item-by-item"
                )
                for index in range(start, stop):
                    try:
                        row = subset[index]
                        item_id = str(row["item_id"])
                        embedding = _encode_images(
                            model, processor, [row["image"]], device=device
                        )
                        encoded_ids.append(item_id)
                        encoded_chunks.append(embedding)
                        split_encoded += 1
                    except Exception as item_error:
                        item_id = source_ids[indices[index]]
                        split_failed.append(item_id)
                        failed_ids.append(item_id)
                        print(
                            f"    failed item={item_id}: "
                            f"{type(item_error).__name__}: {item_error}"
                        )

            if start == 0 or stop == len(subset) or (stop // batch_size) % 50 == 0:
                print(
                    f"  progress {stop:,}/{len(subset):,}; "
                    f"encoded={split_encoded:,}; failed={len(split_failed):,}"
                )

        split_reports[split] = {
            "required_item_count": len(required),
            "source_item_count": len(items_ds),
            "encoded_item_count": split_encoded,
            "failed_item_count": len(split_failed),
            "failed_item_examples": split_failed[:50],
        }

    if not encoded_chunks:
        raise RuntimeError("No FashionCLIP embeddings were produced")
    if len(encoded_ids) != len(set(encoded_ids)):
        raise ValueError("Embedding builder produced duplicate item IDs")

    embeddings = torch.cat(encoded_chunks, dim=0)
    if embeddings.shape != (len(encoded_ids), EXPECTED_EMBEDDING_DIM):
        raise RuntimeError(
            f"Cache shape mismatch: ids={len(encoded_ids)}, shape={tuple(embeddings.shape)}"
        )

    cache_payload = {
        "model_id": model_id,
        "item_ids": encoded_ids,
        "embeddings": embeddings,
        "normalized": True,
        "embedding_version": EXPECTED_EMBEDDING_VERSION,
        "preprocessing_version": preprocessing_version,
        "model_revision": model_revision,
        "failed_item_ids": failed_ids,
    }
    temp_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(cache_payload, temp_cache)
    os.replace(temp_cache, cache_path)
    cache_sha = sha256_file(cache_path)

    manifest = {
        "embedding_version": EXPECTED_EMBEDDING_VERSION,
        "model_name_or_version": model_id,
        "preprocessing_version": preprocessing_version,
        "embedding_dimension": EXPECTED_EMBEDDING_DIM,
        "normalization": "l2",
        "dtype": "float16",
        "item_count": len(encoded_ids),
        "cache_sha256": cache_sha,
        "model_revision": model_revision,
        "source_dataset": DATASET_NAME,
        "source_config": ITEMS_CONFIG,
        "failed_item_count": len(failed_ids),
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    # Image failures are data failures, not a reason to synthesize placeholders.
    # Remove those items and recalculate the MIN2 outfit gate before validation.
    cache_report, usable_ids = inspect_embedding_cache(cache_payload)
    if not cache_report["pass"]:
        raise RuntimeError(f"New embedding cache failed schema checks: {cache_report}")
    repair_reports = (
        _repair_clean_files(runtime_paths, usable_ids) if failed_ids else {}
    )

    return {
        "action": "built",
        "cache_path": str(cache_path),
        "manifest_path": str(manifest_path),
        "cache_sha256": cache_sha,
        "model_id": model_id,
        "model_revision": model_revision,
        "preprocessing_version": preprocessing_version,
        "item_count": len(encoded_ids),
        "failed_item_count": len(failed_ids),
        "failed_item_examples": failed_ids[:50],
        "splits": split_reports,
        "repair": repair_reports,
    }


def ensure_min2_embedding_cache(
    runtime_paths,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device=None,
    force_rebuild: bool = False,
) -> dict:
    """Reuse a complete cache, otherwise build one when it is absent.

    Existing incomplete caches are not silently overwritten. That protects a
    shared/frozen cache if the runtime config points at external benchmark
    storage. Set ``force_rebuild=True`` only when the configured cache is an
    experiment-local artifact that may be replaced.
    """

    required_by_split = _required_item_ids_by_split(runtime_paths.core7_dir)
    required_ids = _all_required_ids(required_by_split)
    status = _existing_cache_status(runtime_paths, required_ids)
    if status["usable"] and not force_rebuild:
        return {
            "action": "reused",
            "cache_path": str(runtime_paths.embedding_cache),
            "manifest_path": str(runtime_paths.embedding_manifest),
            "required_item_count": len(required_ids),
            "missing_required_item_count": 0,
        }

    # A cache built by this helper may intentionally omit failed source images.
    # If clean files were regenerated later, re-apply the recorded repair.
    if not force_rebuild and status.get("reason") == "missing_required_embeddings":
        missing = set(status.get("missing_required_item_ids", []))
        known_failed = set(status.get("known_failed_item_ids", set()))
        if missing and missing.issubset(known_failed):
            repair_reports = _repair_clean_files(
                runtime_paths, set(status.get("usable_item_ids", set()))
            )
            repaired_required = _all_required_ids(
                _required_item_ids_by_split(runtime_paths.core7_dir)
            )
            repaired_status = _existing_cache_status(runtime_paths, repaired_required)
            if repaired_status["usable"]:
                return {
                    "action": "reused_after_repair",
                    "cache_path": str(runtime_paths.embedding_cache),
                    "manifest_path": str(runtime_paths.embedding_manifest),
                    "required_item_count": len(repaired_required),
                    "missing_required_item_count": 0,
                    "repair": repair_reports,
                }

    cache_exists = Path(runtime_paths.embedding_cache).exists()
    manifest_exists = Path(runtime_paths.embedding_manifest).exists()
    if (cache_exists or manifest_exists) and not force_rebuild:
        missing = status.get("missing_required_item_ids", [])
        raise RuntimeError(
            "An embedding cache/manifest already exists but is not sufficient for MIN2. "
            f"missing_required={len(missing)}. Refusing to overwrite it automatically. "
            "Use an experiment-local artifact_root or pass force_rebuild=True."
        )

    return build_min2_fashionclip_cache(
        runtime_paths,
        batch_size=batch_size,
        device=device,
        overwrite=bool(cache_exists or manifest_exists),
    )
