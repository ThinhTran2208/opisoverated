#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUATION3 Test2000 Full -> frozen FashionCLIP -> scorer-ready -> compatibility_logit.

Designed for Google Colab and the repository branch:
    https://github.com/ThinhTran2208/opisoverated/tree/feat/production-inference-v1

Why this path is intentionally simpler than production image inference:
- EVALUATION3 already provides isolated garment images U/B/S/G, so RF-DETR is NOT run.
- EVALUATION3 roles are known semantically, so coarse categories are assigned deterministically:
      U -> TOP, B -> BOTTOM, S -> SHOES, G -> BAG
  rather than re-classifying known roles with zero-shot FashionCLIP.
- The image embedding path matches src/detection/fashionclip.py on the target branch:
      CLIPProcessor -> CLIPModel.get_image_features -> projected feature -> float32 -> L2 normalize.
- The scorer is loaded from its frozen checkpoint and receives:
      item_embeddings [B,4,512]
      coarse_category_ids [B,4]
      item_mask [B,4]
  and returns compatibility_logit [B].

Default inputs on Drive:
    /content/drive/MyDrive/EVALUATION3/test2000_ids.csv
    /content/drive/MyDrive/EVALUATION3/test2000_freeze_manifest.json
    /content/drive/MyDrive/EVALUATION3/outfit.zip
    /content/drive/MyDrive/ML_Final/scorer_runs/type_aware_pairwise_v1/
        final_val_auc_v5_seed42/best.pt

Default outputs:
    /content/drive/MyDrive/EVALUATION3/eval3_test2000_scorer_full_v1/
        EVAL3-Test2000-Full-items.csv
        EVAL3-Test2000-Full-embeddings.pt
        EVAL3-Test2000-Full-scorer-ready.jsonl
        EVAL3-Test2000-Full-predictions.csv
        EVAL3-Test2000-Full-run-manifest.json
        embedding_progress.pt                 # resume checkpoint; removed on success by default

Examples:
    python eval3_test2000_full_preprocess_and_score.py

    python eval3_test2000_full_preprocess_and_score.py --mode encode
    python eval3_test2000_full_preprocess_and_score.py --mode score

    # Reuse the local ZIP left by the overlap audit:
    python eval3_test2000_full_preprocess_and_score.py \
        --local-zip /content/outfit.zip

Dependencies (Colab):
    pip install -q "transformers>=5.1,<6" pandas pillow tqdm

The scorer checkpoint SHA is validated against configs/production_inference_v1.json.
The frozen Test2000 ordered-ID SHA is validated against test2000_freeze_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_URL = "https://github.com/ThinhTran2208/opisoverated.git"
REPO_BRANCH = "feat/production-inference-v1"
DEFAULT_REPO_ROOT = Path("/content/opisoverated")

DEFAULT_E3_ROOT = Path("/content/drive/MyDrive/EVALUATION3")
DEFAULT_ML_ROOT = Path("/content/drive/MyDrive/ML_Final")
DEFAULT_OUTPUT_NAME = "eval3_test2000_scorer_full_v1"

DATASET_NAME = "EVAL3-Test2000-Full"
EXPECTED_TEST_OUTFITS = 2000
EXPECTED_ITEMS_PER_OUTFIT = 4
EXPECTED_TOTAL_ITEMS = EXPECTED_TEST_OUTFITS * EXPECTED_ITEMS_PER_OUTFIT
EXPECTED_EMBEDDING_DIM = 512
EXPECTED_SCORER_VERSION = "type_aware_pairwise_v1"

ROLE_ORDER = ("U", "B", "S", "G")
ROLE_TO_CATEGORY = {
    "U": "TOP",
    "B": "BOTTOM",
    "S": "SHOES",
    "G": "BAG",
}

# The freeze manifest currently records this value. The script still reads the
# manifest dynamically and uses its value as the source of truth.
KNOWN_ORDERED_TEST_IDS_SHA256 = (
    "9aa2935371e86c03d56bb546b57a77690838f031cd93e1258f84b1351858dcf9"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess EVAL3 Test2000 Full and run the frozen compatibility scorer."
    )
    parser.add_argument("--mode", choices=("all", "encode", "score"), default="all")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--branch", default=REPO_BRANCH)
    parser.add_argument("--e3-root", type=Path, default=DEFAULT_E3_ROOT)
    parser.add_argument("--ml-root", type=Path, default=DEFAULT_ML_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--test-ids", type=Path, default=None)
    parser.add_argument("--freeze-manifest", type=Path, default=None)
    parser.add_argument("--drive-zip", type=Path, default=None)
    parser.add_argument(
        "--local-zip",
        type=Path,
        default=None,
        help="Existing local /content copy of outfit.zip. Preferred when available.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--scorer-batch-size", type=int, default=256)
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=20,
        help="Persist embedding resume state to Drive every N encoded batches.",
    )
    parser.add_argument(
        "--keep-progress",
        action="store_true",
        help="Keep embedding_progress.pt after the final embedding cache is complete.",
    )
    parser.add_argument(
        "--no-auto-clone",
        action="store_true",
        help="Do not clone the target branch when --repo-root is absent.",
    )
    parser.add_argument(
        "--no-auto-mount-drive",
        action="store_true",
        help="Do not attempt google.colab.drive.mount when MyDrive is absent.",
    )
    return parser.parse_args()


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def maybe_mount_drive(*, enabled: bool) -> None:
    mydrive = Path("/content/drive/MyDrive")
    if mydrive.is_dir():
        return
    if not enabled:
        raise FileNotFoundError(
            f"{mydrive} is not available. Mount Google Drive or remove --no-auto-mount-drive."
        )
    try:
        from google.colab import drive  # type: ignore
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Google Drive is not mounted and this does not look like Colab. "
            "Mount Drive manually or pass paths on the local filesystem."
        ) from error
    print("[drive] Mounting Google Drive...")
    drive.mount("/content/drive")
    if not mydrive.is_dir():
        raise RuntimeError("Google Drive mount did not expose /content/drive/MyDrive")


def run_command(command: Sequence[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(command),
        cwd=None if cwd is None else str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def ensure_repo(repo_root: Path, branch: str, *, auto_clone: bool) -> str:
    if not repo_root.exists():
        if not auto_clone:
            raise FileNotFoundError(repo_root)
        repo_root.parent.mkdir(parents=True, exist_ok=True)
        print(f"[repo] Cloning {branch} -> {repo_root}")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                REPO_URL,
                str(repo_root),
            ],
            check=True,
        )

    if not (repo_root / ".git").exists():
        raise RuntimeError(f"{repo_root} exists but is not a Git checkout")

    actual_branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    if actual_branch != branch:
        raise RuntimeError(
            f"Repository branch mismatch: expected {branch!r}, got {actual_branch!r}. "
            "Use the requested feat/production-inference-v1 checkout."
        )
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    print(f"[repo] PASS | branch={actual_branch} | commit={commit}")
    return commit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ordered_ids(ids: Sequence[str]) -> str:
    # This exactly reproduces the current freeze hash convention:
    # one outfit ID per line, including the final newline.
    payload = "".join(f"{value}\n" for value in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(torch_module, payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch_module.save(payload, tmp)
    os.replace(tmp, path)


def load_test_ids(test_ids_path: Path, freeze_manifest_path: Path) -> tuple[list[str], dict[str, object]]:
    import pandas as pd

    if not test_ids_path.is_file():
        raise FileNotFoundError(test_ids_path)
    if not freeze_manifest_path.is_file():
        raise FileNotFoundError(freeze_manifest_path)

    frame = pd.read_csv(test_ids_path, dtype=str)
    if "outfit_id" not in frame.columns:
        raise ValueError(f"{test_ids_path} must contain outfit_id")
    ids = frame["outfit_id"].astype(str).str.strip().tolist()
    if len(ids) != EXPECTED_TEST_OUTFITS:
        raise ValueError(f"Expected {EXPECTED_TEST_OUTFITS} test outfits, got {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ValueError("test2000_ids.csv contains duplicate outfit_id values")
    if any(not value for value in ids):
        raise ValueError("test2000_ids.csv contains blank outfit_id")

    freeze = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(freeze, dict):
        raise ValueError("test2000_freeze_manifest.json must be a JSON object")

    expected_count = int(freeze.get("test_outfit_count", -1))
    expected_unique = int(freeze.get("unique_test_outfit_count", -1))
    if expected_count != EXPECTED_TEST_OUTFITS or expected_unique != EXPECTED_TEST_OUTFITS:
        raise ValueError("Freeze manifest does not describe the expected 2,000-outfit split")

    expected_hash = str(freeze.get("ordered_test_ids_sha256", "")).strip()
    actual_hash = sha256_ordered_ids(ids)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            "Frozen ordered Test2000 ID SHA-256 mismatch: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    if expected_hash and expected_hash != KNOWN_ORDERED_TEST_IDS_SHA256:
        print(
            "[freeze] NOTE | manifest SHA differs from the previously frozen project SHA; "
            "using the manifest as source of truth"
        )

    first_expected = str(freeze.get("first_outfit_id", ""))
    last_expected = str(freeze.get("last_outfit_id", ""))
    if first_expected and ids[0] != first_expected:
        raise ValueError(f"First outfit mismatch: expected {first_expected}, got {ids[0]}")
    if last_expected and ids[-1] != last_expected:
        raise ValueError(f"Last outfit mismatch: expected {last_expected}, got {ids[-1]}")

    coverage = freeze.get("image_coverage", {})
    if isinstance(coverage, Mapping):
        present = int(coverage.get("present_item_images", -1))
        complete = int(coverage.get("complete_outfits_4of4", -1))
        if present != EXPECTED_TOTAL_ITEMS or complete != EXPECTED_TEST_OUTFITS:
            raise ValueError(
                "Freeze manifest image coverage is incomplete: "
                f"present_item_images={present}, complete_outfits_4of4={complete}"
            )

    print(
        f"[freeze] PASS | {len(ids):,} outfits | ordered_SHA256={actual_hash} | "
        f"items={EXPECTED_TOTAL_ITEMS:,}"
    )
    return ids, freeze


def load_repo_contract(repo_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    production_path = repo_root / "configs" / "production_inference_v1.json"
    detection_path = repo_root / "configs" / "detection_rfdetr_fashionclip_core7_v1.json"
    if not production_path.is_file():
        raise FileNotFoundError(production_path)
    if not detection_path.is_file():
        raise FileNotFoundError(detection_path)

    production = json.loads(production_path.read_text(encoding="utf-8"))
    detection = json.loads(detection_path.read_text(encoding="utf-8"))
    if not isinstance(production, dict) or not isinstance(detection, dict):
        raise ValueError("Repository config payloads must be JSON objects")

    if str(production.get("scorer_version")) != EXPECTED_SCORER_VERSION:
        raise ValueError("Unexpected scorer_version in production_inference_v1.json")
    if int(production.get("embedding_dim", -1)) != EXPECTED_EMBEDDING_DIM:
        raise ValueError("Unexpected embedding_dim in production_inference_v1.json")
    if str(production.get("embedding_version")) != "fashionclip-512-l2-v1":
        raise ValueError("Unexpected embedding_version in production_inference_v1.json")

    fashionclip = detection.get("fashionclip")
    scorer_handoff = detection.get("scorer_handoff")
    if not isinstance(fashionclip, Mapping) or not isinstance(scorer_handoff, Mapping):
        raise ValueError("Detection config missing fashionclip/scorer_handoff sections")
    if int(fashionclip.get("embedding_dim", -1)) != EXPECTED_EMBEDDING_DIM:
        raise ValueError("Detection FashionCLIP embedding dimension mismatch")

    category_ids = scorer_handoff.get("category_ids")
    if not isinstance(category_ids, Mapping):
        raise ValueError("Detection config missing scorer_handoff.category_ids")
    for role, category in ROLE_TO_CATEGORY.items():
        if category not in category_ids:
            raise ValueError(f"Missing canonical category {category} for role {role}")

    return production, detection


def build_item_records(
    outfit_ids: Sequence[str],
    category_ids: Mapping[str, object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    item_index = 0
    for outfit_index, outfit_id in enumerate(outfit_ids):
        for role in ROLE_ORDER:
            category = ROLE_TO_CATEGORY[role]
            records.append(
                {
                    "item_index": item_index,
                    "outfit_index": outfit_index,
                    "test_position": outfit_index + 1,
                    "outfit_id": outfit_id,
                    "role": role,
                    "item_id": f"E3:{outfit_id}:{role}",
                    "coarse_category": category,
                    "coarse_category_id": int(category_ids[category]),
                    "image_rel_path": f"Outfits/{outfit_id}/{role}.jpg",
                }
            )
            item_index += 1
    if len(records) != EXPECTED_TOTAL_ITEMS:
        raise AssertionError("Internal item record count mismatch")
    return records


def records_sha256(records: Sequence[Mapping[str, object]]) -> str:
    lines = []
    for row in records:
        lines.append(
            "|".join(
                [
                    str(row["item_index"]),
                    str(row["outfit_id"]),
                    str(row["role"]),
                    str(row["coarse_category_id"]),
                ]
            )
        )
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def write_items_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "item_index",
        "outfit_index",
        "test_position",
        "outfit_id",
        "role",
        "item_id",
        "coarse_category",
        "coarse_category_id",
        "image_rel_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in columns})


def discover_local_zip(explicit: Path | None, stage_dir: Path) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            Path("/content/outfit.zip"),
            Path("/content/eval3_overlap_stage/outfit.zip"),
            Path("/content/eval3_test2000_full_stage/outfit.zip"),
            stage_dir / "outfit.zip",
        ]
    )
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file() and path.stat().st_size > 1024 * 1024:
            print(f"[zip] Reusing local ZIP: {path} ({path.stat().st_size / 1024**3:.2f} GB)")
            return path
    return None


def ensure_local_zip(drive_zip: Path, stage_dir: Path, explicit_local: Path | None) -> Path:
    found = discover_local_zip(explicit_local, stage_dir)
    if found is not None:
        return found
    if not drive_zip.is_file():
        raise FileNotFoundError(drive_zip)
    stage_dir.mkdir(parents=True, exist_ok=True)
    destination = stage_dir / "outfit.zip"
    print(
        f"[zip] Copying Drive ZIP sequentially to local SSD: {drive_zip} -> {destination} "
        f"({drive_zip.stat().st_size / 1024**3:.2f} GB)"
    )
    shutil.copyfile(drive_zip, destination)
    if destination.stat().st_size != drive_zip.stat().st_size:
        raise IOError("Local outfit.zip size does not match Drive source")
    return destination


def build_zip_member_map(zf: zipfile.ZipFile, records: Sequence[Mapping[str, object]]) -> dict[int, str]:
    names = set(zf.namelist())
    mapping: dict[int, str] = {}
    missing: list[str] = []
    for row in records:
        outfit_id = str(row["outfit_id"])
        role = str(row["role"])
        candidates = [
            f"Outfits/{outfit_id}/{role}.jpg",
            f"outfit/{outfit_id}/{role}.jpg",
            f"{outfit_id}/{role}.jpg",
            f"Outfits/{outfit_id}/{role}.JPG",
            f"outfit/{outfit_id}/{role}.JPG",
            f"{outfit_id}/{role}.JPG",
        ]
        member = next((value for value in candidates if value in names), None)
        if member is None:
            missing.append(f"{outfit_id}|{role}")
        else:
            mapping[int(row["item_index"])] = member
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"ZIP is missing {len(missing)} required Test2000 item images. First: {preview}"
        )
    print(f"[zip] PASS | resolved {len(mapping):,}/{len(records):,} Test2000 item members")
    return mapping


def read_pil_image_from_zip(zf: zipfile.ZipFile, member: str):
    from PIL import Image

    with zf.open(member, "r") as source:
        raw = source.read()
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image.convert("RGB")


def resolve_device(torch_module, requested: str):
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(requested)


def load_embedding_progress(torch_module, path: Path, *, record_sha: str, model_id: str):
    if not path.is_file():
        embeddings = torch_module.full(
            (EXPECTED_TOTAL_ITEMS, EXPECTED_EMBEDDING_DIM),
            float("nan"),
            dtype=torch_module.float32,
        )
        done = torch_module.zeros(EXPECTED_TOTAL_ITEMS, dtype=torch_module.bool)
        return embeddings, done

    print(f"[resume] Loading embedding progress: {path}")
    payload = torch_module.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("embedding_progress.pt must contain a mapping")
    if str(payload.get("records_sha256")) != record_sha:
        raise ValueError("Embedding progress belongs to a different item manifest")
    if str(payload.get("fashionclip_model_id")) != model_id:
        raise ValueError("Embedding progress belongs to a different FashionCLIP model")
    embeddings = payload.get("embeddings")
    done = payload.get("done_mask")
    if not isinstance(embeddings, torch_module.Tensor) or embeddings.shape != (
        EXPECTED_TOTAL_ITEMS,
        EXPECTED_EMBEDDING_DIM,
    ):
        raise ValueError("Embedding progress tensor has an invalid shape")
    if embeddings.dtype != torch_module.float32:
        embeddings = embeddings.float()
    if not isinstance(done, torch_module.Tensor) or done.shape != (EXPECTED_TOTAL_ITEMS,):
        raise ValueError("Embedding progress done_mask has an invalid shape")
    done = done.bool()
    print(f"[resume] completed item embeddings: {int(done.sum()):,}/{EXPECTED_TOTAL_ITEMS:,}")
    return embeddings, done


def save_embedding_progress(
    torch_module,
    path: Path,
    embeddings,
    done_mask,
    *,
    record_sha: str,
    model_id: str,
    embedding_version: str,
) -> None:
    payload = {
        "dataset": DATASET_NAME,
        "records_sha256": record_sha,
        "fashionclip_model_id": model_id,
        "embedding_version": embedding_version,
        "embedding_dim": EXPECTED_EMBEDDING_DIM,
        "embeddings": embeddings.cpu(),
        "done_mask": done_mask.cpu(),
    }
    atomic_torch_save(torch_module, payload, path)


def encode_fashionclip(
    *,
    repo_root: Path,
    zip_path: Path,
    records: Sequence[Mapping[str, object]],
    member_map: Mapping[int, str],
    output_dir: Path,
    model_id: str,
    embedding_version: str,
    device,
    batch_size: int,
    progress_every_batches: int,
    keep_progress: bool,
    record_sha: str,
):
    import torch
    from tqdm.auto import tqdm
    from transformers import CLIPModel, CLIPProcessor

    sys.path.insert(0, str(repo_root))
    # Reuse the branch helper so projected features behave identically under
    # Transformers v4/v5 return types.
    from src.detection.fashionclip import _extract_feature_tensor

    final_path = output_dir / f"{DATASET_NAME}-embeddings.pt"
    progress_path = output_dir / "embedding_progress.pt"

    if final_path.is_file():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Invalid embedding cache: {final_path}")
        if str(payload.get("records_sha256")) != record_sha:
            raise ValueError("Existing final embedding cache belongs to a different item manifest")
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, torch.Tensor) or embeddings.shape != (
            EXPECTED_TOTAL_ITEMS,
            EXPECTED_EMBEDDING_DIM,
        ):
            raise ValueError("Existing final embedding cache has invalid tensor shape")
        print(f"[embed] Reusing completed cache: {final_path}")
        return embeddings.float().cpu(), final_path

    embeddings, done_mask = load_embedding_progress(
        torch, progress_path, record_sha=record_sha, model_id=model_id
    )
    pending = torch.nonzero(~done_mask, as_tuple=False).flatten().tolist()
    if not pending:
        print("[embed] Progress already contains every item; finalizing cache")
    else:
        print(f"[FashionCLIP] Loading {model_id} on {device}")
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        model.eval().to(device)

        batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
        with zipfile.ZipFile(zip_path, "r") as zf:
            for batch_number, indices in enumerate(
                tqdm(batches, desc="FashionCLIP encode", unit="batch"), start=1
            ):
                images = [read_pil_image_from_zip(zf, member_map[index]) for index in indices]
                inputs = processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                with torch.inference_mode():
                    output = model.get_image_features(pixel_values=pixel_values)
                features = _extract_feature_tensor(output, feature_name="image features")
                features = features.float()
                norms = torch.linalg.vector_norm(features, dim=-1, keepdim=True).clamp_min(1e-12)
                features = features / norms

                if features.ndim != 2 or int(features.shape[1]) != EXPECTED_EMBEDDING_DIM:
                    raise ValueError(
                        "FashionCLIP embedding contract violated: "
                        f"expected [N,{EXPECTED_EMBEDDING_DIM}], got {tuple(features.shape)}"
                    )
                if not bool(torch.isfinite(features).all()):
                    raise ValueError("FashionCLIP produced NaN/Inf embeddings")

                cpu_features = features.detach().cpu()
                embeddings[indices] = cpu_features
                done_mask[indices] = True

                if progress_every_batches > 0 and batch_number % progress_every_batches == 0:
                    save_embedding_progress(
                        torch,
                        progress_path,
                        embeddings,
                        done_mask,
                        record_sha=record_sha,
                        model_id=model_id,
                        embedding_version=embedding_version,
                    )

                del images, inputs, pixel_values, output, features, cpu_features

        save_embedding_progress(
            torch,
            progress_path,
            embeddings,
            done_mask,
            record_sha=record_sha,
            model_id=model_id,
            embedding_version=embedding_version,
        )

        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if int(done_mask.sum()) != EXPECTED_TOTAL_ITEMS:
        raise RuntimeError(
            f"Embedding phase incomplete: {int(done_mask.sum())}/{EXPECTED_TOTAL_ITEMS}"
        )
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("Final embedding cache contains NaN/Inf")
    norms = torch.linalg.vector_norm(embeddings, dim=1)
    max_norm_error = float(torch.max(torch.abs(norms - 1.0)))
    if max_norm_error > 1e-3:
        raise ValueError(f"Embedding L2 norm contract violated; max error={max_norm_error}")

    final_payload = {
        "dataset": DATASET_NAME,
        "records_sha256": record_sha,
        "fashionclip_model_id": model_id,
        "embedding_version": embedding_version,
        "embedding_dim": EXPECTED_EMBEDDING_DIM,
        "role_order": list(ROLE_ORDER),
        "embeddings": embeddings.cpu(),
    }
    atomic_torch_save(torch, final_payload, final_path)
    if progress_path.exists() and not keep_progress:
        progress_path.unlink()
    print(
        f"[embed] PASS | {EXPECTED_TOTAL_ITEMS:,} x {EXPECTED_EMBEDDING_DIM} | "
        f"max_norm_error={max_norm_error:.3e} | {final_path}"
    )
    return embeddings.cpu(), final_path


def load_final_embeddings(path: Path, *, record_sha: str):
    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Embedding cache must contain a mapping")
    if str(payload.get("records_sha256")) != record_sha:
        raise ValueError("Embedding cache item-manifest SHA mismatch")
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.shape != (
        EXPECTED_TOTAL_ITEMS,
        EXPECTED_EMBEDDING_DIM,
    ):
        raise ValueError("Embedding cache tensor has invalid shape")
    embeddings = embeddings.float().cpu()
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("Embedding cache contains NaN/Inf")
    return embeddings


def find_checkpoint(
    explicit: Path | None,
    *,
    repo_root: Path,
    ml_root: Path,
    production: Mapping[str, object],
) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    relative = Path(str(production["checkpoint_path"]))
    candidates.extend(
        [
            ml_root
            / "scorer_runs"
            / "type_aware_pairwise_v1"
            / "final_val_auc_v5_seed42"
            / "best.pt",
            ml_root / relative,
            repo_root / relative,
        ]
    )
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file() and path.stat().st_size > 100_000:
            return path
    raise FileNotFoundError(
        "Frozen scorer checkpoint not found. Tried:\n  - " + "\n  - ".join(map(str, candidates))
    )


def validate_checkpoint_sha(path: Path, production: Mapping[str, object]) -> str:
    expected = str(production.get("checkpoint_sha256", "")).strip()
    actual = sha256_file(path)
    if expected and actual != expected:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    print(f"[checkpoint] PASS | SHA256={actual} | {path}")
    return actual


def write_scorer_ready_jsonl(
    path: Path,
    outfit_ids: Sequence[str],
    records: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_outfit: dict[str, list[Mapping[str, object]]] = {outfit_id: [] for outfit_id in outfit_ids}
    for row in records:
        by_outfit[str(row["outfit_id"])].append(row)

    with path.open("w", encoding="utf-8") as stream:
        for test_position, outfit_id in enumerate(outfit_ids, start=1):
            items = sorted(by_outfit[outfit_id], key=lambda row: ROLE_ORDER.index(str(row["role"])))
            payload = {
                "dataset": DATASET_NAME,
                "test_position": test_position,
                "outfit_id": outfit_id,
                "item_count": len(items),
                "items": [
                    {
                        "item_id": str(row["item_id"]),
                        "role": str(row["role"]),
                        "coarse_category": str(row["coarse_category"]),
                        "coarse_category_id": int(row["coarse_category_id"]),
                        "embedding_index": int(row["item_index"]),
                        "image_rel_path": str(row["image_rel_path"]),
                    }
                    for row in items
                ],
            }
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def score_outfits(
    *,
    repo_root: Path,
    checkpoint_path: Path,
    embeddings,
    outfit_ids: Sequence[str],
    category_id_by_role: Mapping[str, int],
    device,
    batch_size: int,
    output_path: Path,
) -> dict[str, object]:
    import pandas as pd
    import torch
    from tqdm.auto import tqdm

    sys.path.insert(0, str(repo_root))
    from src.scorer.checkpoint import load_checkpoint
    from src.scorer.model import TypeAwarePairwiseScorer

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    if str(payload["scorer_version"]) != EXPECTED_SCORER_VERSION:
        raise ValueError("Unexpected scorer version in checkpoint")
    model = TypeAwarePairwiseScorer.from_config(payload["config"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()

    if embeddings.shape != (EXPECTED_TOTAL_ITEMS, EXPECTED_EMBEDDING_DIM):
        raise ValueError("Embedding tensor shape does not match 2,000 x 4 Test2000 contract")

    outfit_tensor = embeddings.reshape(
        EXPECTED_TEST_OUTFITS, EXPECTED_ITEMS_PER_OUTFIT, EXPECTED_EMBEDDING_DIM
    )
    category_row = torch.tensor(
        [int(category_id_by_role[role]) for role in ROLE_ORDER], dtype=torch.long
    )

    rows: list[dict[str, object]] = []
    for start in tqdm(
        range(0, EXPECTED_TEST_OUTFITS, batch_size),
        desc="Scorer inference",
        unit="batch",
    ):
        end = min(EXPECTED_TEST_OUTFITS, start + batch_size)
        batch_embeddings = outfit_tensor[start:end].to(device=device, dtype=torch.float32)
        batch_categories = category_row.unsqueeze(0).repeat(end - start, 1).to(device)
        item_mask = torch.ones(
            (end - start, EXPECTED_ITEMS_PER_OUTFIT), dtype=torch.bool, device=device
        )
        with torch.inference_mode():
            output = model(
                item_embeddings=batch_embeddings,
                coarse_category_ids=batch_categories,
                item_mask=item_mask,
            )
        if not isinstance(output, Mapping) or "compatibility_logit" not in output:
            raise RuntimeError("Scorer did not return compatibility_logit")
        logits = output["compatibility_logit"]
        if not isinstance(logits, torch.Tensor) or logits.shape != (end - start,):
            raise RuntimeError(
                f"Scorer compatibility_logit shape violation: got {getattr(logits, 'shape', None)}"
            )
        logits_cpu = logits.detach().float().cpu()
        if not bool(torch.isfinite(logits_cpu).all()):
            raise RuntimeError("Scorer returned non-finite compatibility_logit")

        for local_index, logit in enumerate(logits_cpu.tolist()):
            outfit_index = start + local_index
            rows.append(
                {
                    "test_position": outfit_index + 1,
                    "outfit_id": outfit_ids[outfit_index],
                    "dataset": DATASET_NAME,
                    "item_count": EXPECTED_ITEMS_PER_OUTFIT,
                    "role_order": "|".join(ROLE_ORDER),
                    "coarse_category_ids": "|".join(
                        str(int(category_id_by_role[role])) for role in ROLE_ORDER
                    ),
                    "compatibility_logit": float(logit),
                    "scorer_version": str(payload["scorer_version"]),
                    "checkpoint_epoch": int(payload["epoch"]),
                    "checkpoint_best_valid_roc_auc": float(payload["best_valid_roc_auc"]),
                }
            )

    if len(rows) != EXPECTED_TEST_OUTFITS:
        raise RuntimeError(f"Expected 2,000 predictions, got {len(rows)}")
    frame = pd.DataFrame(rows)
    if not frame["compatibility_logit"].map(math.isfinite).all():
        raise RuntimeError("Prediction frame contains non-finite logits")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(
        f"[score] PASS | predictions={len(frame):,} | "
        f"logit_mean={frame['compatibility_logit'].mean():.6f} | "
        f"logit_std={frame['compatibility_logit'].std(ddof=0):.6f} | {output_path}"
    )

    return {
        "scorer_version": str(payload["scorer_version"]),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_best_valid_roc_auc": float(payload["best_valid_roc_auc"]),
        "prediction_count": len(frame),
        "logit_mean": float(frame["compatibility_logit"].mean()),
        "logit_std": float(frame["compatibility_logit"].std(ddof=0)),
        "logit_min": float(frame["compatibility_logit"].min()),
        "logit_max": float(frame["compatibility_logit"].max()),
    }


def main() -> None:
    args = parse_args()
    require_positive("encode_batch_size", args.encode_batch_size)
    require_positive("scorer_batch_size", args.scorer_batch_size)

    maybe_mount_drive(enabled=not args.no_auto_mount_drive)

    e3_root = args.e3_root
    ml_root = args.ml_root
    output_dir = args.output_dir or (e3_root / DEFAULT_OUTPUT_NAME)
    test_ids_path = args.test_ids or (e3_root / "test2000_ids.csv")
    freeze_manifest_path = args.freeze_manifest or (e3_root / "test2000_freeze_manifest.json")
    drive_zip = args.drive_zip or (e3_root / "outfit.zip")
    stage_dir = Path("/content/eval3_test2000_full_stage")

    output_dir.mkdir(parents=True, exist_ok=True)

    commit = ensure_repo(args.repo_root, args.branch, auto_clone=not args.no_auto_clone)
    production, detection = load_repo_contract(args.repo_root)
    outfit_ids, freeze = load_test_ids(test_ids_path, freeze_manifest_path)

    fashionclip = detection["fashionclip"]
    scorer_handoff = detection["scorer_handoff"]
    assert isinstance(fashionclip, Mapping)
    assert isinstance(scorer_handoff, Mapping)
    category_ids = scorer_handoff["category_ids"]
    assert isinstance(category_ids, Mapping)

    model_id = str(fashionclip["model_id"])
    embedding_version = str(production["embedding_version"])
    category_id_by_role = {
        role: int(category_ids[ROLE_TO_CATEGORY[role]]) for role in ROLE_ORDER
    }
    print(
        "[category] fixed EVAL3 role mapping | "
        + " | ".join(
            f"{role}->{ROLE_TO_CATEGORY[role]}:{category_id_by_role[role]}"
            for role in ROLE_ORDER
        )
    )

    records = build_item_records(outfit_ids, category_ids)
    record_sha = records_sha256(records)
    items_csv = output_dir / f"{DATASET_NAME}-items.csv"
    scorer_ready = output_dir / f"{DATASET_NAME}-scorer-ready.jsonl"
    embeddings_path = output_dir / f"{DATASET_NAME}-embeddings.pt"
    predictions_path = output_dir / f"{DATASET_NAME}-predictions.csv"
    run_manifest_path = output_dir / f"{DATASET_NAME}-run-manifest.json"

    write_items_csv(items_csv, records)
    write_scorer_ready_jsonl(scorer_ready, outfit_ids, records)
    print(f"[manifest] items={items_csv}")
    print(f"[manifest] scorer-ready={scorer_ready}")

    import torch

    device = resolve_device(torch, args.device)
    print(f"[device] {device}")

    zip_path: Path | None = None
    embeddings = None
    if args.mode in {"all", "encode"}:
        zip_path = ensure_local_zip(drive_zip, stage_dir, args.local_zip)
        with zipfile.ZipFile(zip_path, "r") as zf:
            member_map = build_zip_member_map(zf, records)
        embeddings, embeddings_path = encode_fashionclip(
            repo_root=args.repo_root,
            zip_path=zip_path,
            records=records,
            member_map=member_map,
            output_dir=output_dir,
            model_id=model_id,
            embedding_version=embedding_version,
            device=device,
            batch_size=args.encode_batch_size,
            progress_every_batches=args.progress_every_batches,
            keep_progress=args.keep_progress,
            record_sha=record_sha,
        )

    if args.mode == "score":
        embeddings = load_final_embeddings(embeddings_path, record_sha=record_sha)

    checkpoint_path: Path | None = None
    checkpoint_sha: str | None = None
    score_summary: dict[str, object] | None = None
    if args.mode in {"all", "score"}:
        assert embeddings is not None
        checkpoint_path = find_checkpoint(
            args.checkpoint,
            repo_root=args.repo_root,
            ml_root=ml_root,
            production=production,
        )
        checkpoint_sha = validate_checkpoint_sha(checkpoint_path, production)
        score_summary = score_outfits(
            repo_root=args.repo_root,
            checkpoint_path=checkpoint_path,
            embeddings=embeddings,
            outfit_ids=outfit_ids,
            category_id_by_role=category_id_by_role,
            device=device,
            batch_size=args.scorer_batch_size,
            output_path=predictions_path,
        )

    manifest = {
        "dataset": DATASET_NAME,
        "source_split_name": freeze.get("split_name"),
        "outfit_count": len(outfit_ids),
        "item_count": len(records),
        "items_per_outfit": EXPECTED_ITEMS_PER_OUTFIT,
        "ordered_test_ids_sha256": sha256_ordered_ids(outfit_ids),
        "records_sha256": record_sha,
        "git": {
            "repository": REPO_URL,
            "branch": args.branch,
            "commit": commit,
        },
        "preprocessing": {
            "fashionclip_model_id": model_id,
            "embedding_version": embedding_version,
            "embedding_dim": EXPECTED_EMBEDDING_DIM,
            "dtype": "float32",
            "normalization": "l2",
            "method": "CLIPProcessor -> CLIPModel.get_image_features -> float32 -> L2",
            "rf_detr_used": False,
            "category_classifier_used": False,
            "category_assignment_reason": "EVALUATION3 U/B/S/G roles are already known",
        },
        "role_order": list(ROLE_ORDER),
        "role_to_category": dict(ROLE_TO_CATEGORY),
        "role_to_category_id": category_id_by_role,
        "scorer": {
            "scorer_version": str(production["scorer_version"]),
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "output": "compatibility_logit",
            "score_summary": score_summary,
        },
        "outputs": {
            "items_csv": str(items_csv),
            "embeddings_pt": str(embeddings_path),
            "scorer_ready_jsonl": str(scorer_ready),
            "predictions_csv": str(predictions_path) if predictions_path.exists() else None,
        },
    }
    atomic_json_write(run_manifest_path, manifest)
    print(f"[run-manifest] {run_manifest_path}")
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
