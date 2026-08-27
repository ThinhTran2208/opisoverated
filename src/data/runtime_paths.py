"""Resolve project and external artifact paths without platform assumptions.

The data pipeline is allowed to run on Colab, a local workstation, or a
server. Core code therefore must not depend on ``google.colab`` or hard-coded
``/content/drive`` paths. Each user can select an artifact root with an
environment variable or an ignored local config file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


ENV_PROJECT_ROOT = "FASHION_PROJECT_ROOT"
ENV_PATHS_CONFIG = "FASHION_PATHS_CONFIG"
ENV_ARTIFACT_ROOT = "FASHION_ARTIFACT_ROOT"
ENV_EMBEDDING_CACHE = "FASHION_EMBEDDING_CACHE"
ENV_EMBEDDING_MANIFEST = "FASHION_EMBEDDING_MANIFEST"
ENV_CORE7_DIR = "FASHION_CORE7_DIR"
ENV_SCORER_READY_DIR = "FASHION_SCORER_READY_DIR"


@dataclass(frozen=True)
class RuntimePaths:
    """Canonical paths used by NB2-NB4 and their CLI equivalents."""

    repo_root: Path
    config_path: Path
    artifact_root: Path
    embedding_cache: Path
    embedding_manifest: Path
    core7_dir: Path
    scorer_ready_dir: Path


def find_repo_root(start: Path | str | None = None) -> Optional[Path]:
    """Find the repository by walking upward from ``start`` or cwd."""

    explicit = os.environ.get(ENV_PROJECT_ROOT)
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if _is_repo_root(candidate):
            return candidate
        raise FileNotFoundError(
            f"{ENV_PROJECT_ROOT} does not point to this repository: {candidate}"
        )

    current = Path(start or Path.cwd()).expanduser().resolve()
    for candidate in (current, *current.parents):
        if _is_repo_root(candidate):
            return candidate
    return None


def load_runtime_paths(
    repo_root: Path | str | None = None,
    config_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> RuntimePaths:
    """Load portable artifact paths.

    Precedence for every path is:

    1. explicit environment variable;
    2. ``configs/data_paths.local.json`` when present;
    3. ``configs/data_paths.example.json`` committed in the repository.

    ``artifact_root`` is resolved relative to the repository. The remaining
    paths are resolved relative to ``artifact_root`` unless they are absolute.
    """

    environment = os.environ if env is None else env
    root = Path(repo_root).expanduser().resolve() if repo_root else find_repo_root()
    if root is None or not _is_repo_root(root):
        raise FileNotFoundError(
            "Cannot find the opisoverated repository. Run the notebook from "
            "the cloned repository or set FASHION_PROJECT_ROOT."
        )

    selected_config = _select_config_path(root, config_path, environment)
    payload = _read_config(selected_config)

    artifact_root_text = environment.get(
        ENV_ARTIFACT_ROOT,
        str(payload.get("artifact_root", "./data")),
    )
    artifact_root = _resolve_path(artifact_root_text, root)

    embedding_cache = _resolve_path(
        environment.get(
            ENV_EMBEDDING_CACHE,
            str(payload.get("embedding_cache", "cache/fashionclip_item_embeddings.pt")),
        ),
        artifact_root,
    )
    embedding_manifest = _resolve_path(
        environment.get(
            ENV_EMBEDDING_MANIFEST,
            str(payload.get("embedding_manifest", "cache/embedding_manifest_v1.json")),
        ),
        artifact_root,
    )
    core7_dir = _resolve_path(
        environment.get(
            ENV_CORE7_DIR,
            str(payload.get("core7_dir", "core7_drop_v2")),
        ),
        artifact_root,
    )
    scorer_ready_dir = _resolve_path(
        environment.get(
            ENV_SCORER_READY_DIR,
            str(payload.get("scorer_ready_dir", "scorer_ready_v2")),
        ),
        artifact_root,
    )

    return RuntimePaths(
        repo_root=root,
        config_path=selected_config,
        artifact_root=artifact_root,
        embedding_cache=embedding_cache,
        embedding_manifest=embedding_manifest,
        core7_dir=core7_dir,
        scorer_ready_dir=scorer_ready_dir,
    )


def _is_repo_root(path: Path) -> bool:
    return (
        (path / "configs" / "category_mapping_core7_v1.json").is_file()
        and (path / "src" / "data" / "prepare_core7_dataset.py").is_file()
    )


def _select_config_path(
    root: Path,
    explicit: Path | str | None,
    environment: Mapping[str, str],
) -> Path:
    requested = explicit or environment.get(ENV_PATHS_CONFIG)
    if requested:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Runtime path config not found: {path}")
        return path

    local = root / "configs" / "data_paths.local.json"
    if local.is_file():
        return local.resolve()

    example = root / "configs" / "data_paths.example.json"
    if not example.is_file():
        raise FileNotFoundError(f"Missing committed path config: {example}")
    return example.resolve()


def _read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime path config must be a JSON object: {path}")
    return payload


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()
