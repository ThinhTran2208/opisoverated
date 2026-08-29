"""Experiment wrapper for building Core-7 scorer artifacts with min_items=2.

This branch intentionally leaves the frozen canonical V2 builder untouched.
The wrapper forces the final scorer-ready validation to accept outfits with
2..8 items while retaining the same category mapping and Negative V1 protocol.
Use an isolated runtime output directory for this experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import build_core7_scorer_dataset as base


MIN_SCORER_ITEMS = 2
MIN_LOO_ORIGINAL_ITEMS = 3
MAX_SCORER_ITEMS = 8
EXPERIMENT_ID = "min-items-2-loo-min3-v1"
EXPERIMENT_MANIFEST = "experiment_manifest_min2.json"


def _require_isolated_output_dir(output_dir: Path) -> None:
    if output_dir.name == "scorer_ready_v2":
        raise ValueError(
            "The min-items-2 experiment must not overwrite frozen scorer_ready_v2; "
            "use an isolated directory such as scorer_ready_v2_min2_exp."
        )


def build_scorer_dataset_min2(**kwargs: Any) -> dict:
    """Run the existing V2 builder with a 2-item final validation boundary.

    ``build_scorer_dataset_v2`` resolves ``validate_all_splits`` and
    ``DEFAULT_MIN_ITEMS`` from module globals at call time. We temporarily bind
    those globals for this isolated experiment and restore them afterwards so
    importing this module does not mutate normal V2 behavior elsewhere.
    """

    if "output_dir" not in kwargs:
        raise TypeError("build_scorer_dataset_min2 requires output_dir")
    output_dir = Path(kwargs["output_dir"])
    _require_isolated_output_dir(output_dir)

    original_default_min = base.DEFAULT_MIN_ITEMS
    original_validate_all_splits = base.validate_all_splits

    def validate_all_splits_min2(*args, **validation_kwargs):
        validation_kwargs.setdefault("min_items", MIN_SCORER_ITEMS)
        return original_validate_all_splits(*args, **validation_kwargs)

    try:
        base.DEFAULT_MIN_ITEMS = MIN_SCORER_ITEMS
        base.validate_all_splits = validate_all_splits_min2
        result = base.build_scorer_dataset_v2(**kwargs)
    finally:
        base.DEFAULT_MIN_ITEMS = original_default_min
        base.validate_all_splits = original_validate_all_splits

    result = dict(result)
    result["experiment_id"] = EXPERIMENT_ID
    result["minimum_outfit_items"] = MIN_SCORER_ITEMS
    result["minimum_loo_original_items"] = MIN_LOO_ORIGINAL_ITEMS

    experiment_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "base_dataset_version": result.get("dataset_version", base.DATASET_VERSION),
        "minimum_scorer_items": MIN_SCORER_ITEMS,
        "maximum_scorer_items": MAX_SCORER_ITEMS,
        "minimum_loo_original_items": MIN_LOO_ORIGINAL_ITEMS,
        "status": result.get("status"),
        "canonical_v2_artifacts_overwritten": False,
    }
    base.write_json(experiment_manifest, output_dir / EXPERIMENT_MANIFEST)
    result["experiment_manifest"] = str(output_dir / EXPERIMENT_MANIFEST)
    return result
