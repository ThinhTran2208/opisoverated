"""Min-items-2 helpers for Core-7 embedding validation/repair.

The canonical validator remains unchanged for the frozen V2 benchmark. This
experiment reuses its strict cache/manifest checks, but any repair after
missing/invalid embeddings must retain outfits once at least two valid items
remain.
"""

from __future__ import annotations

from typing import Sequence

from .validate_core7_embeddings import repair_split


MIN_SCORER_ITEMS = 2


def repair_split_min2(
    positives: Sequence[dict],
    metadata: Sequence[dict],
    usable_item_ids: set[str],
) -> tuple[list[dict], list[dict], dict]:
    """Repair one split using the experiment's two-item minimum."""

    return repair_split(
        positives,
        metadata,
        usable_item_ids,
        min_items=MIN_SCORER_ITEMS,
    )
