# -*- coding: utf-8 -*-
"""Leave-One-Out helpers for outfit diagnosis.

The compatibility scorer experiment accepts 2-item outfits, but LOO must start
from at least 3 items so every residual outfit still contains a valid item pair.
"""

from __future__ import annotations

from typing import Sequence, TypeVar


T = TypeVar("T")
SCORER_MIN_ITEMS = 2
LOO_MIN_ORIGINAL_ITEMS = 3


class LOOInputError(ValueError):
    """Raised when an outfit cannot be diagnosed with leave-one-out scoring."""


def validate_loo_item_count(item_count: int) -> None:
    item_count = int(item_count)
    if item_count < LOO_MIN_ORIGINAL_ITEMS:
        raise LOOInputError(
            "LOO diagnosis requires at least "
            f"{LOO_MIN_ORIGINAL_ITEMS} original items; got {item_count}. "
            "A 2-item original outfit would leave only 1 item after removal, "
            "which has no valid compatibility pair."
        )


def build_leave_one_out_outfits(items: Sequence[T]) -> list[list[T]]:
    """Return one residual outfit per removed item, preserving item order."""

    validate_loo_item_count(len(items))
    residuals = [
        [item for index, item in enumerate(items) if index != removed_index]
        for removed_index in range(len(items))
    ]
    if any(len(residual) < SCORER_MIN_ITEMS for residual in residuals):
        raise RuntimeError("LOO produced a residual below scorer minimum")
    return residuals


def loo_removed_indices(items: Sequence[T]) -> list[int]:
    """Return the removal index corresponding to each generated LOO residual."""

    validate_loo_item_count(len(items))
    return list(range(len(items)))
