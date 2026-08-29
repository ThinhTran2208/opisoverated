"""Leave-One-Out diagnosis boundary for the min-items-2 experiment.

Compatibility scoring accepts outfits with at least two real items. LOO
localization requires at least three items in the original outfit so that each
leave-one-out residual still contains at least two items for the scorer.
"""

from __future__ import annotations

from typing import Sequence, TypeVar


T = TypeVar("T")
MIN_SCORER_ITEMS = 2
MIN_LOO_ORIGINAL_ITEMS = 3


def validate_loo_original_item_count(item_count: int) -> None:
    """Reject LOO diagnosis when the original outfit has fewer than 3 items."""

    if item_count < MIN_LOO_ORIGINAL_ITEMS:
        raise ValueError(
            "LOO diagnosis requires at least "
            f"{MIN_LOO_ORIGINAL_ITEMS} original items; got {item_count}. "
            "Two-item outfits may be compatibility-scored but are not eligible "
            "for LOO problematic-item localization."
        )


def build_loo_subsets(items: Sequence[T]) -> list[tuple[int, list[T]]]:
    """Return ``(removed_index, residual_items)`` for every original item.

    For the minimum valid original size ``n=3``, every residual has exactly two
    items and is therefore valid input to the compatibility scorer.
    """

    original = list(items)
    validate_loo_original_item_count(len(original))

    subsets = [
        (index, original[:index] + original[index + 1 :])
        for index in range(len(original))
    ]
    if any(len(residual) < MIN_SCORER_ITEMS for _, residual in subsets):
        raise AssertionError("LOO residual violated the scorer minimum item count")
    return subsets
