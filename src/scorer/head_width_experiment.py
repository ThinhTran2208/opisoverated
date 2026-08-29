# -*- coding: utf-8 -*-
"""Controlled output-head-width ablation for the frozen V5 scorer recipe.

This module is experiment-only. It does not alter the canonical scorer model or
checkpoint contract. The helper keeps the category/item/pair initialization
identical to the canonical width-16 seed, then changes only the output MLP width
at initialization time.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from .model import TypeAwarePairwiseScorer
from .train import seed_everything


CANONICAL_OUTPUT_HIDDEN_DIM = 16


def build_output_head_width_variant(
    config: Mapping[str, object],
    *,
    output_hidden_dim: int,
    seed: int = 42,
) -> TypeAwarePairwiseScorer:
    """Build a fair head-width variant with shared modules initialized identically.

    The scorer re-initializes category embeddings after constructing the output
    MLP. Therefore simply changing ``output_hidden_dim`` changes RNG consumption
    and would also change category initialization. For this ablation we first
    build the canonical width-16 reference at the requested seed, then build the
    wider variant and copy all shared modules from the reference:

    - category_embedding
    - item_mlp
    - pair_mlp

    Only ``output_mlp`` keeps the width-specific initialization. All parameters
    remain trainable during the subsequent full training run.
    """

    if isinstance(output_hidden_dim, bool) or int(output_hidden_dim) < 1:
        raise ValueError("output_hidden_dim must be a positive integer")

    seed_everything(seed)
    reference_config = deepcopy(dict(config))
    reference_model_config = dict(reference_config["model"])
    reference_model_config["output_hidden_dim"] = CANONICAL_OUTPUT_HIDDEN_DIM
    reference_config["model"] = reference_model_config
    reference = TypeAwarePairwiseScorer.from_config(reference_config)

    seed_everything(seed)
    variant_config = deepcopy(dict(config))
    variant_model_config = dict(variant_config["model"])
    variant_model_config["output_hidden_dim"] = int(output_hidden_dim)
    variant_config["model"] = variant_model_config
    variant = TypeAwarePairwiseScorer.from_config(variant_config)

    variant.category_embedding.load_state_dict(reference.category_embedding.state_dict())
    variant.item_mlp.load_state_dict(reference.item_mlp.state_dict())
    variant.pair_mlp.load_state_dict(reference.pair_mlp.state_dict())

    return variant
