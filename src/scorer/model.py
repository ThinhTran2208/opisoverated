# -*- coding: utf-8 -*-
"""Type-aware Pairwise Scorer V1.

This module implements the locked architecture in ``docs/SCORER_CONTRACT_V1.md``:

FashionCLIP 512-d embedding + learned Core-7 category embedding
    -> item MLP
    -> all valid unordered item pairs
    -> bidirectional symmetric pair MLP
    -> mean over valid pair scores
    -> output MLP
    -> compatibility_logit

The scorer is permutation-invariant with respect to outfit item ordering and
ignores padded positions through ``item_mask``.
"""

from __future__ import annotations

from typing import Mapping

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # Keep source importable in lightweight portability CI.
    torch = None
    nn = None


SCORER_VERSION = "type_aware_pairwise_v1"
EMBEDDING_DIM = 512
CATEGORY_COUNT = 7
CATEGORY_VOCAB_SIZE = 8
CATEGORY_PADDING_IDX = 0
CATEGORY_EMBEDDING_DIM = 32
ITEM_PROJECTION_DIM = 256
ITEM_HIDDEN_DIM = 128
PAIR_HIDDEN_DIM = 128
OUTPUT_HIDDEN_DIM = 16
DROPOUT = 0.2
MIN_ITEMS = 2
MAX_ITEMS = 8

_SUPPORTED_ACTIVATION = "relu"
_SUPPORTED_AGGREGATION = "mean"
_SUPPORTED_PAIR_SYMMETRY = "bidirectional_mean"


def require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for TypeAwarePairwiseScorer. "
            "Install torch in the training environment."
        )


_ModuleBase = nn.Module if nn is not None else object


class TypeAwarePairwiseScorer(_ModuleBase):
    """Locked Type-aware Pairwise V1 scorer."""

    def __init__(
        self,
        *,
        embedding_dim: int = EMBEDDING_DIM,
        category_count: int = CATEGORY_COUNT,
        category_vocab_size: int = CATEGORY_VOCAB_SIZE,
        category_padding_idx: int = CATEGORY_PADDING_IDX,
        category_embedding_dim: int = CATEGORY_EMBEDDING_DIM,
        item_projection_dim: int = ITEM_PROJECTION_DIM,
        item_hidden_dim: int = ITEM_HIDDEN_DIM,
        pair_hidden_dim: int = PAIR_HIDDEN_DIM,
        output_hidden_dim: int = OUTPUT_HIDDEN_DIM,
        dropout: float = DROPOUT,
        activation: str = _SUPPORTED_ACTIVATION,
        aggregation: str = _SUPPORTED_AGGREGATION,
        pair_symmetry: str = _SUPPORTED_PAIR_SYMMETRY,
        min_items: int = MIN_ITEMS,
        max_items: int = MAX_ITEMS,
    ) -> None:
        require_torch()
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if category_embedding_dim <= 0:
            raise ValueError("category_embedding_dim must be positive")
        if category_count != category_vocab_size - 1:
            raise ValueError(
                "Scorer V1 expects category_vocab_size = category_count + 1 "
                "for the PAD category"
            )
        if category_padding_idx != 0:
            raise ValueError("Scorer V1 locks category_padding_idx = 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if min_items < 2 or max_items < min_items:
            raise ValueError("Require 2 <= min_items <= max_items")
        if activation.lower() != _SUPPORTED_ACTIVATION:
            raise ValueError("Scorer V1 supports activation='relu' only")
        if aggregation.lower() != _SUPPORTED_AGGREGATION:
            raise ValueError("Scorer V1 supports aggregation='mean' only")
        if pair_symmetry.lower() != _SUPPORTED_PAIR_SYMMETRY:
            raise ValueError(
                "Scorer V1 supports pair_symmetry='bidirectional_mean' only"
            )

        self.scorer_version = SCORER_VERSION
        self.embedding_dim = int(embedding_dim)
        self.category_count = int(category_count)
        self.category_vocab_size = int(category_vocab_size)
        self.category_padding_idx = int(category_padding_idx)
        self.category_embedding_dim = int(category_embedding_dim)
        self.item_projection_dim = int(item_projection_dim)
        self.item_hidden_dim = int(item_hidden_dim)
        self.pair_hidden_dim = int(pair_hidden_dim)
        self.output_hidden_dim = int(output_hidden_dim)
        self.dropout = float(dropout)
        self.min_items = int(min_items)
        self.max_items = int(max_items)

        # Keep PyTorch's normal module-construction RNG order intact first.
        # We deliberately rescale the category embedding only after every MLP
        # has been constructed. This preserves the downstream Linear
        # initialization that produced the strongest FP32 diagnostic run,
        # while still fixing the category-vs-FashionCLIP feature-scale issue.
        self.category_embedding = nn.Embedding(
            num_embeddings=self.category_vocab_size,
            embedding_dim=self.category_embedding_dim,
            padding_idx=self.category_padding_idx,
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(
                self.embedding_dim + self.category_embedding_dim,
                self.item_projection_dim,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.item_projection_dim, self.item_hidden_dim),
            nn.ReLU(),
        )

        pair_feature_dim = 4 * self.item_hidden_dim + 2 * self.category_embedding_dim
        self.pair_feature_dim = pair_feature_dim

        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_feature_dim, self.pair_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.pair_hidden_dim, 1),
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(1, self.output_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.output_hidden_dim, 1),
        )

        # FashionCLIP inputs are frozen L2-normalized vectors (norm ~= 1).
        # The default 32-d nn.Embedding vectors have norm ~= sqrt(32), which
        # experimentally overwhelmed the FashionCLIP signal.  Rescale only
        # the category table, after downstream module initialization, so the
        # category vectors also start with expected squared norm ~= 1 without
        # perturbing the MLP initialization RNG trajectory.
        self.category_embedding_init_policy = "post_mlp_scale_preserving"
        self.category_embedding_init_std = self.category_embedding_dim ** -0.5
        nn.init.normal_(
            self.category_embedding.weight,
            mean=0.0,
            std=self.category_embedding_init_std,
        )
        with torch.no_grad():
            self.category_embedding.weight[self.category_padding_idx].zero_()

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "TypeAwarePairwiseScorer":
        """Build from the full scorer YAML payload or its ``model`` section."""

        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")

        full_config = config
        model_config = config.get("model", config)
        if not isinstance(model_config, Mapping):
            raise ValueError("config['model'] must be a mapping")

        model_name = str(model_config.get("name", SCORER_VERSION))
        if model_name != SCORER_VERSION:
            raise ValueError(
                f"Expected model.name={SCORER_VERSION!r}, got {model_name!r}"
            )

        data_config = full_config.get("data", {})
        if not isinstance(data_config, Mapping):
            raise ValueError("config['data'] must be a mapping when provided")

        return cls(
            embedding_dim=int(model_config.get("embedding_dim", EMBEDDING_DIM)),
            category_count=int(model_config.get("category_count", CATEGORY_COUNT)),
            category_vocab_size=int(
                model_config.get("category_vocab_size", CATEGORY_VOCAB_SIZE)
            ),
            category_padding_idx=int(
                model_config.get("category_padding_idx", CATEGORY_PADDING_IDX)
            ),
            category_embedding_dim=int(
                model_config.get("category_embedding_dim", CATEGORY_EMBEDDING_DIM)
            ),
            item_projection_dim=int(
                model_config.get("item_projection_dim", ITEM_PROJECTION_DIM)
            ),
            item_hidden_dim=int(model_config.get("item_hidden_dim", ITEM_HIDDEN_DIM)),
            pair_hidden_dim=int(model_config.get("pair_hidden_dim", PAIR_HIDDEN_DIM)),
            output_hidden_dim=int(
                model_config.get("output_hidden_dim", OUTPUT_HIDDEN_DIM)
            ),
            dropout=float(model_config.get("dropout", DROPOUT)),
            activation=str(model_config.get("activation", _SUPPORTED_ACTIVATION)),
            aggregation=str(model_config.get("aggregation", _SUPPORTED_AGGREGATION)),
            pair_symmetry=str(
                model_config.get("pair_symmetry", _SUPPORTED_PAIR_SYMMETRY)
            ),
            min_items=int(data_config.get("min_items", MIN_ITEMS)),
            max_items=int(data_config.get("max_items", MAX_ITEMS)),
        )

    def _validate_inputs(
        self,
        item_embeddings,
        coarse_category_ids,
        item_mask,
        pair_mask,
    ) -> None:
        if not isinstance(item_embeddings, torch.Tensor):
            raise TypeError("item_embeddings must be a torch.Tensor")
        if not isinstance(coarse_category_ids, torch.Tensor):
            raise TypeError("coarse_category_ids must be a torch.Tensor")
        if not isinstance(item_mask, torch.Tensor):
            raise TypeError("item_mask must be a torch.Tensor")

        if item_embeddings.ndim != 3:
            raise ValueError(
                "item_embeddings must have shape [B, L, D], "
                f"got {tuple(item_embeddings.shape)}"
            )
        batch_size, length, embedding_dim = item_embeddings.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, got {embedding_dim}"
            )
        if length > self.max_items:
            raise ValueError(f"Input length={length} exceeds max_items={self.max_items}")
        if coarse_category_ids.shape != (batch_size, length):
            raise ValueError(
                "coarse_category_ids must have shape [B, L] matching embeddings"
            )
        if item_mask.shape != (batch_size, length):
            raise ValueError("item_mask must have shape [B, L] matching embeddings")
        if coarse_category_ids.dtype != torch.long:
            raise ValueError("coarse_category_ids must have dtype torch.long")
        if item_mask.dtype != torch.bool:
            raise ValueError("item_mask must have dtype torch.bool")
        if not item_embeddings.is_floating_point():
            raise ValueError("item_embeddings must be floating point")
        if not torch.isfinite(item_embeddings).all():
            raise ValueError("item_embeddings contains NaN or Inf")

        if torch.any(coarse_category_ids < 0) or torch.any(
            coarse_category_ids >= self.category_vocab_size
        ):
            raise ValueError("coarse_category_ids contains an out-of-vocabulary ID")

        real_category_ids = coarse_category_ids[item_mask]
        if real_category_ids.numel() and torch.any(
            real_category_ids == self.category_padding_idx
        ):
            raise ValueError("Real items may not use the PAD category ID")

        padded_category_ids = coarse_category_ids[~item_mask]
        if padded_category_ids.numel() and torch.any(
            padded_category_ids != self.category_padding_idx
        ):
            raise ValueError("Padded positions must use category ID 0")

        item_counts = item_mask.sum(dim=1)
        if torch.any(item_counts < self.min_items) or torch.any(
            item_counts > self.max_items
        ):
            raise ValueError(
                f"Every outfit must contain [{self.min_items}, {self.max_items}] "
                "real items"
            )

        if pair_mask is not None:
            if not isinstance(pair_mask, torch.Tensor):
                raise TypeError("pair_mask must be a torch.Tensor when provided")
            if pair_mask.shape != (batch_size, length, length):
                raise ValueError("pair_mask must have shape [B, L, L]")
            if pair_mask.dtype != torch.bool:
                raise ValueError("pair_mask must have dtype torch.bool")

    def _pair_indices(self, length: int, device):
        return torch.triu_indices(length, length, offset=1, device=device)

    def _expected_pair_mask(self, item_mask):
        batch_size, length = item_mask.shape
        upper = torch.triu(
            torch.ones(
                (length, length), dtype=torch.bool, device=item_mask.device
            ),
            diagonal=1,
        )
        return (
            item_mask.unsqueeze(2)
            & item_mask.unsqueeze(1)
            & upper.unsqueeze(0).expand(batch_size, -1, -1)
        )

    def forward(
        self,
        item_embeddings,
        coarse_category_ids,
        item_mask,
        pair_mask=None,
    ) -> dict[str, object]:
        """Return ``{\"compatibility_logit\": Tensor[B]}``.

        ``pair_mask`` is optional because it is derivable from ``item_mask``.
        When supplied by the S1 collator, it is validated against the canonical
        ``i < j`` real-item pair mask before being consumed.
        """

        self._validate_inputs(
            item_embeddings,
            coarse_category_ids,
            item_mask,
            pair_mask,
        )

        expected_pair_mask = self._expected_pair_mask(item_mask)
        if pair_mask is None:
            pair_mask = expected_pair_mask
        elif not torch.equal(pair_mask, expected_pair_mask):
            raise ValueError(
                "pair_mask does not match the canonical real-item upper-triangle mask"
            )

        category_vectors = self.category_embedding(coarse_category_ids)
        item_features = torch.cat([item_embeddings, category_vectors], dim=-1)
        item_representations = self.item_mlp(item_features)

        _, length, _ = item_representations.shape
        pair_indices = self._pair_indices(length, item_representations.device)
        left_indices = pair_indices[0]
        right_indices = pair_indices[1]

        h_i = item_representations[:, left_indices, :]
        h_j = item_representations[:, right_indices, :]
        c_i = category_vectors[:, left_indices, :]
        c_j = category_vectors[:, right_indices, :]

        abs_difference = torch.abs(h_i - h_j)
        interaction = h_i * h_j

        forward_features = torch.cat(
            [h_i, h_j, abs_difference, interaction, c_i, c_j], dim=-1
        )
        reverse_features = torch.cat(
            [h_j, h_i, abs_difference, interaction, c_j, c_i], dim=-1
        )

        forward_scores = self.pair_mlp(forward_features).squeeze(-1)
        reverse_scores = self.pair_mlp(reverse_features).squeeze(-1)
        pair_scores = 0.5 * (forward_scores + reverse_scores)

        valid_pairs = pair_mask[:, left_indices, right_indices]
        pair_counts = valid_pairs.sum(dim=1)
        if torch.any(pair_counts == 0):
            raise ValueError("Every outfit must contain at least one valid item pair")

        masked_pair_scores = pair_scores * valid_pairs.to(pair_scores.dtype)
        mean_pair_score = (
            masked_pair_scores.sum(dim=1, keepdim=True)
            / pair_counts.to(pair_scores.dtype).unsqueeze(-1)
        )

        compatibility_logit = self.output_mlp(mean_pair_score).squeeze(-1)
        return {"compatibility_logit": compatibility_logit}
