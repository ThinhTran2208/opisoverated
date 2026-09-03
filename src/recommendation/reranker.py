# -*- coding: utf-8 -*-
"""Frozen V5 compatibility-scorer reranking."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import torch
except ModuleNotFoundError:  # Keep package importable in lightweight CI.
    torch = None

from src.scorer.checkpoint import load_checkpoint, sha256_file
from src.scorer.model import TypeAwarePairwiseScorer


FROZEN_V5_SHA256 = "7b3d0b6e0d44e3de517565f5725ded198bbc762b02a4dece26a58ee145cfed9c"


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for Recommendation V1 reranking")


@dataclass(frozen=True)
class RerankedCandidate:
    item_id: str
    compatibility_logit: float
    improvement_logit: float
    category_id_used: int
    used_category_fallback: bool


class FrozenScorerReranker:
    def __init__(self, scorer, *, batch_size: int = 256) -> None:
        require_torch()
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.scorer = scorer
        self.batch_size = int(batch_size)
        self.scorer.eval()
        for parameter in self.scorer.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def load_v5(
        cls,
        checkpoint_path: Path | str,
        *,
        device: str | object = "cpu",
        batch_size: int = 256,
        expected_sha256: str | None = FROZEN_V5_SHA256,
    ) -> "FrozenScorerReranker":
        require_torch()
        path = Path(checkpoint_path)
        if expected_sha256 is not None:
            actual = sha256_file(path)
            if actual != expected_sha256:
                raise ValueError(
                    f"Frozen V5 checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual}"
                )
        payload = load_checkpoint(path, map_location="cpu")
        scorer = TypeAwarePairwiseScorer.from_config(payload["config"])
        scorer.load_state_dict(payload["model_state_dict"])
        scorer.to(device).eval()
        return cls(scorer, batch_size=batch_size)

    @classmethod
    def load_v5_bytes(
        cls,
        checkpoint_bytes: bytes,
        *,
        device: str | object = "cpu",
        batch_size: int = 256,
        expected_sha256: str | None = FROZEN_V5_SHA256,
    ) -> "FrozenScorerReranker":
        """Load the frozen checkpoint from an in-memory ZIP entry."""

        require_torch()
        if expected_sha256 is not None:
            actual = hashlib.sha256(checkpoint_bytes).hexdigest()
            if actual != expected_sha256:
                raise ValueError(
                    f"Frozen V5 checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual}"
                )
        try:
            payload = torch.load(
                io.BytesIO(checkpoint_bytes),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Frozen V5 checkpoint must contain a mapping")
        required = {"scorer_version", "config", "model_state_dict"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"Frozen V5 checkpoint missing keys: {missing}")
        if payload["scorer_version"] != "type_aware_pairwise_v1":
            raise ValueError("Unexpected scorer version in frozen V5 checkpoint")
        scorer = TypeAwarePairwiseScorer.from_config(payload["config"])
        scorer.load_state_dict(payload["model_state_dict"])
        scorer.to(device).eval()
        return cls(scorer, batch_size=batch_size)

    def _device(self):
        try:
            return next(self.scorer.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _score_batch(self, embeddings, category_ids):
        item_mask = torch.ones(
            category_ids.shape,
            dtype=torch.bool,
            device=category_ids.device,
        )
        with torch.inference_mode():
            output = self.scorer(
                item_embeddings=embeddings,
                coarse_category_ids=category_ids,
                item_mask=item_mask,
            )
        logits = output.get("compatibility_logit") if isinstance(output, dict) else None
        if not isinstance(logits, torch.Tensor) or logits.shape != (embeddings.shape[0],):
            raise RuntimeError("Scorer violated compatibility_logit [B] contract")
        return logits

    def rerank(
        self,
        *,
        outfit_embeddings,
        outfit_category_ids: Sequence[int],
        problematic_index: int,
        candidate_item_ids: Sequence[str],
        candidate_embeddings,
        candidate_category_ids: Sequence[int | None],
    ) -> list[RerankedCandidate]:
        require_torch()
        item_ids = [str(value) for value in candidate_item_ids]
        if len(item_ids) != len(candidate_category_ids):
            raise ValueError("candidate IDs and categories must align")
        device = self._device()
        outfit = torch.as_tensor(outfit_embeddings, dtype=torch.float32, device=device)
        categories = torch.as_tensor(outfit_category_ids, dtype=torch.long, device=device)
        candidates = torch.as_tensor(candidate_embeddings, dtype=torch.float32, device=device)
        if outfit.ndim != 2 or int(outfit.shape[1]) != 512:
            raise ValueError("outfit_embeddings must have shape [N, 512]")
        if categories.shape != (outfit.shape[0],):
            raise ValueError("outfit_category_ids must have shape [N]")
        if candidates.shape != (len(item_ids), 512):
            raise ValueError("candidate_embeddings must have shape [C, 512]")
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("candidate_item_ids must be unique")
        if not bool(torch.isfinite(outfit).all()) or not bool(
            torch.isfinite(candidates).all()
        ):
            raise ValueError("outfit/candidate embeddings contain NaN/Inf")
        all_embeddings = torch.cat((outfit, candidates), dim=0)
        norms = torch.linalg.vector_norm(all_embeddings, dim=1)
        if bool(torch.any(torch.abs(norms - 1.0) > 0.02)):
            raise ValueError("outfit/candidate embeddings must be L2-normalized")
        if not 0 <= problematic_index < outfit.shape[0]:
            raise ValueError("problematic_index is outside the outfit")
        if bool(torch.any(categories < 1)) or bool(torch.any(categories > 7)):
            raise ValueError("outfit categories must be Core-7 IDs in [1, 7]")
        if len(item_ids) == 0:
            return []

        baseline = self._score_batch(outfit.unsqueeze(0), categories.unsqueeze(0))[0]
        baseline_value = float(baseline.detach().cpu())
        target_category = int(categories[problematic_index])
        ranked: list[RerankedCandidate] = []
        for start in range(0, len(item_ids), self.batch_size):
            end = min(start + self.batch_size, len(item_ids))
            size = end - start
            batch_embeddings = outfit.unsqueeze(0).expand(size, -1, -1).clone()
            batch_embeddings[:, problematic_index] = candidates[start:end]
            batch_categories = categories.unsqueeze(0).expand(size, -1).clone()
            used_categories = []
            fallbacks = []
            for offset, candidate_category in enumerate(candidate_category_ids[start:end]):
                fallback = candidate_category is None
                category_id = target_category if fallback else int(candidate_category)
                if not 1 <= category_id <= 7:
                    raise ValueError("candidate category must be a Core-7 ID in [1, 7]")
                batch_categories[offset, problematic_index] = category_id
                used_categories.append(category_id)
                fallbacks.append(fallback)
            logits = self._score_batch(batch_embeddings, batch_categories)
            for offset, logit in enumerate(logits.detach().cpu().tolist()):
                value = float(logit)
                ranked.append(
                    RerankedCandidate(
                        item_id=item_ids[start + offset],
                        compatibility_logit=value,
                        improvement_logit=value - baseline_value,
                        category_id_used=used_categories[offset],
                        used_category_fallback=fallbacks[offset],
                    )
                )
        return sorted(
            ranked,
            key=lambda row: (-row.compatibility_logit, row.item_id),
        )
