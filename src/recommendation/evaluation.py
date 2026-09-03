# -*- coding: utf-8 -*-
"""Evaluation3 metrics for hybrid retrieval and frozen-scorer reranking."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence


EVALUATION_PROTOCOL = "evaluation3-one-item-swap-v1"


class Evaluation3Evaluator:
    """Evaluate ground-truth recovery on one-item-swap negative outfits.

    The known swapped index is used to isolate recommendation quality from LOO
    localization quality.  Each row has exactly one relevant item:
    ``negative_metadata.original_item_id``.
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def evaluate(
        self,
        scorer_ready_records: Sequence[Mapping[str, object]],
        *,
        max_samples: int | None = None,
        split: str = "test",
    ) -> dict[str, object]:
        negatives = sorted(
            (row for row in scorer_ready_records if row.get("label") == 0),
            key=lambda row: str(row.get("sample_id", "")),
        )
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be >= 1")
            negatives = negatives[:max_samples]

        counts = Counter()
        retrieval_hits = {50: 0, 100: 0, 200: 0}
        rerank_hit_1 = 0
        rerank_hit_3 = 0
        reciprocal_rank_sum = 0.0
        records: list[dict[str, object]] = []

        for row in negatives:
            sample_id = str(row.get("sample_id", ""))
            item_ids = [str(value) for value in row.get("items", [])]
            negative_metadata = row.get("negative_metadata")
            if not isinstance(negative_metadata, Mapping):
                counts["missing_negative_metadata"] += 1
                continue
            swapped_index = negative_metadata.get("swapped_item_index")
            ground_truth = str(negative_metadata.get("original_item_id", ""))
            if (
                isinstance(swapped_index, bool)
                or not isinstance(swapped_index, int)
                or not 0 <= swapped_index < len(item_ids)
                or not ground_truth
            ):
                counts["invalid_negative_metadata"] += 1
                continue
            missing_embeddings = [
                item_id for item_id in item_ids if item_id not in self.pipeline.catalog
            ]
            missing_metadata = [
                item_id
                for item_id in item_ids
                if self.pipeline.metadata.category_id(item_id) is None
                or self.pipeline.metadata.master_category(item_id) is None
            ]
            missing_images = [
                item_id
                for item_id in item_ids
                if item_id not in self.pipeline.image_resolver
            ]
            if missing_embeddings:
                counts["missing_outfit_embedding"] += 1
                continue
            if missing_metadata:
                counts["missing_outfit_metadata"] += 1
                continue
            if missing_images:
                counts["missing_outfit_image"] += 1
                continue
            outfit_embeddings = self.pipeline.catalog.get_embeddings(item_ids)
            outfit_categories = [
                int(self.pipeline.metadata.category_id(item_id)) for item_id in item_ids
            ]
            retrieval, reranked = self.pipeline.rank_candidates(
                outfit_item_ids=item_ids,
                outfit_embeddings=outfit_embeddings,
                outfit_category_ids=outfit_categories,
                problematic_index=swapped_index,
            )
            by_item = {candidate.item_id: candidate for candidate in retrieval.candidates}
            gt_candidate = by_item.get(ground_truth)
            retrieval_rank = None
            if gt_candidate is not None:
                ranks = [
                    value
                    for value in (
                        gt_candidate.problematic_rank,
                        gt_candidate.context_rank,
                    )
                    if value is not None
                ]
                retrieval_rank = min(ranks) if ranks else None
            for cutoff in retrieval_hits:
                retrieval_hits[cutoff] += int(
                    retrieval_rank is not None and retrieval_rank <= cutoff
                )

            rerank_rank = next(
                (
                    rank
                    for rank, candidate in enumerate(reranked, start=1)
                    if candidate.item_id == ground_truth
                ),
                None,
            )
            rerank_hit_1 += int(rerank_rank == 1)
            rerank_hit_3 += int(rerank_rank is not None and rerank_rank <= 3)
            if rerank_rank is not None:
                reciprocal_rank_sum += 1.0 / rerank_rank
            counts["evaluated"] += 1
            records.append(
                {
                    "sample_id": sample_id,
                    "ground_truth_item_id": ground_truth,
                    "retrieval_rank": retrieval_rank,
                    "rerank_rank": rerank_rank,
                    "candidate_count": len(reranked),
                }
            )

        evaluated = counts["evaluated"]
        if evaluated == 0:
            raise ValueError("Evaluation3 has no eligible rows")
        return {
            "protocol_version": EVALUATION_PROTOCOL,
            "split": split,
            "requested_negative_count": len(negatives),
            "evaluated_count": evaluated,
            "is_full_split": max_samples is None,
            "skipped": {
                key: value for key, value in sorted(counts.items()) if key != "evaluated"
            },
            "retrieval": {
                "recall_at_50": retrieval_hits[50] / evaluated,
                "recall_at_100": retrieval_hits[100] / evaluated,
                "recall_at_200": retrieval_hits[200] / evaluated,
            },
            "reranking": {
                "hit_at_1": rerank_hit_1 / evaluated,
                "hit_at_3": rerank_hit_3 / evaluated,
                "mrr": reciprocal_rank_sum / evaluated,
            },
            "records": records,
        }

