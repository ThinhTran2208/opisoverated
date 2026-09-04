# -*- coding: utf-8 -*-
"""Polyvore one-item-swap recovery metrics for Recommendation V2."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .pipeline import RecommendationPipeline
from .trace import CandidateTraceWriter, candidate_trace_record

EVALUATION_PROTOCOL = "polyvore-one-item-swap-recovery-v2-conditional-hit"
DEFAULT_EPSILON = 0.0


def _rank(items: Sequence[str], ground_truth: str) -> int | None:
    return next((rank for rank, item in enumerate(items, 1) if item == ground_truth), None)


class Evaluation3Evaluator:
    """Legacy class name; evaluates Polyvore synthetic one-swap recovery."""

    def __init__(self, pipeline, *, epsilon: float = DEFAULT_EPSILON) -> None:
        self.pipeline = pipeline
        self.epsilon = float(epsilon)

    def evaluate(self, scorer_ready_records: Sequence[Mapping[str, object]], *,
                 max_samples: int | None = None, split: str = "test",
                 trace_writer: CandidateTraceWriter | None = None) -> dict[str, object]:
        negatives = sorted((r for r in scorer_ready_records if r.get("label") == 0),
                           key=lambda r: str(r.get("sample_id", "")))
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be >= 1")
            negatives = negatives[:max_samples]

        excluded, failures, coverage = Counter(), Counter(), Counter()
        for key in ("missing_embedding", "missing_metadata", "missing_image",
                    "missing_negative_metadata", "invalid_negative_metadata"):
            excluded[key] = 0
        for key in ("ground_truth_not_in_hybrid_top200",
                    "ground_truth_not_in_full_union",
                    "ground_truth_in_full_union_not_final_top3",
                    "fewer_than_three_final_candidates",
                    "image_read_error", "scorer_error"):
            failures[key] = 0

        hits = {stage: {50: 0, 100: 0, 200: 0}
                for stage in ("item_only", "context_only", "hybrid")}
        full_union_gt_count = 0
        conditional_hit3_count = 0
        successful = evaluated_recommendations = 0
        records: list[dict[str, object]] = []

        ground_truth_ids = [
            str(meta.get("original_item_id", ""))
            for row in negatives
            for meta in [row.get("negative_metadata")]
            if isinstance(meta, Mapping) and meta.get("original_item_id")
        ]
        if hasattr(self.pipeline.image_resolver, "validate_readable"):
            image_read_failures = self.pipeline.image_resolver.validate_readable(ground_truth_ids)
        else:
            image_read_failures = {}

        for row in negatives:
            query_id = str(row.get("sample_id", ""))
            item_ids = [str(x) for x in row.get("items", [])]
            meta = row.get("negative_metadata")
            swapped = meta.get("swapped_item_index") if isinstance(meta, Mapping) else None
            ground_truth = str(meta.get("original_item_id", "")) if isinstance(meta, Mapping) else ""
            replacement = str(meta.get("replacement_item_id", "")) if isinstance(meta, Mapping) else ""
            valid_index = isinstance(swapped, int) and not isinstance(swapped, bool) and 0 <= swapped < len(item_ids)
            problem_id = item_ids[swapped] if valid_index else None
            base = dict(query_id=query_id, source_split=split,
                        problematic_item_index=swapped if valid_index else None,
                        problematic_item_id=problem_id,
                        ground_truth_item_id=ground_truth or None,
                        replacement_item_id=replacement or None)

            reason = None
            if not isinstance(meta, Mapping):
                reason = "missing_negative_metadata"
            elif not valid_index or not ground_truth or not replacement or problem_id != replacement:
                reason = "invalid_negative_metadata"

            required = item_ids + ([ground_truth] if ground_truth else [])
            miss_emb = [x for x in required if x not in self.pipeline.catalog]
            miss_meta = [x for x in required if self.pipeline.metadata.category_id(x) is None or self.pipeline.metadata.master_category(x) is None]
            miss_img = [x for x in required if x not in self.pipeline.image_resolver]
            coverage["required_item_checks"] += len(required)
            coverage["embedding_available"] += len(required) - len(miss_emb)
            coverage["metadata_available"] += len(required) - len(miss_meta)
            coverage["image_available"] += len(required) - len(miss_img)
            if reason is None and miss_emb:
                reason = "missing_embedding"
            if reason is None and miss_meta:
                reason = "missing_metadata"
            if reason is None and miss_img:
                reason = "missing_image"
            if reason:
                excluded[reason] += 1
                trace = candidate_trace_record(**base, failure_reason=reason)
                records.append(trace)
                if trace_writer:
                    trace_writer.append(trace)
                continue

            try:
                if ground_truth in image_read_failures:
                    raise OSError(image_read_failures[ground_truth])
                if not hasattr(self.pipeline.image_resolver, "validate_readable"):
                    self.pipeline.image_resolver.read_bytes(ground_truth)
            except Exception as error:
                failures["image_read_error"] += 1
                trace = candidate_trace_record(**base, failure_reason=f"image_read_error: {type(error).__name__}")
                records.append(trace)
                if trace_writer:
                    trace_writer.append(trace)
                continue

            try:
                embeddings = self.pipeline.catalog.get_embeddings(item_ids)
                categories = [int(self.pipeline.metadata.category_id(x)) for x in item_ids]
                retrieval, reranked = self.pipeline.rank_candidates(
                    outfit_item_ids=item_ids,
                    outfit_embeddings=embeddings,
                    outfit_category_ids=categories,
                    problematic_index=swapped,
                )
            except Exception as error:
                failures["scorer_error"] += 1
                trace = candidate_trace_record(**base, failure_reason=f"scorer_error: {type(error).__name__}")
                records.append(trace)
                if trace_writer:
                    trace_writer.append(trace)
                continue

            item_order = [x.item_id for x in retrieval.problematic_hits][:200]
            context_order = [x.item_id for x in retrieval.context_hits][:200]
            full_union_order = [x.item_id for x in retrieval.candidates]
            hybrid_order = full_union_order[:200]
            all_reranked = [x.item_id for x in reranked]
            final_order = all_reranked[:3]

            if len(final_order) < 3:
                failures["fewer_than_three_final_candidates"] += 1

            for stage, order in (("item_only", item_order),
                                 ("context_only", context_order),
                                 ("hybrid", hybrid_order)):
                rank = _rank(order, ground_truth)
                for k in (50, 100, 200):
                    hits[stage][k] += int(rank is not None and rank <= k)

            if ground_truth not in hybrid_order:
                failures["ground_truth_not_in_hybrid_top200"] += 1

            union_has_gt = ground_truth in full_union_order
            if union_has_gt:
                full_union_gt_count += 1
                rerank_rank = _rank(all_reranked, ground_truth)
                is_conditional_hit3 = rerank_rank is not None and rerank_rank <= 3
                conditional_hit3_count += int(is_conditional_hit3)
                if not is_conditional_hit3:
                    failures["ground_truth_in_full_union_not_final_top3"] += 1
            else:
                failures["ground_truth_not_in_full_union"] += 1

            for candidate in reranked[:3]:
                evaluated_recommendations += 1
                successful += int(candidate.improvement_logit > self.epsilon)

            trace = candidate_trace_record(
                **base,
                item_ids=item_order,
                context_ids=context_order,
                hybrid_ids=hybrid_order,
                final_ids=final_order,
                candidate_counts={
                    "category_pool": retrieval.category_pool_count,
                    "item_retrieval": len(item_order),
                    "context_retrieval": len(context_order),
                    "hybrid_top200": len(hybrid_order),
                    "full_union": len(full_union_order),
                    "reranked": len(reranked),
                    "final": len(final_order),
                },
                excluded_counts={
                    "master_category": retrieval.master_category_filtered_count,
                    "missing_metadata": retrieval.missing_metadata_count,
                    "missing_image": retrieval.missing_image_count,
                    "missing_embedding": retrieval.missing_embedding_count,
                },
            )
            records.append(trace)
            if trace_writer:
                trace_writer.append(trace)

        runtime_failures = failures["scorer_error"] + failures["image_read_error"]
        evaluated = len(negatives) - sum(excluded.values()) - runtime_failures
        if not evaluated:
            raise ValueError("One-item-swap evaluation has no eligible rows")

        ratio = lambda n: n / evaluated
        checks = coverage["required_item_checks"]
        conditional_hit3 = (
            conditional_hit3_count / full_union_gt_count
            if full_union_gt_count else None
        )

        return {
            "protocol_version": EVALUATION_PROTOCOL,
            "split": split,
            "epsilon": self.epsilon,
            "score_type": "compatibility_logit",
            "total_queries": len(negatives),
            "valid_queries": evaluated,
            "excluded_queries": len(negatives) - evaluated,
            "is_full_split": max_samples is None,
            "image_catalog_validation": getattr(self.pipeline, "image_validation", None),
            "coverage": {
                **dict(coverage),
                "embedding_ratio": coverage["embedding_available"] / checks if checks else 0.0,
                "metadata_ratio": coverage["metadata_available"] / checks if checks else 0.0,
                "image_ratio": coverage["image_available"] / checks if checks else 0.0,
            },
            "excluded": dict(sorted(excluded.items())),
            "failures": dict(sorted(failures.items())),
            "retrieval": {
                stage: {f"recall_at_{k}": ratio(n) for k, n in values.items()}
                for stage, values in hits.items()
            },
            "reranking": {
                "metric": "conditional_exact_reference_hit_at_3",
                "conditional_hit_at_3": conditional_hit3,
                "eligible_queries_gt_in_full_union": full_union_gt_count,
                "conditional_hits_at_3": conditional_hit3_count,
                "full_union_gt_coverage": ratio(full_union_gt_count),
            },
            "replacement_quality": {
                "success_rate": successful / evaluated_recommendations if evaluated_recommendations else None,
                "successful": successful,
                "evaluated_recommendations": evaluated_recommendations,
            },
            "records": records,
        }


def report_markdown(result: Mapping[str, object]) -> str:
    retrieval, reranking = result["retrieval"], result["reranking"]
    f = lambda x: "N/A" if x is None else f"{float(x):.6f}"
    rows = []
    for label, key in (("Item-only", "item_only"),
                       ("Context-only", "context_only"),
                       ("Hybrid Top-200", "hybrid")):
        m = retrieval[key]
        rows.append(
            f"| {label} | {f(m['recall_at_50'])} | {f(m['recall_at_100'])} | {f(m['recall_at_200'])} | N/A |"
        )
    rows.append(
        f"| Frozen scorer rerank | N/A | N/A | N/A | {f(reranking['conditional_hit_at_3'])} |"
    )

    return "\n".join([
        "# Recommendation V2 — Conditional Reranking Evaluation",
        "",
        f"Split: `{result['split']}`. Epsilon cố định: `{result['epsilon']}` trên `compatibility_logit`.",
        "",
        "| Stage | Recall@50 | Recall@100 | Recall@200 | Conditional Hit@3 |",
        "| --- | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "Conditional Hit@3 chỉ xét các query mà exact ground-truth item có mặt trong FULL retrieval union mà scorer thực sự được rerank.",
        "Nó đo exact-reference recovery có điều kiện, không phải recommendation accuracy hay human preference.",
        "",
        f"- Tổng query: {result['total_queries']}",
        f"- Query hợp lệ: {result['valid_queries']}",
        f"- Query bị loại: {result['excluded_queries']}",
        f"- GT có mặt trong full union: {reranking['eligible_queries_gt_in_full_union']}",
        f"- Full-union GT coverage: {f(reranking['full_union_gt_coverage'])}",
        f"- Conditional Hit@3 numerator: {reranking['conditional_hits_at_3']}",
        f"- Conditional Hit@3: {f(reranking['conditional_hit_at_3'])}",
        f"- Replacement Success Rate: {f(result['replacement_quality']['success_rate'])}",
        f"- Image catalog: `{json.dumps(result.get('image_catalog_validation'), ensure_ascii=False)}`",
        f"- Coverage: `{json.dumps(result['coverage'], ensure_ascii=False)}`",
        f"- Excluded: `{json.dumps(result['excluded'], ensure_ascii=False)}`",
        f"- Failures: `{json.dumps(result['failures'], ensure_ascii=False)}`",
        "",
        "Ground truth là original item đã bị synthetic swap ra; nhiều replacement khác vẫn có thể hợp lý.",
        "Score chỉ tồn tại trong evaluation nội bộ; public response không chứa score.",
    ]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Recommendation V2 conditional reranking evaluation")
    parser.add_argument("--artifact-root", help="Path to ML_Final directory")
    parser.add_argument("--image-root", help="Path to item-image directory")
    parser.add_argument("--ml-zip", help="Legacy ML_Final ZIP path")
    parser.add_argument("--image-zip", action="append", help="Legacy image ZIP; may be repeated")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "recommendation_category_aware_v2.json"))
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    using_directories = bool(args.artifact_root or args.image_root)
    using_archives = bool(args.ml_zip or args.image_zip)
    if using_directories and using_archives:
        parser.error("Choose directory mode OR ZIP mode, not both")
    if using_directories:
        if not args.artifact_root or not args.image_root:
            parser.error("Directory mode requires both --artifact-root and --image-root")
        pipeline = RecommendationPipeline.load_from_directories(
            args.config,
            artifact_root=args.artifact_root,
            image_root=args.image_root,
            device=args.device,
        )
    else:
        if not args.ml_zip or not args.image_zip:
            parser.error("ZIP mode requires --ml-zip and at least one --image-zip")
        pipeline = RecommendationPipeline.load_from_archives(
            args.config,
            ml_zip_path=args.ml_zip,
            image_zip_paths=args.image_zip,
            device=args.device,
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "recommendation_candidate_records.jsonl"
    trace_path.write_text("", encoding="utf-8")
    result = Evaluation3Evaluator(pipeline).evaluate(
        pipeline.artifact_bundle.load_scorer_ready(args.split),
        split=args.split,
        max_samples=None if args.max_samples == 0 else args.max_samples,
        trace_writer=CandidateTraceWriter(trace_path),
    )
    serializable = dict(result)
    serializable.pop("records", None)
    (out / "recommendation_evaluation_results.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "recommendation_evaluation_report.md").write_text(
        report_markdown(serializable),
        encoding="utf-8",
    )
    print(json.dumps(serializable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
