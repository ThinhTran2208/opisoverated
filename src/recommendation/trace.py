# -*- coding: utf-8 -*-
"""Ordered, internal-only candidate traces for Recommendation V1."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Mapping, Sequence

DEFAULT_TRACE_PATH = Path("outputs/recommendation_candidate_records.jsonl")


class CandidateTraceWriter:
    def __init__(self, path: Path | str = DEFAULT_TRACE_PATH) -> None:
        self.path = Path(path).resolve()
        self._lock = Lock()

    def append(self, record: Mapping[str, object]) -> None:
        line = json.dumps(dict(record), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line + "\n")


def candidate_trace_record(
    *, query_id: str, source_split: str | None,
    problematic_item_index: int | None, problematic_item_id: str | None,
    ground_truth_item_id: str | None, replacement_item_id: str | None = None,
    item_ids: Sequence[str] = (), context_ids: Sequence[str] = (),
    hybrid_ids: Sequence[str] = (), final_ids: Sequence[str] = (),
    candidate_counts: Mapping[str, int] | None = None,
    excluded_counts: Mapping[str, int] | None = None,
    failure_reason: str | None = None,
) -> dict[str, object]:
    return {
        "query_id": str(query_id), "source_split": source_split,
        "problematic_item_index": problematic_item_index,
        "problematic_item_id": problematic_item_id,
        "ground_truth_item_id": ground_truth_item_id,
        "replacement_item_id": replacement_item_id,
        "item_retrieval_top200": list(item_ids)[:200],
        "context_retrieval_top200": list(context_ids)[:200],
        "hybrid_candidates_top200": list(hybrid_ids)[:200],
        "final_top3": list(final_ids)[:3],
        "candidate_counts": dict(candidate_counts or {}),
        "excluded_counts": dict(excluded_counts or {}),
        "failure_reason": failure_reason,
    }
