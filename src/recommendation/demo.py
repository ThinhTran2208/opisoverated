# -*- coding: utf-8 -*-
"""End-to-end ZIP-direct Recommendation V2 demo."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .evaluation import Evaluation3Evaluator
from .pipeline import RecommendationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Category-Aware Hybrid Recommendation V2")
    parser.add_argument("--ml-zip", required=True)
    parser.add_argument("--image-zip", action="append", required=True)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[2]
            / "configs"
            / "recommendation_category_aware_v2.json"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluation-split", choices=("valid", "test"), default="test")
    parser.add_argument("--evaluation-max-samples", type=int, default=50)
    return parser


def _fixture_result(pipeline, records):
    failures = []
    negatives = sorted(
        (row for row in records if row.get("label") == 0),
        key=lambda row: str(row.get("sample_id", "")),
    )
    for row in negatives:
        item_ids = [str(value) for value in row["items"]]
        metadata = row.get("negative_metadata")
        if not isinstance(metadata, dict):
            continue
        swapped_index = metadata.get("swapped_item_index")
        if not isinstance(swapped_index, int):
            continue
        try:
            embeddings = pipeline.catalog.get_embeddings(item_ids)
            categories = [int(pipeline.metadata.category_id(item_id)) for item_id in item_ids]
            result = pipeline.recommend(
                outfit_item_ids=item_ids,
                outfit_embeddings=embeddings,
                outfit_category_ids=categories,
                problematic_index=swapped_index,
            )
            return row, result
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            failures.append(f"{row.get('sample_id')}: {error}")
    raise RuntimeError(
        "No one-item-swap fixture produced three recommendations. "
        f"First failures: {failures[:3]}"
    )


def _render_html(report: dict[str, object], image_files: list[str]) -> str:
    items = report["recommendations"]["items"]
    cards = []
    for item, image_file in zip(items, image_files):
        cards.append(
            "<article>"
            f"<h2>#{int(item['rank'])} {html.escape(str(item['item_id']))}</h2>"
            f"<img src=\"{html.escape(image_file)}\" alt=\"recommendation\">"
            f"<p>{html.escape(str(item['master_category']))} · "
            f"{html.escape(str(item['coarse_category']))}</p>"
            "</article>"
        )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Category-Aware Recommendation V2 Demo</title>
<style>
body{font-family:system-ui;margin:2rem;background:#f5f5f5;color:#171717}
main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem}
article{background:white;padding:1rem;border-radius:12px;box-shadow:0 2px 12px #0001}
img{width:100%;aspect-ratio:1;object-fit:contain;background:#eee}
h1{margin-bottom:.25rem} .meta{color:#555;margin-bottom:1.5rem}
</style></head><body>
<h1>Category-Aware Recommendation V2</h1>
<p class="meta">Scores are intentionally omitted from this demo.</p>
<main>""" + "".join(cards) + """</main></body></html>"""


def run_demo(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = RecommendationPipeline.load_from_archives(
        args.config,
        ml_zip_path=args.ml_zip,
        image_zip_paths=args.image_zip,
        device=args.device,
    )
    bundle = pipeline.artifact_bundle
    scorer_records = bundle.load_scorer_ready(args.evaluation_split)

    first_ref = pipeline.image_resolver.first_ref
    first_bytes = pipeline.image_resolver.read_bytes(first_ref.item_id)
    fixture, result = _fixture_result(pipeline, scorer_records)
    public = result.to_public_dict()

    image_files = []
    for item in result.items:
        filename = f"rank_{item.rank}_{item.item_id}.jpg"
        pipeline.image_resolver.write_selected_image(item.item_id, output_dir / filename)
        image_files.append(filename)

    max_samples = args.evaluation_max_samples
    evaluation = Evaluation3Evaluator(pipeline).evaluate(
        scorer_records,
        max_samples=None if max_samples == 0 else max_samples,
        split=args.evaluation_split,
    )
    evaluation.pop("records", None)
    negative_metadata = fixture["negative_metadata"]
    report = {
        "demo_status": "passed",
        "artifact_mode": "zip-direct",
        "ml_zip": str(Path(args.ml_zip).resolve()),
        "image_zips": [str(Path(path).resolve()) for path in args.image_zip],
        "image_catalog_validation": pipeline.image_validation,
        "stable_first_image": {
            "item_id": first_ref.item_id,
            "archive_path": str(first_ref.archive_path),
            "internal_path": first_ref.internal_path,
            "bytes_read": len(first_bytes),
            "jpeg_signature_valid": first_bytes.startswith(b"\xff\xd8\xff"),
        },
        "one_swap_fixture": {
            "split": args.evaluation_split,
            "sample_id": fixture["sample_id"],
            "outfit_item_ids": fixture["items"],
            "problematic_item_index": negative_metadata["swapped_item_index"],
            "ground_truth_item_id": negative_metadata["original_item_id"],
        },
        "recommendations": public,
        "one_swap_metrics": evaluation,
        "score_fields_exposed": False,
    }
    (output_dir / "demo.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "index.html").write_text(
        _render_html(report, image_files), encoding="utf-8"
    )
    return report


def main() -> int:
    args = build_parser().parse_args()
    report = run_demo(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
