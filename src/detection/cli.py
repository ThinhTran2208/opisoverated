# -*- coding: utf-8 -*-
"""CLI for RF-DETR + FashionCLIP Core-7 garment detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_detection_config
from .pipeline import DetectionPipeline, save_detection_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect outfit garments and prepare canonical Core-7 scorer inputs."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/detection_rfdetr_fashionclip_core7_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    config = load_detection_config(args.config)
    pipeline = DetectionPipeline(config, device=args.device)
    result, image = pipeline.run(args.image)
    saved = save_detection_result(
        result,
        image,
        args.output_dir,
        scorer_min_items=config.scorer_min_items,
        scorer_max_items=config.scorer_max_items,
    )
    summary = {
        "accepted_garments": len(result.garments),
        "rejected_detections": len(result.rejected_detections),
        "categories": [garment.category.coarse_category for garment in result.garments],
        **saved,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
