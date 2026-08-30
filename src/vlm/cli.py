# -*- coding: utf-8 -*-
"""Command-line entrypoint for one grounded VLM explanation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_vlm_config
from .pipeline import VLMExplanationPipeline
from .qwen_backend import Qwen3VLBackend
from .schema import validate_vlm_evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one Qwen3-VL explanation from frozen V5 + LOO evidence"
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vlm_qwen3_vl_4b_instruct_v1.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence = validate_vlm_evidence(
        json.loads(args.evidence.read_text(encoding="utf-8"))
    )
    config = load_vlm_config(args.config)
    backend = Qwen3VLBackend.from_config(config)
    pipeline = VLMExplanationPipeline(backend, config)
    result = pipeline.explain(evidence, args.images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
