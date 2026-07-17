#!/usr/bin/env python3
"""CLI for remote ArtiFixer object refinement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldact.stage2_artifixer.config import ArtiFixerRefinementConfig, load_artifixer_config
from worldact.stage2_artifixer.runner import ArtiFixerRefinementRunner
from worldact.utils.logging import get_logger, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare/submit WorldAct objects to remote ArtiFixer Job API")
    parser.add_argument("--ply", required=True, help="Region/chunk PLY path")
    parser.add_argument("--stage-dir", required=True, help="stage1 directory with objects.json/masks.pt")
    parser.add_argument("--object-id", required=True, help="Object id from objects.json")
    parser.add_argument("--output-dir", required=True, help="Local output directory")
    parser.add_argument("--cameras", default=None, help="Optional cameras.json (default: stage-dir/background/cameras.json)")
    parser.add_argument("--object-ply-dir", default=None, help="Optional directory containing object_*.ply")
    parser.add_argument("--config", default=None, help="YAML config (configs/artifixer/remote.yaml)")
    parser.add_argument("--service-url", default=None, help="Override artifixer_service.base_url")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["prepare", "submit", "wait", "download", "import", "validate", "all"],
    )
    parser.add_argument("--anchor-count", type=int, default=None)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--distillation-steps", type=int, default=None)
    parser.add_argument("--model", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("INFO")
    log = get_logger("artifixer_refinement")

    cfg = load_artifixer_config(args.config) if args.config else ArtiFixerRefinementConfig()
    if args.service_url:
        cfg.service.base_url = args.service_url
    if args.device:
        cfg.device = args.device
    if args.anchor_count is not None:
        cfg.prepare.anchor_count = args.anchor_count
    if args.target_count is not None:
        cfg.prepare.target_count = args.target_count
        # Keep ring counts roughly proportional for smaller smoke runs.
        cfg.prepare.horizontal_targets = max(args.target_count - 8, args.target_count)
        if args.target_count <= 8:
            cfg.prepare.horizontal_targets = max(args.target_count, 1)
            cfg.prepare.high_elevation_targets = 0
            cfg.prepare.low_elevation_targets = 0
    if args.distillation_steps is not None:
        cfg.prepare.distillation_steps = args.distillation_steps
        cfg.service.distillation_steps = args.distillation_steps
    if args.model:
        cfg.prepare.model = args.model
        cfg.service.model = args.model

    runner = ArtiFixerRefinementRunner(
        ply_path=args.ply,
        stage_dir=args.stage_dir,
        object_id=args.object_id,
        output_dir=args.output_dir,
        config=cfg,
        cameras_path=args.cameras,
        object_ply_dir=args.object_ply_dir,
    )
    result = runner.run(args.phase)  # type: ignore[arg-type]
    payload = {
        "phase": result.phase,
        "object_id": result.object_id,
        "output_dir": str(result.output_dir),
        "package_dir": str(result.package_dir) if result.package_dir else None,
        "archive_path": str(result.archive_path) if result.archive_path else None,
        "import_status": result.import_status,
        "job_id": result.job.job_id if result.job else None,
        "details": result.details,
    }
    log.info("done: %s", json.dumps(payload))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
