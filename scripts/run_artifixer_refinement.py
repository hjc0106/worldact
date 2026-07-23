#!/usr/bin/env python3
"""CLI for remote ArtiFixer object refinement."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Stable multi-GPU / NCCL defaults (same as run_art*.sh).
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "0")
os.environ.setdefault("NCCL_CROSS_NIC", "1")

WORLDACT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = WORLDACT_ROOT.parent
for path in (REPO_ROOT, WORLDACT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from worldact.stage2_artifixer.config import ArtiFixerRefinementConfig, load_artifixer_config
from worldact.stage2_artifixer.runner import ArtiFixerRefinementRunner
from worldact.utils.logging import get_logger, setup_logging

try:
    from model_training.utils.gpu_cleanup import (
        cleanup_distributed_cuda,
        install_termination_cleanup_handlers,
        release_cuda_memory,
        restore_signal_handlers,
    )
except ImportError:  # pragma: no cover - host env without ArtiFixer training utils
    def release_cuda_memory() -> None:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

    def cleanup_distributed_cuda() -> None:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        release_cuda_memory()

    def install_termination_cleanup_handlers() -> dict:
        return {}

    def restore_signal_handlers(_previous_handlers: dict) -> None:
        return


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

    runner = None
    try:
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
    except Exception:
        # Match exp.py: free CUDA fragments before the traceback exits the process.
        release_cuda_memory()
        raise
    finally:
        runner = None
        release_cuda_memory()


if __name__ == "__main__":
    previous_signal_handlers = install_termination_cleanup_handlers()
    try:
        raise SystemExit(main())
    finally:
        try:
            cleanup_distributed_cuda()
        finally:
            restore_signal_handlers(previous_signal_handlers)
