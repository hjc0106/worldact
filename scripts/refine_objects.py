#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldact.stage2_object_refinement import ObjectRefinementConfig, load_stage_input, make_backend, refine_stage_objects
from worldact.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="WorldAct Stage 2: foreground object 3DGS refinement/completion")
    parser.add_argument("--ply", required=True, help="Input region/chunk PLY used by decomposition")
    parser.add_argument("--stage-dir", required=True, help="Stage1 decomposition directory containing objects.json and masks.pt")
    parser.add_argument("--cameras", default=None, help="Optional cameras.json; defaults to <stage-dir>/background/cameras.json if present")
    parser.add_argument("--output-dir", default=None, help="Output directory for object refinement artifacts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only-object-id", default=None, help="Process only one object id")
    parser.add_argument("--opacity-min", type=float, default=0.02)
    parser.add_argument("--max-world-scale", type=float, default=0.0, help="Maximum exp(scale) per axis; <=0 disables")
    parser.add_argument("--component-radius", type=float, default=0.5)
    parser.add_argument("--no-keep-largest-component", action="store_true")
    parser.add_argument("--min-component-keep-ratio", type=float, default=0.05)
    parser.add_argument("--min-mask-support", type=int, default=2)
    parser.add_argument("--min-support-ratio", type=float, default=0.0)
    parser.add_argument("--min-mask-keep-ratio", type=float, default=0.01)
    parser.add_argument("--no-depth-gate", action="store_true")
    parser.add_argument("--depth-tolerance", type=float, default=0.08)
    parser.add_argument("--min-alpha", type=float, default=0.05)
    parser.add_argument("--render-alpha-threshold", type=float, default=0.1)
    parser.add_argument("--no-depth-consistency", action="store_true")
    parser.add_argument("--depth-consistency-percentile", type=float, default=0.10)
    parser.add_argument("--depth-consistency-margin", type=float, default=0.3)
    parser.add_argument("--crop-padding", type=int, default=24)
    parser.add_argument("--duplicate-distance", type=float, default=0.03)
    parser.add_argument("--blend-distance", type=float, default=0.05)
    parser.add_argument("--num-generation-views", type=int, default=6)
    parser.add_argument("--generation-resolution", type=int, default=512)
    parser.add_argument("--generation-model", default="sv3d", choices=["sv3d", "zero123plus"])
    parser.add_argument("--reconstruction-model", default="gs_lrm", choices=["gs_lrm", "instant_splat"])
    parser.add_argument("--inpaint-model", default="lama", choices=["lama", "none"])
    parser.add_argument("--icp-iterations", type=int, default=50)
    parser.add_argument("--backend", choices=["none", "external", "sv3d"], default="none")
    parser.add_argument("--backend-command", default=None, help="External command; use {request} and {workdir} placeholders if needed")
    parser.add_argument("--backend-timeout", type=float, default=600.0)
    args = parser.parse_args()

    setup_logging("INFO")
    stage_dir = Path(args.stage_dir)
    cameras = _resolve_cameras(stage_dir, args.cameras)
    output_dir = Path(args.output_dir) if args.output_dir else stage_dir.parent / "stage2_object_refinement"
    cfg = ObjectRefinementConfig(
        opacity_min=float(args.opacity_min),
        max_world_scale=float(args.max_world_scale),
        component_radius=float(args.component_radius),
        keep_largest_component=not bool(args.no_keep_largest_component),
        min_component_keep_ratio=float(args.min_component_keep_ratio),
        min_mask_support=int(args.min_mask_support),
        min_support_ratio=float(args.min_support_ratio),
        min_mask_keep_ratio=float(args.min_mask_keep_ratio),
        use_depth_gate=not bool(args.no_depth_gate),
        depth_tolerance=float(args.depth_tolerance),
        min_alpha=float(args.min_alpha),
        render_alpha_threshold=float(args.render_alpha_threshold),
        use_depth_consistency=not bool(args.no_depth_consistency),
        depth_consistency_percentile=float(args.depth_consistency_percentile),
        depth_consistency_margin=float(args.depth_consistency_margin),
        crop_padding=int(args.crop_padding),
        duplicate_distance=float(args.duplicate_distance),
        blend_distance=float(args.blend_distance),
        num_generation_views=int(args.num_generation_views),
        generation_resolution=int(args.generation_resolution),
        generation_model=args.generation_model,
        reconstruction_model=args.reconstruction_model,
        inpaint_model=args.inpaint_model,
        icp_iterations=int(args.icp_iterations),
    )
    stage_input = load_stage_input(Path(args.ply), stage_dir, cameras_path=cameras, device=str(args.device))
    backend_kwargs = {
        "generation_model": args.generation_model,
        "reconstruction_model": args.reconstruction_model,
        "inpaint_model": args.inpaint_model,
        "num_generation_views": int(args.num_generation_views),
        "generation_resolution": int(args.generation_resolution),
        "icp_iterations": int(args.icp_iterations),
        "device": args.device,
    }
    backend = make_backend(args.backend, command=args.backend_command, timeout=float(args.backend_timeout), **backend_kwargs)
    result = refine_stage_objects(stage_input, output_dir, cfg, backend, only_object_id=args.only_object_id)
    log.info(
        "Processed %d/%d objects; wrote artifacts to %s",
        result.summary["num_processed_objects"],
        result.summary["num_input_objects"],
        output_dir,
    )


def _resolve_cameras(stage_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    candidate = stage_dir / "background" / "cameras.json"
    return candidate if candidate.exists() else None


if __name__ == "__main__":
    main()
