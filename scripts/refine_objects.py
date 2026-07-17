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
    parser.add_argument("--object-ply-dir", default=None, help="Optional directory of target-level object PLY files")
    parser.add_argument("--object-manifest", default=None, help="Optional JSON manifest with object PLY paths and metadata")
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
    parser.add_argument("--generation-model", default="sv3d", choices=["sv3d", "wonder3d", "zero123plus", "scene_multiview"])
    parser.add_argument("--generation-model-path", default=None, help="Optional local path for the generation model")
    parser.add_argument("--wonder3d-crop-size", type=int, default=192)
    parser.add_argument("--wonder3d-guidance-scale", type=float, default=1.0)
    parser.add_argument("--reconstruction-model", default="gs_lrm", choices=["gs_lrm", "instant_splat"])
    parser.add_argument("--inpaint-model", default="lama", choices=["lama", "none"])
    parser.add_argument("--icp-iterations", type=int, default=50)
    parser.add_argument("--no-foreground-confidence", action="store_true")
    parser.add_argument("--min-foreground-confidence", type=float, default=0.35)
    parser.add_argument("--background-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--background-structure-quantile", type=float, default=0.15)
    parser.add_argument("--background-structure-distance", type=float, default=0.08)
    parser.add_argument("--foreground-core-quantile", type=float, default=0.35)
    parser.add_argument("--no-adaptive-cleanup", action="store_true")
    parser.add_argument("--max-off-mask-ratio", type=float, default=0.55)
    parser.add_argument("--robust-core-bbox-quantile", type=float, default=0.04)
    parser.add_argument("--robust-core-bbox-expand", type=float, default=0.18)
    parser.add_argument("--generated-min-view-support", type=int, default=2)
    parser.add_argument("--generated-min-support-ratio", type=float, default=0.5)
    parser.add_argument("--generated-bbox-expand", type=float, default=0.25)
    parser.add_argument("--backend", choices=["none", "external", "sv3d", "wonder3d", "artifixer"], default="none")
    parser.add_argument("--backend-command", default=None, help="External command; use {request} and {workdir} placeholders if needed")
    parser.add_argument("--backend-timeout", type=float, default=600.0)
    parser.add_argument("--artifixer-repo", default=None, help="Path to the official nv-tlabs/ArtiFixer repository")
    parser.add_argument("--artifixer-checkpoint", default=None, help="Path to artifixer-*.pt checkpoint")
    parser.add_argument("--artifixer-python", default="python", help="Python executable inside the ArtiFixer environment")
    parser.add_argument("--artifixer-save-dir", default=None, help="Optional ArtiFixer inference output root")
    parser.add_argument(
        "--artifixer-command",
        default=None,
        help="Optional command template; supports {split}, {save_dir}, {checkpoint}, {repo}, {workdir}, {scene_id}",
    )
    parser.add_argument("--artifixer-prompt-path", default=None, help="Optional precomputed ArtiFixer caption HDF5")
    parser.add_argument(
        "--artifixer-render-trajectory",
        default="trajectory",
        choices=["trajectory", "all_frames", "val_frames"],
        help="ArtiFixer model_eval.run_inference render_trajectory mode",
    )
    parser.add_argument(
        "--refinement-stage",
        default="legacy",
        choices=["legacy", "cleanup_only", "generate_only", "local_refine"],
        help="Stage2 rollout mode: legacy full merge, cleanup only, pseudo-view generation, or local refinement",
    )
    parser.add_argument("--local-refine-iterations", type=int, default=80)
    parser.add_argument("--local-refine-lr", type=float, default=0.03)
    args = parser.parse_args()

    setup_logging("INFO")
    generation_model = "wonder3d" if args.backend == "wonder3d" and args.generation_model == "sv3d" else args.generation_model
    stage_dir = Path(args.stage_dir)
    cameras = _resolve_cameras(stage_dir, args.cameras)
    output_dir = Path(args.output_dir) if args.output_dir else stage_dir.parent / "stage2_object_refinement"
    cfg = ObjectRefinementConfig(
        refinement_stage=args.refinement_stage,
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
        generation_model=generation_model,
        generation_model_path=args.generation_model_path,
        wonder3d_crop_size=int(args.wonder3d_crop_size),
        wonder3d_guidance_scale=float(args.wonder3d_guidance_scale),
        reconstruction_model=args.reconstruction_model,
        inpaint_model=args.inpaint_model,
        icp_iterations=int(args.icp_iterations),
        use_foreground_confidence=not bool(args.no_foreground_confidence),
        min_foreground_confidence=float(args.min_foreground_confidence),
        background_confidence_threshold=float(args.background_confidence_threshold),
        background_structure_quantile=float(args.background_structure_quantile),
        background_structure_distance=float(args.background_structure_distance),
        foreground_core_quantile=float(args.foreground_core_quantile),
        adaptive_cleanup=not bool(args.no_adaptive_cleanup),
        max_off_mask_ratio=float(args.max_off_mask_ratio),
        robust_core_bbox_quantile=float(args.robust_core_bbox_quantile),
        robust_core_bbox_expand=float(args.robust_core_bbox_expand),
        generated_min_view_support=int(args.generated_min_view_support),
        generated_min_support_ratio=float(args.generated_min_support_ratio),
        generated_bbox_expand=float(args.generated_bbox_expand),
        local_refine_iterations=int(args.local_refine_iterations),
        local_refine_lr=float(args.local_refine_lr),
    )
    stage_input = load_stage_input(
        Path(args.ply),
        stage_dir,
        cameras_path=cameras,
        object_ply_dir=Path(args.object_ply_dir) if args.object_ply_dir else None,
        object_manifest=Path(args.object_manifest) if args.object_manifest else None,
        device=str(args.device),
    )
    backend_kwargs = {
        "generation_model": generation_model,
        "generation_model_path": args.generation_model_path,
        "reconstruction_model": args.reconstruction_model,
        "inpaint_model": args.inpaint_model,
        "num_generation_views": int(args.num_generation_views),
        "generation_resolution": int(args.generation_resolution),
        "icp_iterations": int(args.icp_iterations),
        "device": args.device,
        "artifixer_repo": args.artifixer_repo,
        "artifixer_checkpoint": args.artifixer_checkpoint,
        "artifixer_python": args.artifixer_python,
        "artifixer_save_dir": args.artifixer_save_dir,
        "artifixer_command": args.artifixer_command,
        "artifixer_prompt_path": args.artifixer_prompt_path,
        "artifixer_render_trajectory": args.artifixer_render_trajectory,
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
