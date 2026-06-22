#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldact.config import default_config, load_config
from worldact.data.gaussians import GaussianScene
from worldact.stage1_decomposition.aerial_roi import (
    AerialCameraConfig,
    AerialRenderSet,
    build_occupancy_grid,
    export_region_chunks,
    make_aerial_cameras,
    proposals_from_direct_bbox,
    proposals_from_occupancy,
    render_aerial_views,
    save_aerial_render_artifacts,
    save_occupancy_debug,
    scene_bounds,
    segment_aerial_masks,
)
from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
from worldact.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def _load_pipeline_config(args: argparse.Namespace):
    overrides = [
        f"output_dir={args.output_dir}",
        f"device.device={args.device}",
        f"stage1.sam3.device={args.device}",
    ]
    if args.sam3_confidence is not None:
        overrides.append(f"stage1.sam3.confidence_threshold={float(args.sam3_confidence)}")
    if args.sam3_source_dir is not None:
        overrides.append(f"stage1.sam3.source_dir={args.sam3_source_dir}")
    if args.sam3_checkpoint is not None:
        overrides.append(f"stage1.sam3.checkpoint_path={args.sam3_checkpoint}")
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path, overrides) if cfg_path.exists() else default_config(overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _write_empty_outputs(
    output_dir: Path,
    summary: dict[str, object],
    mask_records: list[dict[str, object]],
    reason: str,
    *,
    num_observations: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "aerial_masks.json", "w", encoding="utf-8") as f:
        json.dump(mask_records, f, indent=2)
    with open(output_dir / "region_manifest.json", "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)
    with open(output_dir / "aerial_region_summary.json", "w", encoding="utf-8") as f:
        data = dict(summary)
        data["status"] = reason
        data["num_observations"] = int(num_observations)
        data["num_regions"] = 0
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-guided aerial region segmentation for 3DGS scenes")
    parser.add_argument("--ply", required=True, help="Input standard 3DGS PLY")
    parser.add_argument("--config", default="configs/scenes/outdoor.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", action="append", required=True, help="Region prompt; repeat for multi-prompt fusion")
    parser.add_argument("--up-axis", required=True, help="Scene up axis, e.g. -y for garden/building")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=640)
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--max-views", type=int, default=120)
    parser.add_argument("--grid-cols", type=int, default=None)
    parser.add_argument("--grid-rows", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.35)
    parser.add_argument("--altitude", type=float, default=None)
    parser.add_argument("--clearance", type=float, default=0.8)
    parser.add_argument("--bounds-quantile", type=float, nargs=2, default=[0.01, 0.99])
    parser.add_argument("--bounds-margin", type=float, default=0.04)
    parser.add_argument("--contact-sheet-cols", type=int, default=6)
    parser.add_argument("--candidate-mode", choices=["best", "all", "union"], default="all")
    parser.add_argument("--max-candidates-per-prompt", type=int, default=5)
    parser.add_argument("--min-mask-area-ratio", type=float, default=1e-4)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.75)
    parser.add_argument(
        "--region-mode",
        choices=["bev", "direct_bbox"],
        default="bev",
        help="Region proposal mode: BEV occupancy fusion or direct horizontal bbox from all projected mask points",
    )
    parser.add_argument("--grid-resolution", type=int, default=512)
    parser.add_argument("--occupancy-threshold", type=float, default=0.25)
    parser.add_argument("--min-component-area", type=int, default=16)
    parser.add_argument("--bbox-margin", type=float, default=1.0)
    parser.add_argument("--keep-top-k", type=int, default=3)
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--max-points-per-mask", type=int, default=50000)
    parser.add_argument("--height-mode", choices=["full", "quantile"], default="full")
    parser.add_argument("--height-quantile", type=float, nargs=2, default=[0.02, 0.98])
    parser.add_argument("--height-margin", type=float, default=0.0)
    parser.add_argument("--min-opacity", type=float, default=0.005, help="Minimum sigmoid opacity for Gaussians in exported region chunks; filters out floaters")
    parser.add_argument("--max-scale", type=float, default=None, help="Maximum per-axis scale (after exp) for Gaussians in exported region chunks; filters out oversized splats")
    parser.add_argument(
        "--no-region-camera-config",
        dest="export_region_camera_config",
        action="store_false",
        help="Do not export per-region Stage 2 camera config and center visualization",
    )
    parser.set_defaults(export_region_camera_config=True)
    parser.add_argument(
        "--region-base-config",
        default=None,
        help="Base YAML copied for per-region camera configs; defaults to --config",
    )
    parser.add_argument("--region-camera-config-name", default="region_camera_config.yaml")
    parser.add_argument(
        "--no-region-vlm-prompt",
        dest="region_vlm_prompt",
        action="store_false",
        help="Keep the base config VLM prompt instead of injecting a region-aware assetization prompt",
    )
    parser.add_argument(
        "--region-vlm-max-objects",
        type=int,
        default=12,
        help="Maximum objects requested by the injected per-region Stage 2 VLM prompt",
    )
    parser.set_defaults(region_vlm_prompt=True)
    parser.add_argument(
        "--region-camera-purpose",
        choices=["coverage", "object"],
        default="coverage",
        help="coverage uses region bbox footprint; object uses volume/robust/bbox/hybrid center sources",
    )
    parser.add_argument(
        "--region-camera-coverage-mode",
        choices=["cylinder", "sphere"],
        default="cylinder",
        help="Coverage geometry label for region-level Stage 2 camera planning",
    )
    parser.add_argument("--region-camera-standoff-scale", type=float, default=0.35)
    parser.add_argument("--region-camera-min-standoff", type=float, default=1.0)
    parser.add_argument(
        "--region-camera-center-source",
        choices=["volume", "robust", "bbox", "hybrid"],
        default="volume",
        help="Stage 2 orbit center for --region-camera-purpose=object",
    )
    parser.add_argument(
        "--region-camera-height",
        type=float,
        default=1.0,
        help="Fixed camera center height above the region floor when --region-camera-height-mode=fixed",
    )
    parser.add_argument(
        "--region-camera-height-mode",
        choices=["geometric", "adaptive", "fixed", "median"],
        default="geometric",
        help="How to set the Stage 2 camera center height; geometric uses the midpoint of robust vertical bounds",
    )
    parser.add_argument(
        "--region-camera-floor-quantile",
        type=float,
        default=0.02,
        help="Quantile of region Gaussian heights used as floor when setting camera center height",
    )
    parser.add_argument("--region-camera-height-scale", type=float, default=0.35)
    parser.add_argument("--region-camera-height-min", type=float, default=0.5)
    parser.add_argument("--region-camera-height-max", type=float, default=2.0)
    parser.add_argument("--region-camera-height-quantile", type=float, default=0.98)
    parser.add_argument("--region-camera-radius-scale", type=float, default=0.65)
    parser.add_argument("--region-camera-min-radius", type=float, default=1.0)
    parser.add_argument("--region-camera-max-radius", type=float, default=25.0)
    parser.add_argument(
        "--region-camera-frame-quantile-lo",
        type=float,
        default=0.05,
        help="Lower quantile for volume center frame when --region-camera-center-source=volume",
    )
    parser.add_argument(
        "--region-camera-frame-quantile-hi",
        type=float,
        default=0.95,
        help="Upper quantile for volume center frame when --region-camera-center-source=volume",
    )
    parser.add_argument("--center-visualization-max-points", type=int, default=60000)
    parser.add_argument("--sam3-confidence", type=float, default=None)
    parser.add_argument("--sam3-source-dir", default=None)
    parser.add_argument("--sam3-checkpoint", default=None)
    args = parser.parse_args()

    setup_logging("INFO")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = [prompt.strip() for prompt in args.prompt if prompt.strip()]
    if not prompts:
        raise SystemExit("At least one non-empty --prompt is required")

    cfg = _load_pipeline_config(args)
    scene = GaussianScene.from_ply(args.ply)
    scene.params = scene.params.to(args.device)
    log.info("Loaded %d Gaussians from %s", scene.num_gaussians, args.ply)

    aerial_cfg = AerialCameraConfig(
        up_axis=str(args.up_axis),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        fov_deg=float(args.fov_deg),
        max_views=int(args.max_views),
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        overlap=float(args.overlap),
        altitude=args.altitude,
        clearance=float(args.clearance),
        bounds_quantile=(float(args.bounds_quantile[0]), float(args.bounds_quantile[1])),
        bounds_margin=float(args.bounds_margin),
    )
    cameras, records, aerial_summary = make_aerial_cameras(scene, aerial_cfg)
    cameras = [camera.to(scene.device) for camera in cameras]
    log.info("Rendering %d aerial views", len(cameras))
    views = render_aerial_views(scene, cameras)
    render_set = AerialRenderSet(cameras=cameras, views=views, records=records, summary=aerial_summary)
    save_aerial_render_artifacts(render_set, output_dir, contact_sheet_cols=int(args.contact_sheet_cols))

    segmenter = SAM3Segmenter(cfg.stage1.sam3)
    observations, mask_records = segment_aerial_masks(
        segmenter,
        render_set,
        prompts,
        candidate_mode=str(args.candidate_mode),
        max_candidates_per_prompt=int(args.max_candidates_per_prompt),
        min_mask_area_ratio=float(args.min_mask_area_ratio),
        max_mask_area_ratio=float(args.max_mask_area_ratio),
        output_dir=output_dir,
    )
    with open(output_dir / "aerial_masks.json", "w", encoding="utf-8") as f:
        json.dump(mask_records, f, indent=2)

    bounds = scene_bounds(
        scene.params.means,
        str(args.up_axis),
        quantile=(float(args.bounds_quantile[0]), float(args.bounds_quantile[1])),
        margin_ratio=float(args.bounds_margin),
    )
    if not observations:
        log.warning("SAM3 produced no usable aerial masks for prompts: %s", prompts)
        _write_empty_outputs(output_dir, aerial_summary, mask_records, "no_masks", num_observations=0)
        return

    if str(args.region_mode) == "bev":
        occupancy = build_occupancy_grid(
            render_set,
            observations,
            up_axis=str(args.up_axis),
            bounds=bounds,
            grid_resolution=int(args.grid_resolution),
            alpha_threshold=float(args.alpha_threshold),
            max_points_per_mask=int(args.max_points_per_mask),
        )
        proposals = proposals_from_occupancy(
            occupancy,
            threshold=float(args.occupancy_threshold),
            min_component_area=int(args.min_component_area),
            bbox_margin=float(args.bbox_margin),
            keep_top_k=int(args.keep_top_k),
            prompts=prompts,
        )
        save_occupancy_debug(occupancy, proposals, output_dir / "occupancy.png")
    else:
        proposals = proposals_from_direct_bbox(
            render_set,
            observations,
            up_axis=str(args.up_axis),
            bounds=bounds,
            alpha_threshold=float(args.alpha_threshold),
            max_points_per_mask=int(args.max_points_per_mask),
            bbox_margin=float(args.bbox_margin),
            prompts=prompts,
        )
    if not proposals:
        log.warning("No ROI regions survived region proposal mode %s", args.region_mode)
        _write_empty_outputs(output_dir, aerial_summary, mask_records, "no_regions", num_observations=len(observations))
        return

    region_records = export_region_chunks(
        scene,
        proposals,
        output_dir,
        up_axis=str(args.up_axis),
        height_mode=str(args.height_mode),
        height_quantile=(float(args.height_quantile[0]), float(args.height_quantile[1])),
        height_margin=float(args.height_margin),
        min_opacity=float(args.min_opacity),
        max_scale=float(args.max_scale) if args.max_scale is not None else None,
        export_camera_config=bool(args.export_region_camera_config),
        base_config_path=Path(args.region_base_config) if args.region_base_config is not None else Path(args.config),
        camera_config_name=str(args.region_camera_config_name),
        camera_purpose=str(args.region_camera_purpose),
        camera_coverage_mode=str(args.region_camera_coverage_mode),
        camera_standoff_scale=float(args.region_camera_standoff_scale),
        camera_min_standoff=float(args.region_camera_min_standoff),
        camera_center_source=str(args.region_camera_center_source),
        camera_height_mode=str(args.region_camera_height_mode),
        camera_center_height=float(args.region_camera_height),
        camera_floor_quantile=float(args.region_camera_floor_quantile),
        camera_height_scale=float(args.region_camera_height_scale),
        camera_height_min=float(args.region_camera_height_min),
        camera_height_max=float(args.region_camera_height_max),
        camera_height_quantile=float(args.region_camera_height_quantile),
        camera_radius_scale=float(args.region_camera_radius_scale),
        camera_min_radius=float(args.region_camera_min_radius),
        camera_max_radius=float(args.region_camera_max_radius),
        camera_frame_quantile=(
            float(args.region_camera_frame_quantile_lo),
            float(args.region_camera_frame_quantile_hi),
        ),
        center_visualization_max_points=int(args.center_visualization_max_points),
        region_vlm_prompt=bool(args.region_vlm_prompt),
        region_vlm_max_objects=int(args.region_vlm_max_objects) if args.region_vlm_max_objects is not None else None,
    )
    summary = {
        **aerial_summary,
        "status": "ok",
        "ply": str(args.ply),
        "prompts": prompts,
        "candidate_mode": str(args.candidate_mode),
        "region_mode": str(args.region_mode),
        "num_observations": len(observations),
        "num_regions": len(region_records),
        "occupancy_threshold": float(args.occupancy_threshold),
        "min_component_area": int(args.min_component_area),
        "bbox_margin": float(args.bbox_margin),
        "height_mode": str(args.height_mode),
        "region_camera_config": {
            "enabled": bool(args.export_region_camera_config),
            "base_config": str(args.region_base_config) if args.region_base_config is not None else str(args.config),
            "config_name": str(args.region_camera_config_name),
            "region_vlm_prompt": bool(args.region_vlm_prompt),
            "region_vlm_max_objects": int(args.region_vlm_max_objects)
            if args.region_vlm_max_objects is not None
            else None,
            "purpose": str(args.region_camera_purpose),
            "coverage_mode": str(args.region_camera_coverage_mode),
            "standoff_scale": float(args.region_camera_standoff_scale),
            "min_standoff": float(args.region_camera_min_standoff),
            "center_source": str(args.region_camera_center_source),
            "frame_quantile": [
                float(args.region_camera_frame_quantile_lo),
                float(args.region_camera_frame_quantile_hi),
            ],
            "height_mode": str(args.region_camera_height_mode),
            "fixed_height": float(args.region_camera_height),
            "floor_quantile": float(args.region_camera_floor_quantile),
            "height_scale": float(args.region_camera_height_scale),
            "height_min": float(args.region_camera_height_min),
            "height_max": float(args.region_camera_height_max),
            "height_quantile": float(args.region_camera_height_quantile),
            "radius_scale": float(args.region_camera_radius_scale),
            "min_radius": float(args.region_camera_min_radius),
            "max_radius": float(args.region_camera_max_radius),
        },
        "outputs": {
            "aerial_contact_sheet": str(output_dir / "aerial_contact_sheet.png"),
            "aerial_plan": str(output_dir / "aerial_plan.png"),
            "aerial_masks": str(output_dir / "aerial_masks.json"),
            "occupancy": str(output_dir / "occupancy.png") if str(args.region_mode) == "bev" else None,
            "region_manifest": str(output_dir / "region_manifest.json"),
        },
    }
    with open(output_dir / "aerial_region_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Exported %d region chunks to %s", len(region_records), output_dir / "regions")


if __name__ == "__main__":
    main()
