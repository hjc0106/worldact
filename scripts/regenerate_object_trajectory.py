#!/usr/bin/env python3
"""Regenerate Stage2 prepared-scene trajectories (optionally front-upright).

Copies caption.h5 from an existing prepared scene, regenerates target cameras
(and optionally anchors) from object_local.ply, re-renders RGB/opacity, and
writes QC + a trajectory preview mosaic.

Examples:

  # Upper-hemisphere orbit (previous mode)
  python worldact/scripts/regenerate_object_trajectory.py \\
    --source-scene .../white_pickup_1 \\
    --output-scene .../white_pickup_1_front_views \\
    --mode front_views --replace

  # Front-only upright roll
  python worldact/scripts/regenerate_object_trajectory.py \\
    --source-scene .../white_pickup_1 \\
    --output-scene .../white_pickup_1_front_upright \\
    --mode front_upright --front-yaw-deg 0 --replace
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).resolve().parents[2]
WORLDACT_ROOT = REPO_ROOT / "worldact"
for path in (REPO_ROOT, WORLDACT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_processing.artifixer3d import write_colmap_cameras, write_colmap_images
from data_processing.camera_trajectories import transforms_json, write_json
from data_processing.worldact_object_inputs import (
    _opencv_camera,
    _qvec_from_rotation,
    _read_ply_xyz,
    _render_target_views,
    _validate_caption_h5,
    _write_points3d_from_ply,
    opengl_c2w_to_opencv_w2c,
    worldact_c2w_to_opengl_c2w,
)
from worldact.stage2_artifixer.config import PrepareConfig
from worldact.stage2_artifixer.contracts import TrajectoryDocument
from worldact.stage2_artifixer.trajectory import (
    build_target_trajectory,
    camera_roll_error_deg,
    estimate_up_from_opengl_frames,
    look_at_c2w,
    resolve_front_axis,
    resolve_up_axis,
)
from worldact.stage2_artifixer.up_axis import write_up_axis_json
from worldact.stage2_artifixer.contracts import AnchorsDocument


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument("--ply", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--mode",
        choices=["front_views", "front_upright"],
        default="front_upright",
    )
    parser.add_argument("--target-count", type=int, default=32)
    parser.add_argument("--anchor-count", type=int, default=5)
    parser.add_argument("--front-yaw-deg", type=float, default=0.0)
    parser.add_argument("--front-axis", type=float, nargs=3, default=None)
    parser.add_argument(
        "--up-axis",
        type=float,
        nargs=3,
        default=None,
        help="Manual upright axis in object-local coords (skips auto estimation).",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Stage-1 directory with background/gaussians.ply for scene-ground up.",
    )
    parser.add_argument(
        "--min-up-confidence",
        type=float,
        default=0.45,
        help="Abort if auto up-axis confidence is below this (unless --up-axis set).",
    )
    parser.add_argument("--front-azimuth-half-span", type=float, default=110.0)
    parser.add_argument("--ring-elevation-deg", type=float, default=20.0)
    parser.add_argument("--circumradius-scale", type=float, default=1.25)
    parser.add_argument("--look-at-up-bias", type=float, default=0.12)
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--camera-distance-scale", type=float, default=3.0)
    parser.add_argument("--camera-distance-scale-range", type=float, nargs=2, default=(2.9, 3.2))
    parser.add_argument("--edge-margin", type=float, default=0.15)
    parser.add_argument("--quantile", type=float, nargs=2, default=(0.02, 0.98))
    parser.add_argument("--min-elevation-deg", type=float, default=None)
    parser.add_argument("--keep-source-anchors", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-qc", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source_scene.resolve()
    output = args.output_scene.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source scene not found: {source}")
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"output exists (pass --replace): {output}")
        shutil.rmtree(output)

    scene_id = args.scene_id or output.name
    ply_path = (args.ply or source / "initial_object" / "object_local.ply").resolve()
    if not ply_path.is_file():
        raise FileNotFoundError(f"missing PLY: {ply_path}")

    source_split = _load_json(source / "split.json")
    source_entry = _scene_entry(source_split)
    source_transforms = _load_json(source / "transforms.json")
    source_selected = list(_load_json(source / "selected_indices.json"))
    caption_src = source / str(source_entry["prompt_path"])
    caption_info = _validate_caption_h5(caption_src)

    cfg = _build_config(args)
    import torch

    from worldact.stage2_artifixer.up_axis import _load_gaussian_xyz_opacity_scales

    means_np, opac_np, scales_np = _load_gaussian_xyz_opacity_scales(ply_path)
    points = torch.as_tensor(means_np, dtype=torch.float32)
    opacities = None if opac_np is None else torch.as_tensor(opac_np, dtype=torch.float32)
    scales = None if scales_np is None else torch.as_tensor(scales_np, dtype=torch.float32)
    up_hint = estimate_up_from_opengl_frames(source_transforms["frames"], indices=source_selected)
    local_to_world = None
    norm_path = source / "normalization.json"
    if norm_path.is_file():
        norm = _load_json(norm_path)
        if "local_to_world" in norm:
            local_to_world = torch.as_tensor(norm["local_to_world"], dtype=torch.float32)
    camera_centers_local = None
    if source_transforms.get("frames"):
        cams = []
        for i in source_selected:
            mat = np.asarray(source_transforms["frames"][int(i)]["transform_matrix"], dtype=np.float64)
            cams.append(mat[:3, 3])
        if cams:
            # transforms.json is OpenGL; centers are still valid positions in local frame.
            camera_centers_local = torch.as_tensor(np.stack(cams, axis=0), dtype=torch.float32)

    stage_dir = str(args.stage_dir.resolve()) if args.stage_dir is not None else cfg.stage_dir
    up_axis, up_estimate = resolve_up_axis(
        cfg,
        points_local=points,
        up_hint=up_hint,
        stage_dir=stage_dir,
        local_to_world=local_to_world,
        camera_centers_local=camera_centers_local,
        opacities=opacities,
        scales=scales,
        return_estimate=True,
    )
    if args.up_axis is None and float(up_estimate.confidence) < float(args.min_up_confidence):
        raise RuntimeError(
            f"up-axis confidence {up_estimate.confidence:.3f} < {args.min_up_confidence:.3f} "
            f"(source={up_estimate.source}). Pass --up-axis or lower --min-up-confidence."
        )
    # Lock the estimated axis into cfg so later build_target_trajectory reuse matches.
    cfg.up_axis = tuple(float(x) for x in up_axis.tolist())
    front_axis = resolve_front_axis(cfg, points_local=points, up_axis=up_axis)
    print(
        f"[up_axis] source={up_estimate.source} confidence={up_estimate.confidence:.3f} "
        f"sign={up_estimate.sign_source} axis={up_estimate.up_axis}"
    )

    empty_anchors = AnchorsDocument(schema_version="1.0", image_size=cfg.image_size, anchors=[])
    trajectory = build_target_trajectory(
        empty_anchors,
        cfg,
        local_radius=1.0,
        points_local=points,
        up_hint=up_hint,
    )
    if len(trajectory.frames) != cfg.target_count:
        raise RuntimeError(
            f"expected {cfg.target_count} target frames, got {len(trajectory.frames)}; "
            "relax FOV/distance/min-elevation constraints"
        )

    n_tgt = len(trajectory.frames)
    regenerate_anchors = not args.keep_source_anchors
    if regenerate_anchors:
        anchor_c2ws = _build_anchor_cameras(
            cfg,
            points_local=points,
            front_axis=front_axis,
            count=int(args.anchor_count),
        )
        n_anchors = len(anchor_c2ws)
    else:
        n_anchors = len(source_selected)
        anchor_c2ws = []

    output.mkdir(parents=True)
    images_dir = output / "images"
    render_dir = output / "recon_results" / scene_id / "reconstruction" / scene_id / "ours_0" / "renders"
    opacity_dir = output / "recon_results" / scene_id / "reconstruction" / scene_id / "ours_0" / "opacity"
    captions_dir = output / "captions" / scene_id
    colmap_dir = output / "synthetic_colmap"
    sparse_dir = colmap_dir / "sparse" / "0"
    initial_dir = output / "initial_object"
    nerfstudio_dir = output / "nerfstudio"
    for path in (
        images_dir,
        render_dir,
        opacity_dir,
        captions_dir,
        sparse_dir,
        initial_dir,
        nerfstudio_dir,
        colmap_dir / "images",
    ):
        path.mkdir(parents=True, exist_ok=True)

    shutil.copy2(ply_path, initial_dir / "object_local.ply")
    if (source / "normalization.json").is_file():
        shutil.copy2(source / "normalization.json", output / "normalization.json")
    shutil.copy2(caption_src, captions_dir / "caption.h5")
    _validate_caption_h5(captions_dir / "caption.h5")
    write_up_axis_json(output / "up_axis.json", up_estimate)

    frame_specs: list[dict[str, Any]] = []
    for i, frame in enumerate(trajectory.frames):
        frame_specs.append(
            _spec_from_c2w(
                name=f"{i:05d}.png",
                c2w_opencv=frame.transform_matrix,
                size=cfg.image_size,
                fl_x=frame.fl_x,
                fl_y=frame.fl_y,
                cx=frame.cx,
                cy=frame.cy,
                kind="target",
                kind_label=frame.kind,
            )
        )

    if regenerate_anchors:
        for offset, c2w in enumerate(anchor_c2ws):
            idx = n_tgt + offset
            frame_specs.append(
                _spec_from_c2w(
                    name=f"{idx:05d}.png",
                    c2w_opencv=c2w.tolist(),
                    size=cfg.image_size,
                    fl_x=float(trajectory.fl_x),
                    fl_y=float(trajectory.fl_y),
                    cx=float(trajectory.cx),
                    cy=float(trajectory.cy),
                    kind="anchor",
                    kind_label="anchor",
                )
            )
    else:
        src_render_dir = source / str(source_entry["render_dir"])
        src_opacity_dir = source / str(source_entry["opacity_dir"])
        src_images = source / "images"
        for offset, src_idx in enumerate(source_selected):
            dst_idx = n_tgt + offset
            name = f"{dst_idx:05d}.png"
            src_name = f"{int(src_idx):05d}.png"
            for src_root, dst_root in (
                (src_images, images_dir),
                (src_render_dir, render_dir),
                (src_opacity_dir, opacity_dir),
            ):
                shutil.copy2(src_root / src_name, dst_root / name)
            src_frame = source_transforms["frames"][int(src_idx)]
            frame_specs.append(
                {
                    "kind": "anchor",
                    "kind_label": "anchor",
                    "name": name,
                    "c2w_opencv": None,
                    "c2w_opengl": src_frame["transform_matrix"],
                    "intrinsics": {
                        "camera_model": "OPENCV",
                        "fl_x": float(src_frame["fl_x"]),
                        "fl_y": float(src_frame["fl_y"]),
                        "cx": float(src_frame["cx"]),
                        "cy": float(src_frame["cy"]),
                        "w": int(src_frame["w"]),
                        "h": int(src_frame["h"]),
                        "k1": float(src_frame.get("k1", 0.0)),
                        "k2": float(src_frame.get("k2", 0.0)),
                        "p1": float(src_frame.get("p1", 0.0)),
                        "p2": float(src_frame.get("p2", 0.0)),
                    },
                }
            )

    for i in range(n_tgt):
        PILImage.new("RGB", (cfg.image_size, cfg.image_size), color=(32, 32, 32)).save(
            images_dir / f"{i:05d}.png"
        )

    render_stats: list[dict[str, Any]] = []
    if not args.skip_render:
        render_indices = list(range(n_tgt))
        if regenerate_anchors:
            render_indices.extend(range(n_tgt, n_tgt + n_anchors))
        render_stats = _render_target_views(
            initial_dir / "object_local.ply",
            [frame_specs[i] for i in render_indices],
            render_dir,
            opacity_dir,
            device=args.device,
        )
        if regenerate_anchors:
            # Anchors also serve as conditioning RGB in images/.
            for i in range(n_tgt, n_tgt + n_anchors):
                name = f"{i:05d}.png"
                shutil.copy2(render_dir / name, images_dir / name)

    transforms_frames: list[dict[str, Any]] = []
    colmap_cameras = []
    colmap_images = []
    for index, spec in enumerate(frame_specs):
        intr = dict(spec["intrinsics"])
        if spec.get("c2w_opengl") is not None:
            opengl_c2w = np.asarray(spec["c2w_opengl"], dtype=np.float64)
        else:
            opengl_c2w = worldact_c2w_to_opengl_c2w(spec["c2w_opencv"])
        transforms_frames.append(
            {
                "file_path": f"images/{spec['name']}",
                "transform_matrix": opengl_c2w.tolist(),
                **intr,
            }
        )
        camera_id = index + 1
        colmap_cameras.append(_opencv_camera(camera_id, intr))
        w2c = opengl_c2w_to_opencv_w2c(opengl_c2w)
        colmap_images.append(
            (camera_id, _qvec_from_rotation(w2c[:3, :3]), w2c[:3, 3], camera_id, spec["name"])
        )
        shutil.copy2(images_dir / spec["name"], colmap_dir / "images" / spec["name"])

    top_intrinsics = {k: v for k, v in frame_specs[0]["intrinsics"].items()}
    transforms = transforms_json(top_intrinsics, transforms_frames)
    transforms_path = output / "transforms.json"
    write_json(transforms_path, transforms)
    shutil.copy2(transforms_path, nerfstudio_dir / "transforms.json")

    target_indices = list(range(n_tgt))
    selected_indices = list(range(n_tgt, n_tgt + n_anchors))
    write_json(output / "target_indices.json", target_indices)
    write_json(output / "selected_indices.json", selected_indices)
    write_colmap_cameras(sparse_dir / "cameras.bin", colmap_cameras)
    write_colmap_images(sparse_dir / "images.bin", colmap_images)
    _write_points3d_from_ply(initial_dir / "object_local.ply", sparse_dir / "points3D.bin")

    split_entry = {
        "scene_id": scene_id,
        "transforms_path": "transforms.json",
        "image_root": ".",
        "render_dir": str(render_dir.relative_to(output)),
        "opacity_dir": str(opacity_dir.relative_to(output)),
        "selected_indices_path": "selected_indices.json",
        "target_indices_path": "target_indices.json",
        "prompt_path": str((captions_dir / "caption.h5").relative_to(output)),
        "camera_scale": float(source_entry.get("camera_scale", 0.01)),
        "metric_scale": float(source_entry.get("metric_scale", 1.0)),
        "has_gt": False,
        "initialization_mode": source_entry.get("initialization_mode", "stage1_object"),
        "initial_object_ply": "initial_object/object_local.ply",
        "colmap_dir": "synthetic_colmap",
    }
    write_json(output / "split.json", {"test": {scene_id: split_entry}})

    q_lo, q_hi = float(cfg.quantile[0]), float(cfg.quantile[1])
    pts_np = points.detach().cpu().numpy()
    look_at = 0.5 * (np.quantile(pts_np, q_lo, axis=0) + np.quantile(pts_np, q_hi, axis=0))

    qc: dict[str, Any] = {"skipped": True}
    if not args.skip_qc:
        qc = run_quality_checks(
            output_scene=output,
            transforms=transforms,
            target_indices=target_indices,
            render_dir=render_dir,
            opacity_dir=opacity_dir,
            min_elevation_deg=cfg.min_elevation_deg,
            trajectory=trajectory,
            look_at_center=look_at,
            front_axis=None if front_axis is None else front_axis.detach().cpu().numpy(),
            up_axis=up_axis.detach().cpu().numpy(),
            mode=args.mode,
        )

    summary = {
        "mode": args.mode,
        "source_scene": str(source),
        "output_scene": str(output),
        "scene_id": scene_id,
        "num_targets": n_tgt,
        "num_anchors": n_anchors,
        "regenerated_anchors": regenerate_anchors,
        "target_indices": target_indices,
        "selected_indices": selected_indices,
        "trajectory_kinds": [f.kind for f in trajectory.frames],
        "render_stats": render_stats,
        "caption": {k: caption_info[k] for k in ("caption", "num_tokens", "embedding_dim") if k in caption_info},
        "quality_checks": qc,
        "up_axis_estimate": up_estimate.to_dict(),
        "config": {
            "front_yaw_deg": cfg.front_yaw_deg,
            "front_axis": cfg.front_axis,
            "up_axis": cfg.up_axis,
            "stage_dir": stage_dir,
            "front_azimuth_half_span_deg": cfg.front_azimuth_half_span_deg,
            "min_elevation_deg": cfg.min_elevation_deg,
            "front_targets": cfg.front_targets,
            "side_targets": cfg.side_targets,
            "overhead_targets": cfg.overhead_targets,
            "camera_distance_scale": cfg.camera_distance_scale,
            "camera_distance_scale_range": cfg.camera_distance_scale_range,
            "edge_margin": cfg.edge_margin,
        },
    }
    write_json(output / "regenerate_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not args.skip_qc and not qc.get("passed", False):
        return 2
    return 0


def _build_config(args: argparse.Namespace) -> PrepareConfig:
    if args.mode == "front_upright":
        min_elev = 20.0 if args.min_elevation_deg is None else float(args.min_elevation_deg)
        return PrepareConfig(
            target_count=int(args.target_count),
            horizontal_targets=0,
            high_elevation_targets=0,
            low_elevation_targets=0,
            front_targets=20,
            side_targets=8,
            overhead_targets=4,
            front_elevation_range_deg=(25.0, 45.0),
            side_elevation_range_deg=(25.0, 35.0),
            overhead_elevation_range_deg=(50.0, 60.0),
            min_elevation_deg=min_elev,
            front_yaw_deg=None if args.front_axis is not None else float(args.front_yaw_deg),
            front_axis=tuple(args.front_axis) if args.front_axis is not None else None,
            front_azimuth_half_span_deg=float(args.front_azimuth_half_span),
            sort_trajectory_smooth=False,
            image_size=int(args.image_size),
            fov_deg=float(args.fov_deg),
            camera_distance_scale=float(args.camera_distance_scale),
            camera_distance_scale_range=tuple(args.camera_distance_scale_range),
            edge_margin=float(args.edge_margin),
            quantile=tuple(args.quantile),
            look_at_robust_center=True,
            target_anchor_min_angle_deg=0.0,
            up_axis=tuple(args.up_axis) if args.up_axis is not None else None,
            stage_dir=str(args.stage_dir.resolve()) if args.stage_dir is not None else None,
            min_up_confidence=float(args.min_up_confidence),
        )

    # For sphere-ring / front_views, let Stage-1 anchors define upright; do not
    # force a world-XY yaw that ignores vehicle tilt.
    front_yaw = None if args.mode == "front_views" and args.front_axis is None else (
        None if args.front_axis is not None else float(args.front_yaw_deg)
    )
    min_elev = 8.0 if args.min_elevation_deg is None else float(args.min_elevation_deg)
    return PrepareConfig(
        target_count=int(args.target_count),
        horizontal_targets=int(args.target_count),
        high_elevation_targets=0,
        low_elevation_targets=0,
        sphere_ring=True,
        ring_elevation_deg=float(args.ring_elevation_deg),
        circumradius_scale=float(args.circumradius_scale),
        look_at_up_bias=float(args.look_at_up_bias),
        min_elevation_deg=min_elev,
        front_yaw_deg=front_yaw,
        front_axis=tuple(args.front_axis) if args.front_axis is not None else None,
        image_size=int(args.image_size),
        fov_deg=float(args.fov_deg),
        camera_distance_scale=float(args.camera_distance_scale),
        camera_distance_scale_range=tuple(args.camera_distance_scale_range),
        edge_margin=float(args.edge_margin),
        quantile=tuple(args.quantile),
        look_at_robust_center=True,
        target_anchor_min_angle_deg=0.0,
        sort_trajectory_smooth=False,
        up_axis=tuple(args.up_axis) if args.up_axis is not None else None,
        stage_dir=str(args.stage_dir.resolve()) if args.stage_dir is not None else None,
        min_up_confidence=float(args.min_up_confidence),
    )


def _build_anchor_cameras(
    cfg: PrepareConfig,
    *,
    points_local,
    front_axis,
    count: int,
):
    import torch

    from worldact.stage2_artifixer.trajectory import (
        _object_geometry,
        _projection_fit_distance,
        circumradius,
        look_at_eye_c2w,
        resolve_up_axis,
    )

    center, half_extent, _ = _object_geometry(
        points_local=points_local,
        local_radius=1.0,
        quantile=cfg.quantile,
        look_at_robust_center=True,
    )
    up = resolve_up_axis(cfg, points_local=points_local)
    if front_axis is None:
        front = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    else:
        front = front_axis.float().reshape(3)
    front_h = front - torch.dot(front, up) * up
    if float(front_h.norm()) < 1e-4:
        front = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        front_h = front - torch.dot(front, up) * up
    front_h = front_h / front_h.norm().clamp_min(1e-6)
    right = torch.linalg.cross(up, front_h)
    right = right / right.norm().clamp_min(1e-6)
    front_h = torch.linalg.cross(right, up)
    front_h = front_h / front_h.norm().clamp_min(1e-6)

    fit = _projection_fit_distance(
        half_extent=half_extent, fov_deg=cfg.fov_deg, edge_margin=cfg.edge_margin
    )
    if cfg.sphere_ring:
        R = circumradius(points_local, center=center, quantile=cfg.quantile, fallback=half_extent)
        distance = max(float(cfg.circumradius_scale) * R, fit, 1.5)
        elev = math.radians(float(cfg.ring_elevation_deg))
    else:
        distance = max(float(cfg.camera_distance_scale), fit, 1.5)
        elev = math.radians(35.0)

    elev = max(elev, math.radians(cfg.min_elevation_deg))
    cams = []
    for i in range(count):
        az = 2.0 * math.pi * i / max(count, 1)
        offset = distance * (
            math.cos(elev) * math.cos(az) * front_h
            + math.cos(elev) * math.sin(az) * right
            + math.sin(elev) * up
        )
        cams.append(look_at_eye_c2w(eye=center + offset, target=center, up=up))
    return cams


def _spec_from_c2w(
    *,
    name: str,
    c2w_opencv: list,
    size: int,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    kind: str,
    kind_label: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "kind_label": kind_label,
        "name": name,
        "c2w_opencv": c2w_opencv,
        "intrinsics": {
            "camera_model": "OPENCV",
            "fl_x": float(fl_x),
            "fl_y": float(fl_y),
            "cx": float(cx),
            "cy": float(cy),
            "w": int(size),
            "h": int(size),
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
        },
    }


def run_quality_checks(
    *,
    output_scene: Path,
    transforms: dict[str, Any],
    target_indices: list[int],
    render_dir: Path,
    opacity_dir: Path,
    min_elevation_deg: float,
    trajectory: TrajectoryDocument,
    look_at_center: np.ndarray,
    front_axis: np.ndarray | None,
    up_axis: np.ndarray | None,
    mode: str,
) -> dict[str, Any]:
    import torch

    frames = transforms["frames"]
    center = np.asarray(look_at_center, dtype=np.float64).reshape(3)
    up = None if up_axis is None else np.asarray(up_axis, dtype=np.float64).reshape(3)
    if up is not None:
        up = up / max(np.linalg.norm(up), 1e-8)
    elevations: list[float] = []
    camera_z: list[float] = []
    above_equator = 0
    elev_deltas: list[float] = []
    roll_errors: list[float] = []
    areas: list[float] = []
    edge_hits = 0
    empty_rgb = 0
    empty_opacity = 0
    preview_cells: list[PILImage.Image] = []

    prev_elev = None
    front_t = None if front_axis is None else torch.as_tensor(front_axis, dtype=torch.float32)

    for idx in target_indices:
        frame = frames[idx]
        eye = np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, 3]
        offset = eye - center
        dist = float(np.linalg.norm(offset))
        if up is not None:
            height = float(np.dot(offset, up))
            elev = math.degrees(math.asin(float(np.clip(height / max(dist, 1e-8), -1.0, 1.0))))
            if height > 0:
                above_equator += 1
        else:
            elev = math.degrees(math.asin(float(np.clip(offset[2] / max(dist, 1e-8), -1.0, 1.0))))
            if float(eye[2]) > float(center[2]):
                above_equator += 1
        elevations.append(elev)
        camera_z.append(float(eye[2]))
        if prev_elev is not None:
            elev_deltas.append(abs(elev - prev_elev))
        prev_elev = elev

        # transforms are OpenGL; convert back to OpenCV for roll check.
        from data_processing.worldact_object_inputs import opengl_c2w_to_worldact_c2w

        c2w_cv = torch.as_tensor(opengl_c2w_to_worldact_c2w(frame["transform_matrix"]), dtype=torch.float32)
        if front_t is not None and mode == "front_upright":
            roll_errors.append(camera_roll_error_deg(c2w_cv, front_t))

        rgb_path = render_dir / f"{idx:05d}.png"
        opa_path = opacity_dir / f"{idx:05d}.png"
        if not rgb_path.is_file() or not np.any(np.asarray(PILImage.open(rgb_path).convert("RGB"))):
            empty_rgb += 1
        if not opa_path.is_file():
            empty_opacity += 1
        else:
            alpha = np.asarray(PILImage.open(opa_path).convert("L"), dtype=np.float32) / 255.0
            mask = alpha > 0.01
            areas.append(float(mask.mean()))
            # Ignore 2px soft-gaussian bleed on the border; only flag solid contact.
            inset = 2
            core = mask[inset:-inset, inset:-inset] if mask.shape[0] > 2 * inset else mask
            border = mask.copy()
            border[inset:-inset, inset:-inset] = False
            if mask.any() and int(border.sum()) >= 8 and not bool(core.any() and core.mean() > 0.01):
                edge_hits += 1
            elif mask.any() and int(border.sum()) >= max(32, int(0.002 * mask.size)):
                # Substantial border occupancy even with interior content.
                edge_hits += 1
            elif not mask.any():
                empty_opacity += 1
        if rgb_path.is_file():
            preview_cells.append(PILImage.open(rgb_path).convert("RGB"))

    elev_arr = np.asarray(elevations, dtype=np.float64)
    area_arr = np.asarray(areas, dtype=np.float64) if areas else np.zeros(0)
    roll_arr = np.asarray(roll_errors, dtype=np.float64) if roll_errors else np.zeros(0)
    elev_delta_arr = np.asarray(elev_deltas, dtype=np.float64) if elev_deltas else np.zeros(0)

    mosaic_path = output_scene / "trajectory_preview.jpg"
    if preview_cells:
        _save_mosaic(preview_cells, mosaic_path)

    checks = {
        "min_elevation_ok": bool((elev_arr >= min_elevation_deg - 1e-3).all()) if elev_arr.size else False,
        "cameras_above_vehicle_equator": above_equator == len(target_indices),
        "alpha_not_touching_edges": edge_hits == 0,
        "rgb_non_empty": empty_rgb == 0,
        "opacity_non_empty": empty_opacity == 0,
        "no_low_kind_frames": all(k != "low" for k in (f.kind for f in trajectory.frames)),
    }
    if mode == "front_upright":
        checks["adjacent_elevation_delta_ok"] = bool(
            elev_delta_arr.size == 0 or float(elev_delta_arr.max()) <= 10.0 + 1e-3
        )
        checks["roll_error_ok"] = bool(roll_arr.size > 0 and float(roll_arr.max()) <= 5.0 + 1e-3)
        checks["cameras_above_target"] = bool((np.asarray(camera_z) > float(center[2])).all()) if camera_z else False
    if mode == "front_views":
        checks["no_negative_elevation"] = bool((elev_arr >= -1e-3).all()) if elev_arr.size else False
        checks["target_area_stable"] = bool(area_arr.size > 0 and float(area_arr.std()) <= 0.12)
        # Sphere ring should keep nearly constant elevation.
        checks["ring_elevation_stable"] = bool(
            elev_arr.size > 0 and float(elev_arr.max() - elev_arr.min()) <= 1.0 + 1e-3
        )

    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "look_at_center": center.tolist(),
        "up_axis": None if up is None else up.tolist(),
        "elevation_deg": {
            "min": float(elev_arr.min()) if elev_arr.size else None,
            "max": float(elev_arr.max()) if elev_arr.size else None,
            "mean": float(elev_arr.mean()) if elev_arr.size else None,
        },
        "adjacent_elevation_delta_deg": {
            "max": float(elev_delta_arr.max()) if elev_delta_arr.size else None,
            "mean": float(elev_delta_arr.mean()) if elev_delta_arr.size else None,
        },
        "roll_error_deg": {
            "max": float(roll_arr.max()) if roll_arr.size else None,
            "mean": float(roll_arr.mean()) if roll_arr.size else None,
        },
        "target_area_ratio": {
            "min": float(area_arr.min()) if area_arr.size else None,
            "max": float(area_arr.max()) if area_arr.size else None,
            "mean": float(area_arr.mean()) if area_arr.size else None,
            "std": float(area_arr.std()) if area_arr.size else None,
        },
        "edge_hits": edge_hits,
        "empty_rgb": empty_rgb,
        "empty_opacity": empty_opacity,
        "preview_mosaic": str(mosaic_path.name) if mosaic_path.is_file() else None,
    }
    write_json(output_scene / "trajectory_qc.json", report)
    return report


def _save_mosaic(images: list[PILImage.Image], path: Path, cols: int = 8) -> None:
    if not images:
        return
    w, h = images[0].size
    cols = max(1, min(cols, len(images)))
    rows = int(math.ceil(len(images) / cols))
    canvas = PILImage.new("RGB", (cols * w, rows * h), color=(16, 16, 16))
    for i, image in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(image.resize((w, h)), (c * w, r * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _scene_entry(split: dict[str, Any]) -> dict[str, Any]:
    test = split.get("test") or {}
    if not test:
        raise ValueError("split.json missing test entry")
    return next(iter(test.values()))


if __name__ == "__main__":
    raise SystemExit(main())
