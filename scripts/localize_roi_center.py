#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldact.config import default_config
from worldact.data.gaussians import GaussianScene, RenderedView
from worldact.data.poses import CameraPose
from worldact.rendering.camera_planning import (
    RingCameraSpec,
    compute_scene_frame,
    generate_multi_ring_cameras,
    parse_up_axis,
)
from worldact.stage1_decomposition.roi_guidance import (
    auto_up_axis,
    estimate_radius,
    mask_to_world_points,
    robust_center,
    select_roi_candidate,
)
from worldact.stage1_decomposition.sam3_segmenter import SAM3Candidate, SAM3Segmenter
from worldact.utils.io import save_cameras_json
from worldact.utils.logging import get_logger, setup_logging
from worldact.utils.visualization import save_image

log = get_logger(__name__)

DEFAULT_ROI_BASE_CONFIG = "configs/scenes/indoor_table.yaml"
DEFAULT_ROI_CONFIG_NAME = "roi_camera_config.yaml"


def _auto_up_axis(means: torch.Tensor) -> str:
    means_cpu = means.detach().cpu().float()
    lo = torch.quantile(means_cpu, 0.02, dim=0)
    hi = torch.quantile(means_cpu, 0.98, dim=0)
    extents = hi - lo
    axis = int(torch.argmin(extents).item())
    return ["x", "y", "z"][axis]


def _resolve_probe_radius(args: argparse.Namespace, frame) -> float:
    if float(args.radius) > 0.0:
        return float(args.radius)
    return float(frame.recommended_radii[2])


def _make_intrinsic(image_width: int, image_height: int, fov_deg: float) -> torch.Tensor:
    fov = math.radians(fov_deg)
    fy = image_height / (2.0 * math.tan(fov / 2.0))
    fx = fy
    return torch.tensor(
        [[fx, 0.0, image_width / 2.0], [0.0, fy, image_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _make_probe_camera_cfg(args: argparse.Namespace, up_axis: str, frame) -> SimpleNamespace:
    radius = _resolve_probe_radius(args, frame)
    return SimpleNamespace(
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        fov_deg=float(args.fov_deg),
        pitch_deg=float(args.pitch_deg),
        radius_jitter=radius,
        orbit_radius=radius,
        up_axis=up_axis,
        rings=[
            RingCameraSpec(
                name="probe_level",
                num_cameras=int(args.num_views),
                orbit_radius=radius,
                height_offset=float(args.height_offset),
                look_at_center=True,
            ),
            RingCameraSpec(
                name="probe_high",
                num_cameras=max(4, int(args.num_views) // 2),
                orbit_radius=radius * 0.85,
                height_offset=float(args.height_offset) + float(args.high_offset),
                look_at_center=True,
            ),
        ],
    )


def _save_contact_sheet(image_paths: list[Path], output_path: Path, *, columns: int = 4) -> None:
    if not image_paths:
        return
    import imageio.v3 as iio
    import numpy as np

    images = [iio.imread(path) for path in image_paths]
    height = max(img.shape[0] for img in images)
    width = max(img.shape[1] for img in images)
    rows = math.ceil(len(images) / columns)
    sheet = np.ones((rows * height, columns * width, 3), dtype=np.uint8) * 255
    for idx, img in enumerate(images):
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        img = img[..., :3]
        row = idx // columns
        col = idx % columns
        sheet[row * height : row * height + img.shape[0], col * width : col * width + img.shape[1]] = img
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, sheet)


def _camera_center(camera: CameraPose) -> list[float]:
    return [float(v) for v in camera.c2w[:3, 3].detach().cpu()]


def _render_probe_views(
    scene: GaussianScene,
    cameras: list[CameraPose],
    output_dir: Path,
) -> tuple[list[RenderedView], list[Path]]:
    views_dir = output_dir / "probe_views"
    views_dir.mkdir(parents=True, exist_ok=True)
    views: list[RenderedView] = []
    image_paths: list[Path] = []
    records: list[dict[str, object]] = []
    for idx, camera in enumerate(cameras):
        rendered = scene.render(camera)
        view = RenderedView(
            rgb=rendered.rgb.detach().cpu(),
            depth=rendered.depth.detach().cpu(),
            alpha=rendered.alpha.detach().cpu(),
        )
        views.append(view)
        image_path = views_dir / f"probe_{idx:03d}.png"
        save_image(view.rgb, image_path)
        image_paths.append(image_path)
        records.append(
            {
                "view_index": idx,
                "filename": image_path.name,
                "camera_center": _camera_center(camera),
            }
        )
    with open(views_dir / "probe_views.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    _save_contact_sheet(image_paths, output_dir / "probe_contact_sheet.png", columns=4)
    return views, image_paths


def _candidate_center_score(candidate: SAM3Candidate, image_size: tuple[int, int]) -> float:
    height, width = image_size
    mask = candidate.mask > 0.5
    if not bool(mask.any()):
        return float("-inf")
    ys, xs = mask.nonzero(as_tuple=True)
    cx = float(xs.float().mean().item()) / max(1.0, float(width))
    cy = float(ys.float().mean().item()) / max(1.0, float(height))
    center_bias = 1.0 - min(1.0, math.hypot(cx - 0.5, cy - 0.5) / 0.707)
    area_ratio = float(candidate.area) / max(1.0, float(height * width))
    size_bias = min(1.0, max(0.1, math.sqrt(max(area_ratio, 1e-8)) * 4.0))
    return float(candidate.quality) * (0.5 + 0.35 * center_bias + 0.15 * size_bias)


def _select_roi_candidate(
    segmenter: SAM3Segmenter,
    views: list[RenderedView],
    prompt: str,
) -> tuple[int, SAM3Candidate, float] | None:
    best: tuple[int, SAM3Candidate, float] | None = None
    for view_idx, view in enumerate(views):
        candidate_set = segmenter.segment_candidates(view.rgb, prompt)
        if candidate_set is None or not candidate_set.candidates:
            continue
        for candidate in candidate_set.candidates:
            score = _candidate_center_score(candidate, candidate_set.image_size)
            if best is None or score > best[2]:
                best = (view_idx, candidate, score)
    return best


def _mask_to_world_points(
    view: RenderedView,
    camera: CameraPose,
    mask: torch.Tensor,
    *,
    max_points: int = 50000,
) -> torch.Tensor:
    mask = (mask > 0.5).to(view.depth.device)
    ys, xs = mask.nonzero(as_tuple=True)
    if ys.numel() == 0:
        return torch.zeros(0, 3, dtype=torch.float32, device=view.depth.device)
    depth = view.depth[ys, xs]
    valid = depth > 1e-6
    if not bool(valid.any()):
        return torch.zeros(0, 3, dtype=torch.float32, device=view.depth.device)
    ys = ys[valid].float()
    xs = xs[valid].float()
    depth = depth[valid].float()
    if xs.numel() > max_points:
        step = max(1, xs.numel() // max_points)
        xs = xs[::step]
        ys = ys[::step]
        depth = depth[::step]

    K = camera.intrinsic.to(view.depth.device)
    c2w = camera.c2w.to(view.depth.device)
    x_cam = (xs - K[0, 2]) / K[0, 0] * depth
    y_cam = (ys - K[1, 2]) / K[1, 1] * depth
    cam_pts = torch.stack([x_cam, y_cam, depth], dim=1)
    return (c2w[:3, :3] @ cam_pts.T).T + c2w[:3, 3]


def _robust_center(points: torch.Tensor) -> torch.Tensor:
    center = points.median(dim=0).values
    if points.shape[0] < 32:
        return center
    distances = torch.norm(points - center, dim=1)
    keep = distances <= torch.quantile(distances, 0.85)
    if bool(keep.any()):
        center = points[keep].median(dim=0).values
    return center


def _estimate_radius(points: torch.Tensor, center: torch.Tensor, up_axis: str) -> float:
    up_idx, _ = parse_up_axis(up_axis)
    axes = [0, 1, 2]
    axes.remove(up_idx)
    horizontal = points[:, axes] - center[axes]
    distances = torch.norm(horizontal, dim=1)
    if distances.numel() == 0:
        return 1.0
    radius = float(torch.quantile(distances, 0.90).item())
    return float(max(0.4, min(3.0, radius * 2.5)))


def _overlay_mask(rgb: torch.Tensor, mask: torch.Tensor, color: tuple[float, float, float]) -> torch.Tensor:
    overlay = rgb.detach().cpu().float().clone()
    mask = (mask.detach().cpu().float().clamp(0.0, 1.0)).unsqueeze(-1)
    color_t = torch.tensor(color, dtype=overlay.dtype).view(1, 1, 3)
    return overlay * (1.0 - 0.45 * mask) + color_t * 0.45 * mask


def _axis_vector(axis: int, sign: float = 1.0) -> torch.Tensor:
    vec = torch.zeros(3, dtype=torch.float32)
    vec[int(axis)] = float(sign)
    return vec


def _horizontal_axes(up_axis: str) -> tuple[int, int, int, float]:
    up_idx, up_sign = parse_up_axis(up_axis)
    axes = [0, 1, 2]
    axes.remove(up_idx)
    return axes[0], axes[1], up_idx, up_sign


def _topdown_camera_pose(
    center: torch.Tensor,
    up_axis: str,
    frame,
    *,
    image_width: int,
    image_height: int,
    fov_deg: float,
    radius: float,
) -> CameraPose:
    h0, h1, up_idx, up_sign = _horizontal_axes(up_axis)
    center_cpu = center.detach().cpu().float()
    cam_pos = center_cpu.clone()
    distance = max(float(radius) * 4.0, float(frame.extent_xy) * 0.25, float(frame.recommended_radii[2]) * 2.0, 3.0)
    cam_pos[up_idx] += up_sign * distance
    forward = center_cpu - cam_pos

    # Pick a stable reference axis that is not parallel to the view direction.
    ref = _axis_vector(h0)
    right = torch.cross(ref, forward, dim=0)
    if float(torch.linalg.norm(right).item()) < 1e-6:
        ref = _axis_vector(h1)
        right = torch.cross(ref, forward, dim=0)
    right = F.normalize(right, dim=0)
    forward = F.normalize(forward, dim=0)
    up = torch.cross(right, forward, dim=0)

    R = torch.stack([right, -up, forward], dim=0)
    t = -R @ cam_pos
    extrinsic = torch.eye(4, dtype=torch.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return CameraPose(
        extrinsic=extrinsic,
        intrinsic=_make_intrinsic(image_width, image_height, fov_deg),
        width=image_width,
        height=image_height,
    )


def _project_world_points(points: torch.Tensor, camera: CameraPose) -> torch.Tensor:
    if points.numel() == 0:
        return torch.zeros(0, 2, dtype=torch.float32)
    points_cpu = points.detach().cpu().float()
    ones = torch.ones((points_cpu.shape[0], 1), dtype=torch.float32)
    pts_h = torch.cat([points_cpu, ones], dim=1)
    cam = (camera.extrinsic.detach().cpu().float() @ pts_h.T).T
    z = cam[:, 2]
    valid = z > 1e-6
    if not bool(valid.any()):
        return torch.zeros(0, 2, dtype=torch.float32)
    cam = cam[valid]
    z = z[valid]
    K = camera.intrinsic.detach().cpu().float()
    u = K[0, 0] * (cam[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (cam[:, 1] / z) + K[1, 2]
    return torch.stack([u, v], dim=1)


def _compute_square_crop_bounds(
    points_2d: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.25,
    min_half_size_px: int = 96,
) -> tuple[int, int, int, int] | None:
    if points_2d.numel() == 0:
        return None

    points = points_2d.detach().cpu().float()
    x0 = float(points[:, 0].min().item())
    x1 = float(points[:, 0].max().item())
    y0 = float(points[:, 1].min().item())
    y1 = float(points[:, 1].max().item())
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half_size = max(0.5 * max(x1 - x0, y1 - y0), float(min_half_size_px))
    half_size *= 1.0 + float(padding_ratio)
    half_size = min(half_size, 0.5 * min(image_width, image_height))

    left = int(round(cx - half_size))
    right = int(round(cx + half_size))
    top = int(round(cy - half_size))
    bottom = int(round(cy + half_size))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height

    left = max(0, left)
    top = max(0, top)
    right = min(image_width, max(left + 2, right))
    bottom = min(image_height, max(top + 2, bottom))
    return left, top, right, bottom


def _save_topdown_scene_render(
    scene: GaussianScene,
    center: torch.Tensor,
    roi_points: torch.Tensor,
    *,
    up_axis: str,
    frame,
    radius: float,
    output_path: Path,
    image_width: int,
    image_height: int,
    fov_deg: float,
) -> Path:
    from PIL import Image, ImageDraw
    import numpy as np

    camera = _topdown_camera_pose(
        center,
        up_axis,
        frame,
        image_width=image_width,
        image_height=image_height,
        fov_deg=fov_deg,
        radius=radius,
    ).to(scene.device)
    rendered = scene.render(camera)
    rgb = rendered.rgb.detach().cpu().clamp(0.0, 1.0).numpy()
    base = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    roi_px = _project_world_points(roi_points, camera)
    if roi_px.shape[0] > 0:
        if roi_px.shape[0] > 3000:
            step = max(1, roi_px.shape[0] // 3000)
            roi_px = roi_px[::step]
        for x, y in roi_px.tolist():
            if math.isfinite(x) and math.isfinite(y):
                draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(255, 0, 0, 85))

    h0, h1, _, _ = _horizontal_axes(up_axis)
    radius_points = torch.stack(
        [
            center.detach().cpu().float(),
            center.detach().cpu().float()
            + _axis_vector(h0) * float(radius),
            center.detach().cpu().float()
            - _axis_vector(h0) * float(radius),
            center.detach().cpu().float()
            + _axis_vector(h1) * float(radius),
            center.detach().cpu().float()
            - _axis_vector(h1) * float(radius),
        ],
        dim=0,
    )
    radius_px = _project_world_points(radius_points, camera)
    if radius_px.shape[0] >= 5:
        x0, y0 = radius_px[1:5].min(dim=0).values.tolist()
        x1, y1 = radius_px[1:5].max(dim=0).values.tolist()
        draw.ellipse((x0, y0, x1, y1), outline=(255, 64, 64, 190), width=3)

    center_px = _project_world_points(center.detach().cpu().float().unsqueeze(0), camera)
    if center_px.shape[0] == 1:
        cx, cy = center_px[0].tolist()
        draw.line((cx - 10, cy, cx + 10, cy), fill=(255, 0, 0, 255), width=2)
        draw.line((cx, cy - 10, cx, cy + 10), fill=(255, 0, 0, 255), width=2)
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(255, 0, 0, 230), outline=(255, 255, 255, 255), width=1)
        draw.text((cx + 8, cy - 18), "ROI center", fill=(255, 0, 0, 255))

    draw.text((12, 12), "Top-down scene render with ROI overlay", fill=(0, 0, 0, 255))
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path)

    crop_points = roi_px
    if center_px.shape[0] == 1:
        crop_points = torch.cat([crop_points, center_px], dim=0) if crop_points.numel() > 0 else center_px
    crop_bounds = _compute_square_crop_bounds(
        crop_points,
        image_width=base.size[0],
        image_height=base.size[1],
        padding_ratio=0.18,
        min_half_size_px=max(72, int(0.12 * min(base.size))),
    )
    if crop_bounds is not None:
        closeup = composed.crop(crop_bounds)
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
        closeup = closeup.resize(base.size, resample=resampling)
        closeup_path = output_path.with_name(f"{output_path.stem}_closeup{output_path.suffix}")
        closeup.save(closeup_path)
    return output_path


def _load_sam3_config(args: argparse.Namespace):
    cfg = default_config(["output_dir=/tmp/worldact_roi_probe", f"device.device={args.device}"])
    if args.sam3_source_dir:
        cfg.stage1.sam3.source_dir = args.sam3_source_dir
    if args.sam3_checkpoint:
        cfg.stage1.sam3.checkpoint_path = args.sam3_checkpoint
    cfg.stage1.sam3.device = args.device
    cfg.stage1.sam3.confidence_threshold = float(args.sam3_confidence)
    cfg.stage1.sam3.instance_strategy = "largest"
    return cfg.stage1.sam3


def _default_hybrid_rings(radius: float) -> list[dict[str, object]]:
    return [
        {
            "name": "table_top",
            "num_cameras": 16,
            "orbit_radius": round(radius * 0.57, 6),
            "height_offset": 0.35,
            "look_at_center": True,
        },
        {
            "name": "chair_outer",
            "num_cameras": 16,
            "orbit_radius": round(radius, 6),
            "height_offset": 0.05,
            "look_at_center": True,
        },
        {
            "name": "chair_side",
            "num_cameras": 8,
            "orbit_radius": round(radius * 0.84, 6),
            "height_offset": 0.10,
            "look_at_center": False,
            "pitch_deg": -8.0,
            "look_at_offset_scale": 0.45,
        },
    ]


def _scale_camera_rings(camera_cfg, radius: float) -> None:
    rings = list(camera_cfg.get("rings", []) or [])
    if not rings:
        camera_cfg.rings = _default_hybrid_rings(radius)
        camera_cfg.num_cameras = sum(int(ring["num_cameras"]) for ring in camera_cfg.rings)
        return

    base_radii = [
        float(ring.get("orbit_radius"))
        for ring in rings
        if ring.get("orbit_radius") is not None and float(ring.get("orbit_radius")) > 0.0
    ]
    base_outer = max(base_radii) if base_radii else None
    for ring in rings:
        if ring.get("orbit_radius") is not None and base_outer is not None:
            ring.orbit_radius = round(float(ring.orbit_radius) / base_outer * radius, 6)
        elif ring.get("orbit_radius_scale") is None:
            ring.orbit_radius = round(radius, 6)
    camera_cfg.rings = rings
    camera_cfg.num_cameras = sum(max(1, int(ring.get("num_cameras", 1))) for ring in rings)


def _write_roi_camera_config(
    output_dir: Path,
    *,
    base_config_path: Path,
    config_name: str,
    up_axis: str,
    center: torch.Tensor,
    radius: float,
) -> Path | None:
    if not base_config_path.exists():
        log.warning("Base config %s does not exist; skip ROI camera config export", base_config_path)
        return None

    cfg = OmegaConf.load(str(base_config_path))
    if "stage1" not in cfg:
        cfg.stage1 = {}
    if "camera_gen" not in cfg.stage1:
        cfg.stage1.camera_gen = {}

    center_list = [round(float(v), 6) for v in center.detach().cpu()]
    camera_cfg = cfg.stage1.camera_gen
    camera_cfg.up_axis = up_axis
    camera_cfg.trajectory_mode = "hybrid"
    camera_cfg.auto_center = False
    camera_cfg.center = center_list
    camera_cfg.orbit_radius = round(float(radius), 6)
    camera_cfg.radius_jitter = round(float(radius), 6)
    camera_cfg.look_at_center = True
    _scale_camera_rings(camera_cfg, float(radius))

    output_path = output_dir / config_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(output_path), resolve=False)
    return output_path


def _write_roi_outputs(
    output_dir: Path,
    *,
    ply_path: Path,
    prompt: str,
    up_axis: str,
    best_view_idx: int,
    best_candidate: SAM3Candidate,
    best_score: float,
    center: torch.Tensor,
    radius: float,
    scene: GaussianScene,
    frame,
    roi_points: torch.Tensor,
    views: list[RenderedView],
    base_config_path: Path,
    roi_config_name: str,
    topdown_image_width: int,
    topdown_image_height: int,
    topdown_fov_deg: float,
) -> None:
    overlays_dir = output_dir / "roi_masks"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    overlay = _overlay_mask(views[best_view_idx].rgb, best_candidate.mask, (1.0, 0.0, 0.0))
    overlay_path = overlays_dir / f"best_view_{best_view_idx:03d}_roi_overlay.png"
    mask_path = overlays_dir / f"best_view_{best_view_idx:03d}_roi_mask.png"
    save_image(overlay, overlay_path)
    save_image(best_candidate.mask.unsqueeze(-1).repeat(1, 1, 3), mask_path)

    center_list = [float(v) for v in center.detach().cpu()]
    config_path = _write_roi_camera_config(
        output_dir,
        base_config_path=base_config_path,
        config_name=roi_config_name,
        up_axis=up_axis,
        center=center,
        radius=radius,
    )
    result = {
        "ply": str(ply_path),
        "prompt": prompt,
        "up_axis": up_axis,
        "roi_center": center_list,
        "recommended_orbit_radius": radius,
        "best_view_index": best_view_idx,
        "best_candidate_id": int(best_candidate.candidate_id),
        "best_candidate_score": float(best_score),
        "best_candidate_area": float(best_candidate.area),
        "overlay": str(overlay_path),
        "mask": str(mask_path),
        "topdown_render": str(output_dir / "roi_topdown_render.png"),
        "topdown_closeup_render": str(output_dir / "roi_topdown_render_closeup.png"),
        "base_config": str(base_config_path),
        "generated_config": str(config_path) if config_path is not None else None,
        "config_overrides": [
            f"stage1.camera_gen.up_axis={up_axis}",
            "stage1.camera_gen.trajectory_mode=hybrid",
            "stage1.camera_gen.auto_center=false",
            f"stage1.camera_gen.center=[{center_list[0]},{center_list[1]},{center_list[2]}]",
        ],
    }
    with open(output_dir / "roi_center.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    lines = [
        "Recommended camera overrides:",
        f"stage1.camera_gen.up_axis={up_axis}",
        "stage1.camera_gen.trajectory_mode=hybrid",
        "stage1.camera_gen.auto_center=false",
        f"stage1.camera_gen.center=[{center_list[0]:.6f},{center_list[1]:.6f},{center_list[2]:.6f}]",
        f"# suggested orbit radius around ROI: {radius:.3f}",
    ]
    if config_path is not None:
        lines.extend(["", f"Generated config: {config_path}"])
    (output_dir / "recommended_camera_overrides.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _save_topdown_scene_render(
        scene,
        center,
        roi_points,
        up_axis=up_axis,
        frame=frame,
        radius=radius,
        output_path=output_dir / "roi_topdown_render.png",
        image_width=topdown_image_width,
        image_height=topdown_image_height,
        fov_deg=topdown_fov_deg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Localize a target ROI center for camera trajectory planning")
    parser.add_argument("--ply", required=True, help="Input 3DGS PLY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default=None, help="Optional SAM3 prompt, e.g. 'round table with a vase'")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--up-axis", default="auto", help="auto, x, y, z, -x, -y, or -z")
    parser.add_argument("--num-views", type=int, default=16)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=640)
    parser.add_argument("--fov-deg", type=float, default=80.0)
    parser.add_argument("--pitch-deg", type=float, default=12.0)
    parser.add_argument("--radius", type=float, default=0.0, help="Probe orbit radius; <=0 uses scene-adaptive outer radius")
    parser.add_argument("--height-offset", type=float, default=0.2)
    parser.add_argument("--high-offset", type=float, default=1.2)
    parser.add_argument("--sam3-source-dir", default=None)
    parser.add_argument("--sam3-checkpoint", default=None)
    parser.add_argument("--sam3-confidence", type=float, default=0.45)
    parser.add_argument(
        "--base-config",
        default=DEFAULT_ROI_BASE_CONFIG,
        help="Base scene YAML copied when exporting ROI camera config",
    )
    parser.add_argument(
        "--roi-config-name",
        default=DEFAULT_ROI_CONFIG_NAME,
        help="Filename for the generated YAML under --output-dir",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = Path(args.ply)

    scene = GaussianScene.from_ply(ply_path)
    scene.params = scene.params.to(args.device)
    up_axis = auto_up_axis(scene.params.means) if str(args.up_axis).lower() == "auto" else str(args.up_axis)
    frame = compute_scene_frame(scene.params.means, up_axis=up_axis)
    log.info(
        "Loaded %d Gaussians from %s; probe up_axis=%s, scene_center=%s, extent=%.3f",
        scene.num_gaussians,
        ply_path,
        up_axis,
        [round(float(v), 4) for v in frame.center],
        frame.extent_xy,
    )

    probe_cfg = _make_probe_camera_cfg(args, up_axis, frame)
    cameras = [cam.to(args.device) for cam in generate_multi_ring_cameras(frame.center, probe_cfg, frame=frame)]
    save_cameras_json(cameras, output_dir / "probe_cameras.json")
    views, image_paths = _render_probe_views(scene, cameras, output_dir)
    log.info("Saved %d probe views and contact sheet to %s", len(image_paths), output_dir)

    probe_meta = {
        "ply": str(ply_path),
        "up_axis": up_axis,
        "scene_frame_center": [float(v) for v in frame.center],
        "scene_frame_extent_xy": float(frame.extent_xy),
        "probe_radius": float(probe_cfg.orbit_radius),
        "probe_cameras": str(output_dir / "probe_cameras.json"),
        "contact_sheet": str(output_dir / "probe_contact_sheet.png"),
    }
    with open(output_dir / "probe_meta.json", "w", encoding="utf-8") as f:
        json.dump(probe_meta, f, indent=2)

    if not args.prompt:
        log.info("No --prompt provided; use probe_contact_sheet.png to choose a target view or rerun with --prompt.")
        return

    segmenter = SAM3Segmenter(_load_sam3_config(args))
    best = select_roi_candidate(segmenter, views, args.prompt)
    if best is None:
        raise SystemExit(f"SAM3 produced no ROI mask for prompt {args.prompt!r}; inspect probe_contact_sheet.png")
    best_view_idx, best_candidate, best_score = best
    points = mask_to_world_points(views[best_view_idx], cameras[best_view_idx], best_candidate.mask)
    if points.numel() == 0:
        raise SystemExit("Best ROI mask had no valid depth pixels; inspect probe view depth/overlay")
    center = robust_center(points)
    radius = estimate_radius(points, center, up_axis)
    _write_roi_outputs(
        output_dir,
        ply_path=ply_path,
        prompt=args.prompt,
        up_axis=up_axis,
        best_view_idx=best_view_idx,
        best_candidate=best_candidate,
        best_score=best_score,
        center=center,
        radius=radius,
        scene=scene,
        frame=frame,
        roi_points=points,
        views=views,
        base_config_path=Path(args.base_config),
        roi_config_name=str(args.roi_config_name),
        topdown_image_width=int(args.image_width),
        topdown_image_height=int(args.image_height),
        topdown_fov_deg=float(args.fov_deg),
    )
    log.info("ROI center: %s; suggested radius %.3f", [round(float(v), 5) for v in center], radius)
    log.info("Wrote %s", output_dir / "roi_center.json")


if __name__ == "__main__":
    main()
