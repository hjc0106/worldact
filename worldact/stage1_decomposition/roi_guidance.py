from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from worldact.data.gaussians import GaussianScene, RenderedView
from worldact.data.poses import CameraPose
from worldact.rendering.camera_planning import (
    RingCameraSpec,
    SceneFrame,
    compute_scene_frame,
    generate_multi_ring_cameras,
    parse_up_axis,
)
from worldact.stage1_decomposition.sam3_segmenter import SAM3Candidate, SAM3Segmenter
from worldact.utils.io import save_cameras_json
from worldact.utils.visualization import save_image


DEFAULT_ROI_PROBE_NUM_VIEWS = 16
DEFAULT_ROI_PROBE_IMAGE_WIDTH = 640
DEFAULT_ROI_PROBE_IMAGE_HEIGHT = 640
DEFAULT_ROI_PROBE_FOV_DEG = 80.0
DEFAULT_ROI_PROBE_PITCH_DEG = 12.0
DEFAULT_ROI_PROBE_RADIUS = 0.0
DEFAULT_ROI_PROBE_HEIGHT_OFFSET = 0.2
DEFAULT_ROI_PROBE_HIGH_OFFSET = 1.2


@dataclass
class ROIProbeConfig:
    num_views: int = DEFAULT_ROI_PROBE_NUM_VIEWS
    image_width: int = DEFAULT_ROI_PROBE_IMAGE_WIDTH
    image_height: int = DEFAULT_ROI_PROBE_IMAGE_HEIGHT
    fov_deg: float = DEFAULT_ROI_PROBE_FOV_DEG
    pitch_deg: float = DEFAULT_ROI_PROBE_PITCH_DEG
    radius: float = DEFAULT_ROI_PROBE_RADIUS
    height_offset: float = DEFAULT_ROI_PROBE_HEIGHT_OFFSET
    high_offset: float = DEFAULT_ROI_PROBE_HIGH_OFFSET


@dataclass
class ROISelection:
    prompt: str
    up_axis: str
    frame: SceneFrame
    probe_radius: float
    probe_cameras: list[CameraPose]
    probe_views: list[RenderedView]
    best_view_index: int
    best_candidate: SAM3Candidate
    best_score: float
    roi_points: torch.Tensor
    center: torch.Tensor
    radius: float
    probe_artifacts: dict[str, object] | None = None


def auto_up_axis(means: torch.Tensor) -> str:
    means_cpu = means.detach().cpu().float()
    lo = torch.quantile(means_cpu, 0.02, dim=0)
    hi = torch.quantile(means_cpu, 0.98, dim=0)
    extents = hi - lo
    axis = int(torch.argmin(extents).item())
    return ["x", "y", "z"][axis]


def resolve_up_axis(means: torch.Tensor, up_axis: str | int | None) -> str:
    if up_axis is None:
        return auto_up_axis(means)
    return str(up_axis)


def resolve_probe_radius(requested_radius: float, frame: SceneFrame) -> float:
    if float(requested_radius) > 0.0:
        return float(requested_radius)
    return float(frame.recommended_radii[2])


def build_probe_camera_cfg(frame: SceneFrame, up_axis: str, probe_cfg: ROIProbeConfig) -> tuple[SimpleNamespace, float]:
    radius = resolve_probe_radius(probe_cfg.radius, frame)
    camera_cfg = SimpleNamespace(
        image_width=int(probe_cfg.image_width),
        image_height=int(probe_cfg.image_height),
        fov_deg=float(probe_cfg.fov_deg),
        pitch_deg=float(probe_cfg.pitch_deg),
        radius_jitter=radius,
        orbit_radius=radius,
        up_axis=up_axis,
        rings=[
            RingCameraSpec(
                name="probe_level",
                num_cameras=int(probe_cfg.num_views),
                orbit_radius=radius,
                height_offset=float(probe_cfg.height_offset),
                look_at_center=True,
            ),
            RingCameraSpec(
                name="probe_high",
                num_cameras=max(4, int(probe_cfg.num_views) // 2),
                orbit_radius=radius * 0.85,
                height_offset=float(probe_cfg.height_offset) + float(probe_cfg.high_offset),
                look_at_center=True,
            ),
        ],
    )
    return camera_cfg, radius


def render_probe_views(scene: GaussianScene, cameras: list[CameraPose]) -> list[RenderedView]:
    views: list[RenderedView] = []
    for camera in cameras:
        rendered = scene.render(camera)
        views.append(
            RenderedView(
                rgb=rendered.rgb.detach().cpu(),
                depth=rendered.depth.detach().cpu(),
                alpha=rendered.alpha.detach().cpu(),
            )
        )
    return views


def _save_probe_contact_sheet(image_paths: list[Path], output_path: Path, *, columns: int = 4) -> None:
    if not image_paths:
        return
    from PIL import Image, ImageDraw

    thumbs = [Image.open(path).convert("RGB") for path in image_paths]
    thumb_w = min(256, thumbs[0].width)
    thumb_h = int(thumbs[0].height * (thumb_w / thumbs[0].width))
    label_h = 24
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (path, image) in enumerate(zip(image_paths, thumbs)):
        row, col = divmod(idx, columns)
        image = image.resize((thumb_w, thumb_h))
        x, y = col * thumb_w, row * (thumb_h + label_h)
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 6, y + 5), path.stem, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def save_probe_view_artifacts(
    views: list[RenderedView],
    cameras: list[CameraPose],
    output_dir: Path | str,
    *,
    subdir: str = "probe_views",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    probe_dir = output_dir / subdir
    probe_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    image_paths: list[Path] = []
    for idx, view in enumerate(views):
        image_path = probe_dir / f"probe_{idx:03d}.png"
        save_image(view.rgb, image_path)
        image_paths.append(image_path)
        records.append(
            {
                "index": idx,
                "filename": str(image_path),
                "width": int(view.width),
                "height": int(view.height),
            }
        )

    manifest_path = probe_dir / "probe_views.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    cameras_path = output_dir / "probe_cameras.json"
    save_cameras_json(cameras, cameras_path)

    contact_sheet_path = output_dir / "probe_contact_sheet.png"
    _save_probe_contact_sheet(image_paths, contact_sheet_path, columns=4)

    return {
        "probe_views_dir": str(probe_dir),
        "probe_views_manifest": str(manifest_path),
        "probe_cameras": str(cameras_path),
        "probe_contact_sheet": str(contact_sheet_path),
        "num_probe_views": len(views),
    }


def overlay_roi_mask(rgb: torch.Tensor, mask: torch.Tensor, color: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> torch.Tensor:
    overlay = rgb.detach().cpu().float().clone()
    mask_t = mask.detach().cpu().float().clamp(0.0, 1.0).unsqueeze(-1)
    color_t = torch.tensor(color, dtype=overlay.dtype).view(1, 1, 3)
    return overlay * (1.0 - 0.45 * mask_t) + color_t * 0.45 * mask_t


def save_roi_selection_artifacts(
    selection: ROISelection,
    output_dir: Path | str,
    *,
    subdir: str = "roi_masks",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    masks_dir = output_dir / subdir
    masks_dir.mkdir(parents=True, exist_ok=True)

    view = selection.probe_views[selection.best_view_index]
    candidate = selection.best_candidate
    stem = f"best_view_{selection.best_view_index:03d}"
    rgb_path = masks_dir / f"{stem}_rgb.png"
    mask_path = masks_dir / f"{stem}_roi_mask.png"
    overlay_path = masks_dir / f"{stem}_roi_overlay.png"

    save_image(view.rgb, rgb_path)
    save_image(candidate.mask.unsqueeze(-1).repeat(1, 1, 3), mask_path)
    save_image(overlay_roi_mask(view.rgb, candidate.mask), overlay_path)

    summary = {
        "best_view_rgb": str(rgb_path),
        "best_view_mask": str(mask_path),
        "best_view_overlay": str(overlay_path),
        "best_view_index": int(selection.best_view_index),
        "best_candidate_id": int(candidate.candidate_id),
        "best_candidate_score": float(selection.best_score),
        "best_candidate_area": float(candidate.area),
        "roi_center": [float(v) for v in selection.center.detach().cpu().tolist()],
        "roi_radius": float(selection.radius),
    }
    return summary


def candidate_center_score(candidate: SAM3Candidate, image_size: tuple[int, int]) -> float:
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
    return float(candidate.quality) * (0.2 + 0.5 * center_bias + 0.3 * size_bias)


def select_roi_candidate(
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
            score = candidate_center_score(candidate, candidate_set.image_size)
            if best is None or score > best[2]:
                best = (view_idx, candidate, score)
    return best


def mask_to_world_points(
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


def robust_center(points: torch.Tensor) -> torch.Tensor:
    center = points.median(dim=0).values
    if points.shape[0] < 32:
        return center
    distances = torch.norm(points - center, dim=1)
    keep = distances <= torch.quantile(distances, 0.85)
    if bool(keep.any()):
        center = points[keep].median(dim=0).values
    return center


def estimate_radius(points: torch.Tensor, center: torch.Tensor, up_axis: str) -> float:
    up_idx, _ = parse_up_axis(up_axis)
    axes = [0, 1, 2]
    axes.remove(up_idx)
    horizontal = points[:, axes] - center[axes]
    distances = torch.norm(horizontal, dim=1)
    if distances.numel() == 0:
        return 1.0
    radius = float(torch.quantile(distances, 0.90).item())
    return float(max(0.4, min(3.0, radius * 2.5)))


def localize_prompt_roi(
    scene: GaussianScene,
    segmenter: SAM3Segmenter,
    prompt: str,
    *,
    up_axis: str | int | None = "auto",
    probe_cfg: ROIProbeConfig | None = None,
    probe_output_dir: Path | str | None = None,
) -> ROISelection:
    probe_cfg = probe_cfg or ROIProbeConfig()
    up_axis_resolved = resolve_up_axis(scene.params.means, up_axis)
    frame = compute_scene_frame(scene.params.means, up_axis=up_axis_resolved)
    camera_cfg, probe_radius = build_probe_camera_cfg(frame, up_axis_resolved, probe_cfg)
    probe_cameras = [
        cam.to(scene.device)
        for cam in generate_multi_ring_cameras(frame.center, camera_cfg, frame=frame)
    ]
    probe_views = render_probe_views(scene, probe_cameras)
    probe_artifacts = (
        save_probe_view_artifacts(probe_views, probe_cameras, probe_output_dir)
        if probe_output_dir is not None
        else None
    )
    best = select_roi_candidate(segmenter, probe_views, prompt)
    if best is None:
        raise RuntimeError(f"SAM3 produced no ROI mask for prompt {prompt!r}")

    best_view_idx, best_candidate, best_score = best
    points = mask_to_world_points(probe_views[best_view_idx], probe_cameras[best_view_idx], best_candidate.mask)
    if points.numel() == 0:
        raise RuntimeError("Best ROI mask had no valid depth pixels")

    center = robust_center(points)
    radius = estimate_radius(points, center, up_axis_resolved)
    return ROISelection(
        prompt=prompt,
        up_axis=up_axis_resolved,
        frame=frame,
        probe_radius=probe_radius,
        probe_cameras=probe_cameras,
        probe_views=probe_views,
        best_view_index=best_view_idx,
        best_candidate=best_candidate,
        best_score=best_score,
        roi_points=points,
        center=center,
        radius=radius,
        probe_artifacts=probe_artifacts,
    )
