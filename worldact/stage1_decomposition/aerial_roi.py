from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from worldact.data.gaussians import GaussianScene, RenderedView
from worldact.data.poses import CameraPose
from worldact.rendering.camera_planning import compute_scene_frame, parse_up_axis
from worldact.stage1_decomposition.sam3_segmenter import SAM3Candidate, SAM3Segmenter
from worldact.utils.io import save_cameras_json
from worldact.utils.visualization import save_image


@dataclass
class AerialCameraConfig:
    up_axis: str
    image_width: int = 640
    image_height: int = 640
    fov_deg: float = 70.0
    max_views: int = 96
    grid_cols: int | None = None
    grid_rows: int | None = None
    overlap: float = 0.35
    altitude: float | None = None
    clearance: float = 0.8
    bounds_quantile: tuple[float, float] = (0.01, 0.99)
    bounds_margin: float = 0.04


@dataclass
class AerialViewRecord:
    index: int
    row: int
    col: int
    target: list[float]
    camera_center: list[float]
    distance: float
    rgb: str | None = None


@dataclass
class AerialRenderSet:
    cameras: list[CameraPose]
    views: list[RenderedView]
    records: list[AerialViewRecord]
    summary: dict[str, object]


@dataclass
class AerialMaskObservation:
    view_index: int
    prompt: str
    candidate_id: int
    mask: torch.Tensor
    weight: float
    score: float
    quality: float
    area: float
    area_ratio: float


@dataclass
class OccupancyGrid:
    scores: torch.Tensor
    vote_counts: torch.Tensor
    min0: float
    max0: float
    min1: float
    max1: float
    horizontal_axes: tuple[int, int]
    up_axis: str

    @property
    def normalized(self) -> torch.Tensor:
        max_score = float(self.scores.max().item()) if self.scores.numel() else 0.0
        if max_score <= 1e-8:
            return torch.zeros_like(self.scores)
        return self.scores / max_score


@dataclass
class RegionProposal:
    region_id: str
    cell_bbox: tuple[int, int, int, int]
    bounds: dict[str, list[float]]
    score: float
    mean_score: float
    area_cells: int
    num_gaussians: int = 0
    source_prompts: list[str] = field(default_factory=list)
    ply: str | None = None


@dataclass
class RegionCameraEstimate:
    center: torch.Tensor
    bbox_center: torch.Tensor
    robust_center: torch.Tensor
    orbit_radius: float
    camera_purpose: str
    coverage_mode: str | None
    coverage_radius: float | None
    coverage_standoff: float | None
    aspect_ratio: float
    horizontal_span: tuple[float, float]
    horizontal_diagonal: float
    point_radius_q90: float
    center_source: str
    center_height_above_floor: float
    region_floor_height: float
    floor_quantile: float
    height_mode: str
    height_scale: float | None
    height_quantile: float | None
    height_min: float | None
    height_max: float | None
    region_height_quantile: float | None
    region_height_extent: float | None
    vertical_axis: str


def axis_name(axis: int) -> str:
    return ["x", "y", "z"][int(axis)]


def horizontal_axes(up_axis: str | int | None) -> tuple[int, int, int, float]:
    up_idx, up_sign = parse_up_axis(up_axis)
    axes = [0, 1, 2]
    axes.remove(up_idx)
    return axes[0], axes[1], up_idx, up_sign


def _axis_vector(axis: int, sign: float = 1.0) -> torch.Tensor:
    vec = torch.zeros(3, dtype=torch.float32)
    vec[int(axis)] = float(sign)
    return vec


def _make_intrinsic(image_width: int, image_height: int, fov_deg: float) -> torch.Tensor:
    fov = math.radians(float(fov_deg))
    fy = image_height / (2.0 * math.tan(fov / 2.0))
    fx = fy
    return torch.tensor(
        [[fx, 0.0, image_width / 2.0], [0.0, fy, image_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def scene_bounds(
    means: torch.Tensor,
    up_axis: str,
    *,
    quantile: tuple[float, float],
    margin_ratio: float,
) -> dict[str, object]:
    h0, h1, up_idx, up_sign = horizontal_axes(up_axis)
    means_cpu = means.detach().cpu().float()
    q_lo, q_hi = quantile
    lo = torch.quantile(means_cpu, float(q_lo), dim=0)
    hi = torch.quantile(means_cpu, float(q_hi), dim=0)
    span0 = float(max(1e-4, (hi[h0] - lo[h0]).item()))
    span1 = float(max(1e-4, (hi[h1] - lo[h1]).item()))
    pad = float(margin_ratio) * max(span0, span1)
    up_values = means_cpu[:, up_idx] * float(up_sign)
    return {
        "h0": h0,
        "h1": h1,
        "up_idx": up_idx,
        "up_sign": up_sign,
        "min0": float(lo[h0].item() - pad),
        "max0": float(hi[h0].item() + pad),
        "min1": float(lo[h1].item() - pad),
        "max1": float(hi[h1].item() + pad),
        "floor": float(torch.quantile(up_values, float(q_lo)).item()),
        "ceiling": float(torch.quantile(up_values, float(q_hi)).item()),
        "center": (lo + hi) * 0.5,
    }


def _auto_grid(span0: float, span1: float, cfg: AerialCameraConfig) -> tuple[int, int]:
    max_views = max(1, int(cfg.max_views))
    if cfg.grid_cols is not None or cfg.grid_rows is not None:
        cols = int(cfg.grid_cols) if cfg.grid_cols is not None else max(1, math.ceil(max_views / max(1, int(cfg.grid_rows))))
        rows = int(cfg.grid_rows) if cfg.grid_rows is not None else max(1, math.ceil(max_views / max(1, int(cols))))
        return max(1, cols), max(1, rows)
    aspect = max(1e-4, float(span0) / max(1e-4, float(span1)))
    cols = max(1, int(round(math.sqrt(max_views * aspect))))
    rows = max(1, max_views // cols)
    while cols * rows > max_views and rows > 1:
        rows -= 1
    while cols * rows > max_views and cols > 1:
        cols -= 1
    return cols, rows


def _linspace_centers(min_value: float, max_value: float, count: int) -> list[float]:
    if count <= 1:
        return [0.5 * (float(min_value) + float(max_value))]
    return torch.linspace(float(min_value), float(max_value), int(count)).tolist()


def _camera_pose(
    target: torch.Tensor,
    *,
    distance: float,
    up_axis: str,
    image_width: int,
    image_height: int,
    fov_deg: float,
) -> CameraPose:
    h0, h1, up_idx, up_sign = horizontal_axes(up_axis)
    target_cpu = target.detach().cpu().float()
    cam_pos = target_cpu.clone()
    cam_pos[up_idx] += float(up_sign) * float(distance)
    forward = F.normalize(target_cpu - cam_pos, dim=0)
    ref = _axis_vector(h0)
    right = torch.cross(ref, forward, dim=0)
    if float(torch.linalg.norm(right).item()) < 1e-6:
        ref = _axis_vector(h1)
        right = torch.cross(ref, forward, dim=0)
    right = F.normalize(right, dim=0)
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


def _estimate_distance(
    span0: float,
    span1: float,
    cols: int,
    rows: int,
    cfg: AerialCameraConfig,
    *,
    min_scene_clearance: float,
) -> tuple[float, dict[str, float | str]]:
    if cfg.altitude is not None and float(cfg.altitude) > 0.0:
        return float(cfg.altitude), {"source": "explicit", "footprint0": 0.0, "footprint1": 0.0}
    step0 = float(span0) / max(1, int(cols) - 1) if cols > 1 else float(span0)
    step1 = float(span1) / max(1, int(rows) - 1) if rows > 1 else float(span1)
    overlap = min(0.9, max(0.0, float(cfg.overlap)))
    footprint0 = step0 / max(0.05, 1.0 - overlap)
    footprint1 = step1 / max(0.05, 1.0 - overlap)
    aspect = float(cfg.image_width) / max(1.0, float(cfg.image_height))
    fov = math.radians(float(cfg.fov_deg))
    distance_for_width = footprint0 / max(1e-4, 2.0 * aspect * math.tan(fov / 2.0))
    distance_for_height = footprint1 / max(1e-4, 2.0 * math.tan(fov / 2.0))
    distance = max(distance_for_width, distance_for_height, float(min_scene_clearance), 0.5)
    return distance, {"source": "auto", "footprint0": float(footprint0), "footprint1": float(footprint1)}


def make_aerial_cameras(scene: GaussianScene, cfg: AerialCameraConfig) -> tuple[list[CameraPose], list[AerialViewRecord], dict[str, object]]:
    bounds = scene_bounds(scene.params.means, cfg.up_axis, quantile=cfg.bounds_quantile, margin_ratio=cfg.bounds_margin)
    h0 = int(bounds["h0"])
    h1 = int(bounds["h1"])
    up_idx = int(bounds["up_idx"])
    up_sign = float(bounds["up_sign"])
    min0, max0 = float(bounds["min0"]), float(bounds["max0"])
    min1, max1 = float(bounds["min1"]), float(bounds["max1"])
    span0 = max(1e-4, max0 - min0)
    span1 = max(1e-4, max1 - min1)
    cols, rows = _auto_grid(span0, span1, cfg)
    frame = compute_scene_frame(scene.params.means, up_axis=cfg.up_axis)
    center = bounds["center"].detach().cpu().float()
    target_up_value = float(center[up_idx] * up_sign)
    vertical_clearance = max(0.0, float(bounds["ceiling"]) - target_up_value) + float(cfg.clearance)
    distance, distance_meta = _estimate_distance(span0, span1, cols, rows, cfg, min_scene_clearance=vertical_clearance)

    xs = _linspace_centers(min0, max0, cols)
    ys = _linspace_centers(min1, max1, rows)
    cameras: list[CameraPose] = []
    records: list[AerialViewRecord] = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            target = center.clone()
            target[h0] = float(x)
            target[h1] = float(y)
            camera = _camera_pose(
                target,
                distance=distance,
                up_axis=cfg.up_axis,
                image_width=int(cfg.image_width),
                image_height=int(cfg.image_height),
                fov_deg=float(cfg.fov_deg),
            )
            cam_center = camera.c2w[:3, 3].detach().cpu()
            cameras.append(camera)
            records.append(
                AerialViewRecord(
                    index=len(records),
                    row=row,
                    col=col,
                    target=[float(v) for v in target.tolist()],
                    camera_center=[float(v) for v in cam_center.tolist()],
                    distance=float(distance),
                )
            )

    summary = {
        "up_axis": cfg.up_axis,
        "horizontal_axes": [axis_name(h0), axis_name(h1)],
        "scene_center": [float(v) for v in frame.center.detach().cpu().tolist()],
        "bounds_quantile": [float(cfg.bounds_quantile[0]), float(cfg.bounds_quantile[1])],
        "bounds_margin": float(cfg.bounds_margin),
        "bounds": {
            axis_name(h0): [min0, max0],
            axis_name(h1): [min1, max1],
            "floor": float(bounds["floor"]),
            "ceiling": float(bounds["ceiling"]),
        },
        "grid_cols": int(cols),
        "grid_rows": int(rows),
        "num_cameras": len(cameras),
        "image_width": int(cfg.image_width),
        "image_height": int(cfg.image_height),
        "fov_deg": float(cfg.fov_deg),
        "overlap": float(cfg.overlap),
        "aerial_distance": float(distance),
        "distance_meta": distance_meta,
    }
    return cameras, records, summary


def render_aerial_views(scene: GaussianScene, cameras: list[CameraPose]) -> list[RenderedView]:
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


def save_contact_sheet(image_paths: list[Path], output_path: Path, *, columns: int = 6) -> None:
    if not image_paths:
        return
    from PIL import Image, ImageDraw

    thumbs = [Image.open(path).convert("RGB") for path in image_paths]
    thumb_w = min(220, thumbs[0].width)
    thumb_h = int(thumbs[0].height * thumb_w / max(1, thumbs[0].width))
    label_h = 22
    rows = math.ceil(len(thumbs) / max(1, int(columns)))
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (path, image) in enumerate(zip(image_paths, thumbs)):
        row, col = divmod(idx, columns)
        image = image.resize((thumb_w, thumb_h))
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 6, y + 4), path.stem, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def save_aerial_render_artifacts(
    render_set: AerialRenderSet,
    output_dir: Path | str,
    *,
    contact_sheet_cols: int = 6,
) -> list[Path]:
    output_dir = Path(output_dir)
    views_dir = output_dir / "aerial_views"
    views_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    for view, record in zip(render_set.views, render_set.records):
        image_path = views_dir / f"aerial_{record.index:03d}_r{record.row:02d}_c{record.col:02d}.png"
        save_image(view.rgb, image_path)
        record.rgb = str(image_path)
        image_paths.append(image_path)
    save_cameras_json(render_set.cameras, output_dir / "aerial_cameras.json")
    with open(output_dir / "aerial_views.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": render_set.summary,
                "views": [record.__dict__ for record in render_set.records],
            },
            f,
            indent=2,
        )
    save_contact_sheet(image_paths, output_dir / "aerial_contact_sheet.png", columns=contact_sheet_cols)
    save_plan_view(render_set.summary, render_set.records, output_dir / "aerial_plan.png")
    return image_paths


def overlay_mask(rgb: torch.Tensor, mask: torch.Tensor, color: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> torch.Tensor:
    overlay = rgb.detach().cpu().float().clone()
    mask_t = mask.detach().cpu().float().clamp(0.0, 1.0).unsqueeze(-1)
    color_t = torch.tensor(color, dtype=overlay.dtype).view(1, 1, 3)
    return overlay * (1.0 - 0.45 * mask_t) + color_t * 0.45 * mask_t


def _candidate_weight(candidate: SAM3Candidate, image_size: tuple[int, int]) -> tuple[float, float]:
    height, width = image_size
    area_ratio = float(candidate.area) / max(1.0, float(height * width))
    mask = candidate.mask > 0.5
    if not bool(mask.any()):
        return 0.0, area_ratio
    ys, xs = mask.nonzero(as_tuple=True)
    touches_edge = bool(xs.min() <= 0 or ys.min() <= 0 or xs.max() >= width - 1 or ys.max() >= height - 1)
    edge_weight = 0.7 if touches_edge else 1.0
    size_weight = min(1.0, max(0.25, math.sqrt(max(area_ratio, 1e-8)) * 4.0))
    return float(candidate.quality) * edge_weight * size_weight, area_ratio


def segment_aerial_masks(
    segmenter: SAM3Segmenter,
    render_set: AerialRenderSet,
    prompts: list[str],
    *,
    candidate_mode: str = "all",
    max_candidates_per_prompt: int = 5,
    min_mask_area_ratio: float = 1e-4,
    max_mask_area_ratio: float = 0.75,
    output_dir: Path | str | None = None,
) -> tuple[list[AerialMaskObservation], list[dict[str, object]]]:
    observations: list[AerialMaskObservation] = []
    records: list[dict[str, object]] = []
    masks_dir = Path(output_dir) / "aerial_masks" if output_dir is not None else None
    if masks_dir is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)

    for view_idx, view in enumerate(render_set.views):
        for prompt_idx, prompt in enumerate(prompts):
            candidate_set = segmenter.segment_candidates(view.rgb, prompt)
            if candidate_set is None or not candidate_set.candidates:
                records.append({"view_index": view_idx, "prompt": prompt, "num_candidates": 0, "kept": 0})
                continue
            candidates = sorted(candidate_set.candidates, key=lambda cand: cand.quality, reverse=True)
            if candidate_mode == "best":
                candidates = candidates[:1]
            elif candidate_mode == "all":
                candidates = candidates[: max(1, int(max_candidates_per_prompt))]
            elif candidate_mode == "union":
                candidates = candidates[: max(1, int(max_candidates_per_prompt))]
            else:
                raise ValueError(f"Unsupported candidate_mode {candidate_mode!r}; expected best/all/union")

            kept: list[AerialMaskObservation] = []
            if candidate_mode == "union":
                valid_candidates = []
                for candidate in candidates:
                    weight, area_ratio = _candidate_weight(candidate, candidate_set.image_size)
                    if weight <= 0.0 or area_ratio < min_mask_area_ratio or area_ratio > max_mask_area_ratio:
                        continue
                    valid_candidates.append((candidate, weight, area_ratio))
                if valid_candidates:
                    union_mask = torch.zeros_like(valid_candidates[0][0].mask)
                    for candidate, _, _ in valid_candidates:
                        union_mask = torch.maximum(union_mask, candidate.mask)
                    weight = max(item[1] for item in valid_candidates)
                    area = float((union_mask > 0.5).float().sum().item())
                    obs = AerialMaskObservation(
                        view_index=view_idx,
                        prompt=prompt,
                        candidate_id=-1,
                        mask=union_mask,
                        weight=float(weight),
                        score=max(float(item[0].score) for item in valid_candidates),
                        quality=max(float(item[0].quality) for item in valid_candidates),
                        area=area,
                        area_ratio=area / max(1.0, float(candidate_set.image_size[0] * candidate_set.image_size[1])),
                    )
                    kept.append(obs)
            else:
                for candidate in candidates:
                    weight, area_ratio = _candidate_weight(candidate, candidate_set.image_size)
                    if weight <= 0.0 or area_ratio < min_mask_area_ratio or area_ratio > max_mask_area_ratio:
                        continue
                    kept.append(
                        AerialMaskObservation(
                            view_index=view_idx,
                            prompt=prompt,
                            candidate_id=int(candidate.candidate_id),
                            mask=candidate.mask,
                            weight=float(weight),
                            score=float(candidate.score),
                            quality=float(candidate.quality),
                            area=float(candidate.area),
                            area_ratio=float(area_ratio),
                        )
                    )
            observations.extend(kept)
            mask_path = None
            overlay_path = None
            if masks_dir is not None and kept:
                union_mask = torch.zeros_like(kept[0].mask)
                for obs in kept:
                    union_mask = torch.maximum(union_mask, obs.mask)
                stem = f"aerial_{view_idx:03d}_prompt_{prompt_idx:02d}"
                mask_path = masks_dir / f"{stem}_mask.png"
                overlay_path = masks_dir / f"{stem}_overlay.png"
                save_image(union_mask.unsqueeze(-1).repeat(1, 1, 3), mask_path)
                save_image(overlay_mask(view.rgb, union_mask), overlay_path)
            records.append(
                {
                    "view_index": view_idx,
                    "prompt": prompt,
                    "num_candidates": len(candidate_set.candidates),
                    "kept": len(kept),
                    "mask": str(mask_path) if mask_path is not None else None,
                    "overlay": str(overlay_path) if overlay_path is not None else None,
                    "kept_candidates": [
                        {
                            "candidate_id": obs.candidate_id,
                            "weight": obs.weight,
                            "score": obs.score,
                            "quality": obs.quality,
                            "area": obs.area,
                            "area_ratio": obs.area_ratio,
                        }
                        for obs in kept
                    ],
                }
            )
    return observations, records


def mask_to_world_points(
    view: RenderedView,
    camera: CameraPose,
    mask: torch.Tensor,
    *,
    alpha_threshold: float = 0.02,
    max_points: int = 50000,
) -> torch.Tensor:
    mask_bool = (mask > 0.5).to(view.depth.device)
    if alpha_threshold > 0.0:
        mask_bool = mask_bool & (view.alpha.to(view.depth.device) >= float(alpha_threshold))
    ys, xs = mask_bool.nonzero(as_tuple=True)
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


def _occupancy_shape(min0: float, max0: float, min1: float, max1: float, grid_resolution: int) -> tuple[int, int]:
    span0 = max(1e-4, float(max0) - float(min0))
    span1 = max(1e-4, float(max1) - float(min1))
    max_res = max(32, int(grid_resolution))
    if span0 >= span1:
        width = max_res
        height = max(32, int(round(max_res * span1 / span0)))
    else:
        height = max_res
        width = max(32, int(round(max_res * span0 / span1)))
    return height, width


def build_occupancy_grid(
    render_set: AerialRenderSet,
    observations: list[AerialMaskObservation],
    *,
    up_axis: str,
    bounds: dict[str, object],
    grid_resolution: int,
    alpha_threshold: float,
    max_points_per_mask: int,
) -> OccupancyGrid:
    h0, h1, _, _ = horizontal_axes(up_axis)
    min0, max0 = float(bounds["min0"]), float(bounds["max0"])
    min1, max1 = float(bounds["min1"]), float(bounds["max1"])
    height, width = _occupancy_shape(min0, max0, min1, max1, grid_resolution)
    scores = torch.zeros(height, width, dtype=torch.float32)
    vote_counts = torch.zeros(height, width, dtype=torch.float32)
    span0 = max(1e-4, max0 - min0)
    span1 = max(1e-4, max1 - min1)

    for obs in observations:
        view = render_set.views[obs.view_index]
        camera = render_set.cameras[obs.view_index]
        points = mask_to_world_points(
            view,
            camera,
            obs.mask,
            alpha_threshold=float(alpha_threshold),
            max_points=int(max_points_per_mask),
        )
        if points.numel() == 0:
            continue
        points = points.detach().cpu().float()
        xs = points[:, h0]
        ys = points[:, h1]
        keep = (xs >= min0) & (xs <= max0) & (ys >= min1) & (ys <= max1)
        if not bool(keep.any()):
            continue
        xs = xs[keep]
        ys = ys[keep]
        cols = torch.clamp(((xs - min0) / span0 * (width - 1)).long(), 0, width - 1)
        rows = torch.clamp(((ys - min1) / span1 * (height - 1)).long(), 0, height - 1)
        flat = torch.unique(rows * width + cols)
        flat_rows = torch.div(flat, width, rounding_mode="floor")
        flat_cols = flat % width
        scores[flat_rows, flat_cols] += float(obs.weight)
        vote_counts[flat_rows, flat_cols] += 1.0
    return OccupancyGrid(
        scores=scores,
        vote_counts=vote_counts,
        min0=min0,
        max0=max0,
        min1=min1,
        max1=max1,
        horizontal_axes=(h0, h1),
        up_axis=up_axis,
    )


def _component_cells(mask: torch.Tensor) -> list[list[tuple[int, int]]]:
    import numpy as np

    arr = mask.detach().cpu().numpy().astype(bool)
    height, width = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for row in range(height):
        for col in range(width):
            if not arr[row, col] or visited[row, col]:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cur_row, cur_col = queue.popleft()
                cells.append((cur_row, cur_col))
                for dr, dc in neighbors:
                    nr, nc = cur_row + dr, cur_col + dc
                    if nr < 0 or nr >= height or nc < 0 or nc >= width:
                        continue
                    if visited[nr, nc] or not arr[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))
            components.append(cells)
    return components


def proposals_from_occupancy(
    occupancy: OccupancyGrid,
    *,
    threshold: float,
    min_component_area: int,
    bbox_margin: float,
    keep_top_k: int,
    prompts: list[str],
) -> list[RegionProposal]:
    normalized = occupancy.normalized
    active = normalized >= float(threshold)
    components = _component_cells(active)
    h0, h1 = occupancy.horizontal_axes
    height, width = normalized.shape
    cell_w0 = (occupancy.max0 - occupancy.min0) / max(1, width)
    cell_w1 = (occupancy.max1 - occupancy.min1) / max(1, height)
    proposals: list[RegionProposal] = []
    for cells in components:
        if len(cells) < int(min_component_area):
            continue
        rows = torch.tensor([cell[0] for cell in cells], dtype=torch.long)
        cols = torch.tensor([cell[1] for cell in cells], dtype=torch.long)
        row0, row1 = int(rows.min().item()), int(rows.max().item())
        col0, col1 = int(cols.min().item()), int(cols.max().item())
        values = occupancy.scores[rows, cols]
        score = float(values.sum().item())
        mean_score = float(normalized[rows, cols].mean().item())
        min0 = occupancy.min0 + col0 * cell_w0 - float(bbox_margin)
        max0 = occupancy.min0 + (col1 + 1) * cell_w0 + float(bbox_margin)
        min1 = occupancy.min1 + row0 * cell_w1 - float(bbox_margin)
        max1 = occupancy.min1 + (row1 + 1) * cell_w1 + float(bbox_margin)
        proposals.append(
            RegionProposal(
                region_id=f"region_{len(proposals):03d}",
                cell_bbox=(row0, row1, col0, col1),
                bounds={
                    axis_name(h0): [float(min0), float(max0)],
                    axis_name(h1): [float(min1), float(max1)],
                },
                score=score,
                mean_score=mean_score,
                area_cells=len(cells),
                source_prompts=list(prompts),
            )
        )
    proposals.sort(key=lambda item: (item.score, item.area_cells), reverse=True)
    keep = max(1, int(keep_top_k)) if keep_top_k > 0 else len(proposals)
    proposals = proposals[:keep]
    for idx, proposal in enumerate(proposals):
        proposal.region_id = f"region_{idx:03d}"
    return proposals


def proposals_from_direct_bbox(
    render_set: AerialRenderSet,
    observations: list[AerialMaskObservation],
    *,
    up_axis: str,
    bounds: dict[str, object],
    alpha_threshold: float,
    max_points_per_mask: int,
    bbox_margin: float,
    prompts: list[str],
) -> list[RegionProposal]:
    h0, h1, _, _ = horizontal_axes(up_axis)
    min0, max0 = float(bounds["min0"]), float(bounds["max0"])
    min1, max1 = float(bounds["min1"]), float(bounds["max1"])
    all_points: list[torch.Tensor] = []
    total_score = 0.0
    for obs in observations:
        view = render_set.views[obs.view_index]
        camera = render_set.cameras[obs.view_index]
        points = mask_to_world_points(
            view,
            camera,
            obs.mask,
            alpha_threshold=float(alpha_threshold),
            max_points=int(max_points_per_mask),
        )
        if points.numel() == 0:
            continue
        points = points.detach().cpu().float()
        xs = points[:, h0]
        ys = points[:, h1]
        keep = (xs >= min0) & (xs <= max0) & (ys >= min1) & (ys <= max1)
        if not bool(keep.any()):
            continue
        all_points.append(points[keep])
        total_score += float(obs.weight)
    if not all_points:
        return []
    points_cat = torch.cat(all_points, dim=0)
    axis0 = axis_name(h0)
    axis1 = axis_name(h1)
    min_axis0 = float(points_cat[:, h0].min().item()) - float(bbox_margin)
    max_axis0 = float(points_cat[:, h0].max().item()) + float(bbox_margin)
    min_axis1 = float(points_cat[:, h1].min().item()) - float(bbox_margin)
    max_axis1 = float(points_cat[:, h1].max().item()) + float(bbox_margin)
    area = max(1e-8, (max_axis0 - min_axis0) * (max_axis1 - min_axis1))
    return [
        RegionProposal(
            region_id="region_000",
            cell_bbox=(0, 0, 0, 0),
            bounds={
                axis0: [min_axis0, max_axis0],
                axis1: [min_axis1, max_axis1],
            },
            score=float(total_score),
            mean_score=float(total_score / max(1, len(observations))),
            area_cells=max(1, int(round(area))),
            source_prompts=list(prompts),
        )
    ]


def _proposal_bbox_center(proposal: RegionProposal, points: torch.Tensor, up_axis: str) -> torch.Tensor:
    h0, h1, up_idx, _ = horizontal_axes(up_axis)
    axis0 = axis_name(h0)
    axis1 = axis_name(h1)
    min0, max0 = proposal.bounds[axis0]
    min1, max1 = proposal.bounds[axis1]
    center = torch.zeros(3, dtype=torch.float32)
    center[h0] = 0.5 * (float(min0) + float(max0))
    center[h1] = 0.5 * (float(min1) + float(max1))
    if points.numel() > 0:
        center[up_idx] = points.detach().cpu().float()[:, up_idx].median()
    return center


def _robust_point_center(points: torch.Tensor) -> torch.Tensor:
    points_cpu = points.detach().cpu().float()
    center = points_cpu.median(dim=0).values
    if points_cpu.shape[0] < 64:
        return center
    distances = torch.norm(points_cpu - center, dim=1)
    keep = distances <= torch.quantile(distances, 0.85)
    if bool(keep.any()):
        center = points_cpu[keep].median(dim=0).values
    return center


def _clamp_center_to_proposal(center: torch.Tensor, proposal: RegionProposal, up_axis: str) -> torch.Tensor:
    h0, h1, _, _ = horizontal_axes(up_axis)
    axis0 = axis_name(h0)
    axis1 = axis_name(h1)
    min0, max0 = proposal.bounds[axis0]
    min1, max1 = proposal.bounds[axis1]
    clamped = center.detach().cpu().float().clone()
    clamped[h0] = float(max(float(min0), min(float(max0), float(clamped[h0]))))
    clamped[h1] = float(max(float(min1), min(float(max1), float(clamped[h1]))))
    return clamped


def _region_floor_height(points: torch.Tensor, up_axis: str, floor_quantile: float) -> float:
    _, _, up_idx, up_sign = horizontal_axes(up_axis)
    points_cpu = points.detach().cpu().float()
    q = min(1.0, max(0.0, float(floor_quantile)))
    signed_heights = points_cpu[:, up_idx] * float(up_sign)
    return float(torch.quantile(signed_heights, q).item())


def _resolve_region_camera_height(
    points: torch.Tensor,
    up_axis: str,
    *,
    mode: str,
    floor_quantile: float,
    fixed_height: float,
    height_scale: float,
    height_min: float,
    height_max: float,
    height_quantile: float,
) -> tuple[float, dict[str, float | str | None]]:
    _, _, up_idx, up_sign = horizontal_axes(up_axis)
    points_cpu = points.detach().cpu().float()
    signed_heights = points_cpu[:, up_idx] * float(up_sign)
    floor_q = min(1.0, max(0.0, float(floor_quantile)))
    top_q = min(1.0, max(floor_q, float(height_quantile)))
    floor_height = float(torch.quantile(signed_heights, floor_q).item())
    region_top = float(torch.quantile(signed_heights, top_q).item())
    region_extent = max(0.0, region_top - floor_height)

    resolved_mode = str(mode).strip().lower()
    if resolved_mode == "geometric":
        height = 0.5 * region_extent
    elif resolved_mode == "fixed":
        height = float(fixed_height)
    elif resolved_mode == "median":
        median_height = float(torch.quantile(signed_heights, 0.5).item())
        height = max(0.0, median_height - floor_height)
    elif resolved_mode == "adaptive":
        height = float(height_scale) * region_extent
        height = max(float(height_min), height)
        if float(height_max) > 0.0:
            height = min(float(height_max), height)
    else:
        raise ValueError(f"Unsupported region camera height mode {mode!r}; expected geometric/adaptive/fixed/median")

    return float(height), {
        "height_mode": resolved_mode,
        "region_floor_height": float(floor_height),
        "floor_quantile": float(floor_q),
        "height_scale": float(height_scale) if resolved_mode == "adaptive" else None,
        "height_quantile": float(top_q),
        "height_min": float(height_min) if resolved_mode == "adaptive" else None,
        "height_max": float(height_max) if resolved_mode == "adaptive" else None,
        "region_height_quantile": float(region_top),
        "region_height_extent": float(region_extent),
    }


def _estimate_region_camera_from_volume(
    proposal: RegionProposal,
    points_cpu: torch.Tensor,
    *,
    up_axis: str,
    span0: float,
    span1: float,
    diagonal: float,
    frame_quantile: tuple[float, float],
    radius_scale: float,
    min_radius: float,
    max_radius: float,
) -> tuple[torch.Tensor, float, dict[str, float | str | None], float, float]:
    h0, h1, up_idx, up_sign = horizontal_axes(up_axis)
    frame = compute_scene_frame(points_cpu, up_axis=up_axis, quantile=tuple(frame_quantile))
    center = _clamp_center_to_proposal(frame.center.clone(), proposal, up_axis)
    floor_height = float(frame.floor_z)
    region_top = float(frame.ceiling_z)
    region_extent = max(0.0, region_top - floor_height)
    signed_center = float(center[up_idx].item()) * float(up_sign)
    resolved_height = signed_center - floor_height
    height_meta: dict[str, float | str | None] = {
        "height_mode": "volume",
        "region_floor_height": floor_height,
        "floor_quantile": float(frame_quantile[0]),
        "height_scale": None,
        "height_quantile": float(frame_quantile[1]),
        "height_min": None,
        "height_max": None,
        "region_height_quantile": region_top,
        "region_height_extent": region_extent,
    }

    horizontal = points_cpu[:, [h0, h1]] - center[[h0, h1]]
    distances = torch.linalg.norm(horizontal, dim=1)
    point_radius = float(torch.quantile(distances, 0.90).item()) if distances.numel() else 0.0
    frame_outer = float(frame.recommended_radii[2])
    radius = max(
        float(min_radius),
        float(radius_scale) * float(diagonal),
        point_radius * 1.35,
        frame_outer,
    )
    if float(max_radius) > 0.0:
        radius = min(radius, float(max_radius))
    return center, float(resolved_height), height_meta, float(point_radius), float(radius)


def _estimate_region_camera_from_coverage(
    proposal: RegionProposal,
    points_cpu: torch.Tensor,
    *,
    up_axis: str,
    diagonal: float,
    coverage_mode: str,
    standoff_scale: float,
    min_standoff: float,
    min_radius: float,
    max_radius: float,
    height_mode: str,
    center_height: float,
    floor_quantile: float,
    height_scale: float,
    height_min: float,
    height_max: float,
    height_quantile: float,
) -> tuple[torch.Tensor, float, dict[str, float | str | None], float, float, float, float]:
    h0, h1, up_idx, up_sign = horizontal_axes(up_axis)
    center = _proposal_bbox_center(proposal, points_cpu, up_axis)
    center = _clamp_center_to_proposal(center, proposal, up_axis)
    resolved_height, height_meta = _resolve_region_camera_height(
        points_cpu,
        up_axis,
        mode=str(height_mode),
        floor_quantile=float(floor_quantile),
        fixed_height=float(center_height),
        height_scale=float(height_scale),
        height_min=float(height_min),
        height_max=float(height_max),
        height_quantile=float(height_quantile),
    )
    floor_height = float(height_meta["region_floor_height"])
    center[up_idx] = float(up_sign) * (floor_height + float(resolved_height))

    horizontal = points_cpu[:, [h0, h1]] - center[[h0, h1]]
    distances = torch.linalg.norm(horizontal, dim=1)
    point_radius = float(torch.quantile(distances, 0.90).item()) if distances.numel() else 0.0
    coverage_radius = 0.5 * float(diagonal)
    standoff = max(float(min_standoff), float(standoff_scale) * coverage_radius)
    radius = max(float(min_radius), coverage_radius + standoff)
    if float(max_radius) > 0.0:
        radius = min(radius, float(max_radius))
    return (
        center,
        float(resolved_height),
        height_meta,
        float(point_radius),
        float(radius),
        float(coverage_radius),
        float(standoff),
    )


def estimate_region_camera(
    proposal: RegionProposal,
    points: torch.Tensor,
    *,
    up_axis: str,
    camera_purpose: str = "coverage",
    coverage_mode: str = "cylinder",
    standoff_scale: float = 0.35,
    min_standoff: float = 1.0,
    center_source: str = "volume",
    height_mode: str = "geometric",
    center_height: float = 1.0,
    floor_quantile: float = 0.02,
    height_scale: float = 0.35,
    height_min: float = 0.5,
    height_max: float = 2.0,
    height_quantile: float = 0.98,
    radius_scale: float = 0.65,
    min_radius: float = 1.0,
    max_radius: float = 25.0,
    frame_quantile: tuple[float, float] = (0.05, 0.95),
) -> RegionCameraEstimate | None:
    if points.numel() == 0:
        return None
    h0, h1, up_idx, up_sign = horizontal_axes(up_axis)
    axis0 = axis_name(h0)
    axis1 = axis_name(h1)
    min0, max0 = proposal.bounds[axis0]
    min1, max1 = proposal.bounds[axis1]
    span0 = max(1e-4, float(max0) - float(min0))
    span1 = max(1e-4, float(max1) - float(min1))
    diagonal = math.hypot(span0, span1)
    aspect_ratio = max(span0, span1) / max(1e-4, min(span0, span1))

    points_cpu = points.detach().cpu().float()
    bbox_center = _proposal_bbox_center(proposal, points_cpu, up_axis)
    robust_center = _robust_point_center(points_cpu)
    purpose = str(camera_purpose).strip().lower()
    coverage_kind = str(coverage_mode).strip().lower()
    if purpose == "coverage":
        mode = "coverage_bbox"
        center, resolved_height, height_meta, point_radius, radius, coverage_radius, coverage_standoff = (
            _estimate_region_camera_from_coverage(
                proposal,
                points_cpu,
                up_axis=up_axis,
                diagonal=diagonal,
                coverage_mode=coverage_kind,
                standoff_scale=float(standoff_scale),
                min_standoff=float(min_standoff),
                min_radius=float(min_radius),
                max_radius=float(max_radius),
                height_mode=str(height_mode),
                center_height=float(center_height),
                floor_quantile=float(floor_quantile),
                height_scale=float(height_scale),
                height_min=float(height_min),
                height_max=float(height_max),
                height_quantile=float(height_quantile),
            )
        )
    elif purpose != "object":
        raise ValueError(f"Unsupported region camera purpose {camera_purpose!r}; expected coverage/object")
    else:
        mode = str(center_source).strip().lower()
        coverage_radius = None
        coverage_standoff = None

    if purpose == "object" and mode == "volume":
        center, resolved_height, height_meta, point_radius, radius = _estimate_region_camera_from_volume(
            proposal,
            points_cpu,
            up_axis=up_axis,
            span0=span0,
            span1=span1,
            diagonal=diagonal,
            frame_quantile=tuple(frame_quantile),
            radius_scale=float(radius_scale),
            min_radius=float(min_radius),
            max_radius=float(max_radius),
        )
    elif purpose == "object" and mode == "bbox":
        center = bbox_center
        center = _clamp_center_to_proposal(center, proposal, up_axis)
        resolved_height, height_meta = _resolve_region_camera_height(
            points_cpu,
            up_axis,
            mode=str(height_mode),
            floor_quantile=float(floor_quantile),
            fixed_height=float(center_height),
            height_scale=float(height_scale),
            height_min=float(height_min),
            height_max=float(height_max),
            height_quantile=float(height_quantile),
        )
        floor_height = float(height_meta["region_floor_height"])
        center[up_idx] = float(up_sign) * (float(floor_height) + float(resolved_height))
        horizontal = points_cpu[:, [h0, h1]] - center[[h0, h1]]
        distances = torch.linalg.norm(horizontal, dim=1)
        point_radius = float(torch.quantile(distances, 0.90).item()) if distances.numel() else 0.0
        radius = max(float(min_radius), float(radius_scale) * diagonal, point_radius * 1.35)
        if float(max_radius) > 0.0:
            radius = min(radius, float(max_radius))
    elif purpose == "object" and mode == "hybrid":
        center = robust_center.clone()
        center[h0] = 0.5 * (float(robust_center[h0]) + float(bbox_center[h0]))
        center[h1] = 0.5 * (float(robust_center[h1]) + float(bbox_center[h1]))
        center = _clamp_center_to_proposal(center, proposal, up_axis)
        resolved_height, height_meta = _resolve_region_camera_height(
            points_cpu,
            up_axis,
            mode=str(height_mode),
            floor_quantile=float(floor_quantile),
            fixed_height=float(center_height),
            height_scale=float(height_scale),
            height_min=float(height_min),
            height_max=float(height_max),
            height_quantile=float(height_quantile),
        )
        floor_height = float(height_meta["region_floor_height"])
        center[up_idx] = float(up_sign) * (float(floor_height) + float(resolved_height))
        horizontal = points_cpu[:, [h0, h1]] - center[[h0, h1]]
        distances = torch.linalg.norm(horizontal, dim=1)
        point_radius = float(torch.quantile(distances, 0.90).item()) if distances.numel() else 0.0
        radius = max(float(min_radius), float(radius_scale) * diagonal, point_radius * 1.35)
        if float(max_radius) > 0.0:
            radius = min(radius, float(max_radius))
    elif purpose == "object" and mode == "robust":
        center = robust_center
        center = _clamp_center_to_proposal(center, proposal, up_axis)
        resolved_height, height_meta = _resolve_region_camera_height(
            points_cpu,
            up_axis,
            mode=str(height_mode),
            floor_quantile=float(floor_quantile),
            fixed_height=float(center_height),
            height_scale=float(height_scale),
            height_min=float(height_min),
            height_max=float(height_max),
            height_quantile=float(height_quantile),
        )
        floor_height = float(height_meta["region_floor_height"])
        center[up_idx] = float(up_sign) * (float(floor_height) + float(resolved_height))
        horizontal = points_cpu[:, [h0, h1]] - center[[h0, h1]]
        distances = torch.linalg.norm(horizontal, dim=1)
        point_radius = float(torch.quantile(distances, 0.90).item()) if distances.numel() else 0.0
        radius = max(float(min_radius), float(radius_scale) * diagonal, point_radius * 1.35)
        if float(max_radius) > 0.0:
            radius = min(radius, float(max_radius))
    elif purpose == "object":
        raise ValueError(
            f"Unsupported region camera center source {center_source!r}; "
            "expected volume/robust/bbox/hybrid"
        )

    return RegionCameraEstimate(
        center=center,
        bbox_center=bbox_center,
        robust_center=robust_center,
        orbit_radius=float(radius),
        camera_purpose=purpose,
        coverage_mode=coverage_kind if purpose == "coverage" else None,
        coverage_radius=coverage_radius,
        coverage_standoff=coverage_standoff,
        aspect_ratio=float(aspect_ratio),
        horizontal_span=(float(span0), float(span1)),
        horizontal_diagonal=float(diagonal),
        point_radius_q90=float(point_radius),
        center_source=mode,
        center_height_above_floor=float(resolved_height),
        region_floor_height=float(height_meta["region_floor_height"]),
        floor_quantile=float(height_meta["floor_quantile"]),
        height_mode=str(height_meta["height_mode"]),
        height_scale=height_meta["height_scale"] if height_meta["height_scale"] is None else float(height_meta["height_scale"]),
        height_quantile=height_meta["height_quantile"] if height_meta["height_quantile"] is None else float(height_meta["height_quantile"]),
        height_min=height_meta["height_min"] if height_meta["height_min"] is None else float(height_meta["height_min"]),
        height_max=height_meta["height_max"] if height_meta["height_max"] is None else float(height_meta["height_max"]),
        region_height_quantile=float(height_meta["region_height_quantile"]),
        region_height_extent=float(height_meta["region_height_extent"]),
        vertical_axis=axis_name(up_idx),
    )


def _default_region_camera_rings(radius: float) -> list[dict[str, object]]:
    radius = max(0.1, float(radius))
    low_h = max(0.25, min(2.0, radius * 0.08))
    mid_h = max(0.6, min(3.2, radius * 0.14))
    high_h = max(1.2, min(5.0, radius * 0.25))
    return [
        {
            "name": "region_inner",
            "num_cameras": 12,
            "orbit_radius": round(radius * 0.55, 6),
            "height_offset": round(low_h, 6),
            "look_at_center": True,
        },
        {
            "name": "region_outer",
            "num_cameras": 16,
            "orbit_radius": round(radius, 6),
            "height_offset": round(mid_h, 6),
            "look_at_center": True,
        },
        {
            "name": "region_high",
            "num_cameras": 8,
            "orbit_radius": round(radius * 0.85, 6),
            "height_offset": round(high_h, 6),
            "look_at_center": False,
            "pitch_deg": -12.0,
            "look_at_offset_scale": 0.35,
        },
    ]


def _default_region_coverage_rings(radius: float, center_height: float) -> list[dict[str, object]]:
    radius = max(0.1, float(radius))
    h = max(0.2, float(center_height))
    return [
        {
            "name": "coverage_inner",
            "num_cameras": 8,
            "orbit_radius": round(radius * 0.55, 6),
            "height_offset": round(max(0.25, h * 0.50), 6),
            "look_at_center": True,
        },
        {
            "name": "coverage_outer",
            "num_cameras": 16,
            "orbit_radius": round(radius, 6),
            "height_offset": round(max(0.45, h * 1.00), 6),
            "look_at_center": True,
        },
        {
            "name": "coverage_high",
            "num_cameras": 8,
            "orbit_radius": round(radius * 0.80, 6),
            "height_offset": round(max(0.8, h * 2.00), 6),
            "look_at_center": True,
        },
    ]


def _scale_or_create_region_camera_rings(camera_cfg, radius: float) -> None:
    rings = list(camera_cfg.get("rings", []) or [])
    if not rings:
        camera_cfg.rings = _default_region_camera_rings(radius)
        camera_cfg.num_cameras = sum(int(ring["num_cameras"]) for ring in camera_cfg.rings)
        return

    base_radii: list[float] = []
    ring_dicts: list[dict[str, object]] = []
    for ring in rings:
        ring_dict = OmegaConf.to_container(ring, resolve=False) if OmegaConf.is_config(ring) else dict(ring)
        ring_dicts.append(ring_dict)
        value = ring_dict.get("orbit_radius")
        if value is not None and float(value) > 0.0:
            base_radii.append(float(value))
    base_outer = max(base_radii) if base_radii else None
    for ring in ring_dicts:
        value = ring.get("orbit_radius")
        if value is not None and base_outer is not None and float(value) > 0.0:
            ring["orbit_radius"] = round(float(value) / base_outer * float(radius), 6)
        elif ring.get("orbit_radius_scale") is None:
            ring["orbit_radius"] = round(float(radius), 6)
    camera_cfg.rings = ring_dicts
    camera_cfg.num_cameras = sum(max(1, int(ring.get("num_cameras", 1))) for ring in ring_dicts)


def _apply_region_camera_rings(camera_cfg, estimate: RegionCameraEstimate) -> None:
    if estimate.camera_purpose == "coverage":
        camera_cfg.rings = _default_region_coverage_rings(
            float(estimate.orbit_radius),
            float(estimate.center_height_above_floor),
        )
        camera_cfg.num_cameras = sum(int(ring["num_cameras"]) for ring in camera_cfg.rings)
        return
    _scale_or_create_region_camera_rings(camera_cfg, float(estimate.orbit_radius))


def _region_assetization_prompts(source_prompts: list[str]) -> tuple[str, str]:
    context = ", ".join(prompt.strip() for prompt in source_prompts if prompt.strip())
    if not context:
        context = "unspecified Stage 1 region"
    system = (
        "You select assetizable object instances for downstream SAM3 and 3D Gaussian decomposition. "
        "Do not produce a complete scene inventory. Treat \"portable\" as \"keep_for_assetization\": "
        "set it to true for compact, separately editable physical objects or fixtures, even if they "
        "are fixed in the real world. Set it to false for background surfaces, large structures, "
        "terrain, vegetation masses, shadows, and ambiguous noisy blobs. Return only valid JSON."
    )
    user = (
        "Inspect these {num_keyframes} multi-view renderings of a Stage 1 region. "
        f"The coarse region context is: {context}.\n\n"
        "Return a JSON array of visible, well-bounded object instances suitable for object-level "
        "assetization. Prioritize objects that can be independently selected, edited, removed, "
        "replaced, or converted into standalone 3D assets. Include compact outdoor fixtures when "
        "they are clearly separable, such as vehicles, signs, trash bins, light poles, bollards, "
        "gates, barriers, planters, benches, chairs, tables, or small equipment. Include vegetation "
        "only when it is a clearly separated individual plant or tree and would make sense as a "
        "standalone asset.\n\n"
        "Do not include pavement, roads, parking lines, ground, grass regions, building facades, "
        "walls, roofs, canopy structures, shadows, blurry floaters, or large vegetation masses. "
        "For repeated objects, choose only the clearest and most spatially separable instances. "
        "Avoid listing many nearly identical adjacent objects unless each one has a stable location "
        "cue and clear boundary.\n\n"
        "Each item must have keys: \"name\", \"category\", \"portable\", \"recognizability\", "
        "\"text_prompt\". The text_prompt must be a concise SAM3 noun phrase for one instance, "
        "including appearance and stable location. Use recognizability values \"precise\", "
        "\"subtle\", or \"unrecognizable\". Return at most {max_objects} items and no explanatory "
        "text outside the JSON array."
    )
    return system, user


def _apply_region_assetization_vlm_config(
    cfg,
    *,
    source_prompts: list[str],
    max_objects: int | None,
) -> None:
    if "vlm" not in cfg.stage1:
        cfg.stage1.vlm = {}
    if "prompts" not in cfg.stage1.vlm:
        cfg.stage1.vlm.prompts = {}
    system, user = _region_assetization_prompts(source_prompts)
    cfg.stage1.vlm.prompts.system = system
    cfg.stage1.vlm.prompts.user = user
    if max_objects is not None:
        cfg.stage1.vlm.max_objects = int(max(1, max_objects))


def _apply_region_mapping_defaults(cfg) -> None:
    if "sam3" not in cfg.stage1:
        cfg.stage1.sam3 = {}
    if "component_filter" not in cfg.stage1.sam3:
        cfg.stage1.sam3.component_filter = {}
    cfg.stage1.sam3.component_filter.enabled = True
    cfg.stage1.sam3.component_filter.min_area_ratio = 0.0002
    cfg.stage1.sam3.component_filter.max_area_ratio = 0.18
    cfg.stage1.sam3.component_filter.min_component_area_px = 32
    cfg.stage1.sam3.component_filter.reject_multi_component_union = True
    cfg.stage1.sam3.max_candidates_per_view = 5
    cfg.stage1.sam3.min_candidate_quality = 0.0

    if "candidate_support" not in cfg.stage1:
        cfg.stage1.candidate_support = {}
    cfg.stage1.candidate_support.enabled = True
    cfg.stage1.candidate_support.depth_gate_enabled = True
    cfg.stage1.candidate_support.depth_tolerance = 0.08
    cfg.stage1.candidate_support.min_alpha = 0.05
    cfg.stage1.candidate_support.max_candidates_per_view = 5
    cfg.stage1.candidate_support.min_candidate_quality = 0.0
    cfg.stage1.candidate_support.component_count_max_points = 4096
    cfg.stage1.candidate_support.support_duplicate_iou = 0.45
    cfg.stage1.candidate_support.support_duplicate_containment = 0.7
    cfg.stage1.candidate_support.identity_suppression_enabled = False
    cfg.stage1.candidate_support.identity_suppression_min_reliability_gap = 0.02

    if "prefilter" not in cfg.stage1:
        cfg.stage1.prefilter = {}
    cfg.stage1.prefilter.enabled = True
    cfg.stage1.prefilter.core_erode_px = 4
    cfg.stage1.prefilter.min_view_hits = 3
    cfg.stage1.prefilter.min_hit_ratio = 0.35
    cfg.stage1.prefilter.depth_gate_enabled = True
    cfg.stage1.prefilter.depth_tolerance = 0.08
    cfg.stage1.prefilter.min_alpha = 0.05

    if "soft_assignment" not in cfg.stage1:
        cfg.stage1.soft_assignment = {}
    cfg.stage1.soft_assignment.depth_gate_enabled = True
    cfg.stage1.soft_assignment.depth_tolerance = 0.08
    cfg.stage1.soft_assignment.min_alpha = 0.05
    cfg.stage1.soft_assignment.use_mask_reliability_weights = True

    if "object_cleanup" not in cfg.stage1:
        cfg.stage1.object_cleanup = {}
    cfg.stage1.object_cleanup.enabled = True
    cfg.stage1.object_cleanup.keep_largest_component = True
    cfg.stage1.object_cleanup.radius = 1.2
    cfg.stage1.object_cleanup.min_component_gaussians = 128
    cfg.stage1.object_cleanup.min_component_ratio = 0.08

    if "view_propagation" not in cfg.stage1:
        cfg.stage1.view_propagation = {}
    cfg.stage1.view_propagation.fallback_to_raw_on_empty_graph = True
    cfg.stage1.view_propagation.fallback_to_raw_on_no_association = True
    cfg.stage1.view_propagation.conflict_min_views = 1

    if "object_dedup" not in cfg.stage1:
        cfg.stage1.object_dedup = {}
    cfg.stage1.object_dedup.enabled = True
    cfg.stage1.object_dedup.suppress = False
    cfg.stage1.object_dedup.mode = "adaptive"
    cfg.stage1.object_dedup.duplicate_score_threshold = 0.62
    cfg.stage1.object_dedup.fragment_containment_threshold = 0.85
    cfg.stage1.object_dedup.fragment_max_count_ratio = 0.7
    cfg.stage1.object_dedup.max_centroid_distance = 1.0
    cfg.stage1.object_dedup.min_horizontal_iou = 0.45
    cfg.stage1.object_dedup.min_3d_iou = 0.35
    cfg.stage1.object_dedup.min_containment = 0.6
    cfg.stage1.object_dedup.prefer_larger = True
    cfg.stage1.object_dedup.up_axis = "y"


def write_region_camera_config(
    region_dir: Path,
    *,
    base_config_path: Path | str | None,
    config_name: str,
    up_axis: str,
    estimate: RegionCameraEstimate,
    source_prompts: list[str] | None = None,
    region_vlm_prompt: bool = True,
    region_vlm_max_objects: int | None = 12,
) -> Path | None:
    if base_config_path is None:
        return None
    base_path = Path(base_config_path)
    if not base_path.exists():
        return None

    cfg = OmegaConf.load(str(base_path))
    if "stage1" not in cfg:
        cfg.stage1 = {}
    if "camera_gen" not in cfg.stage1:
        cfg.stage1.camera_gen = {}
    if "chunking" not in cfg.stage1:
        cfg.stage1.chunking = {}

    center_list = [round(float(v), 6) for v in estimate.center.detach().cpu().tolist()]
    camera_cfg = cfg.stage1.camera_gen
    camera_cfg.up_axis = up_axis
    camera_cfg.trajectory_mode = "hybrid"
    camera_cfg.auto_center = False
    camera_cfg.center = center_list
    camera_cfg.orbit_radius = round(float(estimate.orbit_radius), 6)
    camera_cfg.radius_jitter = round(float(estimate.orbit_radius), 6)
    camera_cfg.look_at_center = True
    camera_cfg.keyframe_strategy = "diverse"
    _apply_region_camera_rings(camera_cfg, estimate)
    cfg.stage1.chunking.enabled = False
    _apply_region_mapping_defaults(cfg)
    if bool(region_vlm_prompt):
        _apply_region_assetization_vlm_config(
            cfg,
            source_prompts=list(source_prompts or []),
            max_objects=region_vlm_max_objects,
        )

    output_path = region_dir / config_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(output_path), resolve=False)
    return output_path


def write_region_camera_overrides(region_dir: Path, *, up_axis: str, estimate: RegionCameraEstimate) -> Path:
    center = [float(v) for v in estimate.center.detach().cpu().tolist()]
    path = region_dir / "recommended_camera_overrides.txt"
    lines = [
        "Recommended Stage 2 camera overrides:",
        f"# camera_purpose={estimate.camera_purpose}, center_source={estimate.center_source}",
        f"stage1.camera_gen.up_axis={up_axis}",
        "stage1.camera_gen.trajectory_mode=hybrid",
        "stage1.camera_gen.auto_center=false",
        f"stage1.camera_gen.center=[{center[0]:.6f},{center[1]:.6f},{center[2]:.6f}]",
        f"stage1.camera_gen.orbit_radius={estimate.orbit_radius:.6f}",
        f"stage1.camera_gen.radius_jitter={estimate.orbit_radius:.6f}",
        "stage1.camera_gen.look_at_center=true",
        "stage1.camera_gen.keyframe_strategy=diverse",
        "stage1.chunking.enabled=false",
        "",
        "Use region_camera_config.yaml when possible; it also stores the generated hybrid rings.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_region_center_visualization(
    proposal: RegionProposal,
    points: torch.Tensor,
    estimate: RegionCameraEstimate,
    output_path: Path | str,
    *,
    up_axis: str,
    max_points: int = 60000,
) -> Path:
    from PIL import Image, ImageDraw

    h0, h1, _, _ = horizontal_axes(up_axis)
    axis0 = axis_name(h0)
    axis1 = axis_name(h1)
    min0, max0 = proposal.bounds[axis0]
    min1, max1 = proposal.bounds[axis1]
    span0 = max(1e-4, float(max0) - float(min0))
    span1 = max(1e-4, float(max1) - float(min1))
    margin = 0.12 * max(span0, span1)
    plot_min0 = float(min0) - margin
    plot_max0 = float(max0) + margin
    plot_min1 = float(min1) - margin
    plot_max1 = float(max1) + margin
    plot_span0 = max(1e-4, plot_max0 - plot_min0)
    plot_span1 = max(1e-4, plot_max1 - plot_min1)
    width = 1200
    height = 900
    pad = 70

    def to_px(x: float, y: float) -> tuple[int, int]:
        px = pad + int(round((float(x) - plot_min0) / plot_span0 * (width - 2 * pad)))
        py = height - pad - int(round((float(y) - plot_min1) / plot_span1 * (height - 2 * pad)))
        return px, py

    def draw_cross(draw: ImageDraw.ImageDraw, center: torch.Tensor, color: tuple[int, int, int], label: str) -> None:
        x, y = to_px(float(center[h0]), float(center[h1]))
        draw.line((x - 12, y, x + 12, y), fill=color, width=3)
        draw.line((x, y - 12, x, y + 12), fill=color, width=3)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(255, 255, 255), width=1)
        draw.text((x + 10, y - 18), label, fill=color)

    image = Image.new("RGB", (width, height), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    draw.rectangle((pad, pad, width - pad, height - pad), outline=(40, 40, 40), width=2)

    points_cpu = points.detach().cpu().float()
    if points_cpu.shape[0] > max_points:
        sample_idx = torch.linspace(0, points_cpu.shape[0] - 1, int(max_points)).long()
        points_cpu = points_cpu[sample_idx]
    for x, y in points_cpu[:, [h0, h1]].tolist():
        px, py = to_px(float(x), float(y))
        if pad <= px <= width - pad and pad <= py <= height - pad:
            draw.point((px, py), fill=(95, 95, 95))

    bbox0 = to_px(float(min0), float(min1))
    bbox1 = to_px(float(max0), float(max1))
    draw.rectangle((bbox0[0], bbox1[1], bbox1[0], bbox0[1]), outline=(0, 145, 90), width=4)
    draw.text((bbox0[0] + 8, bbox1[1] + 8), "region bbox", fill=(0, 100, 65))

    center = estimate.center.detach().cpu().float()
    radius = float(estimate.orbit_radius)
    if estimate.coverage_radius is not None:
        coverage_radius = float(estimate.coverage_radius)
        coverage_left = to_px(float(center[h0]) - coverage_radius, float(center[h1]) - coverage_radius)
        coverage_right = to_px(float(center[h0]) + coverage_radius, float(center[h1]) + coverage_radius)
        draw.ellipse(
            (coverage_left[0], coverage_right[1], coverage_right[0], coverage_left[1]),
            outline=(35, 120, 210),
            width=3,
        )
        draw.text((coverage_right[0] + 4, coverage_right[1] + 4), "coverage radius", fill=(35, 90, 170))
    radius_left = to_px(float(center[h0]) - radius, float(center[h1]) - radius)
    radius_right = to_px(float(center[h0]) + radius, float(center[h1]) + radius)
    draw.ellipse((radius_left[0], radius_right[1], radius_right[0], radius_left[1]), outline=(220, 40, 40), width=3)
    draw_cross(draw, estimate.bbox_center, (40, 100, 220), "bbox center")
    draw_cross(draw, estimate.robust_center, (230, 125, 20), "robust center")
    center_label = "volume center" if str(estimate.center_source) == "volume" else "camera center"
    draw_cross(draw, center, (210, 20, 20), center_label)

    draw.text(
        (pad, 20),
        f"{proposal.region_id}: purpose={estimate.camera_purpose}, center={estimate.center_source}, up={up_axis}, plane={axis0}-{axis1}",
        fill=(0, 0, 0),
    )
    draw.text(
        (pad, height - 46),
        (
            f"camera center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})  "
            f"orbit_radius={estimate.orbit_radius:.3f}  "
            f"span=({estimate.horizontal_span[0]:.3f}, {estimate.horizontal_span[1]:.3f})"
        ),
        fill=(0, 0, 0),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def export_region_chunks(
    scene: GaussianScene,
    proposals: list[RegionProposal],
    output_dir: Path | str,
    *,
    up_axis: str,
    height_mode: str = "full",
    height_quantile: tuple[float, float] = (0.02, 0.98),
    height_margin: float = 0.0,
    min_opacity: float = 0.005,
    max_scale: float | None = None,
    export_camera_config: bool = True,
    base_config_path: Path | str | None = None,
    camera_config_name: str = "region_camera_config.yaml",
    camera_purpose: str = "coverage",
    camera_coverage_mode: str = "cylinder",
    camera_standoff_scale: float = 0.35,
    camera_min_standoff: float = 1.0,
    camera_center_source: str = "volume",
    camera_height_mode: str = "geometric",
    camera_center_height: float = 1.0,
    camera_floor_quantile: float = 0.02,
    camera_height_scale: float = 0.35,
    camera_height_min: float = 0.5,
    camera_height_max: float = 2.0,
    camera_height_quantile: float = 0.98,
    camera_radius_scale: float = 0.65,
    camera_min_radius: float = 1.0,
    camera_max_radius: float = 25.0,
    camera_frame_quantile: tuple[float, float] = (0.05, 0.95),
    center_visualization_max_points: int = 60000,
    region_vlm_prompt: bool = True,
    region_vlm_max_objects: int | None = 12,
) -> list[dict[str, object]]:
    output_dir = Path(output_dir)
    regions_dir = output_dir / "regions"
    regions_dir.mkdir(parents=True, exist_ok=True)
    h0, h1, up_idx, _ = horizontal_axes(up_axis)
    means = scene.params.means.detach().cpu()
    opacities = torch.sigmoid(scene.params.opacities.squeeze(-1)).detach().cpu()
    scales = torch.exp(scene.params.scales).detach().cpu()
    records: list[dict[str, object]] = []
    for proposal in proposals:
        axis0 = axis_name(h0)
        axis1 = axis_name(h1)
        min0, max0 = proposal.bounds[axis0]
        min1, max1 = proposal.bounds[axis1]
        keep = (means[:, h0] >= min0) & (means[:, h0] <= max0) & (means[:, h1] >= min1) & (means[:, h1] <= max1)
        keep = keep & (opacities >= float(min_opacity))
        if max_scale is not None:
            max_scale_per_gaussian = scales.max(dim=-1).values
            keep = keep & (max_scale_per_gaussian <= float(max_scale))
        height_bounds = None
        if str(height_mode) == "quantile" and bool(keep.any()):
            up_values = means[keep, up_idx]
            q0, q1 = float(height_quantile[0]), float(height_quantile[1])
            lo = float(torch.quantile(up_values.detach().cpu().float(), q0).item()) - float(height_margin)
            hi = float(torch.quantile(up_values.detach().cpu().float(), q1).item()) + float(height_margin)
            keep = keep & (means[:, up_idx] >= lo) & (means[:, up_idx] <= hi)
            height_bounds = {axis_name(up_idx): [lo, hi]}
        indices = keep.nonzero(as_tuple=True)[0]
        proposal.num_gaussians = int(indices.numel())
        region_dir = regions_dir / proposal.region_id
        region_dir.mkdir(parents=True, exist_ok=True)
        ply_path = region_dir / "chunk.ply"
        camera_record = None
        if indices.numel() > 0:
            region_points = means[indices].detach().cpu().float()
            scene.subset(indices.to(scene.device)).to_ply(ply_path)
            proposal.ply = str(ply_path)
            if bool(export_camera_config):
                # Re-estimate orbit geometry from the exported chunk Gaussian cloud, not mask back-projection.
                estimate = estimate_region_camera(
                    proposal,
                    region_points,
                    up_axis=up_axis,
                    camera_purpose=str(camera_purpose),
                    coverage_mode=str(camera_coverage_mode),
                    standoff_scale=float(camera_standoff_scale),
                    min_standoff=float(camera_min_standoff),
                    center_source=str(camera_center_source),
                    height_mode=str(camera_height_mode),
                    center_height=float(camera_center_height),
                    floor_quantile=float(camera_floor_quantile),
                    height_scale=float(camera_height_scale),
                    height_min=float(camera_height_min),
                    height_max=float(camera_height_max),
                    height_quantile=float(camera_height_quantile),
                    radius_scale=float(camera_radius_scale),
                    min_radius=float(camera_min_radius),
                    max_radius=float(camera_max_radius),
                    frame_quantile=tuple(camera_frame_quantile),
                )
                if estimate is not None:
                    config_path = write_region_camera_config(
                        region_dir,
                        base_config_path=base_config_path,
                        config_name=str(camera_config_name),
                        up_axis=up_axis,
                        estimate=estimate,
                        source_prompts=list(proposal.source_prompts),
                        region_vlm_prompt=bool(region_vlm_prompt),
                        region_vlm_max_objects=region_vlm_max_objects,
                    )
                    overrides_path = write_region_camera_overrides(region_dir, up_axis=up_axis, estimate=estimate)
                    visualization_path = save_region_center_visualization(
                        proposal,
                        region_points,
                        estimate,
                        region_dir / "region_center_visualization.png",
                        up_axis=up_axis,
                        max_points=int(center_visualization_max_points),
                    )
                    camera_record = {
                        "center": [float(v) for v in estimate.center.detach().cpu().tolist()],
                        "volume_center": [float(v) for v in estimate.center.detach().cpu().tolist()]
                        if str(estimate.center_source) == "volume"
                        else None,
                        "bbox_center": [float(v) for v in estimate.bbox_center.detach().cpu().tolist()],
                        "robust_center": [float(v) for v in estimate.robust_center.detach().cpu().tolist()],
                        "camera_purpose": estimate.camera_purpose,
                        "coverage_mode": estimate.coverage_mode,
                        "coverage_radius": estimate.coverage_radius,
                        "coverage_standoff": estimate.coverage_standoff,
                        "aspect_ratio": float(estimate.aspect_ratio),
                        "center_source": estimate.center_source,
                        "frame_quantile": list(camera_frame_quantile)
                        if str(estimate.center_source) == "volume"
                        else None,
                        "recommended_orbit_radius": float(estimate.orbit_radius),
                        "horizontal_span": [float(estimate.horizontal_span[0]), float(estimate.horizontal_span[1])],
                        "horizontal_diagonal": float(estimate.horizontal_diagonal),
                        "point_radius_q90": float(estimate.point_radius_q90),
                        "center_height_above_floor": float(estimate.center_height_above_floor),
                        "region_floor_height": float(estimate.region_floor_height),
                        "floor_quantile": float(estimate.floor_quantile),
                        "height_mode": estimate.height_mode,
                        "height_scale": estimate.height_scale,
                        "height_quantile": estimate.height_quantile,
                        "height_min": estimate.height_min,
                        "height_max": estimate.height_max,
                        "region_height_quantile": estimate.region_height_quantile,
                        "region_height_extent": estimate.region_height_extent,
                        "vertical_axis": estimate.vertical_axis,
                        "base_config": str(base_config_path) if base_config_path is not None else None,
                        "generated_config": str(config_path) if config_path is not None else None,
                        "recommended_overrides": str(overrides_path),
                        "center_visualization": str(visualization_path),
                    }
                    with open(region_dir / "region_camera.json", "w", encoding="utf-8") as f:
                        json.dump(camera_record, f, indent=2)
        bbox = {
            "region_id": proposal.region_id,
            "up_axis": up_axis,
            "horizontal_axes": [axis0, axis1],
            "height_mode": str(height_mode),
            "bounds": proposal.bounds,
            "height_bounds": height_bounds,
            "score": proposal.score,
            "mean_score": proposal.mean_score,
            "area_cells": proposal.area_cells,
            "num_gaussians": proposal.num_gaussians,
            "source_prompts": proposal.source_prompts,
            "ply": proposal.ply,
            "camera": camera_record,
        }
        with open(region_dir / "bbox.json", "w", encoding="utf-8") as f:
            json.dump(bbox, f, indent=2)
        records.append(bbox)
    with open(output_dir / "region_manifest.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return records


def save_plan_view(summary: dict[str, object], records: list[AerialViewRecord], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    up_idx, _ = parse_up_axis(str(summary["up_axis"]))
    axes = [0, 1, 2]
    axes.remove(up_idx)
    h0, h1 = axes
    bounds = summary["bounds"]
    axis0 = axis_name(h0)
    axis1 = axis_name(h1)
    min0, max0 = bounds[axis0]
    min1, max1 = bounds[axis1]
    width = 1200
    height = 900
    pad = 48
    span0 = max(1e-4, float(max0) - float(min0))
    span1 = max(1e-4, float(max1) - float(min1))

    def to_px(x: float, y: float) -> tuple[int, int]:
        px = pad + int(round((float(x) - float(min0)) / span0 * (width - 2 * pad)))
        py = height - pad - int(round((float(y) - float(min1)) / span1 * (height - 2 * pad)))
        return px, py

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((pad, pad, width - pad, height - pad), outline=(30, 30, 30), width=2)
    for record in records:
        target = record.target
        px, py = to_px(float(target[h0]), float(target[h1]))
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(220, 50, 47), outline=(255, 255, 255))
        draw.text((px + 5, py - 10), str(record.index), fill=(120, 0, 0))
    draw.text(
        (pad, 16),
        f"aerial grid: {summary['grid_cols']} x {summary['grid_rows']}, up={summary['up_axis']}, plane={axis0}-{axis1}",
        fill=(0, 0, 0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def save_occupancy_debug(
    occupancy: OccupancyGrid,
    proposals: list[RegionProposal],
    output_path: Path | str,
) -> None:
    from PIL import Image, ImageDraw
    import numpy as np

    normalized = occupancy.normalized.detach().cpu().clamp(0.0, 1.0)
    arr = (normalized.numpy() * 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    img = Image.fromarray(rgb, mode="RGB").resize((900, 900), resample=Image.NEAREST)
    draw = ImageDraw.Draw(img)
    src_h, src_w = normalized.shape
    scale_x = 900 / max(1, src_w)
    scale_y = 900 / max(1, src_h)
    colors = [(255, 0, 0), (0, 190, 80), (0, 100, 255), (255, 180, 0), (180, 0, 255)]
    for idx, proposal in enumerate(proposals):
        row0, row1, col0, col1 = proposal.cell_bbox
        color = colors[idx % len(colors)]
        box = (col0 * scale_x, row0 * scale_y, (col1 + 1) * scale_x, (row1 + 1) * scale_y)
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 4, box[1] + 4), proposal.region_id, fill=color)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
