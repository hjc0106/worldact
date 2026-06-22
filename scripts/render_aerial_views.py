#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldact.data.gaussians import GaussianScene
from worldact.data.poses import CameraPose
from worldact.rendering.camera_planning import compute_scene_frame, parse_up_axis
from worldact.utils.io import save_cameras_json
from worldact.utils.logging import get_logger, setup_logging
from worldact.utils.visualization import save_image

log = get_logger(__name__)


def _axis_vector(axis: int, sign: float = 1.0) -> torch.Tensor:
    vec = torch.zeros(3, dtype=torch.float32)
    vec[int(axis)] = float(sign)
    return vec


def _horizontal_axes(up_axis: str) -> tuple[int, int, int, float]:
    up_idx, up_sign = parse_up_axis(up_axis)
    axes = [0, 1, 2]
    axes.remove(up_idx)
    return axes[0], axes[1], up_idx, up_sign


def _axis_name(axis: int) -> str:
    return ["x", "y", "z"][int(axis)]


def _make_intrinsic(image_width: int, image_height: int, fov_deg: float) -> torch.Tensor:
    fov = math.radians(float(fov_deg))
    fy = image_height / (2.0 * math.tan(fov / 2.0))
    fx = fy
    return torch.tensor(
        [[fx, 0.0, image_width / 2.0], [0.0, fy, image_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _scene_bounds(
    means: torch.Tensor,
    up_axis: str,
    *,
    quantile: tuple[float, float],
    margin_ratio: float,
) -> dict[str, object]:
    h0, h1, up_idx, up_sign = _horizontal_axes(up_axis)
    means_cpu = means.detach().cpu().float()
    q_lo, q_hi = quantile
    lo = torch.quantile(means_cpu, float(q_lo), dim=0)
    hi = torch.quantile(means_cpu, float(q_hi), dim=0)
    span0 = float(max(1e-4, (hi[h0] - lo[h0]).item()))
    span1 = float(max(1e-4, (hi[h1] - lo[h1]).item()))
    pad = float(margin_ratio) * max(span0, span1)
    up_values = means_cpu[:, up_idx] * float(up_sign)
    floor = float(torch.quantile(up_values, float(q_lo)).item())
    ceiling = float(torch.quantile(up_values, float(q_hi)).item())
    return {
        "h0": h0,
        "h1": h1,
        "up_idx": up_idx,
        "up_sign": up_sign,
        "min0": float(lo[h0].item() - pad),
        "max0": float(hi[h0].item() + pad),
        "min1": float(lo[h1].item() - pad),
        "max1": float(hi[h1].item() + pad),
        "floor": floor,
        "ceiling": ceiling,
        "center": ((lo + hi) * 0.5),
    }


def _auto_grid(
    span0: float,
    span1: float,
    *,
    max_views: int,
    grid_cols: int | None,
    grid_rows: int | None,
) -> tuple[int, int]:
    max_views = max(1, int(max_views))
    if grid_cols is not None or grid_rows is not None:
        cols = int(grid_cols) if grid_cols is not None else max(1, math.ceil(max_views / max(1, int(grid_rows))))
        rows = int(grid_rows) if grid_rows is not None else max(1, math.ceil(max_views / max(1, int(cols))))
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
    h0, h1, up_idx, up_sign = _horizontal_axes(up_axis)
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
    *,
    image_width: int,
    image_height: int,
    fov_deg: float,
    overlap: float,
    altitude: float | None,
    min_scene_clearance: float,
) -> tuple[float, dict[str, float]]:
    if altitude is not None and float(altitude) > 0.0:
        return float(altitude), {"source": "explicit", "footprint0": 0.0, "footprint1": 0.0}

    step0 = float(span0) / max(1, int(cols) - 1) if cols > 1 else float(span0)
    step1 = float(span1) / max(1, int(rows) - 1) if rows > 1 else float(span1)
    overlap = min(0.9, max(0.0, float(overlap)))
    footprint0 = step0 / max(0.05, 1.0 - overlap)
    footprint1 = step1 / max(0.05, 1.0 - overlap)
    aspect = float(image_width) / max(1.0, float(image_height))
    fov = math.radians(float(fov_deg))
    distance_for_width = footprint0 / max(1e-4, 2.0 * aspect * math.tan(fov / 2.0))
    distance_for_height = footprint1 / max(1e-4, 2.0 * math.tan(fov / 2.0))
    distance = max(distance_for_width, distance_for_height, float(min_scene_clearance), 0.5)
    return distance, {"source": "auto", "footprint0": footprint0, "footprint1": footprint1}


def _make_aerial_cameras(scene: GaussianScene, args: argparse.Namespace) -> tuple[list[CameraPose], list[dict[str, object]], dict[str, object]]:
    up_axis = str(args.up_axis)
    bounds = _scene_bounds(
        scene.params.means,
        up_axis,
        quantile=(float(args.bounds_quantile[0]), float(args.bounds_quantile[1])),
        margin_ratio=float(args.bounds_margin),
    )
    h0 = int(bounds["h0"])
    h1 = int(bounds["h1"])
    up_idx = int(bounds["up_idx"])
    up_sign = float(bounds["up_sign"])
    min0, max0 = float(bounds["min0"]), float(bounds["max0"])
    min1, max1 = float(bounds["min1"]), float(bounds["max1"])
    span0 = max(1e-4, max0 - min0)
    span1 = max(1e-4, max1 - min1)
    cols, rows = _auto_grid(
        span0,
        span1,
        max_views=int(args.max_views),
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
    )

    frame = compute_scene_frame(scene.params.means, up_axis=up_axis)
    center = bounds["center"].detach().cpu().float()
    target_up_value = float(center[up_idx] * up_sign)
    vertical_clearance = max(0.0, float(bounds["ceiling"]) - target_up_value) + float(args.clearance)
    distance, distance_meta = _estimate_distance(
        span0,
        span1,
        cols,
        rows,
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        fov_deg=float(args.fov_deg),
        overlap=float(args.overlap),
        altitude=args.altitude,
        min_scene_clearance=vertical_clearance,
    )

    xs = _linspace_centers(min0, max0, cols)
    ys = _linspace_centers(min1, max1, rows)
    cameras: list[CameraPose] = []
    records: list[dict[str, object]] = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            target = center.clone()
            target[h0] = float(x)
            target[h1] = float(y)
            camera = _camera_pose(
                target,
                distance=distance,
                up_axis=up_axis,
                image_width=int(args.image_width),
                image_height=int(args.image_height),
                fov_deg=float(args.fov_deg),
            )
            cam_center = camera.c2w[:3, 3].detach().cpu()
            cameras.append(camera)
            records.append(
                {
                    "index": len(records),
                    "row": row,
                    "col": col,
                    "target": [float(v) for v in target.tolist()],
                    "camera_center": [float(v) for v in cam_center.tolist()],
                    "distance": float(distance),
                }
            )

    summary = {
        "up_axis": up_axis,
        "horizontal_axes": [_axis_name(h0), _axis_name(h1)],
        "scene_center": [float(v) for v in frame.center.detach().cpu().tolist()],
        "bounds_quantile": [float(args.bounds_quantile[0]), float(args.bounds_quantile[1])],
        "bounds_margin": float(args.bounds_margin),
        "bounds": {
            _axis_name(h0): [min0, max0],
            _axis_name(h1): [min1, max1],
            "floor": float(bounds["floor"]),
            "ceiling": float(bounds["ceiling"]),
        },
        "grid_cols": int(cols),
        "grid_rows": int(rows),
        "num_cameras": len(cameras),
        "image_width": int(args.image_width),
        "image_height": int(args.image_height),
        "fov_deg": float(args.fov_deg),
        "overlap": float(args.overlap),
        "aerial_distance": float(distance),
        "distance_meta": distance_meta,
    }
    return cameras, records, summary


def _save_contact_sheet(image_paths: list[Path], output_path: Path, *, columns: int) -> None:
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


def _save_plan_view(scene: GaussianScene, records: list[dict[str, object]], summary: dict[str, object], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    up_idx, _ = parse_up_axis(str(summary["up_axis"]))
    axes = [0, 1, 2]
    axes.remove(up_idx)
    h0, h1 = axes
    bounds = summary["bounds"]
    axis0 = _axis_name(h0)
    axis1 = _axis_name(h1)
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
    means = scene.params.means.detach().cpu()
    max_points = 70000
    if means.shape[0] > max_points:
        means = means[:: max(1, math.ceil(means.shape[0] / max_points))]
    xs = means[:, h0]
    ys = means[:, h1]
    keep = (xs >= float(min0)) & (xs <= float(max0)) & (ys >= float(min1)) & (ys <= float(max1))
    for x, y in zip(xs[keep].tolist(), ys[keep].tolist()):
        draw.point(to_px(x, y), fill=(190, 190, 190))

    draw.rectangle((pad, pad, width - pad, height - pad), outline=(30, 30, 30), width=2)
    for record in records:
        target = record["target"]
        px, py = to_px(float(target[h0]), float(target[h1]))
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(220, 50, 47), outline=(255, 255, 255))
        draw.text((px + 5, py - 10), str(record["index"]), fill=(120, 0, 0))
    draw.text(
        (pad, 16),
        f"aerial grid: {summary['grid_cols']} x {summary['grid_rows']}, up={summary['up_axis']}, plane={axis0}-{axis1}",
        fill=(0, 0, 0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render dense aerial top-down views for a 3DGS PLY scene")
    parser.add_argument("--ply", required=True, help="Input standard 3DGS PLY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--up-axis", default="z", help="Scene up axis: z/y/x or -z/-y/-x")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=640)
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--max-views", type=int, default=96, help="Auto grid uses at most this many aerial cameras")
    parser.add_argument("--grid-cols", type=int, default=None, help="Override auto grid columns")
    parser.add_argument("--grid-rows", type=int, default=None, help="Override auto grid rows")
    parser.add_argument("--overlap", type=float, default=0.35, help="Approximate neighboring footprint overlap for auto altitude")
    parser.add_argument("--altitude", type=float, default=None, help="Explicit camera distance above each target")
    parser.add_argument("--clearance", type=float, default=0.8, help="Minimum distance above the quantile scene ceiling")
    parser.add_argument("--bounds-quantile", type=float, nargs=2, default=[0.01, 0.99])
    parser.add_argument("--bounds-margin", type=float, default=0.04)
    parser.add_argument("--contact-sheet-cols", type=int, default=6)
    parser.add_argument("--save-alpha", action="store_true")
    args = parser.parse_args()

    setup_logging("INFO")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = output_dir / "aerial_views"
    views_dir.mkdir(parents=True, exist_ok=True)

    scene = GaussianScene.from_ply(args.ply)
    scene.params = scene.params.to(args.device)
    log.info("Loaded %d Gaussians from %s", scene.num_gaussians, args.ply)

    cameras, records, summary = _make_aerial_cameras(scene, args)
    cameras = [camera.to(scene.device) for camera in cameras]
    log.info("Rendering %d aerial views (%s x %s)", len(cameras), summary["grid_cols"], summary["grid_rows"])

    image_paths: list[Path] = []
    for idx, (camera, record) in enumerate(zip(cameras, records)):
        view = scene.render(camera)
        image_path = views_dir / f"aerial_{idx:03d}_r{record['row']:02d}_c{record['col']:02d}.png"
        save_image(view.rgb, image_path)
        record["rgb"] = str(image_path)
        if args.save_alpha:
            alpha_path = views_dir / f"aerial_{idx:03d}_alpha.png"
            save_image(view.alpha.unsqueeze(-1).repeat(1, 1, 3), alpha_path)
            record["alpha"] = str(alpha_path)
        image_paths.append(image_path)

    save_cameras_json(cameras, output_dir / "aerial_cameras.json")
    with open(output_dir / "aerial_views.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "views": records}, f, indent=2)
    _save_contact_sheet(image_paths, output_dir / "aerial_contact_sheet.png", columns=max(1, int(args.contact_sheet_cols)))
    _save_plan_view(scene, records, summary, output_dir / "aerial_plan.png")
    log.info("Saved aerial views to %s", output_dir)


if __name__ == "__main__":
    main()
