#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldact.config import load_config
from worldact.data.gaussians import GaussianScene
from worldact.rendering.camera_planning import (
    generate_camera_trajectory,
    parse_up_axis,
    resolve_trajectory_center,
    select_keyframes,
)
from worldact.utils.logging import get_logger, setup_logging
from worldact.utils.visualization import save_image

log = get_logger(__name__)

RING_COLORS = [
    (43, 108, 176),   # blue
    (221, 107, 32),   # orange
    (47, 133, 90),    # green
    (128, 90, 213),   # purple
    (214, 69, 65),    # red
    (49, 151, 149),   # teal
]
RING_COLOR_BY_NAME = {
    "table_top": (43, 108, 176),
    "chair_outer": (221, 107, 32),
    "chair_side": (47, 133, 90),
    "single_ring": (43, 108, 176),
}


def _horizontal_axes(up_axis: int) -> tuple[int, int]:
    axes = [0, 1, 2]
    axes.remove(up_axis)
    return axes[0], axes[1]


def _axis_name(axis: int) -> str:
    return ["x", "y", "z"][axis]


def _ring_labels(cam_cfg) -> list[str]:
    mode = str(getattr(cam_cfg, "trajectory_mode", "single_ring"))
    rings = list(getattr(cam_cfg, "rings", []) or [])
    if mode not in {"multi_ring", "hybrid"} or not rings:
        return ["single_ring"] * int(getattr(cam_cfg, "num_cameras", 0))
    names = [str(getattr(ring, "name", f"ring_{idx}")) for idx, ring in enumerate(rings)]
    counts = [max(1, int(getattr(ring, "num_cameras", 1))) for ring in rings]
    labels: list[str] = []
    for yaw_idx in range(max(counts)):
        for name, count in zip(names, counts):
            if yaw_idx < count:
                labels.append(name)
    return labels


def _camera_metadata(
    camera,
    center: torch.Tensor,
    order: int,
    *,
    up_axis: int,
    up_sign: float,
    ring_name: str | None = None,
) -> dict:
    pos = camera.c2w[:3, 3].detach().cpu()
    center_cpu = center.detach().cpu()
    h0, h1 = _horizontal_axes(up_axis)
    dx = float(pos[h0] - center_cpu[h0])
    dy = float(pos[h1] - center_cpu[h1])
    yaw_deg = math.degrees(math.atan2(dy, dx))
    radius = math.hypot(dx, dy)
    height = float((pos[up_axis] - center_cpu[up_axis]) * up_sign)
    forward = camera.c2w[:3, 2].detach().cpu()
    return {
        "order": order,
        "ring": ring_name,
        "position": [float(pos[0]), float(pos[1]), float(pos[2])],
        "horizontal_axes": [_axis_name(h0), _axis_name(h1)],
        "up_axis": ("-" if up_sign < 0 else "") + _axis_name(up_axis),
        "yaw_deg": yaw_deg,
        "radius": radius,
        "height_offset": height,
        "forward": [float(forward[0]), float(forward[1]), float(forward[2])],
    }


def _save_camera_positions_csv(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "order",
                "ring",
                "x",
                "y",
                "z",
                "yaw_deg",
                "radius",
                "height_offset",
                "forward_x",
                "forward_y",
                "forward_z",
            ]
        )
        for record in records:
            pos = record["position"]
            forward = record["forward"]
            writer.writerow(
                [
                    record["order"],
                    record.get("ring") or "",
                    pos[0],
                    pos[1],
                    pos[2],
                    record["yaw_deg"],
                    record["radius"],
                    record["height_offset"],
                    forward[0],
                    forward[1],
                    forward[2],
                ]
            )


def _ring_color_map(records: list[dict]) -> dict[str, tuple[int, int, int]]:
    names: list[str] = []
    for record in records:
        name = str(record.get("ring") or "single_ring")
        if name not in names:
            names.append(name)
    colors: dict[str, tuple[int, int, int]] = {}
    for idx, name in enumerate(names):
        colors[name] = RING_COLOR_BY_NAME.get(name, RING_COLORS[idx % len(RING_COLORS)])
    return colors


def _draw_arrow(
    draw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    width: int = 2,
) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_len = 8
    head_angle = math.radians(28)
    p1 = (
        int(end[0] - head_len * math.cos(angle - head_angle)),
        int(end[1] - head_len * math.sin(angle - head_angle)),
    )
    p2 = (
        int(end[0] - head_len * math.cos(angle + head_angle)),
        int(end[1] - head_len * math.sin(angle + head_angle)),
    )
    draw.polygon([end, p1, p2], fill=fill)


def _draw_legend(
    draw,
    color_by_ring: dict[str, tuple[int, int, int]],
    *,
    xy: tuple[int, int],
    title: str = "rings",
) -> None:
    x, y = xy
    line_h = 20
    width = max(130, max((len(name) for name in color_by_ring), default=4) * 8 + 50)
    height = 28 + line_h * len(color_by_ring)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=6, fill=(255, 255, 255), outline=(190, 190, 190))
    draw.text((x + 10, y + 8), title, fill=(20, 20, 20))
    for idx, (name, color) in enumerate(color_by_ring.items()):
        yy = y + 28 + idx * line_h
        draw.rectangle((x + 10, yy + 4, x + 24, yy + 18), fill=color)
        draw.text((x + 30, yy + 2), name, fill=(20, 20, 20))


def _draw_scene_points(
    draw,
    means: torch.Tensor,
    *,
    axes: tuple[int, int],
    bounds: tuple[float, float, float, float],
    to_px,
    color: tuple[int, int, int] = (185, 185, 185),
    max_points: int = 45000,
) -> None:
    if means.numel() == 0:
        return
    h0, h1 = axes
    x0, x1, y0, y1 = bounds
    points = means.detach().cpu()
    if points.shape[0] > max_points:
        stride = max(1, int(math.ceil(points.shape[0] / max_points)))
        points = points[::stride]
    xs = points[:, h0]
    ys = points[:, h1]
    keep = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    for x, y in zip(xs[keep].tolist(), ys[keep].tolist()):
        px, py = to_px(float(x), float(y))
        draw.point((px, py), fill=color)


def _topdown_bounds_from_scene(scene: GaussianScene, *, up_axis: int) -> tuple[float, float, float, float]:
    means = scene.params.means.detach().cpu()
    h0, h1 = _horizontal_axes(up_axis)
    xs = means[:, h0]
    ys = means[:, h1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    pad = 0.15 * max(x1 - x0, y1 - y0, 1e-3)
    return x0 - pad, x1 + pad, y0 - pad, y1 + pad


def _topdown_bounds_from_records(
    records: list[dict],
    center: torch.Tensor,
    *,
    up_axis: int,
    padding_scale: float = 0.35,
) -> tuple[float, float, float, float]:
    h0, h1 = _horizontal_axes(up_axis)
    xs = [float(center[h0])]
    ys = [float(center[h1])]
    for record in records:
        pos = record["position"]
        xs.append(float(pos[h0]))
        ys.append(float(pos[h1]))
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span = max(x1 - x0, y1 - y0, 1e-3)
    pad = max(0.35, padding_scale * span)
    return x0 - pad, x1 + pad, y0 - pad, y1 + pad


def _save_topdown_diagram(
    scene: GaussianScene,
    center: torch.Tensor,
    output_path: Path,
    *,
    up_axis: int,
    up_sign: float,
    records: list[dict],
    focused: bool = False,
) -> None:
    from PIL import Image, ImageDraw

    h0, h1 = _horizontal_axes(up_axis)
    if focused:
        x0, x1, y0, y1 = _topdown_bounds_from_records(records, center, up_axis=up_axis)
        title = "ROI topdown"
        scene_point_color = (205, 205, 205)
    else:
        x0, x1, y0, y1 = _topdown_bounds_from_scene(scene, up_axis=up_axis)
        title = "scene topdown"
        scene_point_color = (190, 190, 190)

    size = 800
    image = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    color_by_ring = _ring_color_map(records)

    def to_px(x: float, y: float) -> tuple[int, int]:
        px = int((x - x0) / max(x1 - x0, 1e-6) * (size - 1))
        py = int((1.0 - (y - y0) / max(y1 - y0, 1e-6)) * (size - 1))
        return px, py

    _draw_scene_points(
        draw,
        scene.params.means,
        axes=(h0, h1),
        bounds=(x0, x1, y0, y1),
        to_px=to_px,
        color=scene_point_color,
    )

    cx, cy = to_px(float(center[h0]), float(center[h1]))
    draw.text(
        (12, 12),
        f"{title}: {_axis_name(h0)}-{_axis_name(h1)} plane, up={('-' if up_sign < 0 else '') + _axis_name(up_axis)}",
        fill=(0, 0, 0),
    )

    radii_by_ring: dict[str, list[float]] = {}
    for record in records:
        ring = str(record.get("ring") or "single_ring")
        radii_by_ring.setdefault(ring, []).append(float(record["radius"]))
    if focused:
        for ring, radii in radii_by_ring.items():
            if not radii:
                continue
            radius = sum(radii) / len(radii)
            rx, _ = to_px(float(center[h0]) + radius, float(center[h1]))
            r_px = abs(rx - cx)
            color = color_by_ring.get(ring, (30, 90, 220))
            draw.ellipse((cx - r_px, cy - r_px, cx + r_px, cy + r_px), outline=color, width=2)

    for record in records:
        order = int(record["order"])
        ring = str(record.get("ring") or "single_ring")
        color = color_by_ring.get(ring, (30, 90, 220))
        pos = record["position"]
        px, py = to_px(float(pos[h0]), float(pos[h1]))
        forward = record["forward"]
        arrow_len = 0.22 * max(float(record["radius"]), 0.5)
        tip_x = float(pos[h0]) + arrow_len * float(forward[h0])
        tip_y = float(pos[h1]) + arrow_len * float(forward[h1])
        tx, ty = to_px(tip_x, tip_y)
        _draw_arrow(draw, (px, py), (tx, ty), fill=(35, 35, 35), width=1)
        r = 5 if focused else 4
        draw.ellipse((px - r, py - r, px + r, py + r), fill=color, outline=(255, 255, 255))
        draw.text((px + 6, py + 5), str(order), fill=(0, 0, 0))

    draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(220, 40, 40), outline=(255, 255, 255))
    draw.text((cx + 8, cy + 8), "center", fill=(120, 0, 0))
    _draw_legend(draw, color_by_ring, xy=(12, 44))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _save_side_diagram(
    records: list[dict],
    output_path: Path,
    *,
    up_axis_label: str,
) -> None:
    from PIL import Image, ImageDraw

    size = (920, 520)
    margin_l, margin_r, margin_t, margin_b = 82, 34, 52, 72
    image = Image.new("RGB", size, (247, 247, 247))
    draw = ImageDraw.Draw(image)
    color_by_ring = _ring_color_map(records)

    radii = [float(record["radius"]) for record in records] or [1.0]
    heights = [float(record["height_offset"]) for record in records] or [0.0]
    r_min = 0.0
    r_max = max(max(radii), 1e-3) * 1.15
    h_min = min(0.0, min(heights))
    h_max = max(0.0, max(heights))
    h_pad = max(0.15, 0.25 * max(h_max - h_min, 1e-3))
    h_min -= h_pad
    h_max += h_pad

    plot_w = size[0] - margin_l - margin_r
    plot_h = size[1] - margin_t - margin_b

    def to_px(radius: float, height: float) -> tuple[int, int]:
        px = margin_l + int((radius - r_min) / max(r_max - r_min, 1e-6) * plot_w)
        py = margin_t + int((1.0 - (height - h_min) / max(h_max - h_min, 1e-6)) * plot_h)
        return px, py

    draw.text((12, 12), f"side view: radius vs height offset, up={up_axis_label}", fill=(0, 0, 0))
    x_axis_y = to_px(0.0, 0.0)[1]
    draw.line((margin_l, x_axis_y, size[0] - margin_r, x_axis_y), fill=(170, 170, 170), width=1)
    draw.line((margin_l, margin_t, margin_l, size[1] - margin_b), fill=(60, 60, 60), width=2)
    draw.line((margin_l, size[1] - margin_b, size[0] - margin_r, size[1] - margin_b), fill=(60, 60, 60), width=2)
    draw.text((size[0] // 2 - 42, size[1] - 34), "orbit radius", fill=(0, 0, 0))
    draw.text((10, size[1] // 2 - 10), "height", fill=(0, 0, 0))

    for tick in range(5):
        r_value = r_min + (r_max - r_min) * tick / 4.0
        px, py0 = to_px(r_value, h_min)
        _, py1 = to_px(r_value, h_max)
        draw.line((px, py0, px, py1), fill=(225, 225, 225), width=1)
        draw.text((px - 14, size[1] - margin_b + 8), f"{r_value:.1f}", fill=(60, 60, 60))

    for tick in range(5):
        h_value = h_min + (h_max - h_min) * tick / 4.0
        px0, py = to_px(r_min, h_value)
        px1, _ = to_px(r_max, h_value)
        draw.line((px0, py, px1, py), fill=(225, 225, 225), width=1)
        draw.text((margin_l - 58, py - 7), f"{h_value:.2f}", fill=(60, 60, 60))

    records_by_ring: dict[str, list[dict]] = {}
    for record in records:
        ring = str(record.get("ring") or "single_ring")
        records_by_ring.setdefault(ring, []).append(record)

    for ring, ring_records in records_by_ring.items():
        color = color_by_ring.get(ring, (30, 90, 220))
        radius = sum(float(record["radius"]) for record in ring_records) / len(ring_records)
        height = sum(float(record["height_offset"]) for record in ring_records) / len(ring_records)
        px0, py = to_px(0.0, height)
        px, _ = to_px(radius, height)
        draw.line((px0, py, px, py), fill=color, width=3)
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=color, outline=(255, 255, 255), width=2)
        label = f"{ring}: {len(ring_records)} cams, r={radius:.2f}, h={height:.2f}"
        label_x = min(px + 12, size[0] - margin_r - 245)
        draw.text((label_x, py - 9), label, fill=(20, 20, 20))

    _draw_legend(draw, color_by_ring, xy=(size[0] - 190, 48))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize generated camera trajectory and keyframes")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--config", default="configs/scenes/indoor_table.yaml")
    parser.add_argument("--output-dir", default="outputs/camera_preview")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-keyframes", type=int, default=None)
    parser.add_argument("--save-renders", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    setup_logging("INFO")
    overrides = list(args.overrides)
    overrides.append(f"device.device={args.device}")
    if args.num_keyframes is not None:
        overrides.append(f"stage1.vlm.num_keyframes={args.num_keyframes}")

    cfg = load_config(Path(args.config), overrides)
    scene = GaussianScene.from_ply(args.ply)
    scene.params = scene.params.to(cfg.device.device)
    cam_cfg = cfg.stage1.camera_gen
    scene.camera_trajectory = [
        cam.to(scene.device)
        for cam in generate_camera_trajectory(scene.params.means, cam_cfg)
    ]
    center, _ = resolve_trajectory_center(scene.params.means, cam_cfg)
    up_axis, up_sign = parse_up_axis(getattr(cam_cfg, "up_axis", "z"))
    ring_labels = _ring_labels(cam_cfg)
    keyframe_indices = select_keyframes(
        scene.camera_trajectory,
        int(cfg.stage1.vlm.num_keyframes),
        cam_cfg,
        center=center,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_records = [
        _camera_metadata(
            camera,
            center,
            order,
            up_axis=up_axis,
            up_sign=up_sign,
            ring_name=ring_labels[order] if order < len(ring_labels) else None,
        )
        for order, camera in enumerate(scene.camera_trajectory)
    ]
    keyframe_records = []
    for order, view_idx in enumerate(keyframe_indices):
        record = {
            "order": order,
            "view_index": view_idx,
            **_camera_metadata(
                scene.camera_trajectory[view_idx],
                center,
                view_idx,
                up_axis=up_axis,
                up_sign=up_sign,
                ring_name=ring_labels[view_idx] if view_idx < len(ring_labels) else None,
            ),
        }
        keyframe_records.append(record)

    with open(output_dir / "cameras.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "center": center.tolist(),
                "up_axis": getattr(cam_cfg, "up_axis", "z"),
                "trajectory_mode": str(getattr(cam_cfg, "trajectory_mode", "single_ring")),
                "num_cameras": len(scene.camera_trajectory),
                "cameras": camera_records,
            },
            f,
            indent=2,
        )
    with open(output_dir / "camera_positions.json", "w", encoding="utf-8") as f:
        json.dump(camera_records, f, indent=2)
    _save_camera_positions_csv(camera_records, output_dir / "camera_positions.csv")
    with open(output_dir / "keyframes.json", "w", encoding="utf-8") as f:
        json.dump(keyframe_records, f, indent=2)

    _save_topdown_diagram(
        scene,
        center,
        output_dir / "camera_topdown.png",
        up_axis=up_axis,
        up_sign=up_sign,
        records=camera_records,
        focused=False,
    )
    _save_topdown_diagram(
        scene,
        center,
        output_dir / "camera_topdown_roi.png",
        up_axis=up_axis,
        up_sign=up_sign,
        records=camera_records,
        focused=True,
    )
    _save_side_diagram(
        camera_records,
        output_dir / "camera_side.png",
        up_axis_label=("-" if up_sign < 0 else "") + _axis_name(up_axis),
    )
    log.info(
        "Saved camera diagrams to %s, %s and %s",
        output_dir / "camera_topdown.png",
        output_dir / "camera_topdown_roi.png",
        output_dir / "camera_side.png",
    )
    log.info("Saved camera positions to %s and %s", output_dir / "camera_positions.json", output_dir / "camera_positions.csv")

    if args.save_renders:
        render_dir = output_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        for order, view_idx in enumerate(keyframe_indices):
            view = scene.render(scene.camera_trajectory[view_idx])
            save_image(view.rgb, render_dir / f"keyframe_{order:02d}_view_{view_idx:02d}.png")
        log.info("Saved %d keyframe renders to %s", len(keyframe_indices), render_dir)


if __name__ == "__main__":
    main()
