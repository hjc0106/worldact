#!/usr/bin/env python3
"""Visualize region ring cameras and Gaussian point cloud in 3D.

Reads ``region_camera_config.yaml`` (or any pipeline config with ``stage1.camera_gen``)
and optionally a region ``chunk.ply``, then renders camera positions, view directions,
orbit rings, and the Gaussian cloud so you can inspect horizontal/vertical alignment.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldact.config import load_config
from worldact.data.gaussians import GaussianScene
from worldact.rendering.camera_planning import (
    RingCameraSpec,
    _horizontal_axes,
    _look_at_target,
    generate_camera_trajectory,
    parse_up_axis,
)

RING_COLORS = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FFC107", "#00BCD4"]
RING_MARKERS = ["^", "s", "D", "o", "v", "P"]


def _ring_specs(cam_cfg) -> list[RingCameraSpec]:
    from worldact.rendering.camera_planning import _ring_specs_from_cfg

    return _ring_specs_from_cfg(cam_cfg)


def _ring_labels(cam_cfg) -> list[str]:
    rings = _ring_specs(cam_cfg)
    if not rings:
        return ["single_ring"]
    names = [ring.name for ring in rings]
    counts = [max(1, int(ring.num_cameras)) for ring in rings]
    labels: list[str] = []
    for yaw_idx in range(max(counts)):
        for name, count in zip(names, counts):
            if yaw_idx < count:
                labels.append(name)
    return labels


def _camera_record(
    camera,
    center: torch.Tensor,
    order: int,
    *,
    up_axis: int,
    up_sign: float,
    ring_name: str | None,
) -> dict:
    pos = camera.c2w[:3, 3].detach().cpu()
    center_cpu = center.detach().cpu()
    h0, h1 = _horizontal_axes(up_axis)
    dx = float(pos[h0] - center_cpu[h0])
    dy = float(pos[h1] - center_cpu[h1])
    forward = camera.c2w[:3, 2].detach().cpu()
    return {
        "order": order,
        "ring": ring_name,
        "position": pos.numpy().astype(np.float64),
        "forward": forward.numpy().astype(np.float64),
        "yaw_deg": math.degrees(math.atan2(dy, dx)),
        "radius": math.hypot(dx, dy),
        "height_offset": float((pos[up_axis] - center_cpu[up_axis]) * up_sign),
    }


def _load_gaussian_points(ply_path: Path | None, max_points: int) -> np.ndarray | None:
    if ply_path is None or not ply_path.exists():
        return None
    scene = GaussianScene.from_ply(str(ply_path))
    means = scene.params.means.detach().cpu().numpy()
    if means.shape[0] > max_points:
        step = max(1, means.shape[0] // max_points)
        means = means[::step]
    return means


def _gaussian_stats(points: np.ndarray, up_axis: int) -> dict[str, object]:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return {
        "centroid": points.mean(axis=0),
        "median": np.median(points, axis=0),
        "min": lo,
        "max": hi,
        "up_span": float(hi[up_axis] - lo[up_axis]),
        "up_min": float(lo[up_axis]),
        "up_max": float(hi[up_axis]),
    }


def _ring_style(ring_name: str, index: int) -> tuple[str, str, int]:
    return RING_COLORS[index % len(RING_COLORS)], RING_MARKERS[index % len(RING_MARKERS)], 70 + 10 * (index % 3)


def _compute_look_at_points(
    center: torch.Tensor,
    ring: RingCameraSpec,
    *,
    up_axis: int,
) -> np.ndarray:
    n = max(1, int(ring.num_cameras))
    orbit_radius = float(ring.orbit_radius or 0.0)
    targets = []
    for i in range(n):
        yaw = 2.0 * math.pi * i / n
        target = _look_at_target(center, yaw, orbit_radius, ring, up_axis=up_axis)
        targets.append(target.detach().cpu().numpy())
    return np.asarray(targets, dtype=np.float64)


def _plot_points(ax3d, ax_xy, ax_xz, ax_yz, pts: np.ndarray, up_axis: int, *, alpha: float = 0.15) -> None:
    y_norm = (pts[:, up_axis] - pts[:, up_axis].min()) / max(1e-6, pts[:, up_axis].max() - pts[:, up_axis].min())
    colors = plt.cm.coolwarm(y_norm)
    colors[:, 3] = alpha

    # 3D axes: X, horizontal-1 as depth, up_axis as vertical
    h0, h1 = _horizontal_axes(up_axis)
    ax3d.scatter(pts[:, 0], pts[:, h1], pts[:, up_axis], c=colors, s=0.5, linewidths=0, rasterized=True)
    ax_xy.scatter(pts[:, h0], pts[:, h1], c=colors, s=0.6, alpha=alpha, linewidths=0, rasterized=True)
    ax_xz.scatter(pts[:, h0], pts[:, up_axis], c=colors, s=0.6, alpha=alpha, linewidths=0, rasterized=True)
    ax_yz.scatter(pts[:, h1], pts[:, up_axis], c=colors, s=0.6, alpha=alpha, linewidths=0, rasterized=True)


def _style_axes(fig, axes_flat, title: str, up_axis_label: str) -> None:
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes_flat:
        if hasattr(ax, "set_facecolor"):
            ax.set_facecolor("#0f0f23")
        if hasattr(ax, "tick_params"):
            ax.tick_params(colors="white")
        if hasattr(ax, "spines"):
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
        if hasattr(ax, "grid"):
            ax.grid(True, color="#333355", linewidth=0.5, alpha=0.6)
    axes_flat[0].set_title(title, color="white", fontsize=10)


def _draw_target_marker(
    ax3d,
    ax_xy,
    ax_xz,
    ax_yz,
    point: np.ndarray,
    *,
    up_axis: int,
    color: str,
    marker: str,
    size: float,
    label: str | None = None,
) -> None:
    h0, h1 = _horizontal_axes(up_axis)
    ax3d.scatter([point[0]], [point[h1]], [point[up_axis]], c=color, s=size, marker=marker, zorder=10, label=label)
    ax_xy.scatter([point[h0]], [point[h1]], c=color, s=size * 0.9, marker=marker, zorder=10)
    ax_xz.scatter([point[h0]], [point[up_axis]], c=color, s=size * 0.9, marker=marker, zorder=10)
    ax_yz.scatter([point[h1]], [point[up_axis]], c=color, s=size * 0.9, marker=marker, zorder=10)


def _draw_cameras_and_rings(
    ax3d,
    ax_xy,
    ax_xz,
    ax_yz,
    *,
    center: np.ndarray,
    records: list[dict],
    rings: list[RingCameraSpec],
    up_axis: int,
    up_sign: float,
    arrow_scale: float,
) -> None:
    h0, h1 = _horizontal_axes(up_axis)
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    records_by_ring: dict[str, list[dict]] = {}
    for record in records:
        ring = str(record.get("ring") or "ring")
        records_by_ring.setdefault(ring, []).append(record)

    for ring_idx, ring in enumerate(rings):
        ring_name = ring.name
        ring_records = records_by_ring.get(ring_name, [])
        if not ring_records:
            continue
        color, marker, size = _ring_style(ring_name, ring_idx)
        positions = np.stack([rec["position"] for rec in ring_records], axis=0)
        forwards = np.stack([rec["forward"] for rec in ring_records], axis=0)

        label = (
            f"{ring_name} (n={ring.num_cameras}, r={float(ring.orbit_radius or 0.0):.2f}, "
            f"h={float(ring.height_offset):.2f})"
        )
        ax3d.scatter(
            positions[:, 0],
            positions[:, h1],
            positions[:, up_axis],
            c=color,
            s=size,
            marker=marker,
            zorder=5,
            label=label,
            depthshade=False,
        )
        ax_xy.scatter(positions[:, h0], positions[:, h1], c=color, s=size * 0.8, marker=marker, zorder=5)
        ax_xz.scatter(positions[:, h0], positions[:, up_axis], c=color, s=size * 0.8, marker=marker, zorder=5, label=label)
        ax_yz.scatter(positions[:, h1], positions[:, up_axis], c=color, s=size * 0.8, marker=marker, zorder=5)

        # orbit circle at camera height (average)
        theta = np.linspace(0.0, 2.0 * math.pi, 200)
        orbit_r = float(ring.orbit_radius or ring_records[0]["radius"])
        height = float(np.mean([rec["height_offset"] for rec in ring_records]))
        circle = np.zeros((200, 3), dtype=np.float64)
        circle[:, h0] = center[h0] + orbit_r * np.cos(theta)
        circle[:, h1] = center[h1] + orbit_r * np.sin(theta)
        circle[:, up_axis] = center[up_axis] + up_sign * height
        ax3d.plot(circle[:, 0], circle[:, h1], circle[:, up_axis], color=color, alpha=0.45, linewidth=1.0)
        ax_xy.plot(circle[:, h0], circle[:, h1], color=color, alpha=0.55, linewidth=1.0)

        # view direction arrows
        for pos, fwd in zip(positions, forwards):
            fwd = fwd / max(np.linalg.norm(fwd), 1e-6)
            end = pos + fwd * arrow_scale
            ax3d.plot(
                [pos[0], end[0]],
                [pos[h1], end[h1]],
                [pos[up_axis], end[up_axis]],
                color=color,
                alpha=0.55,
                linewidth=1.2,
            )
            ax_xz.plot([pos[h0], end[h0]], [pos[up_axis], end[up_axis]], color=color, alpha=0.35, linewidth=0.8)
            ax_yz.plot([pos[h1], end[h1]], [pos[up_axis], end[up_axis]], color=color, alpha=0.35, linewidth=0.8)
            ax3d.plot([center[0], pos[0]], [center[h1], pos[h1]], [center[up_axis], pos[up_axis]], color=color, alpha=0.08, linewidth=0.5)

        # look-at targets (when ring uses look-at logic)
        look_targets = _compute_look_at_points(torch.tensor(center, dtype=torch.float32), ring, up_axis=up_axis)
        ax3d.scatter(
            look_targets[:, 0],
            look_targets[:, h1],
            look_targets[:, up_axis],
            c=color,
            s=18,
            marker="x",
            alpha=0.7,
            zorder=4,
        )
        ax_xz.scatter(look_targets[:, h0], look_targets[:, up_axis], c=color, s=14, marker="x", alpha=0.7)
        ax_yz.scatter(look_targets[:, h1], look_targets[:, up_axis], c=color, s=14, marker="x", alpha=0.7)


def visualize_region_cameras(
    *,
    config_path: Path,
    ply_path: Path | None,
    output_path: Path,
    max_gauss_points: int,
    arrow_scale: float,
) -> None:
    cfg = load_config(config_path)
    cam_cfg = cfg.stage1.camera_gen
    up_axis, up_sign = parse_up_axis(getattr(cam_cfg, "up_axis", "z"))
    h0, h1 = _horizontal_axes(up_axis)
    up_axis_label = ("-" if up_sign < 0 else "") + ["x", "y", "z"][up_axis]

    scene = GaussianScene.from_ply(str(ply_path)) if ply_path and ply_path.exists() else None
    if scene is not None:
        scene.params = scene.params.to("cpu")

    center_cfg = getattr(cam_cfg, "center", None)
    if center_cfg is None:
        raise SystemExit("Config must define stage1.camera_gen.center for region visualization.")
    center = torch.tensor(list(center_cfg), dtype=torch.float32)
    means = scene.params.means if scene is not None else torch.zeros(1, 3)
    cameras = generate_camera_trajectory(means, cam_cfg, ignore_config_center=False)
    ring_labels = _ring_labels(cam_cfg)
    rings = _ring_specs(cam_cfg)

    records = [
        _camera_record(
            camera,
            center,
            order,
            up_axis=up_axis,
            up_sign=up_sign,
            ring_name=ring_labels[order] if order < len(ring_labels) else None,
        )
        for order, camera in enumerate(cameras)
    ]

    pts = _load_gaussian_points(ply_path, max_gauss_points)
    gstats = _gaussian_stats(pts, up_axis) if pts is not None else None
    center_np = center.detach().cpu().numpy()

    print(f"Config center ({up_axis_label} up): {center_np.tolist()}")
    print(f"Generated {len(cameras)} cameras across {len(rings)} rings")
    for ring in rings:
        ring_recs = [r for r in records if r.get("ring") == ring.name]
        if not ring_recs:
            continue
        heights = [r["height_offset"] for r in ring_recs]
        print(
            f"  {ring.name}: n={len(ring_recs)}, "
            f"height_offset={heights[0]:.4f} (world {['x','y','z'][up_axis]}="
            f"{center_np[up_axis] + up_sign * heights[0]:.4f})"
        )
    if gstats is not None:
        print(
            f"Gaussian median: {gstats['median'].tolist()}  "
            f"up-axis span [{gstats['up_min']:.4f}, {gstats['up_max']:.4f}]"
        )
        delta_up = float(gstats["median"][up_axis] - center_np[up_axis])
        print(f"Median vs config center on up axis: {delta_up:+.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_name("camera_positions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": str(config_path),
                "ply": str(ply_path) if ply_path else None,
                "center": center_np.tolist(),
                "up_axis": up_axis_label,
                "gaussian_stats": (
                    {
                        "centroid": gstats["centroid"].tolist(),
                        "median": gstats["median"].tolist(),
                        "min": gstats["min"].tolist(),
                        "max": gstats["max"].tolist(),
                    }
                    if gstats
                    else None
                ),
                "cameras": [
                    {
                        "order": rec["order"],
                        "ring": rec["ring"],
                        "position": rec["position"].tolist(),
                        "forward": rec["forward"].tolist(),
                        "yaw_deg": rec["yaw_deg"],
                        "radius": rec["radius"],
                        "height_offset": rec["height_offset"],
                    }
                    for rec in records
                ],
            },
            f,
            indent=2,
        )

    fig = plt.figure(figsize=(22, 17))
    ax3d: Axes3D = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)
    _style_axes(fig, [ax3d, ax_xy, ax_xz, ax_yz], "Region cameras + Gaussian cloud", up_axis_label)

    if pts is not None:
        _plot_points(ax3d, ax_xy, ax_xz, ax_yz, pts, up_axis)
        _draw_target_marker(
            ax3d,
            ax_xy,
            ax_xz,
            ax_yz,
            gstats["median"],
            up_axis=up_axis,
            color="#E040FB",
            marker="o",
            size=120,
            label="Gaussian median",
        )
        _draw_target_marker(
            ax3d,
            ax_xy,
            ax_xz,
            ax_yz,
            gstats["centroid"],
            up_axis=up_axis,
            color="#80DEEA",
            marker=".",
            size=80,
            label="Gaussian centroid",
        )

    _draw_target_marker(
        ax3d,
        ax_xy,
        ax_xz,
        ax_yz,
        center_np,
        up_axis=up_axis,
        color="yellow",
        marker="*",
        size=260,
        label="config center",
    )

    _draw_cameras_and_rings(
        ax3d,
        ax_xy,
        ax_xz,
        ax_yz,
        center=center_np,
        records=records,
        rings=rings,
        up_axis=up_axis,
        up_sign=up_sign,
        arrow_scale=arrow_scale,
    )

    # height reference lines on side/front views
    ax_xz.axhline(center_np[up_axis], color="yellow", alpha=0.55, linewidth=0.9, linestyle=":")
    ax_yz.axhline(center_np[up_axis], color="yellow", alpha=0.55, linewidth=0.9, linestyle=":")
    if gstats is not None:
        ax_xz.axhline(float(gstats["median"][up_axis]), color="#E040FB", alpha=0.45, linewidth=0.9, linestyle="--")
        ax_yz.axhline(float(gstats["median"][up_axis]), color="#E040FB", alpha=0.45, linewidth=0.9, linestyle="--")

    ax3d.set_xlabel("X", color="white", labelpad=4)
    ax3d.set_ylabel(["X", "Y", "Z"][h1], color="white", labelpad=4)
    ax3d.set_zlabel(f"{['X','Y','Z'][up_axis]} (up={up_axis_label})", color="white", labelpad=4)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#333355")
    ax3d.yaxis.pane.set_edgecolor("#333355")
    ax3d.zaxis.pane.set_edgecolor("#333355")
    ax3d.legend(loc="upper left", fontsize=7, facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white")

    ax_xy.set_xlabel(["X", "Y", "Z"][h0], color="white")
    ax_xy.set_ylabel(["X", "Y", "Z"][h1], color="white")
    ax_xy.set_title(f"Top view ({['X','Y','Z'][h0]}-{['X','Y','Z'][h1]})", color="white", fontsize=10)

    ax_xz.set_xlabel(["X", "Y", "Z"][h0], color="white")
    ax_xz.set_ylabel(f"{['X','Y','Z'][up_axis]} (world)", color="white")
    ax_xz.set_title(f"Front view ({['X','Y','Z'][h0]}-up)", color="white", fontsize=10)
    ax_xz.invert_yaxis()
    ax_xz.legend(fontsize=6, facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white")

    ax_yz.set_xlabel(["X", "Y", "Z"][h1], color="white")
    ax_yz.set_ylabel(f"{['X','Y','Z'][up_axis]} (world)", color="white")
    ax_yz.set_title(f"Side view ({['X','Y','Z'][h1]}-up)", color="white", fontsize=10)
    ax_yz.invert_yaxis()

    delta_text = ""
    if gstats is not None:
        delta_up = float(gstats["median"][up_axis] - center_np[up_axis])
        delta_text = f"  |  median-center Δup={delta_up:+.4f}"
    fig.suptitle(
        f"Region camera layout  |  up={up_axis_label}  |  center={center_np.tolist()}{delta_text}\n"
        f"Lines/arrows = view direction, x = look-at targets per camera",
        color="white",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved camera positions → {json_path}")
    print(f"Saved visualization   → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize region ring cameras against a Gaussian chunk.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "outputs/garden/aerial_regions_table_bev/regions/region_000/region_camera_config.yaml"
        ),
        help="Region camera config YAML (stage1.camera_gen)",
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=Path("outputs/garden/aerial_regions_table_bev/regions/region_000/chunk.ply"),
        help="Region chunk PLY for Gaussian overlay (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/garden/aerial_regions_table_bev/regions/region_000/camera_3d_visualization.png"
        ),
        help="Output PNG path",
    )
    parser.add_argument("--max-gauss-points", type=int, default=30000, help="Subsample Gaussian points for display")
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=1.2,
        help="Length of view-direction arrows in world units",
    )
    parser.add_argument("--no-ply", action="store_true", help="Skip Gaussian overlay even if PLY exists")
    args = parser.parse_args()

    ply_path = None if args.no_ply else args.ply
    visualize_region_cameras(
        config_path=args.config,
        ply_path=ply_path,
        output_path=args.output,
        max_gauss_points=args.max_gauss_points,
        arrow_scale=args.arrow_scale,
    )


if __name__ == "__main__":
    main()
