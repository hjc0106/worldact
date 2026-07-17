#!/usr/bin/env python3
"""Compare generated camera positions from two region camera configs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    _horizontal_axes,
    _ring_specs_from_cfg,
    generate_camera_trajectory,
    parse_up_axis,
)


def _ring_labels(cam_cfg) -> list[str]:
    rings = _ring_specs_from_cfg(cam_cfg)
    if not rings:
        return ["single_ring"] * int(getattr(cam_cfg, "num_cameras", 1))
    names = [ring.name for ring in rings]
    counts = [max(1, int(ring.num_cameras)) for ring in rings]
    labels: list[str] = []
    for yaw_idx in range(max(counts)):
        for name, count in zip(names, counts):
            if yaw_idx < count:
                labels.append(name)
    return labels


def _load_camera_positions(config_path: Path, means: torch.Tensor) -> dict[str, object]:
    cfg = load_config(config_path)
    cam_cfg = cfg.stage1.camera_gen
    center = torch.tensor(list(getattr(cam_cfg, "center")), dtype=torch.float32)
    up_axis, up_sign = parse_up_axis(getattr(cam_cfg, "up_axis", "z"))
    cameras = generate_camera_trajectory(means, cam_cfg, ignore_config_center=False)
    labels = _ring_labels(cam_cfg)
    positions = np.asarray([cam.c2w[:3, 3].detach().cpu().numpy() for cam in cameras], dtype=np.float64)
    rings = _ring_specs_from_cfg(cam_cfg)
    return {
        "path": str(config_path),
        "name": config_path.stem,
        "center": center.detach().cpu().numpy().astype(np.float64),
        "up_axis": up_axis,
        "up_sign": up_sign,
        "positions": positions,
        "labels": labels,
        "rings": [
            {
                "name": ring.name,
                "num_cameras": int(ring.num_cameras),
                "orbit_radius": float(ring.orbit_radius or 0.0),
                "height_offset": float(ring.height_offset),
            }
            for ring in rings
        ],
    }


def _stats(item: dict[str, object]) -> dict[str, object]:
    positions = item["positions"]
    assert isinstance(positions, np.ndarray)
    return {
        "config": item["path"],
        "num_cameras": int(positions.shape[0]),
        "center": item["center"].tolist(),
        "axis_min": positions.min(axis=0).tolist(),
        "axis_max": positions.max(axis=0).tolist(),
        "axis_mean": positions.mean(axis=0).tolist(),
        "axis_std": positions.std(axis=0).tolist(),
        "rings": item["rings"],
    }


def _style(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for spine in getattr(ax, "spines", {}).values():
        spine.set_alpha(0.35)


def _plot_compare(items: list[dict[str, object]], output: Path) -> None:
    colors = ["#1f77b4", "#d62728"]
    fig = plt.figure(figsize=(18, 14))
    ax3d = fig.add_subplot(3, 3, 1, projection="3d")
    ax_xy = fig.add_subplot(3, 3, 2)
    ax_xz = fig.add_subplot(3, 3, 3)
    ax_yz = fig.add_subplot(3, 3, 4)
    hist_axes = [fig.add_subplot(3, 3, 5), fig.add_subplot(3, 3, 6), fig.add_subplot(3, 3, 8)]
    text_ax = fig.add_subplot(3, 3, 7)
    text_ax.axis("off")

    summary_lines = []
    for idx, item in enumerate(items):
        color = colors[idx % len(colors)]
        name = str(item["name"])
        pos = item["positions"]
        center = item["center"]
        labels = item["labels"]
        up_axis = int(item["up_axis"])
        h0, h1 = _horizontal_axes(up_axis)
        assert isinstance(pos, np.ndarray)
        assert isinstance(center, np.ndarray)
        unique_rings = list(dict.fromkeys(str(label) for label in labels))

        ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=34, alpha=0.78, color=color, label=f"{name} ({len(pos)})")
        ax3d.scatter([center[0]], [center[1]], [center[2]], marker="*", s=130, color=color, edgecolor="black")
        ax_xy.scatter(pos[:, 0], pos[:, 1], s=32, alpha=0.72, color=color, label=name)
        ax_xz.scatter(pos[:, 0], pos[:, 2], s=32, alpha=0.72, color=color, label=name)
        ax_yz.scatter(pos[:, 1], pos[:, 2], s=32, alpha=0.72, color=color, label=name)

        for axis_idx, ax in enumerate(hist_axes):
            ax.hist(pos[:, axis_idx], bins=min(18, max(5, len(pos) // 2)), alpha=0.42, color=color, label=name)
            ax.axvline(center[axis_idx], color=color, linestyle="--", linewidth=1.2, alpha=0.8)

        mins = pos.min(axis=0)
        maxs = pos.max(axis=0)
        summary_lines.append(
            f"{name}\n"
            f"  cameras={len(pos)}, rings={', '.join(unique_rings)}\n"
            f"  X [{mins[0]:.3f}, {maxs[0]:.3f}], span={maxs[0] - mins[0]:.3f}\n"
            f"  Y [{mins[1]:.3f}, {maxs[1]:.3f}], span={maxs[1] - mins[1]:.3f}\n"
            f"  Z [{mins[2]:.3f}, {maxs[2]:.3f}], span={maxs[2] - mins[2]:.3f}\n"
            f"  horizontal axes={['X','Y','Z'][h0]}-{['X','Y','Z'][h1]}, up={['X','Y','Z'][up_axis]}"
        )

    ax3d.set_title("3D camera positions")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend(fontsize=8)

    for ax, title, xlabel, ylabel in [
        (ax_xy, "X-Y projection", "X", "Y"),
        (ax_xz, "X-Z projection", "X", "Z"),
        (ax_yz, "Y-Z projection", "Y", "Z"),
    ]:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        _style(ax)

    for axis_idx, ax in enumerate(hist_axes):
        axis_name = "XYZ"[axis_idx]
        ax.set_title(f"{axis_name} distribution")
        ax.set_xlabel(axis_name)
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
        _style(ax)

    text_ax.text(0.0, 1.0, "\n\n".join(summary_lines), va="top", ha="left", family="monospace", fontsize=8.5)
    fig.suptitle("Region camera config position comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True, help="Config path. Pass exactly two.")
    parser.add_argument("--ply", type=Path, default=None, help="Optional region chunk PLY used for trajectory radius bounds.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    if len(args.config) != 2:
        raise SystemExit("--config must be passed exactly twice")

    if args.ply is not None:
        scene = GaussianScene.from_ply(str(args.ply))
        means = scene.params.means.detach().cpu()
    else:
        means = torch.zeros(1, 3)

    items = [_load_camera_positions(path, means) for path in args.config]
    _plot_compare(items, args.output)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump([_stats(item) for item in items], f, indent=2)
    print(f"Saved comparison visualization -> {args.output}")
    if args.json_output is not None:
        print(f"Saved comparison stats -> {args.json_output}")


if __name__ == "__main__":
    main()
