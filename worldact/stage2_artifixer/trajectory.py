"""Generate object-centric target camera trajectories in local coordinates."""

from __future__ import annotations

import math

import torch

from worldact.data.poses import CameraPose
from worldact.stage2_artifixer.config import PrepareConfig
from worldact.stage2_artifixer.contracts import (
    SCHEMA_VERSION,
    AnchorsDocument,
    TrajectoryDocument,
    TrajectoryFrame,
)


def build_target_trajectory(
    anchors: AnchorsDocument,
    cfg: PrepareConfig,
    *,
    local_radius: float = 1.0,
) -> TrajectoryDocument:
    """Create ~32 look-at-origin cameras; drop those within min angle of an anchor."""
    size = int(cfg.image_size)
    fov = math.radians(cfg.fov_deg)
    fy = size / (2.0 * math.tan(fov / 2.0))
    fx = fy
    cx = (size - 1) * 0.5
    cy = (size - 1) * 0.5
    # Fit the local AABB corners (radius * sqrt(3)) inside the vertical FOV.
    half_extent = max(local_radius, 1.0) * math.sqrt(3.0)
    distance = max(
        cfg.camera_distance_scale * local_radius,
        half_extent / max(math.tan(fov * 0.5), 1e-4) * 1.15,
        1.5,
    )

    specs: list[tuple[str, float, float]] = []
    for i in range(cfg.horizontal_targets):
        specs.append(("horizontal", 2.0 * math.pi * i / cfg.horizontal_targets, 0.0))
    for i in range(cfg.high_elevation_targets):
        specs.append(
            (
                "high",
                2.0 * math.pi * i / max(cfg.high_elevation_targets, 1),
                math.radians(cfg.high_elevation_deg),
            )
        )
    for i in range(cfg.low_elevation_targets):
        specs.append(
            (
                "low",
                2.0 * math.pi * i / max(cfg.low_elevation_targets, 1),
                math.radians(cfg.low_elevation_deg),
            )
        )

    anchor_dirs = [_c2w_dir(a.transform_matrix) for a in anchors.anchors]
    min_cos = math.cos(math.radians(cfg.target_anchor_min_angle_deg))

    frames: list[TrajectoryFrame] = []
    for kind, azimuth, elevation in specs:
        c2w = look_at_c2w(azimuth=azimuth, elevation=elevation, distance=distance)
        direction = c2w[:3, 3] / c2w[:3, 3].norm().clamp_min(1e-6)
        if any(float(torch.dot(direction, ad)) >= min_cos for ad in anchor_dirs):
            continue
        cam_dist = float(c2w[:3, 3].norm().item())
        if cam_dist < max(local_radius, 1.0) * 1.05:
            continue
        if not _object_projects_inside(c2w, fx, fy, cx, cy, size, local_radius=local_radius):
            continue
        idx = len(frames)
        frames.append(
            TrajectoryFrame(
                file_path=f"{idx:05d}.png",
                transform_matrix=c2w.tolist(),
                fl_x=fx,
                fl_y=fy,
                cx=cx,
                cy=cy,
                w=size,
                h=size,
                kind=kind,  # type: ignore[arg-type]
            )
        )
        if len(frames) >= cfg.target_count:
            break

    if len(frames) < max(1, min(4, cfg.target_count)):
        raise ValueError(f"only generated {len(frames)} target frames; check FOV/distance settings")

    return TrajectoryDocument(
        schema_version=SCHEMA_VERSION,
        camera_model="OPENCV",
        fl_x=fx,
        fl_y=fy,
        cx=cx,
        cy=cy,
        w=size,
        h=size,
        frames=frames,
    )


def look_at_c2w(
    *,
    azimuth: float,
    elevation: float,
    distance: float,
    up: torch.Tensor | None = None,
) -> torch.Tensor:
    """Camera at spherical coords looking at origin; OpenCV-ish +Y down / +Z forward via W2C."""
    if up is None:
        up = torch.tensor([0.0, 0.0, 1.0])
    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)
    eye = torch.tensor([x, y, z], dtype=torch.float32)
    target = torch.zeros(3, dtype=torch.float32)
    forward = target - eye
    forward = forward / forward.norm().clamp_min(1e-6)
    right = torch.linalg.cross(forward, up.float())
    if float(right.norm()) < 1e-6:
        right = torch.linalg.cross(forward, torch.tensor([0.0, 1.0, 0.0]))
    right = right / right.norm().clamp_min(1e-6)
    true_up = torch.linalg.cross(right, forward)
    true_up = true_up / true_up.norm().clamp_min(1e-6)
    # c2w columns: right, -up (OpenCV), forward
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = -true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def trajectory_to_cameras(doc: TrajectoryDocument) -> list[CameraPose]:
    cameras: list[CameraPose] = []
    for frame in doc.frames:
        c2w = torch.tensor(frame.transform_matrix, dtype=torch.float32)
        k = torch.tensor(
            [[frame.fl_x, 0.0, frame.cx], [0.0, frame.fl_y, frame.cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        cameras.append(CameraPose(extrinsic=torch.linalg.inv(c2w), intrinsic=k, width=frame.w, height=frame.h))
    return cameras


def _c2w_dir(matrix: list[list[float]]) -> torch.Tensor:
    t = torch.tensor(matrix, dtype=torch.float32)[:3, 3]
    return t / t.norm().clamp_min(1e-6)


def _object_projects_inside(
    c2w: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    size: int,
    margin: float = 2.0,
    local_radius: float = 1.0,
) -> bool:
    """Check local AABB corners project inside the image with positive depth."""
    r = float(local_radius)
    corners = torch.tensor(
        [[a, b, c] for a in (-r, r) for b in (-r, r) for c in (-r, r)],
        dtype=torch.float32,
    )
    w2c = torch.linalg.inv(c2w)
    ones = torch.ones(corners.shape[0], 1)
    cam = (w2c @ torch.cat([corners, ones], dim=-1).T).T[:, :3]
    if not bool((cam[:, 2] > 0.05).all()):
        return False
    u = fx * (cam[:, 0] / cam[:, 2]) + cx
    v = fy * (cam[:, 1] / cam[:, 2]) + cy
    return bool(
        (u >= -margin).all()
        and (v >= -margin).all()
        and (u <= size - 1 + margin).all()
        and (v <= size - 1 + margin).all()
    )
