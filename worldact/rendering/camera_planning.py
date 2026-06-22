from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F

from worldact.data.poses import CameraPose
from worldact.rendering.camera import generate_interior_cameras, sample_keyframes


@dataclass
class SceneFrame:
    center: torch.Tensor
    floor_z: float
    ceiling_z: float
    extent_xy: float
    up_axis: int = 2
    up_sign: float = 1.0
    recommended_radii: tuple[float, float, float] = (0.8, 1.5, 2.2)

    def clamp_radius(self, radius: float, *, min_r: float = 0.3, max_r: float | None = None) -> float:
        upper = max_r if max_r is not None else 0.5 * self.extent_xy
        return float(max(min_r, min(radius, upper)))


@dataclass
class RingCameraSpec:
    name: str = "ring"
    num_cameras: int = 8
    orbit_radius: Optional[float] = None
    orbit_radius_scale: Optional[float] = None
    height_offset: float = 0.0
    pitch_deg: Optional[float] = None
    look_at_center: bool = True
    look_at_offset_xy: list[float] = field(default_factory=lambda: [0.0, 0.0])
    look_at_offset_scale: float = 0.0


def parse_up_axis(up_axis: str | int | None) -> tuple[int, float]:
    if up_axis is None:
        return 2, 1.0
    if isinstance(up_axis, int):
        if up_axis not in (0, 1, 2):
            raise ValueError(f"up_axis integer must be 0, 1, or 2, got {up_axis}")
        return int(up_axis), 1.0
    text = str(up_axis).strip().lower()
    sign = -1.0 if text.startswith("-") else 1.0
    axis_name = text[1:] if text.startswith("-") else text
    mapping = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2}
    if axis_name not in mapping:
        raise ValueError(f"Unsupported up_axis {up_axis!r}; expected x/y/z or -x/-y/-z")
    return mapping[axis_name], sign


def _horizontal_axes(up_axis: int) -> tuple[int, int]:
    axes = [0, 1, 2]
    axes.remove(up_axis)
    return axes[0], axes[1]


def _axis_vector(axis: int, sign: float = 1.0) -> torch.Tensor:
    vec = torch.zeros(3, dtype=torch.float32)
    vec[axis] = float(sign)
    return vec


def compute_scene_frame(
    means: torch.Tensor,
    quantile: tuple[float, float] = (0.02, 0.98),
    up_axis: str | int | None = None,
) -> SceneFrame:
    means_cpu = means.detach().cpu().float()
    up_idx, up_sign = parse_up_axis(up_axis)
    h0, h1 = _horizontal_axes(up_idx)
    q_lo, q_hi = quantile
    lo = torch.quantile(means_cpu, q_lo, dim=0)
    hi = torch.quantile(means_cpu, q_hi, dim=0)
    center = (lo + hi) * 0.5
    up_values = means_cpu[:, up_idx] * float(up_sign)
    floor_z = float(torch.quantile(up_values, q_lo).item())
    ceiling_z = float(torch.quantile(up_values, q_hi).item())
    x_range = float(hi[h0] - lo[h0])
    y_range = float(hi[h1] - lo[h1])
    extent_xy = max(x_range, y_range, 1e-6)
    r_inner = max(0.15 * extent_xy, 0.3)
    r_mid = max(0.30 * extent_xy, 0.6)
    r_outer = max(0.45 * extent_xy, 1.0)
    return SceneFrame(
        center=center,
        up_axis=up_idx,
        up_sign=up_sign,
        floor_z=floor_z,
        ceiling_z=ceiling_z,
        extent_xy=extent_xy,
        recommended_radii=(r_inner, r_mid, r_outer),
    )


def resolve_trajectory_center(
    means: torch.Tensor,
    cfg: Any,
    *,
    ignore_config_center: bool = False,
) -> tuple[torch.Tensor, SceneFrame | None]:
    frame = compute_scene_frame(means, up_axis=getattr(cfg, "up_axis", "z"))
    explicit_center = getattr(cfg, "center", None)
    auto_center = bool(getattr(cfg, "auto_center", False))
    if explicit_center is not None and not auto_center and not ignore_config_center:
        center = torch.tensor(list(explicit_center), dtype=torch.float32)
        return center, frame
    return frame.center.clone(), frame


def _make_intrinsic(image_width: int, image_height: int, fov_deg: float) -> torch.Tensor:
    fov = math.radians(fov_deg)
    fy = image_height / (2.0 * math.tan(fov / 2.0))
    fx = fy
    return torch.tensor(
        [[fx, 0.0, image_width / 2.0], [0.0, fy, image_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _camera_pose_from_basis(
    cam_pos: torch.Tensor,
    forward: torch.Tensor,
    intrinsic: torch.Tensor,
    image_width: int,
    image_height: int,
    *,
    up_axis: int = 2,
    up_sign: float = 1.0,
) -> CameraPose:
    world_up = _axis_vector(up_axis, up_sign)
    forward = F.normalize(forward, dim=0)
    right = F.normalize(torch.cross(forward, world_up, dim=0), dim=0)
    up = torch.cross(right, forward, dim=0)
    R = torch.stack([right, -up, forward], dim=0)
    t = -R @ cam_pos
    extrinsic = torch.eye(4, dtype=torch.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return CameraPose(extrinsic=extrinsic, intrinsic=intrinsic.clone(), width=image_width, height=image_height)


def _resolve_ring_radius(ring: RingCameraSpec, frame: SceneFrame | None, fallback: float) -> float:
    if ring.orbit_radius is not None:
        radius = float(ring.orbit_radius)
    elif ring.orbit_radius_scale is not None and frame is not None:
        radius = float(ring.orbit_radius_scale) * frame.extent_xy
    else:
        radius = fallback
    if frame is not None:
        return frame.clamp_radius(radius)
    return radius


def _look_at_target(
    center: torch.Tensor,
    yaw: float,
    orbit_radius: float,
    ring: RingCameraSpec,
    *,
    up_axis: int = 2,
) -> torch.Tensor:
    target = center.to(dtype=torch.float32).clone()
    h0, h1 = _horizontal_axes(up_axis)
    offset_xy = ring.look_at_offset_xy or [0.0, 0.0]
    target[h0] += float(offset_xy[0]) if len(offset_xy) > 0 else 0.0
    target[h1] += float(offset_xy[1]) if len(offset_xy) > 1 else 0.0
    if float(ring.look_at_offset_scale) != 0.0:
        scale = float(ring.look_at_offset_scale) * orbit_radius
        target[h0] += scale * math.cos(yaw)
        target[h1] += scale * math.sin(yaw)
    return target


def _uses_look_at(ring: RingCameraSpec) -> bool:
    offset_xy = ring.look_at_offset_xy or [0.0, 0.0]
    has_xy_offset = any(abs(float(v)) > 1e-8 for v in offset_xy[:2])
    return bool(ring.look_at_center) or float(ring.look_at_offset_scale) != 0.0 or has_xy_offset


def generate_ring_cameras(
    center: torch.Tensor,
    ring: RingCameraSpec,
    *,
    image_width: int,
    image_height: int,
    fov_deg: float,
    default_pitch_deg: float,
    fallback_radius: float,
    frame: SceneFrame | None = None,
) -> list[CameraPose]:
    intrinsic = _make_intrinsic(image_width, image_height, fov_deg)
    orbit_radius = _resolve_ring_radius(ring, frame, fallback_radius)
    pitch_deg = float(ring.pitch_deg if ring.pitch_deg is not None else default_pitch_deg)
    pitch = math.radians(pitch_deg)
    cameras: list[CameraPose] = []
    num_cameras = max(1, int(ring.num_cameras))
    up_idx = int(frame.up_axis) if frame is not None else 2
    up_sign = float(frame.up_sign) if frame is not None else 1.0
    h0, h1 = _horizontal_axes(up_idx)

    for i in range(num_cameras):
        yaw = 2.0 * math.pi * i / num_cameras
        cam_pos = center.to(dtype=torch.float32).clone()
        cam_pos[h0] = center[h0] + orbit_radius * math.cos(yaw)
        cam_pos[h1] = center[h1] + orbit_radius * math.sin(yaw)
        cam_pos[up_idx] = center[up_idx] + up_sign * float(ring.height_offset)
        if _uses_look_at(ring):
            look_at = _look_at_target(center, yaw, orbit_radius, ring, up_axis=up_idx)
            forward = look_at - cam_pos
        else:
            forward = torch.zeros(3, dtype=torch.float32)
            forward[h0] = math.cos(yaw) * math.cos(pitch)
            forward[h1] = math.sin(yaw) * math.cos(pitch)
            forward[up_idx] = -up_sign * math.sin(pitch)
        cameras.append(
            _camera_pose_from_basis(
                cam_pos,
                forward,
                intrinsic,
                image_width,
                image_height,
                up_axis=up_idx,
                up_sign=up_sign,
            )
        )
    return cameras


def yaw_interleave_ring_cameras(ring_cameras: list[list[CameraPose]]) -> list[CameraPose]:
    if not ring_cameras:
        return []
    max_count = max(len(ring) for ring in ring_cameras)
    interleaved: list[CameraPose] = []
    for yaw_idx in range(max_count):
        for ring in ring_cameras:
            if yaw_idx < len(ring):
                interleaved.append(ring[yaw_idx])
    return interleaved


def _ring_specs_from_cfg(cfg: Any) -> list[RingCameraSpec]:
    rings_cfg = getattr(cfg, "rings", None) or []
    specs: list[RingCameraSpec] = []
    for item in rings_cfg:
        if isinstance(item, RingCameraSpec):
            specs.append(item)
            continue
        specs.append(
            RingCameraSpec(
                name=str(getattr(item, "name", "ring")),
                num_cameras=int(getattr(item, "num_cameras", 8)),
                orbit_radius=getattr(item, "orbit_radius", None),
                orbit_radius_scale=getattr(item, "orbit_radius_scale", None),
                height_offset=float(getattr(item, "height_offset", 0.0)),
                pitch_deg=getattr(item, "pitch_deg", None),
                look_at_center=bool(getattr(item, "look_at_center", True)),
                look_at_offset_xy=list(getattr(item, "look_at_offset_xy", [0.0, 0.0])),
                look_at_offset_scale=float(getattr(item, "look_at_offset_scale", 0.0)),
            )
        )
    return specs


def generate_multi_ring_cameras(
    center: torch.Tensor,
    cfg: Any,
    *,
    frame: SceneFrame | None = None,
) -> list[CameraPose]:
    rings = _ring_specs_from_cfg(cfg)
    if not rings:
        raise ValueError("multi_ring/hybrid trajectory requires at least one ring in camera_gen.rings")

    fallback_radius = float(cfg.orbit_radius if cfg.orbit_radius is not None else cfg.radius_jitter)
    ring_lists: list[list[CameraPose]] = []
    for ring in rings:
        ring_lists.append(
            generate_ring_cameras(
                center,
                ring,
                image_width=int(cfg.image_width),
                image_height=int(cfg.image_height),
                fov_deg=float(cfg.fov_deg),
                default_pitch_deg=float(cfg.pitch_deg),
                fallback_radius=fallback_radius,
                frame=frame,
            )
        )
    return yaw_interleave_ring_cameras(ring_lists)


def generate_camera_trajectory(
    means: torch.Tensor,
    cfg: Any,
    *,
    ignore_config_center: bool = False,
) -> list[CameraPose]:
    center, frame = resolve_trajectory_center(means, cfg, ignore_config_center=ignore_config_center)
    mode = str(getattr(cfg, "trajectory_mode", "single_ring"))
    if mode in {"multi_ring", "hybrid"}:
        return generate_multi_ring_cameras(center, cfg, frame=frame)
    return generate_interior_cameras(
        center=center,
        num_cameras=int(cfg.num_cameras),
        image_width=int(cfg.image_width),
        image_height=int(cfg.image_height),
        fov_deg=float(cfg.fov_deg),
        pitch_deg=float(cfg.pitch_deg),
        radius_jitter=float(cfg.radius_jitter),
        height_offset=float(cfg.height_offset),
        orbit_radius=cfg.orbit_radius,
        look_at_center=bool(cfg.look_at_center),
    )


def _angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % (2.0 * math.pi)
    return min(diff, 2.0 * math.pi - diff)


def _camera_pose_features(
    camera: CameraPose,
    center: torch.Tensor,
    *,
    up_axis: int = 2,
    up_sign: float = 1.0,
) -> tuple[float, float, float]:
    pos = camera.c2w[:3, 3].detach().cpu().float()
    center_cpu = center.detach().cpu().float()
    h0, h1 = _horizontal_axes(up_axis)
    dx = float(pos[h0] - center_cpu[h0])
    dy = float(pos[h1] - center_cpu[h1])
    yaw = math.atan2(dy, dx)
    radius = math.hypot(dx, dy)
    height = float((pos[up_axis] - center_cpu[up_axis]) * up_sign)
    return yaw, radius, height


def select_diverse_keyframes(
    cameras: list[CameraPose],
    num_frames: int,
    min_angle_deg: float,
    center: torch.Tensor,
    up_axis: str | int | None = None,
) -> list[int]:
    if not cameras:
        return []
    if num_frames >= len(cameras):
        return list(range(len(cameras)))

    up_idx, up_sign = parse_up_axis(up_axis)
    features = [_camera_pose_features(cam, center, up_axis=up_idx, up_sign=up_sign) for cam in cameras]
    selected = [0]
    remaining = set(range(1, len(cameras)))

    while len(selected) < num_frames and remaining:
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            yaw, radius, height = features[idx]
            min_sep = min(_angle_diff(yaw, features[s][0]) for s in selected)
            ring_bonus = sum(
                1
                for s in selected
                if abs(features[s][1] - radius) > 0.1 or abs(features[s][2] - height) > 0.05
            )
            score = min_sep + 0.05 * ring_bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    if len(selected) < num_frames:
        for idx in range(len(cameras)):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= num_frames:
                break

    return sorted(selected[:num_frames])


def select_keyframes(
    cameras: list[CameraPose],
    num_frames: int,
    cfg: Any,
    *,
    center: torch.Tensor | None = None,
) -> list[int]:
    strategy = str(getattr(cfg, "keyframe_strategy", "uniform"))
    if strategy == "diverse" and center is not None:
        min_angle = float(getattr(cfg, "keyframe_min_angle_deg", 20.0))
        return select_diverse_keyframes(cameras, num_frames, min_angle, center, up_axis=getattr(cfg, "up_axis", "z"))
    return sample_keyframes(cameras, num_frames)


def trajectory_center_for_keyframes(
    means: torch.Tensor,
    cfg: Any,
    *,
    ignore_config_center: bool = False,
) -> torch.Tensor:
    center, _ = resolve_trajectory_center(means, cfg, ignore_config_center=ignore_config_center)
    return center
