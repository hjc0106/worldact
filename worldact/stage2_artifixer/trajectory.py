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
    points_local: torch.Tensor | None = None,
    up_hint: torch.Tensor | None = None,
) -> TrajectoryDocument:
    """Create ~32 look-at-center cameras; drop those within min angle of an anchor."""
    size = int(cfg.image_size)
    fov = math.radians(cfg.fov_deg)
    fy = size / (2.0 * math.tan(fov / 2.0))
    fx = fy
    cx = (size - 1) * 0.5
    cy = (size - 1) * 0.5

    center, half_extent, bbox_z_min = _object_geometry(
        points_local=points_local,
        local_radius=local_radius,
        quantile=cfg.quantile,
        look_at_robust_center=cfg.look_at_robust_center,
    )
    min_fit_distance = _projection_fit_distance(
        half_extent=half_extent,
        fov_deg=cfg.fov_deg,
        edge_margin=cfg.edge_margin,
    )
    up_axis = resolve_up_axis(cfg, points_local=points_local, up_hint=up_hint)
    front_axis = resolve_front_axis(cfg, points_local=points_local, up_axis=up_axis)

    if cfg.sphere_ring:
        return _build_sphere_ring_trajectory(
            cfg,
            center=center,
            half_extent=half_extent,
            bbox_z_min=bbox_z_min,
            min_fit_distance=min_fit_distance,
            front_axis=front_axis,
            up_axis=up_axis,
            points_local=points_local,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            size=size,
            local_radius=local_radius,
        )

    specs = _trajectory_specs(cfg)
    anchor_dirs = [_c2w_dir(a.transform_matrix, center=center) for a in anchors.anchors]
    min_cos = math.cos(math.radians(cfg.target_anchor_min_angle_deg))

    frames: list[TrajectoryFrame] = []
    for kind, azimuth, elevation, distance_scale in specs:
        elevation = max(elevation, math.radians(cfg.min_elevation_deg))
        distance = max(
            float(distance_scale) * max(local_radius, 1.0),
            min_fit_distance,
            1.5,
        )
        c2w = look_at_c2w(
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            target=center,
            front_axis=front_axis,
        )
        if not _passes_hard_constraints(
            c2w,
            center=center,
            bbox_z_min=bbox_z_min,
            min_elevation_deg=cfg.min_elevation_deg,
        ):
            continue
        direction = (c2w[:3, 3] - center) / (c2w[:3, 3] - center).norm().clamp_min(1e-6)
        if cfg.target_anchor_min_angle_deg > 0 and any(
            float(torch.dot(direction, ad)) >= min_cos for ad in anchor_dirs
        ):
            continue
        cam_dist = float((c2w[:3, 3] - center).norm().item())
        if cam_dist < max(local_radius, 1.0) * 1.05:
            continue
        if not _object_projects_inside(
            c2w,
            fx,
            fy,
            cx,
            cy,
            size,
            local_radius=local_radius,
            center=center,
            half_extent=half_extent,
            edge_margin=0.0,
            margin=2.0,
        ):
            continue
        frames.append(
            TrajectoryFrame(
                file_path=f"{len(frames):05d}.png",
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

    if cfg.sort_trajectory_smooth and frames:
        frames = _sort_frames_smooth(frames, center=center)
        for i, frame in enumerate(frames):
            frame.file_path = f"{i:05d}.png"

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


def resolve_up_axis(
    cfg: PrepareConfig,
    *,
    points_local: torch.Tensor | None = None,
    up_hint: torch.Tensor | None = None,
    stage_dir: str | None = None,
    local_to_world: torch.Tensor | None = None,
    camera_centers_local: torch.Tensor | None = None,
    opacities: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    return_estimate: bool = False,
):
    """Vehicle upright axis.

    Priority:
      1. ``cfg.up_axis`` manual override
      2. Scene ground plane from Stage-1 ``stage_dir`` (shared gravity)
      3. Robust PCA + support-surface scoring on the object
      4. ``up_hint`` (e.g. mean image-up of Stage-1 anchors)
      5. +Z fallback

    For cars, PCA order is typically length > width > height, so the *smallest*
    eigenvector is upright — not the mid-axis most aligned with +Z.
    """
    from worldact.stage2_artifixer.up_axis import estimate_up_axis

    pts_np = None
    if points_local is not None and points_local.numel() >= 9:
        pts_np = points_local.detach().float().cpu().numpy().reshape(-1, 3)
    hint_np = None
    if up_hint is not None and float(up_hint.float().norm()) > 1e-6:
        hint_np = up_hint.detach().float().cpu().numpy().reshape(3)
    l2w_np = None
    if local_to_world is not None:
        l2w_np = local_to_world.detach().float().cpu().numpy()
    cams_np = None
    if camera_centers_local is not None and camera_centers_local.numel() > 0:
        cams_np = camera_centers_local.detach().float().cpu().numpy().reshape(-1, 3)
    opac_np = None if opacities is None else opacities.detach().float().cpu().numpy().reshape(-1)
    scales_np = None if scales is None else scales.detach().float().cpu().numpy().reshape(-1, 3)

    estimate = estimate_up_axis(
        stage_dir=stage_dir or getattr(cfg, "stage_dir", None),
        points_local=pts_np,
        opacities=opac_np,
        scales=scales_np,
        camera_centers_local=cams_np,
        local_to_world=l2w_np,
        manual_up=cfg.up_axis,
        camera_up_hint=hint_np,
        min_confidence=float(getattr(cfg, "min_up_confidence", 0.45)),
    )
    up = torch.tensor(estimate.up_axis, dtype=torch.float32)
    up = up / up.norm().clamp_min(1e-6)
    if return_estimate:
        return up, estimate
    return up


def resolve_front_axis(
    cfg: PrepareConfig,
    *,
    points_local: torch.Tensor | None = None,
    up_axis: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return a unit front axis in local coords, or None for legacy world-Z up.

    When ``front_yaw_deg`` / ``front_axis`` are unset but points are available,
    use the PCA length axis projected onto the vehicle equator (so azimuth 0
    faces along the vehicle, not raw +X).
    """
    if cfg.front_axis is not None:
        axis = torch.tensor(list(cfg.front_axis), dtype=torch.float32)
        return axis / axis.norm().clamp_min(1e-6)
    if cfg.front_yaw_deg is not None:
        yaw = math.radians(float(cfg.front_yaw_deg))
        axis = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], dtype=torch.float32)
        # If we know up, project yaw-defined front onto the equator so it stays
        # horizontal in the vehicle frame.
        if up_axis is not None:
            up = up_axis.float().reshape(3)
            up = up / up.norm().clamp_min(1e-6)
            axis = axis - torch.dot(axis, up) * up
            if float(axis.norm()) < 1e-4:
                return None
        return axis / axis.norm().clamp_min(1e-6)
    if points_local is not None and points_local.numel() >= 9:
        pts = points_local.detach().float().reshape(-1, 3)
        centered = pts - pts.mean(dim=0)
        cov = centered.T @ centered / max(pts.shape[0], 1)
        _evals, evecs = torch.linalg.eigh(cov)
        length = evecs[:, -1]
        up = up_axis if up_axis is not None else torch.tensor([0.0, 0.0, 1.0])
        up = up.float().reshape(3)
        up = up / up.norm().clamp_min(1e-6)
        front = length - torch.dot(length, up) * up
        if float(front.norm()) < 1e-4:
            return None
        front = front / front.norm().clamp_min(1e-6)
        # Prefer the end with greater extent / taller mass as "front" (+cab).
        if float(torch.dot(front, torch.tensor([1.0, 0.0, 0.0]))) < 0:
            front = -front
        return front
    return None


def estimate_up_from_opengl_frames(
    frames: list[dict],
    *,
    indices: list[int] | None = None,
) -> torch.Tensor | None:
    """Mean image-up from prepared OpenGL frames (anchor cameras)."""
    from data_processing.worldact_object_inputs import opengl_c2w_to_worldact_c2w

    idxs = indices if indices is not None else list(range(len(frames)))
    ups: list[torch.Tensor] = []
    for i in idxs:
        c2w = torch.as_tensor(opengl_c2w_to_worldact_c2w(frames[int(i)]["transform_matrix"]), dtype=torch.float32)
        ups.append(-c2w[:3, 1])
    if not ups:
        return None
    mean = torch.stack(ups, dim=0).mean(dim=0)
    if float(mean.norm()) < 1e-6:
        return None
    return mean / mean.norm().clamp_min(1e-6)


def circumradius(
    points_local: torch.Tensor | None,
    *,
    center: torch.Tensor,
    quantile: tuple[float, float],
    fallback: float,
) -> float:
    """Robust circumsphere radius about ``center`` (quantile of point distances)."""
    if points_local is None or points_local.numel() == 0:
        return max(float(fallback), 1.0)
    pts = points_local.detach().float().reshape(-1, 3)
    dist = (pts - center.reshape(1, 3)).norm(dim=1)
    q_hi = float(quantile[1])
    return float(torch.quantile(dist, q_hi).clamp_min(fallback).item())


def _build_sphere_ring_trajectory(
    cfg: PrepareConfig,
    *,
    center: torch.Tensor,
    half_extent: float,
    bbox_z_min: float,
    min_fit_distance: float,
    front_axis: torch.Tensor | None,
    up_axis: torch.Tensor,
    points_local: torch.Tensor | None,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    size: int,
    local_radius: float,
) -> TrajectoryDocument:
    """Place cameras on a slightly elevated planar ring of the (scaled) circumsphere."""
    up = up_axis / up_axis.norm().clamp_min(1e-6)
    if front_axis is None:
        # Default front: project +X onto the equator; fall back to +Y.
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
    # Re-orthogonalize front against right/up.
    front_h = torch.linalg.cross(right, up)
    front_h = front_h / front_h.norm().clamp_min(1e-6)

    R = circumradius(
        points_local,
        center=center,
        quantile=cfg.quantile,
        fallback=max(half_extent, local_radius),
    )
    # Concentric sphere: on/near circumsphere, but never closer than FOV fit.
    cam_radius = max(float(cfg.circumradius_scale) * R, float(min_fit_distance), 1.5)
    elev = math.radians(float(cfg.ring_elevation_deg))
    elev = max(elev, math.radians(float(cfg.min_elevation_deg)))
    look_at = center + float(cfg.look_at_up_bias) * R * up

    n = int(cfg.target_count)
    frames: list[TrajectoryFrame] = []
    for i in range(n):
        az = 2.0 * math.pi * i / n
        offset = cam_radius * (
            math.cos(elev) * math.cos(az) * front_h
            + math.cos(elev) * math.sin(az) * right
            + math.sin(elev) * up
        )
        eye = center + offset
        # Reject any sample that falls below the vehicle equator (underside).
        if float(torch.dot(offset, up)) <= 0.0:
            continue
        c2w = look_at_eye_c2w(eye=eye, target=look_at, up=up)
        if float(c2w[2, 3].item()) <= float(bbox_z_min) and float(torch.dot(offset, up)) < 1e-3:
            continue
        frames.append(
            TrajectoryFrame(
                file_path=f"{len(frames):05d}.png",
                transform_matrix=c2w.tolist(),
                fl_x=fx,
                fl_y=fy,
                cx=cx,
                cy=cy,
                w=size,
                h=size,
                kind="horizontal",
            )
        )

    if len(frames) < max(1, min(4, cfg.target_count)):
        raise ValueError(f"sphere ring only produced {len(frames)} frames; check up/front axes")

    return TrajectoryDocument(
        schema_version=SCHEMA_VERSION,
        camera_model="OPENCV",
        fl_x=fx,
        fl_y=fy,
        cx=cx,
        cy=cy,
        w=size,
        h=size,
        frames=frames[: cfg.target_count],
    )


def look_at_eye_c2w(
    *,
    eye: torch.Tensor,
    target: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """OpenCV look-at from an explicit eye position with a preferred up direction."""
    eye = eye.float().reshape(3)
    target = target.float().reshape(3)
    up = up.float().reshape(3)
    forward = target - eye
    forward = forward / forward.norm().clamp_min(1e-6)
    # Keep image-up aligned with the projection of ``up``.
    projected = up - torch.dot(up, forward) * forward
    if float(projected.norm()) < 1e-4:
        fallback = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        projected = fallback - torch.dot(fallback, forward) * forward
    true_up = projected / projected.norm().clamp_min(1e-6)
    right = torch.linalg.cross(forward, true_up)
    right = right / right.norm().clamp_min(1e-6)
    true_up = torch.linalg.cross(right, forward)
    true_up = true_up / true_up.norm().clamp_min(1e-6)
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = -true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def look_at_c2w(
    *,
    azimuth: float,
    elevation: float,
    distance: float,
    target: torch.Tensor | None = None,
    up: torch.Tensor | None = None,
    front_axis: torch.Tensor | None = None,
) -> torch.Tensor:
    """Camera at spherical coords looking at target; OpenCV-ish +Y down / +Z forward via W2C.

    When ``front_axis`` is provided, roll is chosen so the projection of that
    axis onto the image plane aligns with image-up (vehicle nose toward the
    top of the frame). Near-singular views fall back to a fixed reference axis.
    """
    if target is None:
        target = torch.zeros(3, dtype=torch.float32)
    else:
        target = target.float().reshape(3)
    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)
    eye = target + torch.tensor([x, y, z], dtype=torch.float32)
    forward = target - eye
    forward = forward / forward.norm().clamp_min(1e-6)

    if front_axis is not None:
        true_up = _image_up_from_front_axis(forward, front_axis.float().reshape(3))
    else:
        if up is None:
            up = torch.tensor([0.0, 0.0, 1.0])
        right = torch.linalg.cross(forward, up.float())
        if float(right.norm()) < 1e-6:
            right = torch.linalg.cross(forward, torch.tensor([0.0, 1.0, 0.0]))
        right = right / right.norm().clamp_min(1e-6)
        true_up = torch.linalg.cross(right, forward)
        true_up = true_up / true_up.norm().clamp_min(1e-6)

    right = torch.linalg.cross(forward, true_up)
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


def camera_roll_error_deg(c2w: torch.Tensor, front_axis: torch.Tensor) -> float:
    """Absolute angle (deg) between desired image-up and camera image-up."""
    forward = c2w[:3, 2]
    desired = _image_up_from_front_axis(forward, front_axis.float().reshape(3))
    actual = -c2w[:3, 1]
    actual = actual / actual.norm().clamp_min(1e-6)
    cos = float(torch.clamp(torch.dot(desired, actual), -1.0, 1.0).item())
    return math.degrees(math.acos(cos))


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


def _image_up_from_front_axis(forward: torch.Tensor, front_axis: torch.Tensor) -> torch.Tensor:
    """Project front_axis onto the image plane; fall back near singularities."""
    forward = forward / forward.norm().clamp_min(1e-6)
    front = front_axis / front_axis.norm().clamp_min(1e-6)
    projected = front - torch.dot(front, forward) * forward
    if float(projected.norm()) < 1e-3:
        # Looking along the front axis: use world-Z, else a side axis.
        ref = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        projected = ref - torch.dot(ref, forward) * forward
        if float(projected.norm()) < 1e-3:
            side = torch.tensor([-front[1], front[0], 0.0], dtype=torch.float32)
            if float(side.norm()) < 1e-6:
                side = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
            projected = side - torch.dot(side, forward) * forward
    return projected / projected.norm().clamp_min(1e-6)


def _trajectory_specs(cfg: PrepareConfig) -> list[tuple[str, float, float, float]]:
    """Return (kind, azimuth_rad, elevation_rad, distance_scale) specs."""
    specs: list[tuple[str, float, float, float]] = []
    scale_lo, scale_hi = _distance_scale_bounds(cfg)
    front_yaw = 0.0 if cfg.front_yaw_deg is None else math.radians(float(cfg.front_yaw_deg))
    if cfg.front_axis is not None and cfg.front_yaw_deg is None:
        axis = torch.tensor(list(cfg.front_axis), dtype=torch.float32)
        front_yaw = math.atan2(float(axis[1]), float(axis[0]))
    half_span = cfg.front_azimuth_half_span_deg

    def _elevations(count: int, fixed_deg: float, elev_range: tuple[float, float] | None) -> list[float]:
        if count <= 0:
            return []
        if elev_range is None:
            return [math.radians(fixed_deg)] * count
        lo, hi = float(elev_range[0]), float(elev_range[1])
        if count == 1:
            return [math.radians(0.5 * (lo + hi))]
        return [math.radians(lo + (hi - lo) * i / (count - 1)) for i in range(count)]

    def _scales(count: int) -> list[float]:
        if count <= 0:
            return []
        if count == 1 or abs(scale_hi - scale_lo) < 1e-8:
            return [0.5 * (scale_lo + scale_hi)] * count
        return [scale_lo + (scale_hi - scale_lo) * i / (count - 1) for i in range(count)]

    def _azimuths_full(count: int) -> list[float]:
        if count <= 0:
            return []
        return [2.0 * math.pi * i / count for i in range(count)]

    def _azimuths_front_band(count: int, span_scale: float = 1.0) -> list[float]:
        if count <= 0:
            return []
        if half_span is None:
            return [front_yaw + 2.0 * math.pi * i / count for i in range(count)]
        span = math.radians(float(half_span)) * float(span_scale)
        if count == 1:
            return [front_yaw]
        return [front_yaw + (-span + 2.0 * span * i / (count - 1)) for i in range(count)]

    # Legacy rings (disabled by setting counts to 0).
    horiz_elevs = _elevations(cfg.horizontal_targets, 0.0, cfg.horizontal_elevation_range_deg)
    for i, (elev, scale) in enumerate(zip(horiz_elevs, _scales(cfg.horizontal_targets))):
        az = _azimuths_full(cfg.horizontal_targets)[i]
        if half_span is not None and _angle_diff(az, front_yaw) > math.radians(float(half_span)) + 1e-6:
            continue
        specs.append(("horizontal", az, elev, scale))

    high_elevs = _elevations(cfg.high_elevation_targets, cfg.high_elevation_deg, cfg.high_elevation_range_deg)
    for i, (elev, scale) in enumerate(zip(high_elevs, _scales(cfg.high_elevation_targets))):
        az = _azimuths_full(cfg.high_elevation_targets)[i]
        if half_span is not None and _angle_diff(az, front_yaw) > math.radians(float(half_span)) + 1e-6:
            continue
        specs.append(("high", az, elev, scale))

    if cfg.low_elevation_targets > 0:
        for i, scale in enumerate(_scales(cfg.low_elevation_targets)):
            az = _azimuths_full(cfg.low_elevation_targets)[i]
            if half_span is not None and _angle_diff(az, front_yaw) > math.radians(float(half_span)) + 1e-6:
                continue
            specs.append(("low", az, math.radians(cfg.low_elevation_deg), scale))

    # Front-upright rings. Prefer one continuous path when the three front
    # rings alone make up the full target count (smooth elev/az motion).
    front_total = int(cfg.front_targets) + int(cfg.side_targets) + int(cfg.overhead_targets)
    if (
        front_total > 0
        and cfg.horizontal_targets == 0
        and cfg.high_elevation_targets == 0
        and cfg.low_elevation_targets == 0
    ):
        specs.extend(
            _front_upright_path_specs(
                cfg,
                front_yaw=front_yaw,
                half_span_deg=float(half_span if half_span is not None else 110.0),
                scales=_scales(front_total),
            )
        )
        return specs

    front_elevs = _elevations(cfg.front_targets, 35.0, cfg.front_elevation_range_deg)
    for i, (elev, scale) in enumerate(zip(front_elevs, _scales(cfg.front_targets))):
        # Keep the core front band slightly tighter than the absolute half-span.
        az = _azimuths_front_band(cfg.front_targets, span_scale=0.75)[i]
        specs.append(("front", az, elev, scale))

    side_elevs = _elevations(cfg.side_targets, 30.0, cfg.side_elevation_range_deg)
    side_scales = _scales(cfg.side_targets)
    if cfg.side_targets > 0:
        span = math.radians(float(half_span if half_span is not None else 110.0))
        # Place half the side cams on each flank near the span edge.
        left_n = cfg.side_targets // 2
        right_n = cfg.side_targets - left_n
        side_az: list[float] = []
        for i in range(left_n):
            t = (i + 1) / (left_n + 1)
            side_az.append(front_yaw - span * (0.75 + 0.25 * t))
        for i in range(right_n):
            t = (i + 1) / (right_n + 1)
            side_az.append(front_yaw + span * (0.75 + 0.25 * t))
        for az, elev, scale in zip(side_az, side_elevs, side_scales):
            specs.append(("side", az, elev, scale))

    overhead_elevs = _elevations(cfg.overhead_targets, 55.0, cfg.overhead_elevation_range_deg)
    for i, (elev, scale) in enumerate(zip(overhead_elevs, _scales(cfg.overhead_targets))):
        az = _azimuths_front_band(cfg.overhead_targets, span_scale=0.25)[i]
        specs.append(("overhead", az, elev, scale))

    return specs


def _front_upright_path_specs(
    cfg: PrepareConfig,
    *,
    front_yaw: float,
    half_span_deg: float,
    scales: list[float],
) -> list[tuple[str, float, float, float]]:
    """Build 32 cameras as one left→right sweep with slow elevation changes.

    Layout intent:
      - 20 front-hemisphere views (elev 25–45)
      - 8 side-transition views near ±span (elev 25–35)
      - 4 overhead views near the front axis (elev 50–60)
    """
    n_front = int(cfg.front_targets)
    n_side = int(cfg.side_targets)
    n_over = int(cfg.overhead_targets)
    n = n_front + n_side + n_over
    if len(scales) < n:
        scales = scales + [scales[-1] if scales else float(cfg.camera_distance_scale)] * (n - len(scales))

    span = math.radians(float(half_span_deg))
    front_span = 0.75 * span
    specs: list[tuple[str, float, float, float]] = []

    # Side cameras first on the left flank, then front sweep L→R, then right flank,
    # then overhead near center — but elevation is scheduled to change ≤10°/step.
    left_side = n_side // 2
    right_side = n_side - left_side

    samples: list[tuple[str, float, float]] = []
    # left side: az from -span to -front_span
    for i in range(left_side):
        t = i / max(left_side - 1, 1) if left_side > 1 else 0.0
        az = front_yaw + (-span + (span - front_span) * t)
        elev = math.radians(25.0 + 10.0 * t)  # 25 → 35
        samples.append(("side", az, elev))

    # front band: az -front_span → +front_span, elev oscillates 25–45 slowly
    for i in range(n_front):
        t = i / max(n_front - 1, 1)
        az = front_yaw + (-front_span + 2.0 * front_span * t)
        # One slow half-sine keeps adjacent Δelev ≤ 10° for n_front≥20.
        elev = math.radians(25.0 + 20.0 * math.sin(math.pi * t))  # 25→45→25
        samples.append(("front", az, elev))

    # right side
    for i in range(right_side):
        t = i / max(right_side - 1, 1) if right_side > 1 else 0.0
        az = front_yaw + (front_span + (span - front_span) * t)
        elev = math.radians(35.0 - 10.0 * t)  # 35 → 25
        samples.append(("side", az, elev))

    # overhead climb from the previous elevation with ≤10° steps toward 50–60.
    prev_elev = samples[-1][2] if samples else math.radians(30.0)
    over_targets = [
        math.radians(50.0 + 10.0 * (i / max(n_over - 1, 1))) if n_over > 1 else math.radians(55.0)
        for i in range(n_over)
    ]
    for i, target_elev in enumerate(over_targets):
        t = i / max(n_over - 1, 1) if n_over > 1 else 0.0
        az = front_yaw + (-0.15 * span + 0.30 * span * t)
        # Step toward target without exceeding max adjacent Δelev once enforced.
        elev = min(target_elev, prev_elev + math.radians(10.0))
        if i == n_over - 1:
            elev = target_elev
        samples.append(("overhead", az, elev))
        prev_elev = elev

    samples = _enforce_elev_step(samples, max_step_deg=10.0)

    for (kind, az, elev), scale in zip(samples, scales):
        specs.append((kind, az, elev, scale))
    return specs


def _enforce_elev_step(
    samples: list[tuple[str, float, float]],
    *,
    max_step_deg: float,
) -> list[tuple[str, float, float]]:
    """Clamp consecutive elevation jumps by stretching intermediate samples."""
    if len(samples) < 2:
        return samples
    out: list[tuple[str, float, float]] = [samples[0]]
    max_step = math.radians(max_step_deg)
    for kind, az, elev in samples[1:]:
        prev_kind, prev_az, prev_elev = out[-1]
        delta = elev - prev_elev
        if abs(delta) <= max_step + 1e-6:
            out.append((kind, az, elev))
            continue
        # Walk elevation toward the target in max_step increments while
        # interpolating azimuth; keep the original kind on the final sample.
        steps = int(math.ceil(abs(delta) / max_step))
        for s in range(1, steps + 1):
            t = s / steps
            mid_az = prev_az + (az - prev_az) * t
            mid_elev = prev_elev + delta * t
            mid_kind = kind if s == steps else prev_kind
            out.append((mid_kind, mid_az, mid_elev))
        # Trim extras later if we exceeded budget — for now keep fidelity to
        # the elev constraint; caller requests exact target_count via filtering.
    # If enforcement expanded the list, downsample uniformly back to original N.
    if len(out) == len(samples):
        return out
    idx = np_linspace_indices(len(out), len(samples))
    return [out[i] for i in idx]


def np_linspace_indices(n_src: int, n_dst: int) -> list[int]:
    if n_dst <= 1:
        return [0]
    return [int(round(i * (n_src - 1) / (n_dst - 1))) for i in range(n_dst)]


def _sort_frames_smooth(frames: list[TrajectoryFrame], *, center: torch.Tensor) -> list[TrajectoryFrame]:
    """Order frames by signed azimuth about front, then elevation — a continuous sweep."""

    def _key(frame: TrajectoryFrame) -> tuple[float, float]:
        eye = torch.tensor(frame.transform_matrix, dtype=torch.float32)[:3, 3]
        offset = eye - center
        az = math.atan2(float(offset[1]), float(offset[0]))
        elev = math.asin(float(torch.clamp(offset[2] / offset.norm().clamp_min(1e-6), -1.0, 1.0)))
        return (az, elev)

    return sorted(frames, key=_key)


def _angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


def _distance_scale_bounds(cfg: PrepareConfig) -> tuple[float, float]:
    if cfg.camera_distance_scale_range is None:
        s = float(cfg.camera_distance_scale)
        return s, s
    lo, hi = float(cfg.camera_distance_scale_range[0]), float(cfg.camera_distance_scale_range[1])
    return (lo, hi) if lo <= hi else (hi, lo)


def _object_geometry(
    *,
    points_local: torch.Tensor | None,
    local_radius: float,
    quantile: tuple[float, float],
    look_at_robust_center: bool,
) -> tuple[torch.Tensor, float, float]:
    """Return (look-at center, half_extent, bbox_z_min)."""
    radius = max(float(local_radius), 1.0)
    if points_local is None or points_local.numel() == 0:
        center = torch.zeros(3, dtype=torch.float32)
        half_extent = radius * math.sqrt(3.0)
        return center, half_extent, float(-radius)

    pts = points_local.detach().float().reshape(-1, 3)
    q_lo = float(quantile[0])
    q_hi = float(quantile[1])
    lo = torch.quantile(pts, q_lo, dim=0)
    hi = torch.quantile(pts, q_hi, dim=0)
    robust_center = 0.5 * (lo + hi)
    center = robust_center if look_at_robust_center else torch.zeros(3, dtype=torch.float32)
    half = 0.5 * (hi - lo)
    half_extent = float(half.norm().clamp_min(radius * 0.5).item())
    return center, half_extent, float(lo[2].item())


def _projection_fit_distance(
    *,
    half_extent: float,
    fov_deg: float,
    edge_margin: float,
) -> float:
    """Minimum distance so a sphere of radius half_extent fits with edge margin."""
    fov = math.radians(fov_deg)
    margin = float(max(0.0, min(0.45, edge_margin)))
    usable_tan = math.tan(fov * 0.5) * max(1e-4, 1.0 - 2.0 * margin)
    return float(half_extent) / usable_tan


def _passes_hard_constraints(
    c2w: torch.Tensor,
    *,
    center: torch.Tensor,
    bbox_z_min: float,
    min_elevation_deg: float,
) -> bool:
    eye = c2w[:3, 3]
    offset = eye - center
    dist = float(offset.norm().clamp_min(1e-6).item())
    elevation_deg = math.degrees(math.asin(float(torch.clamp(offset[2] / dist, -1.0, 1.0).item())))
    if elevation_deg < float(min_elevation_deg) - 1e-3:
        return False
    if float(eye[2].item()) <= float(center[2].item()):
        return False
    if float(eye[2].item()) <= float(bbox_z_min):
        return False
    return True


def _c2w_dir(matrix: list[list[float]], center: torch.Tensor | None = None) -> torch.Tensor:
    t = torch.tensor(matrix, dtype=torch.float32)[:3, 3]
    if center is not None:
        t = t - center.float().reshape(3)
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
    center: torch.Tensor | None = None,
    half_extent: float | None = None,
    edge_margin: float = 0.0,
) -> bool:
    """Check object AABB corners project inside the image with positive depth."""
    if center is None:
        center = torch.zeros(3, dtype=torch.float32)
    if half_extent is None:
        r = float(local_radius)
        half = torch.tensor([r, r, r], dtype=torch.float32)
    else:
        r = float(half_extent) / math.sqrt(3.0)
        half = torch.tensor([r, r, r], dtype=torch.float32)
    corners = torch.tensor(
        [[a, b, c] for a in (-1.0, 1.0) for b in (-1.0, 1.0) for c in (-1.0, 1.0)],
        dtype=torch.float32,
    )
    corners = center.reshape(1, 3) + corners * half.reshape(1, 3)
    w2c = torch.linalg.inv(c2w)
    ones = torch.ones(corners.shape[0], 1)
    cam = (w2c @ torch.cat([corners, ones], dim=-1).T).T[:, :3]
    if not bool((cam[:, 2] > 0.05).all()):
        return False
    u = fx * (cam[:, 0] / cam[:, 2]) + cx
    v = fy * (cam[:, 1] / cam[:, 2]) + cy
    border = max(float(margin), float(edge_margin) * float(size))
    return bool(
        (u >= border).all()
        and (v >= border).all()
        and (u <= size - 1 - border).all()
        and (v <= size - 1 - border).all()
    )
