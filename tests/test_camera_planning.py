from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from worldact.config import CameraGenConfig, RingSpec
from worldact.rendering.camera import generate_interior_cameras, sample_keyframes
from worldact.rendering.camera_planning import (
    RingCameraSpec,
    compute_scene_frame,
    generate_camera_trajectory,
    generate_multi_ring_cameras,
    generate_ring_cameras,
    parse_up_axis,
    resolve_trajectory_center,
    select_diverse_keyframes,
    select_keyframes,
    yaw_interleave_ring_cameras,
)


def _make_means() -> torch.Tensor:
    xs, ys = torch.meshgrid(torch.linspace(-2, 2, 5), torch.linspace(-2, 2, 5), indexing="ij")
    zs = torch.zeros_like(xs)
    return torch.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)], dim=-1)


def test_compute_scene_frame_extent_and_center() -> None:
    means = _make_means()
    frame = compute_scene_frame(means)
    assert frame.extent_xy > 3.0
    assert frame.center.shape == (3,)
    assert frame.floor_z <= frame.center[2] <= frame.ceiling_z


def test_parse_up_axis_supports_signed_axes() -> None:
    assert parse_up_axis("z") == (2, 1.0)
    assert parse_up_axis("-y") == (1, -1.0)


def test_y_up_ring_uses_xz_orbit_plane() -> None:
    center = torch.tensor([1.0, 2.0, 3.0])
    frame = compute_scene_frame(_make_means(), up_axis="y")
    frame.center = center
    ring = RingCameraSpec(num_cameras=1, orbit_radius=2.0, height_offset=0.5, look_at_center=True)
    cameras = generate_ring_cameras(
        center,
        ring,
        image_width=64,
        image_height=64,
        fov_deg=60.0,
        default_pitch_deg=10.0,
        fallback_radius=1.0,
        frame=frame,
    )
    pos = cameras[0].c2w[:3, 3]
    assert torch.allclose(pos, torch.tensor([3.0, 2.5, 3.0]), atol=1e-5)


def test_resolve_trajectory_center_prefers_explicit_center() -> None:
    means = _make_means()
    cfg = SimpleNamespace(center=[1.0, 2.0, 0.5], auto_center=False)
    center, frame = resolve_trajectory_center(means, cfg)
    assert frame is not None
    assert center.tolist() == [1.0, 2.0, 0.5]


def test_resolve_trajectory_center_auto_center_overrides_explicit() -> None:
    means = _make_means()
    cfg = SimpleNamespace(center=[1.0, 2.0, 0.5], auto_center=True)
    center, frame = resolve_trajectory_center(means, cfg)
    assert frame is not None
    assert center.tolist() != [1.0, 2.0, 0.5]


def test_single_ring_backward_compatible() -> None:
    center = torch.tensor([0.0, 0.0, 0.0])
    cfg = CameraGenConfig(
        trajectory_mode="single_ring",
        num_cameras=8,
        image_width=64,
        image_height=64,
        fov_deg=60.0,
        pitch_deg=10.0,
        radius_jitter=0.4,
        height_offset=0.2,
        orbit_radius=1.0,
        look_at_center=True,
    )
    direct = generate_interior_cameras(
        center=center,
        num_cameras=cfg.num_cameras,
        image_width=cfg.image_width,
        image_height=cfg.image_height,
        fov_deg=cfg.fov_deg,
        pitch_deg=cfg.pitch_deg,
        radius_jitter=cfg.radius_jitter,
        height_offset=cfg.height_offset,
        orbit_radius=cfg.orbit_radius,
        look_at_center=cfg.look_at_center,
    )
    planned = generate_camera_trajectory(_make_means(), cfg)
    assert len(planned) == len(direct) == 8
    for a, b in zip(direct, planned):
        assert torch.allclose(a.extrinsic, b.extrinsic, atol=1e-5)
        assert torch.allclose(a.intrinsic, b.intrinsic, atol=1e-5)


def test_multi_ring_total_count_and_interleaving() -> None:
    center = torch.tensor([0.0, 0.0, 0.0])
    cfg = SimpleNamespace(
        image_width=64,
        image_height=64,
        fov_deg=60.0,
        pitch_deg=10.0,
        radius_jitter=0.5,
        orbit_radius=1.0,
        rings=[
            RingCameraSpec(name="inner", num_cameras=4, orbit_radius=1.0, height_offset=0.3, look_at_center=True),
            RingCameraSpec(name="outer", num_cameras=2, orbit_radius=2.0, height_offset=0.1, look_at_center=True),
        ],
    )
    inner = generate_ring_cameras(
        center,
        cfg.rings[0],
        image_width=64,
        image_height=64,
        fov_deg=60.0,
        default_pitch_deg=10.0,
        fallback_radius=1.0,
    )
    outer = generate_ring_cameras(
        center,
        cfg.rings[1],
        image_width=64,
        image_height=64,
        fov_deg=60.0,
        default_pitch_deg=10.0,
        fallback_radius=1.0,
    )
    interleaved = yaw_interleave_ring_cameras([inner, outer])
    assert len(interleaved) == 6
    assert interleaved[0].c2w[0, 3].item() == inner[0].c2w[0, 3].item()
    assert interleaved[1].c2w[0, 3].item() == outer[0].c2w[0, 3].item()


def test_look_at_center_forward_points_to_center() -> None:
    center = torch.tensor([0.0, 0.0, 0.0])
    ring = RingCameraSpec(num_cameras=1, orbit_radius=1.0, height_offset=0.0, look_at_center=True)
    cameras = generate_ring_cameras(
        center,
        ring,
        image_width=64,
        image_height=64,
        fov_deg=60.0,
        default_pitch_deg=10.0,
        fallback_radius=1.0,
    )
    cam = cameras[0]
    pos = cam.c2w[:3, 3]
    forward = cam.c2w[:3, 2]
    to_center = center - pos
    to_center = to_center / to_center.norm()
    assert torch.dot(forward, to_center).item() > 0.99


def test_look_at_offset_scale_points_outward() -> None:
    center = torch.tensor([0.0, 0.0, 0.0])
    ring = RingCameraSpec(
        num_cameras=1,
        orbit_radius=1.0,
        height_offset=0.0,
        look_at_center=False,
        look_at_offset_scale=0.5,
    )
    cameras = generate_ring_cameras(
        center,
        ring,
        image_width=64,
        image_height=64,
        fov_deg=60.0,
        default_pitch_deg=10.0,
        fallback_radius=1.0,
    )
    cam = cameras[0]
    pos = cam.c2w[:3, 3]
    forward = cam.c2w[:3, 2]
    expected_target = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32)
    to_target = expected_target - pos
    to_target = to_target / to_target.norm()
    assert torch.dot(forward, to_target).item() > 0.99


def test_hybrid_config_generates_expected_camera_count() -> None:
    cfg = CameraGenConfig(
        trajectory_mode="hybrid",
        auto_center=False,
        center=[0.0, 0.0, 0.0],
        image_width=64,
        image_height=64,
        fov_deg=60.0,
        pitch_deg=10.0,
        radius_jitter=0.5,
        rings=[
            RingSpec(name="a", num_cameras=4, orbit_radius=1.0, height_offset=0.2, look_at_center=True),
            RingSpec(name="b", num_cameras=2, orbit_radius=1.5, height_offset=0.1, look_at_center=True),
        ],
    )
    cameras = generate_camera_trajectory(_make_means(), cfg)
    assert len(cameras) == 6


def test_select_diverse_keyframes_covers_multiple_radii() -> None:
    center = torch.tensor([0.0, 0.0, 0.0])
    cfg = SimpleNamespace(
        image_width=64,
        image_height=64,
        fov_deg=60.0,
        pitch_deg=10.0,
        radius_jitter=0.5,
        orbit_radius=1.0,
        rings=[
            RingSpec(name="inner", num_cameras=8, orbit_radius=1.0, height_offset=0.3, look_at_center=True),
            RingSpec(name="outer", num_cameras=8, orbit_radius=2.0, height_offset=0.1, look_at_center=True),
        ],
    )
    cameras = generate_multi_ring_cameras(center, cfg)
    indices = select_diverse_keyframes(cameras, num_frames=4, min_angle_deg=30.0, center=center)
    assert len(indices) == 4
    radii = []
    for idx in indices:
        pos = cameras[idx].c2w[:3, 3]
        radii.append(float(torch.norm(pos[:2]).item()))
    assert max(radii) - min(radii) > 0.5


def test_select_keyframes_uniform_matches_sample_keyframes() -> None:
    center = torch.tensor([0.0, 0.0, 0.0])
    cameras = generate_interior_cameras(
        center=center,
        num_cameras=12,
        image_width=64,
        image_height=64,
        fov_deg=60.0,
        pitch_deg=10.0,
        radius_jitter=0.5,
        height_offset=0.0,
        orbit_radius=1.0,
        look_at_center=True,
    )
    cfg = SimpleNamespace(keyframe_strategy="uniform")
    assert select_keyframes(cameras, 5, cfg, center=center) == sample_keyframes(cameras, 5)


def test_ring_radius_scale_uses_scene_extent() -> None:
    means = _make_means()
    frame = compute_scene_frame(means)
    center = torch.zeros(3)
    ring = RingCameraSpec(num_cameras=1, orbit_radius_scale=0.5, height_offset=0.0, look_at_center=True)
    cameras = generate_ring_cameras(
        center,
        ring,
        image_width=64,
        image_height=64,
        fov_deg=60.0,
        default_pitch_deg=10.0,
        fallback_radius=1.0,
        frame=frame,
    )
    pos = cameras[0].c2w[:3, 3]
    expected_r = frame.clamp_radius(0.5 * frame.extent_xy)
    assert math.isclose(float(torch.norm(pos[:2]).item()), expected_r, rel_tol=1e-4)
