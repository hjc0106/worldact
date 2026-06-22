from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from worldact.data.poses import CameraPose


def sample_keyframes(cameras: list[CameraPose], num_frames: int) -> list[int]:
    if not cameras:
        return []
    if num_frames >= len(cameras):
        return list(range(len(cameras)))
    step = len(cameras) / num_frames
    return [int(i * step) for i in range(num_frames)]


def generate_interior_cameras(
    center: torch.Tensor,
    num_cameras: int,
    image_width: int,
    image_height: int,
    fov_deg: float,
    pitch_deg: float,
    radius_jitter: float,
    height_offset: float,
    orbit_radius: Optional[float] = None,
    look_at_center: bool = False,
) -> list[CameraPose]:
    fov = math.radians(fov_deg)
    fy = image_height / (2.0 * math.tan(fov / 2.0))
    fx = fy
    K = torch.tensor(
        [[fx, 0.0, image_width / 2.0], [0.0, fy, image_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    pitch = math.radians(pitch_deg)
    cameras: list[CameraPose] = []
    num_center = max(1, num_cameras // 2)
    num_offset = max(1, num_cameras - num_center)
    world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    for i in range(num_cameras):
        if look_at_center:
            radius = float(orbit_radius if orbit_radius is not None else radius_jitter)
            yaw = 2.0 * math.pi * i / num_cameras
            cam_pos = torch.tensor(
                [
                    center[0] + radius * math.cos(yaw),
                    center[1] + radius * math.sin(yaw),
                    center[2] + height_offset,
                ],
                dtype=torch.float32,
            )
            forward = F.normalize(center.to(dtype=torch.float32) - cam_pos, dim=0)
        elif i < num_center:
            yaw = 2.0 * math.pi * i / num_center
            cam_pos = torch.tensor([center[0], center[1], center[2] + height_offset], dtype=torch.float32)
            forward = torch.tensor(
                [math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)],
                dtype=torch.float32,
            )
        else:
            j = i - num_center
            yaw = 2.0 * math.pi * j / num_offset
            cam_pos = torch.tensor(
                [
                    center[0] + radius_jitter * math.cos(yaw),
                    center[1] + radius_jitter * math.sin(yaw),
                    center[2] + height_offset,
                ],
                dtype=torch.float32,
            )
            forward = torch.tensor(
                [math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)],
                dtype=torch.float32,
            )
        forward = F.normalize(forward, dim=0)
        right = F.normalize(torch.cross(forward, world_up, dim=0), dim=0)
        up = torch.cross(right, forward, dim=0)
        R = torch.stack([right, -up, forward], dim=0)
        t = -R @ cam_pos
        extrinsic = torch.eye(4, dtype=torch.float32)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        cameras.append(CameraPose(extrinsic=extrinsic, intrinsic=K.clone(), width=image_width, height=image_height))
    return cameras
