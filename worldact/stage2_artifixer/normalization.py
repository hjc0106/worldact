"""Object-centric world <-> local coordinate transforms for ArtiFixer packages."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from worldact.data.gaussians import GaussianParams, GaussianScene
from worldact.data.poses import CameraPose
from worldact.stage2_artifixer.contracts import SCHEMA_VERSION, NormalizationTransform


def robust_bbox_center_radius(
    means: torch.Tensor,
    quantile: tuple[float, float] = (0.02, 0.98),
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, float]:
    """center = midpoint of robust AABB; radius = max extent / 2."""
    means_f = means.detach().float()
    q_lo, q_hi = quantile
    lo = torch.quantile(means_f, q_lo, dim=0)
    hi = torch.quantile(means_f, q_hi, dim=0)
    center = (lo + hi) * 0.5
    extent = (hi - lo).clamp_min(eps)
    radius = float(extent.max().item() * 0.5)
    radius = max(radius, eps)
    return center, radius


def build_normalization(
    params: GaussianParams,
    quantile: tuple[float, float] = (0.02, 0.98),
) -> NormalizationTransform:
    center, radius = robust_bbox_center_radius(params.means, quantile)
    center_list = [float(x) for x in center.tolist()]
    local_to_world = [
        [radius, 0.0, 0.0, center_list[0]],
        [0.0, radius, 0.0, center_list[1]],
        [0.0, 0.0, radius, center_list[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    inv_r = 1.0 / radius
    world_to_local = [
        [inv_r, 0.0, 0.0, -center_list[0] * inv_r],
        [0.0, inv_r, 0.0, -center_list[1] * inv_r],
        [0.0, 0.0, inv_r, -center_list[2] * inv_r],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return NormalizationTransform(
        schema_version=SCHEMA_VERSION,
        center=center_list,
        radius=radius,
        local_to_world=local_to_world,
        world_to_local=world_to_local,
        quantile=quantile,
        num_gaussians=params.num_gaussians,
    )


def world_to_local_params(params: GaussianParams, norm: NormalizationTransform) -> GaussianParams:
    center = torch.tensor(norm.center, dtype=params.means.dtype, device=params.means.device)
    radius = float(norm.radius)
    means = (params.means - center) / radius
    # PLY stores log-scales; world scale *= radius => log_world = log_local + log(radius)
    # Going world -> local: log_local = log_world - log(radius)
    scales = params.scales - math.log(radius)
    return GaussianParams(
        means=means,
        scales=scales,
        rotations=params.rotations.clone(),
        opacities=params.opacities.clone(),
        colors=params.colors.clone(),
    )


def local_to_world_params(params: GaussianParams, norm: NormalizationTransform) -> GaussianParams:
    center = torch.tensor(norm.center, dtype=params.means.dtype, device=params.means.device)
    radius = float(norm.radius)
    means = params.means * radius + center
    scales = params.scales + math.log(radius)
    return GaussianParams(
        means=means,
        scales=scales,
        rotations=params.rotations.clone(),
        opacities=params.opacities.clone(),
        colors=params.colors.clone(),
    )


def world_to_local_scene(scene: GaussianScene, norm: NormalizationTransform) -> GaussianScene:
    return GaussianScene(
        params=world_to_local_params(scene.params, norm),
        camera_trajectory=[world_to_local_camera(cam, norm) for cam in scene.camera_trajectory],
        metadata=dict(scene.metadata),
    )


def local_to_world_scene(scene: GaussianScene, norm: NormalizationTransform) -> GaussianScene:
    return GaussianScene(
        params=local_to_world_params(scene.params, norm),
        camera_trajectory=[local_to_world_camera(cam, norm) for cam in scene.camera_trajectory],
        metadata=dict(scene.metadata),
    )


def world_to_local_camera(camera: CameraPose, norm: NormalizationTransform) -> CameraPose:
    """Scale/translate c2w translation into local object coordinates; intrinsics unchanged."""
    c2w = camera.c2w.clone()
    center = torch.tensor(norm.center, dtype=c2w.dtype, device=c2w.device)
    radius = float(norm.radius)
    c2w[:3, 3] = (c2w[:3, 3] - center) / radius
    w2c = torch.linalg.inv(c2w)
    return CameraPose(
        extrinsic=w2c,
        intrinsic=camera.intrinsic.clone(),
        width=camera.width,
        height=camera.height,
    )


def local_to_world_camera(camera: CameraPose, norm: NormalizationTransform) -> CameraPose:
    c2w = camera.c2w.clone()
    center = torch.tensor(norm.center, dtype=c2w.dtype, device=c2w.device)
    radius = float(norm.radius)
    c2w[:3, 3] = c2w[:3, 3] * radius + center
    w2c = torch.linalg.inv(c2w)
    return CameraPose(
        extrinsic=w2c,
        intrinsic=camera.intrinsic.clone(),
        width=camera.width,
        height=camera.height,
    )


def c2w_matrix_list(camera: CameraPose) -> list[list[float]]:
    return camera.c2w.detach().cpu().float().tolist()


def camera_from_c2w(
    c2w: Sequence[Sequence[float]] | torch.Tensor,
    intrinsic: Sequence[Sequence[float]] | torch.Tensor,
    width: int,
    height: int,
) -> CameraPose:
    c2w_t = torch.as_tensor(c2w, dtype=torch.float32)
    k = torch.as_tensor(intrinsic, dtype=torch.float32)
    return CameraPose(extrinsic=torch.linalg.inv(c2w_t), intrinsic=k, width=width, height=height)


def position_roundtrip_error(params: GaussianParams, norm: NormalizationTransform) -> float:
    restored = local_to_world_params(world_to_local_params(params, norm), norm)
    return float((restored.means - params.means).abs().max().item())
