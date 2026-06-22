from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import ObjectMask
from worldact.stage2_object_refinement.common import StageObjectRecord
from worldact.stage2_object_refinement.utils import project_points


@dataclass
class AlignmentResult:
    transform: torch.Tensor
    scale: float
    rotation: torch.Tensor
    translation: torch.Tensor
    stats: dict[str, Any]


def compute_alignment_transform(
    seed_scene: GaussianScene,
    generated_scene: GaussianScene,
    obj_mask: ObjectMask | None,
    record: StageObjectRecord,
    source_scene: GaussianScene,
    *,
    icp_iterations: int = 50,
    icp_tolerance: float = 1e-6,
    icp_max_correspondence_distance: float = 0.1,
    device: str = "cuda",
) -> AlignmentResult:
    target_device = torch.device(device)
    seed_on_device = _ensure_device(seed_scene, target_device)
    gen_on_device = _ensure_device(generated_scene, target_device)
    generated_to_world = _coarse_alignment(seed_on_device, gen_on_device)
    if icp_iterations > 0 and seed_on_device.num_gaussians > 3 and gen_on_device.num_gaussians > 3:
        generated_refined = _apply_transform(gen_on_device, generated_to_world, device=device)
        icp_transform = _icp_alignment(
            seed_on_device,
            generated_refined,
            max_iterations=icp_iterations,
            tolerance=icp_tolerance,
            max_correspondence_distance=icp_max_correspondence_distance,
        )
        generated_to_world = icp_transform @ generated_to_world
    generated_to_world = generated_to_world.to(target_device)
    scale = float(torch.norm(generated_to_world[:3, :3], dim=0).mean().item())
    rotation = generated_to_world[:3, :3] / scale
    translation = generated_to_world[:3, 3]
    return AlignmentResult(
        transform=generated_to_world,
        scale=scale,
        rotation=rotation,
        translation=translation,
        stats={
            "num_seed_gaussians": seed_on_device.num_gaussians,
            "num_generated_gaussians": gen_on_device.num_gaussians,
            "icp_iterations": icp_iterations,
        },
    )


def _ensure_device(scene: GaussianScene, device: torch.device) -> GaussianScene:
    if scene.params.means.device == device:
        return scene
    return GaussianScene(
        params=scene.params.to(device),
        camera_trajectory=[cam.to(device) for cam in scene.camera_trajectory],
        metadata=dict(scene.metadata),
    )


def _coarse_alignment(seed_scene: GaussianScene, generated_scene: GaussianScene) -> torch.Tensor:
    seed_center = seed_scene.params.means.mean(dim=0)
    gen_center = generated_scene.params.means.mean(dim=0)
    seed_centered = seed_scene.params.means - seed_center
    gen_centered = generated_scene.params.means - gen_center
    seed_scale = float(torch.norm(seed_centered, dim=1).quantile(0.9))
    gen_scale = float(torch.norm(gen_centered, dim=1).quantile(0.9))
    scale_factor = seed_scale / max(gen_scale, 1e-8)
    R = _pca_alignment(seed_centered, gen_centered)
    M = torch.eye(4, dtype=seed_scene.params.means.dtype, device=seed_scene.device)
    M[:3, :3] = R * scale_factor
    M[:3, 3] = seed_center - R @ (gen_center * scale_factor)
    return M


def _pca_alignment(source_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    source_cov = (source_points - source_points.mean(0)).T @ (source_points - source_points.mean(0))
    target_cov = (target_points - target_points.mean(0)).T @ (target_points - target_points.mean(0))
    try:
        _, _, source_Vt = torch.svd(source_cov)
        _, _, target_Vt = torch.svd(target_cov)
    except Exception:
        return torch.eye(3, dtype=source_points.dtype, device=source_points.device)
    R = source_Vt @ target_Vt.T
    det = torch.det(R)
    if det < 0:
        source_Vt_adj = source_Vt.clone()
        source_Vt_adj[:, 2] = -source_Vt_adj[:, 2]
        R = source_Vt_adj @ target_Vt.T
    return R


def _icp_alignment(
    source_scene: GaussianScene,
    target_scene: GaussianScene,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float = 0.1,
) -> torch.Tensor:
    src = source_scene.params.means.clone().detach()
    tgt = target_scene.params.means.clone().detach()
    device = src.device
    dtype = src.dtype
    cumulative_transform = torch.eye(4, dtype=dtype, device=device)
    prev_loss = float("inf")
    max_corr_sq = max_correspondence_distance ** 2

    for _ in range(max_iterations):
        diffs = src.unsqueeze(1) - tgt.unsqueeze(0)
        dist_sq = (diffs ** 2).sum(dim=2)
        dist_sq_clamped = dist_sq.clamp(max=max_corr_sq)
        nearest = dist_sq_clamped.argmin(dim=1)
        dists = dist_sq[torch.arange(src.shape[0], device=device), nearest]
        valid = dists < max_correspondence_distance
        if not valid.any():
            break
        src_valid = src[valid]
        tgt_valid = tgt[nearest[valid]]
        loss = float(dists[valid].mean().item())
        if abs(prev_loss - loss) < tolerance:
            break
        prev_loss = loss
        step = _procrustes(src_valid, tgt_valid)
        src = (step[:3, :3] @ src.T + step[:3, 3:4]).T
        cumulative_transform = step @ cumulative_transform

    return cumulative_transform


def _procrustes(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    src_center = source.mean(dim=0)
    tgt_center = target.mean(dim=0)
    src_c = source - src_center
    tgt_c = target - tgt_center
    H = src_c.T @ tgt_c
    try:
        U, _, Vt = torch.svd(H)
    except Exception:
        M = torch.eye(4, dtype=source.dtype, device=source.device)
        M[:3, 3] = tgt_center - src_center
        return M
    R = Vt @ U.T
    det = torch.det(R)
    if det < 0:
        Vt_adj = Vt.clone()
        Vt_adj[:, 2] = -Vt_adj[:, 2]
        R = Vt_adj @ U.T
    t = tgt_center - R @ src_center
    M = torch.eye(4, dtype=source.dtype, device=source.device)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _apply_transform(scene: GaussianScene, transform: torch.Tensor, *, device: str = "cuda") -> GaussianScene:
    params = scene.params.to(device)
    M = transform.to(device=device, dtype=params.means.dtype)
    ones = torch.ones(params.num_gaussians, 1, dtype=params.means.dtype, device=device)
    homo = torch.cat([params.means, ones], dim=1)
    new_means = (homo @ M.T)[:, :3]
    from worldact.data.gaussians import GaussianParams
    new_params = GaussianParams(
        means=new_means,
        scales=params.scales,
        rotations=params.rotations,
        opacities=params.opacities,
        colors=params.colors,
    )
    return GaussianScene(params=new_params, camera_trajectory=list(scene.camera_trajectory), metadata=dict(scene.metadata))


def write_alignment_metadata(result: AlignmentResult, metadata_path: str | torch.Tensor) -> dict[str, Any]:
    transform = result.transform.cpu().tolist()
    metadata = {
        "transform": transform,
        "scale": result.scale,
        "icp_iterations": result.stats.get("icp_iterations", 0),
    }
    if isinstance(metadata_path, str):
        import json
        from pathlib import Path
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    return metadata