from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from worldact.data.gaussians import GaussianParams, GaussianScene
from worldact.data.masks import ObjectMask
from worldact.stage2_object_refinement.common import ObjectRefinementConfig
from worldact.stage2_object_refinement.utils import concat_gaussian_params, project_points


@dataclass
class MergeResult:
    scene: GaussianScene
    stats: dict[str, Any] = field(default_factory=dict)


def merge_generated_object(
    refined_scene: GaussianScene,
    generated_ply: Path | None,
    obj_mask: ObjectMask | None,
    cfg: ObjectRefinementConfig,
    *,
    metadata_path: Path | None = None,
    best_view_index: int = -1,
) -> MergeResult:
    if generated_ply is None or not generated_ply.exists():
        return MergeResult(
            scene=refined_scene,
            stats={
                "mode": "refined_only",
                "refined_gaussians": refined_scene.num_gaussians,
                "generated_gaussians": 0,
                "kept_generated_gaussians": 0,
            },
        )

    generated = GaussianScene.from_ply(generated_ply).to(refined_scene.device) if hasattr(GaussianScene, "to") else GaussianScene.from_ply(generated_ply)
    generated = generated.to(refined_scene.device) if hasattr(generated, "to") else _scene_to(generated, refined_scene.device)
    generated.camera_trajectory = list(refined_scene.camera_trajectory)
    metadata = _load_metadata(metadata_path)
    generated = _apply_metadata_transform(generated, metadata)
    keep = _generated_keep_mask(refined_scene, generated, obj_mask, cfg, best_view_index=best_view_index)
    kept_generated = generated.subset(keep.nonzero(as_tuple=True)[0])
    if kept_generated.num_gaussians == 0:
        return MergeResult(
            scene=refined_scene,
            stats={
                "mode": "refined_only_empty_generated_keep",
                "refined_gaussians": refined_scene.num_gaussians,
                "generated_gaussians": generated.num_gaussians,
                "kept_generated_gaussians": 0,
            },
        )
    blended_params = _blend_overlap_region(
        refined_scene.params, kept_generated.params, float(cfg.blend_distance)
    )
    merged = GaussianScene(
        params=blended_params,
        camera_trajectory=list(refined_scene.camera_trajectory),
        metadata=dict(refined_scene.metadata),
    )
    return MergeResult(
        scene=merged,
        stats={
            "mode": "merged",
            "refined_gaussians": refined_scene.num_gaussians,
            "generated_gaussians": generated.num_gaussians,
            "kept_generated_gaussians": kept_generated.num_gaussians,
            "total_gaussians": merged.num_gaussians,
            "duplicate_distance": float(cfg.duplicate_distance),
            "blend_distance": float(cfg.blend_distance),
        },
    )


def _scene_to(scene: GaussianScene, device: torch.device | str) -> GaussianScene:
    return GaussianScene(scene.params.to(device), [cam.to(device) for cam in scene.camera_trajectory], dict(scene.metadata))


def _load_metadata(metadata_path: Path | None) -> dict[str, Any]:
    if metadata_path is None or not metadata_path.exists():
        return {}
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def _apply_metadata_transform(scene: GaussianScene, metadata: dict[str, Any]) -> GaussianScene:
    transform = metadata.get("transform")
    if transform is None:
        return scene
    matrix = torch.tensor(transform, dtype=scene.params.means.dtype, device=scene.device)
    if matrix.shape != (4, 4):
        return scene
    ones = torch.ones(scene.num_gaussians, 1, dtype=scene.params.means.dtype, device=scene.device)
    homo = torch.cat([scene.params.means, ones], dim=1)
    params = scene.params
    params.means = (homo @ matrix.T)[:, :3]
    return scene


def _generated_keep_mask(
    refined_scene: GaussianScene,
    generated_scene: GaussianScene,
    obj_mask: ObjectMask | None,
    cfg: ObjectRefinementConfig,
    *,
    best_view_index: int,
) -> torch.Tensor:
    keep = _not_duplicate_mask(refined_scene.params.means, generated_scene.params.means, float(cfg.duplicate_distance))
    if obj_mask is None or not obj_mask.masks or not generated_scene.camera_trajectory:
        return keep
    view_idx = best_view_index if best_view_index in obj_mask.masks else next(iter(obj_mask.masks))
    if view_idx >= len(generated_scene.camera_trajectory):
        return keep
    try:
        refined_render = refined_scene.render(refined_scene.camera_trajectory[view_idx])
    except Exception:
        return keep
    valid, px, py, z = project_points(generated_scene, view_idx)
    mask = obj_mask.masks[view_idx].to(generated_scene.device).float() > 0.5
    in_mask = valid & mask[py.clamp(0, mask.shape[0] - 1), px.clamp(0, mask.shape[1] - 1)]
    alpha = refined_render.alpha.to(generated_scene.device)
    depth = refined_render.depth.to(generated_scene.device)
    alpha_at_px = alpha[py.clamp(0, alpha.shape[0] - 1), px.clamp(0, alpha.shape[1] - 1)]
    depth_at_px = depth[py.clamp(0, depth.shape[0] - 1), px.clamp(0, depth.shape[1] - 1)]
    high_alpha = alpha_at_px >= float(cfg.render_alpha_threshold)
    depth_near_object = (z - depth_at_px).abs() <= float(cfg.depth_tolerance) * depth_at_px.abs().clamp(min=0.1)
    at_surface = high_alpha & depth_near_object
    rendered_object_mask = mask.to(alpha.device) & (alpha >= float(cfg.render_alpha_threshold))
    if rendered_object_mask.any():
        object_depths = depth[rendered_object_mask]
        z_min = float(object_depths.min())
        z_max = float(object_depths.max())
        z_margin = max(0.1, (z_max - z_min) * 0.3)
    else:
        z_min = 0.0
        z_max = float("inf")
        z_margin = 0.0
    reasonable_depth = (z >= z_min - z_margin) & (z <= z_max + z_margin)
    in_gap = ~high_alpha & reasonable_depth
    return keep & in_mask & (at_surface | in_gap)


def _not_duplicate_mask(refined_means: torch.Tensor, generated_means: torch.Tensor, distance: float) -> torch.Tensor:
    if distance <= 0.0 or refined_means.numel() == 0 or generated_means.numel() == 0:
        return torch.ones(generated_means.shape[0], dtype=torch.bool, device=generated_means.device)
    result = torch.ones(generated_means.shape[0], dtype=torch.bool, device=generated_means.device)
    chunk = 4096
    r = refined_means.to(generated_means.device)
    for start in range(0, generated_means.shape[0], chunk):
        part = generated_means[start : start + chunk]
        dists = torch.cdist(part, r)
        result[start : start + chunk] = dists.min(dim=1).values > float(distance)
    return result


def _blend_overlap_region(
    refined_params: GaussianParams,
    generated_params: GaussianParams,
    blend_distance: float,
) -> GaussianParams:
    if blend_distance <= 0.0 or refined_params.num_gaussians == 0 or generated_params.num_gaussians == 0:
        return concat_gaussian_params(refined_params, generated_params)

    refined_means = refined_params.means
    generated_means = generated_params.means.to(refined_params.device)
    chunk = 4096
    min_dists = torch.ones(generated_params.num_gaussians, dtype=torch.float32, device=refined_params.device)
    for start in range(0, generated_means.shape[0], chunk):
        part = generated_means[start : start + chunk]
        dists = torch.cdist(part, refined_means)
        min_dists[start : start + chunk] = dists.min(dim=1).values

    blend_weights = (1.0 - (min_dists / blend_distance).clamp(min=0.0, max=1.0)).unsqueeze(1)

    gen_opacity = torch.sigmoid(generated_params.opacities.to(refined_params.device))
    gen_opacity_adjusted = torch.logit(gen_opacity * (1.0 - blend_weights * 0.5) + 1e-6, eps=1e-6)
    gen_scales = generated_params.scales.to(refined_params.device)
    scale_shrink = 1.0 - blend_weights * 0.3
    gen_scales_adjusted = gen_scales + torch.log(scale_shrink.clamp(min=0.1) + 1e-8)

    adjusted_generated = GaussianParams(
        means=generated_params.means.to(refined_params.device),
        scales=gen_scales_adjusted,
        rotations=generated_params.rotations.to(refined_params.device),
        opacities=gen_opacity_adjusted,
        colors=generated_params.colors.to(refined_params.device),
    )
    return concat_gaussian_params(refined_params, adjusted_generated)
