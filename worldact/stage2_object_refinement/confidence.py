from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import ObjectMask
from worldact.stage2_object_refinement.common import ObjectRefinementConfig
from worldact.stage2_object_refinement.utils import depth_gate, project_points, radius_components


@dataclass
class ConfidenceResult:
    foreground: torch.Tensor
    background: torch.Tensor
    mask_support_ratio: torch.Tensor
    background_structure: torch.Tensor
    stats: dict[str, Any] = field(default_factory=dict)


def compute_foreground_confidence(
    object_scene: GaussianScene,
    obj_mask: ObjectMask | None,
    cfg: ObjectRefinementConfig,
    *,
    source_scene: GaussianScene | None = None,
) -> ConfidenceResult:
    n = object_scene.num_gaussians
    device = object_scene.device
    if n == 0:
        empty = torch.zeros(0, dtype=torch.float32, device=device)
        return ConfidenceResult(empty, empty, empty, empty, {"input": 0, "valid_views": 0})

    mask_ratio, valid_views = _mask_support_ratio(object_scene, obj_mask, cfg, source_scene=source_scene)
    opacity = torch.sigmoid(object_scene.params.opacities.reshape(-1)).detach().float()
    opacity_score = _normalize01(opacity)
    scale_score = _scale_score(object_scene)
    core_score = _core_score(object_scene)
    component_score, component_stats = _main_component_score(object_scene, cfg)
    background_structure = _background_structure_score(object_scene, mask_ratio, cfg)

    foreground = (
        0.45 * mask_ratio
        + 0.20 * opacity_score
        + 0.15 * core_score
        + 0.10 * scale_score
        + 0.10 * component_score
        - 0.30 * background_structure * (1.0 - mask_ratio)
    ).clamp(0.0, 1.0)
    background = (
        0.45 * (1.0 - mask_ratio)
        + 0.35 * background_structure
        + 0.10 * (1.0 - core_score)
        + 0.10 * (1.0 - opacity_score)
    ).clamp(0.0, 1.0)

    stats = {
        "input": int(n),
        "valid_views": int(valid_views),
        "mean_foreground_confidence": float(foreground.mean().item()),
        "mean_background_confidence": float(background.mean().item()),
        "mean_mask_support_ratio": float(mask_ratio.mean().item()),
        "background_structure_ratio": float(
            (background_structure >= float(cfg.background_confidence_threshold)).float().mean().item()
        ),
        "low_foreground_ratio": float(
            (foreground < float(cfg.min_foreground_confidence)).float().mean().item()
        ),
        "component": component_stats,
    }
    return ConfidenceResult(foreground, background, mask_ratio, background_structure, stats)


def foreground_confidence_keep(
    object_scene: GaussianScene,
    obj_mask: ObjectMask | None,
    cfg: ObjectRefinementConfig,
    *,
    source_scene: GaussianScene | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    confidence = compute_foreground_confidence(object_scene, obj_mask, cfg, source_scene=source_scene)
    n = object_scene.num_gaussians
    if n == 0:
        return torch.zeros(0, dtype=torch.bool, device=object_scene.device), confidence.stats
    if confidence.stats.get("valid_views", 0) == 0:
        stats = dict(confidence.stats)
        stats.update({"applied": False, "reason": "no_valid_mask_views", "kept": int(n)})
        return torch.ones(n, dtype=torch.bool, device=object_scene.device), stats

    foreground = confidence.foreground
    background = confidence.background
    keep = (foreground >= float(cfg.min_foreground_confidence)) | (
        background < float(cfg.background_confidence_threshold)
    )
    keep_ratio = float(keep.sum().item() / max(1, n))
    fallback = False
    if not bool(keep.any()) or keep_ratio < float(cfg.min_mask_keep_ratio):
        top_k = max(1, int(round(max(float(cfg.min_mask_keep_ratio), 0.01) * n)))
        top_indices = foreground.argsort(descending=True)[:top_k]
        keep = torch.zeros(n, dtype=torch.bool, device=object_scene.device)
        keep[top_indices] = True
        fallback = True

    stats = dict(confidence.stats)
    stats.update({
        "applied": True,
        "kept": int(keep.sum().item()),
        "keep_ratio": float(keep.sum().item() / max(1, n)),
        "min_foreground_confidence": float(cfg.min_foreground_confidence),
        "background_confidence_threshold": float(cfg.background_confidence_threshold),
        "fallback": fallback,
    })
    return keep, stats


def _mask_support_ratio(
    object_scene: GaussianScene,
    obj_mask: ObjectMask | None,
    cfg: ObjectRefinementConfig,
    *,
    source_scene: GaussianScene | None,
) -> tuple[torch.Tensor, int]:
    n = object_scene.num_gaussians
    support = torch.zeros(n, dtype=torch.float32, device=object_scene.device)
    valid_views = 0
    if obj_mask is None or not obj_mask.masks:
        return support, valid_views
    render_cache = {}
    for view_idx, mask in obj_mask.masks.items():
        if view_idx >= len(object_scene.camera_trajectory):
            continue
        valid, px, py, z = project_points(object_scene, view_idx)
        if not bool(valid.any()):
            continue
        if cfg.use_depth_gate and source_scene is not None and view_idx < len(source_scene.camera_trajectory):
            if view_idx not in render_cache:
                render_cache[view_idx] = source_scene.render(source_scene.camera_trajectory[view_idx])
            rendered = render_cache[view_idx]
            valid = depth_gate(
                valid,
                px,
                py,
                z,
                rendered.depth.to(object_scene.device),
                rendered.alpha.to(object_scene.device),
                float(cfg.depth_tolerance),
                float(cfg.min_alpha),
            )
        mask_dev = mask.to(object_scene.device).float() > 0.5
        hit = valid & mask_dev[py.clamp(0, mask_dev.shape[0] - 1), px.clamp(0, mask_dev.shape[1] - 1)]
        support += hit.float()
        valid_views += 1
    if valid_views == 0:
        return support, valid_views
    return support / max(1, valid_views), valid_views


def _scale_score(scene: GaussianScene) -> torch.Tensor:
    world_scale = torch.exp(scene.params.scales.detach()).max(dim=1).values.float()
    if world_scale.numel() <= 1:
        return torch.ones_like(world_scale)
    high = torch.quantile(world_scale, 0.90).clamp(min=1e-6)
    return (1.0 - (world_scale / high).clamp(0.0, 1.0)).clamp(0.0, 1.0)


def _core_score(scene: GaussianScene) -> torch.Tensor:
    points = scene.params.means.detach().float()
    if points.shape[0] <= 1:
        return torch.ones(points.shape[0], dtype=torch.float32, device=scene.device)
    center = points.median(dim=0).values
    dist = torch.norm(points - center, dim=1)
    high = torch.quantile(dist, 0.90).clamp(min=1e-6)
    return (1.0 - (dist / high).clamp(0.0, 1.0)).clamp(0.0, 1.0)


def _main_component_score(scene: GaussianScene, cfg: ObjectRefinementConfig) -> tuple[torch.Tensor, dict[str, Any]]:
    n = scene.num_gaussians
    if n == 0:
        return torch.zeros(0, dtype=torch.float32, device=scene.device), {"num_components": 0}
    if float(cfg.component_radius) <= 0.0 or n == 1:
        return torch.ones(n, dtype=torch.float32, device=scene.device), {"num_components": 1, "largest_component": n}
    components = radius_components(scene.params.means.detach(), float(cfg.component_radius))
    score = torch.zeros(n, dtype=torch.float32, device=scene.device)
    if not components:
        return torch.ones(n, dtype=torch.float32, device=scene.device), {"num_components": 0}
    largest = max(components, key=lambda item: int(item.numel()))
    score[largest] = 1.0
    for component in components:
        if component is largest:
            continue
        score[component] = float(component.numel() / max(1, largest.numel()))
    return score, {
        "num_components": int(len(components)),
        "largest_component": int(largest.numel()),
        "largest_ratio": float(largest.numel() / max(1, n)),
    }


def _background_structure_score(
    scene: GaussianScene,
    mask_ratio: torch.Tensor,
    cfg: ObjectRefinementConfig,
) -> torch.Tensor:
    points = scene.params.means.detach().float()
    n = points.shape[0]
    if n < 4:
        return torch.zeros(n, dtype=torch.float32, device=scene.device)
    candidate = mask_ratio <= torch.quantile(mask_ratio, float(cfg.background_structure_quantile)).clamp(0.0, 1.0)
    if int(candidate.sum().item()) < 4:
        candidate = torch.ones(n, dtype=torch.bool, device=scene.device)
    candidate_points = points[candidate]
    centered = candidate_points - candidate_points.mean(dim=0, keepdim=True)
    try:
        cov = centered.T @ centered / max(1, candidate_points.shape[0] - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        normal = eigvecs[:, eigvals.argmin()]
    except Exception:
        return torch.zeros(n, dtype=torch.float32, device=scene.device)
    normal = normal / normal.norm().clamp(min=1e-6)
    offsets = candidate_points @ normal
    plane_offset = torch.median(offsets)
    distance = torch.abs(points @ normal - plane_offset)
    structure_distance = float(cfg.background_structure_distance)
    if structure_distance <= 0.0:
        scale = torch.quantile(distance, 0.25).clamp(min=1e-6)
    else:
        scale = torch.tensor(structure_distance, dtype=distance.dtype, device=distance.device)
    return (1.0 - (distance / scale).clamp(0.0, 1.0)).clamp(0.0, 1.0)


def _normalize01(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.ones_like(values, dtype=torch.float32)
    lo = torch.quantile(values, 0.05)
    hi = torch.quantile(values, 0.95)
    return ((values - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0)
