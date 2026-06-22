from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import ObjectMask
from worldact.stage2_object_refinement.common import ObjectRefinementConfig
from worldact.stage2_object_refinement.utils import depth_gate, project_points, radius_components


@dataclass
class PruneResult:
    scene: GaussianScene
    kept_local_indices: torch.Tensor
    stats: dict[str, Any] = field(default_factory=dict)


def prune_object(
    object_scene: GaussianScene,
    obj_mask: ObjectMask | None,
    cfg: ObjectRefinementConfig,
    *,
    source_scene: GaussianScene | None = None,
) -> PruneResult:
    n = object_scene.num_gaussians
    if n == 0:
        return PruneResult(object_scene, torch.zeros(0, dtype=torch.long, device=object_scene.device), {"input": 0, "output": 0})

    keep = torch.ones(n, dtype=torch.bool, device=object_scene.device)
    stats: dict[str, Any] = {"input": int(n)}
    keep = keep & _opacity_keep(object_scene, cfg)
    stats["after_opacity"] = int(keep.sum().item())
    scale_keep = _scale_keep(object_scene, cfg)
    keep = keep & scale_keep
    stats["after_scale"] = int(keep.sum().item())

    if obj_mask is not None and obj_mask.masks:
        support_keep, support_stats = _mask_support_keep(object_scene, obj_mask, cfg, source_scene=source_scene)
        keep = keep & support_keep
        stats["mask_support"] = support_stats
        stats["after_mask_support"] = int(keep.sum().item())

    if obj_mask is not None and obj_mask.masks and cfg.use_depth_consistency:
        consistency_keep, consistency_stats = _depth_consistency_keep(object_scene, obj_mask, cfg)
        keep = keep & consistency_keep
        stats["depth_consistency"] = consistency_stats
        stats["after_depth_consistency"] = int(keep.sum().item())

    if bool(keep.any()) and cfg.component_radius > 0.0:
        component_keep, component_stats = _component_keep(object_scene, keep, cfg)
        keep = keep & component_keep
        stats["component"] = component_stats
    stats["after_component"] = int(keep.sum().item())

    if not bool(keep.any()):
        top_idx = torch.sigmoid(object_scene.params.opacities.reshape(-1)).argmax()
        keep = torch.zeros(n, dtype=torch.bool, device=object_scene.device)
        keep[top_idx] = True
        stats["fallback"] = "all_filters_removed_object_kept_highest_opacity"
    kept_local = keep.nonzero(as_tuple=True)[0]
    stats["output"] = int(kept_local.numel())
    stats["keep_ratio"] = float(kept_local.numel() / max(1, n))
    return PruneResult(object_scene.subset(kept_local), kept_local, stats)


def _opacity_keep(scene: GaussianScene, cfg: ObjectRefinementConfig) -> torch.Tensor:
    if cfg.opacity_min <= 0.0:
        return torch.ones(scene.num_gaussians, dtype=torch.bool, device=scene.device)
    return torch.sigmoid(scene.params.opacities.reshape(-1)) >= float(cfg.opacity_min)


def _scale_keep(scene: GaussianScene, cfg: ObjectRefinementConfig) -> torch.Tensor:
    if cfg.max_world_scale <= 0.0:
        return torch.ones(scene.num_gaussians, dtype=torch.bool, device=scene.device)
    world_scale = torch.exp(scene.params.scales).max(dim=1).values
    return world_scale <= float(cfg.max_world_scale)


def _component_keep(scene: GaussianScene, base_keep: torch.Tensor, cfg: ObjectRefinementConfig) -> tuple[torch.Tensor, dict[str, Any]]:
    kept_local = base_keep.nonzero(as_tuple=True)[0]
    if kept_local.numel() == 0:
        return base_keep, {"applied": False, "reason": "empty_input"}
    points = scene.params.means[kept_local].detach()
    components = radius_components(points, float(cfg.component_radius))
    if not components:
        return base_keep, {"applied": False, "reason": "no_components"}
    if cfg.keep_largest_component:
        selected = max(components, key=lambda item: int(item.numel()))
        keep_ratio = float(selected.numel() / max(1, kept_local.numel()))
        if keep_ratio < float(cfg.min_component_keep_ratio):
            return base_keep, {
                "applied": False,
                "reason": "largest_component_too_small",
                "num_components": int(len(components)),
                "largest_component": int(selected.numel()),
                "input": int(kept_local.numel()),
                "keep_ratio": keep_ratio,
            }
        result = torch.zeros_like(base_keep)
        result[kept_local[selected]] = True
        return result, {
            "applied": True,
            "num_components": int(len(components)),
            "largest_component": int(selected.numel()),
            "input": int(kept_local.numel()),
            "keep_ratio": keep_ratio,
        }
    return base_keep, {"applied": False, "reason": "keep_largest_disabled", "num_components": int(len(components))}


def _mask_support_keep(
    object_scene: GaussianScene,
    obj_mask: ObjectMask,
    cfg: ObjectRefinementConfig,
    *,
    source_scene: GaussianScene | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    support = torch.zeros(object_scene.num_gaussians, dtype=torch.int32, device=object_scene.device)
    valid_views = 0
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
        support += hit.to(dtype=torch.int32)
        valid_views += 1
    if valid_views == 0:
        return torch.ones(object_scene.num_gaussians, dtype=torch.bool, device=object_scene.device), {
            "valid_views": 0,
            "applied": False,
        }
    ratio = support.float() / max(1, valid_views)
    keep = support >= int(cfg.min_mask_support)
    if float(cfg.min_support_ratio) > 0.0:
        keep = keep | (ratio >= float(cfg.min_support_ratio))
    keep_ratio = float(keep.sum().item() / max(1, object_scene.num_gaussians))
    if not bool(keep.any()):
        keep = torch.zeros(object_scene.num_gaussians, dtype=torch.bool, device=object_scene.device)
        keep[torch.argmax(support)] = True
        fallback = True
    elif keep_ratio < float(cfg.min_mask_keep_ratio):
        top_k = min(1, object_scene.num_gaussians)
        top_indices = support.argsort(descending=True)[:top_k]
        keep = torch.zeros(object_scene.num_gaussians, dtype=torch.bool, device=object_scene.device)
        keep[top_indices] = True
        fallback = True
    else:
        fallback = False
    return keep, {
        "valid_views": int(valid_views),
        "min_mask_support": int(cfg.min_mask_support),
        "min_support_ratio": float(cfg.min_support_ratio),
        "kept": int(keep.sum().item()),
        "raw_kept": int((support >= int(cfg.min_mask_support)).sum().item()),
        "raw_keep_ratio": keep_ratio,
        "fallback": fallback,
    }


def _depth_consistency_keep(
    object_scene: GaussianScene,
    obj_mask: ObjectMask,
    cfg: ObjectRefinementConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    n = object_scene.num_gaussians
    if n == 0:
        return torch.ones(0, dtype=torch.bool, device=object_scene.device), {"applied": False, "reason": "empty"}

    depth_ok_count = torch.zeros(n, dtype=torch.float32, device=object_scene.device)
    view_count = torch.zeros(n, dtype=torch.float32, device=object_scene.device)

    pct_lo = float(cfg.depth_consistency_percentile)
    pct_hi = 1.0 - pct_lo
    margin_factor = float(cfg.depth_consistency_margin)

    for view_idx, mask in obj_mask.masks.items():
        if view_idx >= len(object_scene.camera_trajectory):
            continue
        valid, px, py, z = project_points(object_scene, view_idx)
        if not bool(valid.any()):
            continue
        mask_dev = mask.to(object_scene.device).float() > 0.5
        in_mask = valid & mask_dev[py.clamp(0, mask_dev.shape[0] - 1), px.clamp(0, mask_dev.shape[1] - 1)]

        in_mask_z = z[in_mask]
        if in_mask_z.numel() < 3:
            view_count += valid.float()
            depth_ok_count += valid.float()
            continue

        z_lo = torch.quantile(in_mask_z, pct_lo)
        z_hi = torch.quantile(in_mask_z, pct_hi)
        depth_range = float(z_hi - z_lo)
        margin = max(0.1, depth_range * margin_factor)
        z_min = float(z_lo) - margin
        z_max = float(z_hi) + margin

        depth_ok = (z >= z_min) & (z <= z_max) & valid
        view_count += valid.float()
        depth_ok_count += depth_ok.float()

    if view_count.max() == 0:
        return torch.ones(object_scene.num_gaussians, dtype=torch.bool, device=object_scene.device), {"applied": False, "reason": "no_valid_views"}

    consistency_ratio = depth_ok_count / view_count.clamp(min=1.0)
    keep = consistency_ratio >= 0.5
    keep_ratio = float(keep.sum().item() / max(1, n))
    if not bool(keep.any()):
        top_idx = consistency_ratio.argmax()
        keep = torch.zeros(n, dtype=torch.bool, device=object_scene.device)
        keep[top_idx] = True
        fallback = True
    else:
        fallback = False

    return keep, {
        "applied": True,
        "kept": int(keep.sum().item()),
        "total": n,
        "keep_ratio": keep_ratio,
        "fallback": fallback,
    }
