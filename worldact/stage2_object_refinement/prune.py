from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import ObjectMask
from worldact.stage2_object_refinement.common import ObjectRefinementConfig
from worldact.stage2_object_refinement.confidence import ConfidenceResult, compute_foreground_confidence
from worldact.stage2_object_refinement.utils import depth_gate, project_points, radius_components


@dataclass
class PruneResult:
    scene: GaussianScene
    kept_local_indices: torch.Tensor
    stats: dict[str, Any] = field(default_factory=dict)
    object_core_indices: torch.Tensor | None = None
    uncertain_shell_indices: torch.Tensor | None = None
    background_reject_indices: torch.Tensor | None = None


def prune_object(
    object_scene: GaussianScene,
    obj_mask: ObjectMask | None,
    cfg: ObjectRefinementConfig,
    *,
    source_scene: GaussianScene | None = None,
) -> PruneResult:
    n = object_scene.num_gaussians
    if n == 0:
        empty = torch.zeros(0, dtype=torch.long, device=object_scene.device)
        return PruneResult(
            object_scene,
            empty,
            {"input": 0, "output": 0},
            object_core_indices=empty,
            uncertain_shell_indices=empty,
            background_reject_indices=empty,
        )

    keep = torch.ones(n, dtype=torch.bool, device=object_scene.device)
    stats: dict[str, Any] = {"input": int(n)}
    confidence: ConfidenceResult | None = None
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

    if bool(cfg.use_foreground_confidence):
        confidence = compute_foreground_confidence(object_scene, obj_mask, cfg, source_scene=source_scene)
        confidence_keep, confidence_stats = _foreground_confidence_keep_from_result(object_scene, confidence, cfg)
        keep = keep & confidence_keep
        stats["foreground_confidence"] = confidence_stats
        stats["after_foreground_confidence"] = int(keep.sum().item())

        if bool(cfg.adaptive_cleanup):
            keep, adaptive_stats = _adaptive_cleanup_keep(object_scene, keep, confidence, cfg)
            stats["adaptive_cleanup"] = adaptive_stats
            stats["after_adaptive_cleanup"] = int(keep.sum().item())
        else:
            planar_keep, planar_stats = _planar_background_keep_from_confidence(object_scene, confidence, cfg)
            keep = keep & planar_keep
            stats["planar_background"] = planar_stats
            stats["after_planar_background"] = int(keep.sum().item())

            bbox_keep, bbox_stats = _robust_core_bbox_keep(object_scene, keep, confidence, cfg)
            keep = keep & bbox_keep
            stats["robust_core_bbox"] = bbox_stats
            stats["after_robust_core_bbox"] = int(keep.sum().item())

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
    core_indices, uncertain_indices, reject_indices = _classify_pruned_indices(keep, confidence, cfg)
    stats["separation"] = {
        "object_core": int(core_indices.numel()),
        "uncertain_shell": int(uncertain_indices.numel()),
        "background_reject": int(reject_indices.numel()),
    }
    stats["output"] = int(kept_local.numel())
    stats["keep_ratio"] = float(kept_local.numel() / max(1, n))
    return PruneResult(
        object_scene.subset(kept_local),
        kept_local,
        stats,
        object_core_indices=core_indices,
        uncertain_shell_indices=uncertain_indices,
        background_reject_indices=reject_indices,
    )


def _foreground_confidence_keep_from_result(
    object_scene: GaussianScene,
    confidence: ConfidenceResult,
    cfg: ObjectRefinementConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
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


def _planar_background_keep_from_confidence(
    object_scene: GaussianScene,
    confidence: ConfidenceResult,
    cfg: ObjectRefinementConfig,
    *,
    strong_mask_threshold: float = 0.90,
) -> tuple[torch.Tensor, dict[str, Any]]:
    n = int(confidence.foreground.shape[0])
    if n == 0 or confidence.stats.get("valid_views", 0) == 0:
        return torch.ones(n, dtype=torch.bool, device=confidence.foreground.device), {
            "applied": False,
            "reason": "no_confidence",
        }
    opacity = torch.sigmoid(object_scene.params.opacities.reshape(-1)).detach().float()
    world_scale = torch.exp(object_scene.params.scales.detach()).max(dim=1).values.float()
    low_opacity = opacity <= torch.quantile(opacity, 0.35)
    large_scale = world_scale >= torch.quantile(world_scale, 0.80)
    low_or_weak_foreground = confidence.foreground < min(0.95, float(cfg.min_foreground_confidence) + 0.20)
    not_strong_mask = confidence.mask_support_ratio < float(strong_mask_threshold)
    planar_background = (
        confidence.background_structure >= float(cfg.background_confidence_threshold)
    ) & not_strong_mask & (
        low_or_weak_foreground | low_opacity | large_scale
    )
    keep = ~planar_background
    return keep, {
        "applied": True,
        "rejected": int(planar_background.sum().item()),
        "keep_ratio": float(keep.sum().item() / max(1, n)),
        "background_confidence_threshold": float(cfg.background_confidence_threshold),
        "strong_mask_threshold": float(strong_mask_threshold),
    }


def _robust_core_bbox_keep(
    object_scene: GaussianScene,
    base_keep: torch.Tensor,
    confidence: ConfidenceResult,
    cfg: ObjectRefinementConfig,
    *,
    quantile: float | None = None,
    expand: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    n = object_scene.num_gaussians
    if n < 4 or confidence.stats.get("valid_views", 0) == 0:
        return torch.ones(n, dtype=torch.bool, device=object_scene.device), {
            "applied": False,
            "reason": "insufficient_points_or_views",
        }
    core_score_threshold = torch.quantile(
        confidence.foreground[base_keep],
        float(cfg.foreground_core_quantile),
    ) if bool(base_keep.any()) else torch.tensor(float(cfg.min_foreground_confidence), device=object_scene.device)
    core = base_keep & (confidence.foreground >= core_score_threshold) & (
        confidence.background < float(cfg.background_confidence_threshold)
    )
    if int(core.sum().item()) < 4:
        return torch.ones(n, dtype=torch.bool, device=object_scene.device), {
            "applied": False,
            "reason": "too_few_core_points",
            "core_points": int(core.sum().item()),
        }
    q = max(0.0, min(0.20, float(cfg.robust_core_bbox_quantile if quantile is None else quantile)))
    points = object_scene.params.means.detach()
    core_points = points[core]
    lo = torch.quantile(core_points, q, dim=0)
    hi = torch.quantile(core_points, 1.0 - q, dim=0)
    extent = (hi - lo).clamp(min=1e-6)
    bbox_expand = max(0.0, float(cfg.robust_core_bbox_expand if expand is None else expand))
    margin = extent * bbox_expand
    keep = ((points >= lo - margin) & (points <= hi + margin)).all(dim=1)
    keep_ratio = float(keep.sum().item() / max(1, n))
    # Avoid catastrophic clipping when masks are too noisy to define a stable core.
    if keep_ratio < max(float(cfg.min_mask_keep_ratio), 0.25):
        return torch.ones(n, dtype=torch.bool, device=object_scene.device), {
            "applied": False,
            "reason": "bbox_too_aggressive",
            "raw_keep_ratio": keep_ratio,
            "core_points": int(core.sum().item()),
        }
    return keep, {
        "applied": True,
        "core_points": int(core.sum().item()),
        "rejected": int((~keep).sum().item()),
        "keep_ratio": keep_ratio,
        "quantile": q,
        "expand": bbox_expand,
        "bbox_min": [float(v) for v in (lo - margin).detach().cpu().tolist()],
        "bbox_max": [float(v) for v in (hi + margin).detach().cpu().tolist()],
    }


def _adaptive_cleanup_keep(
    object_scene: GaussianScene,
    base_keep: torch.Tensor,
    confidence: ConfidenceResult,
    cfg: ObjectRefinementConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    policies = [
        {"name": "conservative", "strong_mask": 0.85, "bbox_q": 0.01, "bbox_expand": 0.40},
        {"name": "balanced", "strong_mask": 0.90, "bbox_q": 0.02, "bbox_expand": 0.30},
        {"name": "strict", "strong_mask": 0.90, "bbox_q": 0.04, "bbox_expand": 0.18},
        {"name": "very_strict", "strong_mask": 0.92, "bbox_q": 0.06, "bbox_expand": 0.12},
    ]
    base_indices = base_keep.nonzero(as_tuple=True)[0]
    if base_indices.numel() == 0:
        return base_keep, {"applied": False, "reason": "empty_base_keep"}

    candidates: list[dict[str, Any]] = []
    best_keep = base_keep
    best_score = float("-inf")
    best_name = "base"
    base_fg_sum = confidence.foreground[base_keep].sum().clamp(min=1e-6)
    base_extent = _bbox_extent(object_scene.params.means.detach(), base_keep)
    for policy in policies:
        planar_keep, planar_stats = _planar_background_keep_from_confidence(
            object_scene,
            confidence,
            cfg,
            strong_mask_threshold=float(policy["strong_mask"]),
        )
        after_planar = base_keep & planar_keep
        bbox_keep, bbox_stats = _robust_core_bbox_keep(
            object_scene,
            after_planar,
            confidence,
            cfg,
            quantile=float(policy["bbox_q"]),
            expand=float(policy["bbox_expand"]),
        )
        candidate_keep = after_planar & bbox_keep
        candidate_stats = _score_cleanup_candidate(
            object_scene,
            base_keep,
            candidate_keep,
            confidence,
            base_fg_sum,
            base_extent,
        )
        score = float(candidate_stats["score"])
        entry = {
            "policy": str(policy["name"]),
            "score": score,
            "planar": planar_stats,
            "bbox": bbox_stats,
        }
        entry.update(candidate_stats)
        candidates.append(entry)
        if score > best_score:
            best_score = score
            best_keep = candidate_keep
            best_name = str(policy["name"])

    return best_keep, {
        "applied": True,
        "selected_policy": best_name,
        "selected_score": best_score,
        "candidates": candidates,
    }


def _score_cleanup_candidate(
    object_scene: GaussianScene,
    base_keep: torch.Tensor,
    candidate_keep: torch.Tensor,
    confidence: ConfidenceResult,
    base_fg_sum: torch.Tensor,
    base_extent: torch.Tensor,
) -> dict[str, Any]:
    n_base = int(base_keep.sum().item())
    n_candidate = int(candidate_keep.sum().item())
    if n_candidate == 0:
        return {
            "score": -1e6,
            "keep_ratio_vs_base": 0.0,
            "foreground_keep_ratio": 0.0,
            "mask_support_mean": 0.0,
            "background_mean": 1.0,
            "background_structure_mean": 1.0,
            "bbox_min_axis_ratio": 0.0,
            "rejected": n_base,
        }
    fg_keep_ratio = float((confidence.foreground[candidate_keep].sum() / base_fg_sum).clamp(max=1.0).item())
    keep_ratio = float(n_candidate / max(1, n_base))
    mask_support = float(confidence.mask_support_ratio[candidate_keep].mean().item())
    background = float(confidence.background[candidate_keep].mean().item())
    background_structure = float(confidence.background_structure[candidate_keep].mean().item())
    extent = _bbox_extent(object_scene.params.means.detach(), candidate_keep)
    axis_ratio = (extent / base_extent.clamp(min=1e-6)).clamp(max=1.0)
    min_axis_ratio = float(axis_ratio.min().item())
    # Prefer removing structured background, but strongly penalize losing object extent/foreground support.
    score = (
        1.2 * mask_support
        + 1.4 * fg_keep_ratio
        - 1.4 * background
        - 0.8 * background_structure
        - 3.5 * max(0.0, 0.92 - fg_keep_ratio)
        - 3.0 * max(0.0, 0.82 - keep_ratio)
        - 4.0 * max(0.0, 0.72 - min_axis_ratio)
        + 0.15 * min(max(0.0, 1.0 - keep_ratio), 0.25)
    )
    return {
        "score": float(score),
        "keep_ratio_vs_base": keep_ratio,
        "foreground_keep_ratio": fg_keep_ratio,
        "mask_support_mean": mask_support,
        "background_mean": background,
        "background_structure_mean": background_structure,
        "bbox_min_axis_ratio": min_axis_ratio,
        "rejected": int(n_base - n_candidate),
    }


def _bbox_extent(points: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    if not bool(keep.any()):
        return torch.ones(3, dtype=points.dtype, device=points.device)
    selected = points[keep]
    return (selected.max(dim=0).values - selected.min(dim=0).values).clamp(min=1e-6)


def _classify_pruned_indices(
    keep: torch.Tensor,
    confidence: ConfidenceResult | None,
    cfg: ObjectRefinementConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if confidence is None or confidence.foreground.numel() != keep.numel():
        core = keep
    else:
        core = keep & (confidence.foreground >= float(cfg.min_foreground_confidence)) & (
            confidence.background < float(cfg.background_confidence_threshold)
        )
    uncertain = keep & ~core
    rejected = ~keep
    return (
        core.nonzero(as_tuple=True)[0],
        uncertain.nonzero(as_tuple=True)[0],
        rejected.nonzero(as_tuple=True)[0],
    )


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
    visible = torch.zeros(object_scene.num_gaussians, dtype=torch.int32, device=object_scene.device)
    off_mask = torch.zeros(object_scene.num_gaussians, dtype=torch.int32, device=object_scene.device)
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
        outside = valid & ~mask_dev[py.clamp(0, mask_dev.shape[0] - 1), px.clamp(0, mask_dev.shape[1] - 1)]
        support += hit.to(dtype=torch.int32)
        visible += valid.to(dtype=torch.int32)
        off_mask += outside.to(dtype=torch.int32)
        valid_views += 1
    if valid_views == 0:
        return torch.ones(object_scene.num_gaussians, dtype=torch.bool, device=object_scene.device), {
            "valid_views": 0,
            "applied": False,
        }
    ratio = support.float() / max(1, valid_views)
    off_ratio = off_mask.float() / visible.clamp(min=1).float()
    keep = support >= int(cfg.min_mask_support)
    if float(cfg.min_support_ratio) > 0.0:
        keep = keep | (ratio >= float(cfg.min_support_ratio))
    negative = (visible >= int(cfg.min_mask_support)) & (off_ratio > float(cfg.max_off_mask_ratio))
    keep = keep & ~negative
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
        "off_mask_rejected": int(negative.sum().item()),
        "max_off_mask_ratio": float(cfg.max_off_mask_ratio),
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
