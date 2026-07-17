from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import ObjectMask
from worldact.stage2_object_refinement.common import ObjectRefinementConfig, StageObjectRecord
from worldact.stage2_object_refinement.confidence import compute_foreground_confidence


@dataclass
class ObjectDiagnostics:
    object_id: str
    name: str
    num_gaussians: int
    bbox_min: list[float]
    bbox_max: list[float]
    mean_opacity: float
    max_world_scale: float
    mean_world_scale: float
    mean_mask_iou: float | None = None
    mean_coverage: float | None = None
    mean_leakage: float | None = None
    mean_foreground_confidence: float | None = None
    mean_background_confidence: float | None = None
    background_structure_ratio: float | None = None
    mask_support_ratio: float | None = None
    low_foreground_ratio: float | None = None
    component_stats: dict[str, Any] | None = None
    rendered_views: int = 0
    render_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "name": self.name,
            "num_gaussians": self.num_gaussians,
            "bbox_min": self.bbox_min,
            "bbox_max": self.bbox_max,
            "mean_opacity": self.mean_opacity,
            "max_world_scale": self.max_world_scale,
            "mean_world_scale": self.mean_world_scale,
            "mean_mask_iou": self.mean_mask_iou,
            "mean_coverage": self.mean_coverage,
            "mean_leakage": self.mean_leakage,
            "mean_foreground_confidence": self.mean_foreground_confidence,
            "mean_background_confidence": self.mean_background_confidence,
            "background_structure_ratio": self.background_structure_ratio,
            "mask_support_ratio": self.mask_support_ratio,
            "low_foreground_ratio": self.low_foreground_ratio,
            "component_stats": self.component_stats,
            "rendered_views": self.rendered_views,
            "render_error": self.render_error,
        }


def diagnose_object(
    scene: GaussianScene,
    record: StageObjectRecord,
    obj_mask: ObjectMask | None,
    cfg: ObjectRefinementConfig,
) -> ObjectDiagnostics:
    indices = record.source_gaussian_indices.to(scene.device)
    object_scene = scene.subset(indices) if indices.numel() else scene.subset(indices)
    p = object_scene.params
    if object_scene.num_gaussians == 0:
        return ObjectDiagnostics(
            object_id=record.object_id,
            name=record.name,
            num_gaussians=0,
            bbox_min=[],
            bbox_max=[],
            mean_opacity=0.0,
            max_world_scale=0.0,
            mean_world_scale=0.0,
            render_error="empty_object",
        )

    scales = torch.exp(p.scales.detach())
    opacities = torch.sigmoid(p.opacities.detach().reshape(-1))
    diagnostics = ObjectDiagnostics(
        object_id=record.object_id,
        name=record.name,
        num_gaussians=object_scene.num_gaussians,
        bbox_min=[float(v) for v in p.means.min(dim=0).values.detach().cpu().tolist()],
        bbox_max=[float(v) for v in p.means.max(dim=0).values.detach().cpu().tolist()],
        mean_opacity=float(opacities.mean().item()),
        max_world_scale=float(scales.max().item()),
        mean_world_scale=float(scales.mean().item()),
    )
    if obj_mask is None or not obj_mask.masks:
        return diagnostics

    confidence = compute_foreground_confidence(object_scene, obj_mask, cfg, source_scene=scene)
    diagnostics.mean_foreground_confidence = confidence.stats.get("mean_foreground_confidence")
    diagnostics.mean_background_confidence = confidence.stats.get("mean_background_confidence")
    diagnostics.background_structure_ratio = confidence.stats.get("background_structure_ratio")
    diagnostics.mask_support_ratio = confidence.stats.get("mean_mask_support_ratio")
    diagnostics.low_foreground_ratio = confidence.stats.get("low_foreground_ratio")
    diagnostics.component_stats = confidence.stats.get("component")

    ious: list[float] = []
    coverages: list[float] = []
    leakages: list[float] = []
    try:
        for view_idx, mask in obj_mask.masks.items():
            if view_idx >= len(object_scene.camera_trajectory):
                continue
            rendered = object_scene.render(object_scene.camera_trajectory[view_idx])
            alpha = rendered.alpha.to(mask.device)
            gt = mask.float() > 0.5
            pred = alpha >= float(cfg.render_alpha_threshold)
            inter = (pred & gt).sum().float()
            union = (pred | gt).sum().float().clamp_min(1.0)
            mask_area = gt.sum().float().clamp_min(1.0)
            pred_area = pred.sum().float().clamp_min(1.0)
            ious.append(float((inter / union).item()))
            coverages.append(float((inter / mask_area).item()))
            leakages.append(float(((pred & ~gt).sum().float() / pred_area).item()))
    except Exception as exc:  # Rendering is optional for package-only workflows.
        diagnostics.render_error = str(exc)
        return diagnostics

    if ious:
        diagnostics.mean_mask_iou = float(sum(ious) / len(ious))
        diagnostics.mean_coverage = float(sum(coverages) / len(coverages))
        diagnostics.mean_leakage = float(sum(leakages) / len(leakages))
        diagnostics.rendered_views = len(ious)
    return diagnostics
