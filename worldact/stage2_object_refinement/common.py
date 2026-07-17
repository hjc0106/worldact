from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ObjectRefinementConfig:
    refinement_stage: str = "legacy"
    opacity_min: float = 0.02
    max_world_scale: float = 0.0
    component_radius: float = 0.5
    keep_largest_component: bool = True
    min_component_keep_ratio: float = 0.05
    min_mask_support: int = 2
    min_support_ratio: float = 0.0
    min_mask_keep_ratio: float = 0.01
    use_depth_gate: bool = True
    depth_tolerance: float = 0.08
    min_alpha: float = 0.05
    render_alpha_threshold: float = 0.1
    use_depth_consistency: bool = True
    depth_consistency_percentile: float = 0.10
    depth_consistency_margin: float = 0.3
    crop_padding: int = 24
    crop_target_fill_ratio: float = 0.72
    sv3d_max_elevation_deg: float = 25.0
    sv3d_condition_elevation_deg: float = 15.0
    duplicate_distance: float = 0.03
    blend_distance: float = 0.05
    num_generation_views: int = 6
    generation_resolution: int = 512
    generation_model: str = "sv3d"
    generation_model_path: str | None = None
    wonder3d_crop_size: int = 192
    wonder3d_guidance_scale: float = 1.0
    reconstruction_model: str = "gs_lrm"
    inpaint_model: str = "lama"
    icp_iterations: int = 50
    use_foreground_confidence: bool = True
    min_foreground_confidence: float = 0.35
    background_confidence_threshold: float = 0.65
    background_structure_quantile: float = 0.15
    background_structure_distance: float = 0.08
    foreground_core_quantile: float = 0.35
    adaptive_cleanup: bool = True
    max_off_mask_ratio: float = 0.55
    robust_core_bbox_quantile: float = 0.04
    robust_core_bbox_expand: float = 0.18
    generated_min_view_support: int = 2
    generated_min_support_ratio: float = 0.5
    generated_bbox_expand: float = 0.25
    local_refine_iterations: int = 80
    local_refine_lr: float = 0.03
    local_refine_anchor_weight: float = 0.10
    local_refine_alpha_weight: float = 0.20


@dataclass
class StageObjectRecord:
    object_id: str
    name: str
    source_gaussian_indices: torch.Tensor
    best_view_index: int = -1
    best_view_score: float = 0.0
    metadata: dict | None = None
    memory_state: dict | None = None
    object_ply_path: str | None = None

    @classmethod
    def from_json(cls, item: dict) -> "StageObjectRecord":
        metadata = dict(item.get("metadata") or {})
        object_ply_path = item.get("object_ply_path") or metadata.get("object_ply_path")
        return cls(
            object_id=str(item.get("object_id", "")),
            name=str(item.get("name", "object")),
            source_gaussian_indices=torch.tensor(item.get("source_gaussian_indices", []), dtype=torch.long),
            best_view_index=int(item.get("best_view_index", -1)),
            best_view_score=float(item.get("best_view_score", 0.0) or 0.0),
            metadata=metadata,
            memory_state=dict(item.get("memory_state") or {}),
            object_ply_path=str(object_ply_path) if object_ply_path else None,
        )


@dataclass
class BackendRunResult:
    generated_ply: str | None
    metadata_path: str | None
    status: str
    message: str = ""


def slugify_name(name: str) -> str:
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip().lower())
    return slug.strip("_") or "object"
