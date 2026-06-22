from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene, RenderedView
from worldact.stage1_decomposition.sam3_segmenter import SAM3CandidateSet


@dataclass
class Candidate3DSupport:
    object_id: str
    object_name: str
    prompt: str
    view_idx: int
    candidate_id: int
    gaussian_indices: torch.Tensor
    support_score: float
    visible_count: int
    mask_area: float
    depth_consistency: float
    reliability: float
    component_count: int
    overlap_penalty: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_name": self.object_name,
            "prompt": self.prompt,
            "view_idx": int(self.view_idx),
            "candidate_id": int(self.candidate_id),
            "num_gaussians": int(self.gaussian_indices.numel()),
            "support_score": float(self.support_score),
            "visible_count": int(self.visible_count),
            "mask_area": float(self.mask_area),
            "depth_consistency": float(self.depth_consistency),
            "reliability": float(self.reliability),
            "component_count": int(self.component_count),
            "overlap_penalty": float(self.overlap_penalty),
        }


def build_candidate_supports(
    scene: GaussianScene,
    view_idx: int,
    rendered_view: RenderedView,
    object_id: str,
    object_name: str,
    prompt: str,
    candidate_set: SAM3CandidateSet,
    cfg: Any,
) -> list[Candidate3DSupport]:
    max_candidates = int(getattr(cfg, "max_candidates_per_view", 5))
    min_quality = float(getattr(cfg, "min_candidate_quality", 0.0))
    candidates = sorted(candidate_set.candidates, key=lambda cand: cand.quality, reverse=True)
    supports: list[Candidate3DSupport] = []
    projection = project_gaussians_to_view(scene, view_idx, rendered_view, cfg)
    for candidate in candidates[:max_candidates]:
        if float(candidate.quality) < min_quality:
            continue
        mask = candidate.mask.to(scene.device) > 0.5
        if not bool(mask.any()):
            continue
        valid = projection["valid"]
        px = projection["px"]
        py = projection["py"]
        in_mask = torch.zeros_like(valid)
        valid_idx = valid.nonzero(as_tuple=True)[0]
        if valid_idx.numel() > 0:
            in_mask[valid_idx] = mask[py[valid_idx], px[valid_idx]]
        gaussian_indices = in_mask.nonzero(as_tuple=True)[0].detach().cpu()
        visible_count = int(valid.sum().item())
        mask_area = float(mask.float().sum().item())
        support_ratio = float(gaussian_indices.numel()) / max(1.0, min(float(visible_count), mask_area))
        depth_consistency = float(projection["depth_consistency"])
        component_count = _support_component_count(scene, gaussian_indices, cfg)
        reliability = _candidate_reliability(
            quality=float(candidate.quality),
            support_ratio=support_ratio,
            depth_consistency=depth_consistency,
            component_count=component_count,
        )
        supports.append(
            Candidate3DSupport(
                object_id=str(object_id),
                object_name=object_name,
                prompt=prompt,
                view_idx=int(view_idx),
                candidate_id=int(candidate.candidate_id),
                gaussian_indices=gaussian_indices,
                support_score=support_ratio,
                visible_count=visible_count,
                mask_area=mask_area,
                depth_consistency=depth_consistency,
                reliability=reliability,
                component_count=component_count,
            )
        )
    return supports


def project_gaussians_to_view(
    scene: GaussianScene,
    view_idx: int,
    rendered_view: RenderedView,
    cfg: Any,
) -> dict[str, torch.Tensor | float]:
    cam = scene.camera_trajectory[view_idx]
    means = scene.params.means
    device = means.device
    extrinsic = cam.extrinsic.to(device)
    intrinsic = cam.intrinsic.to(device)
    cam_pts = (extrinsic[:3, :3] @ means.T + extrinsic[:3, 3:4]).T
    z = cam_pts[:, 2]
    valid = z > 1e-6
    x = intrinsic[0, 0] * cam_pts[:, 0] / z.clamp_min(1e-6) + intrinsic[0, 2]
    y = intrinsic[1, 1] * cam_pts[:, 1] / z.clamp_min(1e-6) + intrinsic[1, 2]
    px = torch.round(x).long()
    py = torch.round(y).long()
    valid = valid & (px >= 0) & (py >= 0) & (px < cam.width) & (py < cam.height)
    depth = rendered_view.depth.to(device)
    alpha = rendered_view.alpha.to(device)
    if bool(getattr(cfg, "depth_gate_enabled", True)) and bool(valid.any()):
        depth_at = depth[py.clamp(0, cam.height - 1), px.clamp(0, cam.width - 1)]
        alpha_at = alpha[py.clamp(0, cam.height - 1), px.clamp(0, cam.width - 1)]
        tolerance = float(getattr(cfg, "depth_tolerance", 0.08))
        min_alpha = float(getattr(cfg, "min_alpha", 0.05))
        depth_ok = depth_at > 1e-6
        if tolerance > 0.0:
            depth_ok = depth_ok & ((z - depth_at).abs() <= tolerance)
        valid = valid & depth_ok & (alpha_at >= min_alpha)
        denom = max(1, int((depth_at > 1e-6).sum().item()))
        depth_consistency = float(valid.sum().item()) / float(denom)
    else:
        depth_consistency = 1.0 if bool(valid.any()) else 0.0
    return {"valid": valid, "px": px, "py": py, "depth_consistency": depth_consistency}


def support_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    a_set = set(int(v) for v in a.detach().cpu().tolist())
    b_set = set(int(v) for v in b.detach().cpu().tolist())
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return float(inter / max(1, union))


def support_containment(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    a_set = set(int(v) for v in a.detach().cpu().tolist())
    b_set = set(int(v) for v in b.detach().cpu().tolist())
    inter = len(a_set & b_set)
    return float(inter / max(1, min(len(a_set), len(b_set))))


def compute_mask_overlap_penalties(supports_by_object: dict[str, list[Candidate3DSupport]]) -> None:
    per_view: dict[int, list[Candidate3DSupport]] = {}
    for supports in supports_by_object.values():
        for support in supports:
            per_view.setdefault(support.view_idx, []).append(support)
    for supports in per_view.values():
        for idx, support in enumerate(supports):
            penalty = 0.0
            for other in supports[:idx] + supports[idx + 1 :]:
                if support.object_id == other.object_id:
                    continue
                penalty = max(penalty, support_containment(support.gaussian_indices, other.gaussian_indices))
            support.overlap_penalty = penalty
            support.reliability = max(0.05, support.reliability * (1.0 - 0.5 * penalty))


def summarize_supports(supports_by_object: dict[str, list[Candidate3DSupport]]) -> dict[str, Any]:
    return {
        object_id: [support.to_json() for support in supports]
        for object_id, supports in sorted(supports_by_object.items(), key=lambda item: item[0])
    }


def _candidate_reliability(
    *,
    quality: float,
    support_ratio: float,
    depth_consistency: float,
    component_count: int,
) -> float:
    component_penalty = 1.0 / max(1.0, float(component_count))
    value = 0.35 * quality + 0.30 * min(1.0, support_ratio) + 0.25 * depth_consistency + 0.10 * component_penalty
    return float(max(0.05, min(1.0, value)))


def _support_component_count(scene: GaussianScene, indices: torch.Tensor, cfg: Any) -> int:
    if indices.numel() == 0:
        return 0
    max_points = int(getattr(cfg, "component_count_max_points", 4096))
    if indices.numel() > max_points:
        indices = indices[:max_points]
    points = scene.params.means[indices.to(scene.device)].detach()
    if points.shape[0] <= 1:
        return int(points.shape[0])
    radius = _adaptive_radius(points)
    coords = torch.floor(points / radius).to(torch.int64)
    unique = torch.unique(coords, dim=0)
    return int(max(1, min(points.shape[0], unique.shape[0])))


def _adaptive_radius(points: torch.Tensor) -> float:
    extent = torch.clamp(points.max(dim=0).values - points.min(dim=0).values, min=1e-4)
    scale = float(torch.median(extent).item())
    return max(1e-3, scale / 8.0)
