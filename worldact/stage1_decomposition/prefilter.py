from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import MaskSet
from worldact.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PrefilterStats:
    num_gaussians: int
    kept_gaussians: int
    keep_ratio: float
    num_objects: int
    num_views: int
    depth_gated: bool = False
    depth_gated_views: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "num_gaussians": self.num_gaussians,
            "kept_gaussians": self.kept_gaussians,
            "keep_ratio": self.keep_ratio,
            "num_objects": self.num_objects,
            "num_views": self.num_views,
            "depth_gated": self.depth_gated,
            "depth_gated_views": self.depth_gated_views,
        }


@dataclass
class PrefilterResult:
    scene: GaussianScene
    keep_indices: torch.Tensor
    stats: PrefilterStats


class GaussianMaskPrefilter:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @staticmethod
    def _pool_mask(mask: torch.Tensor, radius: int, *, mode: str) -> torch.Tensor:
        if radius <= 0:
            return mask
        x = mask.float().unsqueeze(0).unsqueeze(0)
        kernel = 2 * radius + 1
        if mode == "dilate":
            pooled = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius)
        elif mode == "erode":
            pooled = 1.0 - F.max_pool2d(1.0 - x, kernel_size=kernel, stride=1, padding=radius)
        else:
            raise ValueError(f"Unknown mask pooling mode: {mode}")
        return pooled.squeeze(0).squeeze(0) > 0.5

    def _build_core_mask(self, mask: torch.Tensor) -> torch.Tensor:
        foreground = mask.float() > 0.5
        if not foreground.any():
            return torch.zeros_like(foreground)
        core = self._pool_mask(foreground, int(getattr(self.cfg, "core_erode_px", 0)), mode="erode")
        min_ratio = float(getattr(self.cfg, "min_core_area_ratio", 0.0))
        if core.sum().float() < foreground.sum().float() * min_ratio:
            core = foreground
        return core

    @staticmethod
    def _project_points(
        scene: GaussianScene, view_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cam = scene.camera_trajectory[view_idx]
        means = scene.params.means
        device = means.device
        extrinsic = cam.extrinsic.to(device)
        intrinsic = cam.intrinsic.to(device)
        cam_pts = (extrinsic[:3, :3] @ means.T + extrinsic[:3, 3:4]).T
        z = cam_pts[:, 2]
        valid = z > 1e-6
        x = intrinsic[0, 0] * cam_pts[:, 0] / z + intrinsic[0, 2]
        y = intrinsic[1, 1] * cam_pts[:, 1] / z + intrinsic[1, 2]
        px = torch.round(x).long()
        py = torch.round(y).long()
        valid = valid & (px >= 0) & (py >= 0) & (px < cam.width) & (py < cam.height)
        return valid, px, py, z

    @staticmethod
    def _depth_gate(
        valid: torch.Tensor,
        px: torch.Tensor,
        py: torch.Tensor,
        z: torch.Tensor,
        depth: torch.Tensor,
        alpha: torch.Tensor,
        max_depth_error: float,
        min_alpha: float,
    ) -> torch.Tensor:
        """Keep only Gaussians whose projected depth matches the rendered depth at that pixel.

        ``depth`` is the camera-space z produced by gsplat ``RGB+D`` rendering, the same convention
        used by :func:`worldact.stage1_decomposition.view_graph_propagation._warp_mask`. A Gaussian
        sitting behind the visible surface (e.g. ground under a car) has a much larger ``z`` than the
        rendered depth at its projected pixel, so it is gated out and can never receive positive
        assignment downstream. ``max_depth_error <= 0`` disables gating.
        """
        if max_depth_error <= 0.0 and min_alpha <= 0.0:
            return valid
        rendered = depth[py.clamp(0, depth.shape[0] - 1), px.clamp(0, depth.shape[1] - 1)]
        rendered_alpha = alpha[py.clamp(0, alpha.shape[0] - 1), px.clamp(0, alpha.shape[1] - 1)]
        depth_ok = rendered > 1e-6
        if max_depth_error > 0.0:
            depth_ok = depth_ok & ((z - rendered).abs() <= float(max_depth_error))
        alpha_ok = rendered_alpha >= float(min_alpha)
        return valid & depth_ok & alpha_ok

    def apply(self, scene: GaussianScene, masks: MaskSet) -> PrefilterResult:
        if not bool(getattr(self.cfg, "enabled", False)):
            all_indices = torch.arange(scene.num_gaussians, device=scene.device)
            stats = PrefilterStats(
                num_gaussians=scene.num_gaussians,
                kept_gaussians=scene.num_gaussians,
                keep_ratio=1.0,
                num_objects=len(masks.object_masks),
                num_views=len(masks.all_view_indices()),
            )
            return PrefilterResult(scene=scene, keep_indices=all_indices, stats=stats)

        views = masks.all_view_indices()
        if not views:
            all_indices = torch.arange(scene.num_gaussians, device=scene.device)
            stats = PrefilterStats(
                num_gaussians=scene.num_gaussians,
                kept_gaussians=scene.num_gaussians,
                keep_ratio=1.0,
                num_objects=len(masks.object_masks),
                num_views=0,
            )
            return PrefilterResult(scene=scene, keep_indices=all_indices, stats=stats)

        num_gaussians = scene.num_gaussians
        num_objects = len(masks.object_masks)
        object_view_counts = torch.zeros(num_objects, dtype=torch.int32, device=scene.device)
        support = torch.zeros(num_gaussians, num_objects, dtype=torch.int32, device=scene.device)

        core_masks: dict[tuple[int, int], torch.Tensor] = {}
        for obj_idx, obj_mask in enumerate(masks.object_masks):
            for view_idx, mask in obj_mask.masks.items():
                core_masks[(obj_idx, view_idx)] = self._build_core_mask(mask.to(scene.device))
                if core_masks[(obj_idx, view_idx)].any():
                    object_view_counts[obj_idx] += 1

        depth_gated = bool(getattr(self.cfg, "depth_gate_enabled", False))
        depth_gated_views = 0
        for view_idx in views:
            valid, px, py, point_depth = self._project_points(scene, view_idx)
            if not bool(valid.any()):
                continue
            if depth_gated:
                rendered = scene.render(scene.camera_trajectory[view_idx])
                valid = self._depth_gate(
                    valid,
                    px,
                    py,
                    point_depth,
                    rendered.depth.to(scene.device),
                    rendered.alpha.to(scene.device),
                    float(getattr(self.cfg, "depth_tolerance", 0.08)),
                    float(getattr(self.cfg, "min_alpha", 0.05)),
                )
                depth_gated_views += 1
                if not bool(valid.any()):
                    continue
            for obj_idx in range(num_objects):
                core = core_masks.get((obj_idx, view_idx))
                if core is None or not bool(core.any()):
                    continue
                hit = valid & core[py.clamp(0, core.shape[0] - 1), px.clamp(0, core.shape[1] - 1)]
                support[:, obj_idx] += hit.to(dtype=torch.int32)

        if num_objects == 0:
            keep = torch.zeros(num_gaussians, dtype=torch.bool, device=scene.device)
        else:
            best_support, best_obj = support.max(dim=1)
            best_obj_views = object_view_counts[best_obj].clamp_min(1).to(dtype=best_support.dtype)
            hit_ratio = best_support.to(dtype=torch.float32) / best_obj_views.to(dtype=torch.float32)
            min_view_hits = int(getattr(self.cfg, "min_view_hits", 1))
            min_hit_ratio = float(getattr(self.cfg, "min_hit_ratio", 0.0))
            keep = (best_support >= min_view_hits) | (hit_ratio >= min_hit_ratio)

        keep_indices = keep.nonzero(as_tuple=True)[0]
        if keep_indices.numel() == 0:
            log.warning("Prefilter removed all Gaussians; falling back to original scene")
            keep_indices = torch.arange(scene.num_gaussians, device=scene.device)
        elif keep_indices.numel() < max(1, int(0.01 * scene.num_gaussians)):
            log.warning(
                "Prefilter kept only %d/%d Gaussians; falling back to original scene",
                int(keep_indices.numel()),
                scene.num_gaussians,
            )
            keep_indices = torch.arange(scene.num_gaussians, device=scene.device)

        filtered = scene.subset(keep_indices)
        stats = PrefilterStats(
            num_gaussians=scene.num_gaussians,
            kept_gaussians=int(keep_indices.numel()),
            keep_ratio=float(keep_indices.numel() / max(1, scene.num_gaussians)),
            num_objects=num_objects,
            num_views=len(views),
            depth_gated=depth_gated,
            depth_gated_views=depth_gated_views,
        )
        log.info(
            "Prefilter kept %d/%d Gaussians (%.2f%%) using %d objects across %d views",
            stats.kept_gaussians,
            stats.num_gaussians,
            stats.keep_ratio * 100.0,
            stats.num_objects,
            stats.num_views,
        )
        return PrefilterResult(scene=filtered, keep_indices=keep_indices, stats=stats)
