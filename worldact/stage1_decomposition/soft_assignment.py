from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import MaskSet
from worldact.rendering.rasterizer import render_scores
from worldact.utils.logging import get_logger

log = get_logger(__name__)


class SoftAssignment(nn.Module):
    def __init__(self, num_gaussians: int, num_objects: int, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.logits = nn.Parameter(torch.zeros(num_gaussians, num_objects))

    @property
    def scores(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def _assignment_device(self) -> torch.device:
        return self.logits.device

    def _instance_cfg(self) -> Any | None:
        cfg = getattr(self.cfg, "instance_separation", None)
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return None
        return cfg

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

    def _build_trimap(self, mask: torch.Tensor, cfg: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        foreground = mask.float() > 0.5
        if not foreground.any():
            empty = torch.zeros_like(foreground)
            return empty, empty, empty

        core = self._pool_mask(foreground, int(getattr(cfg, "core_erode_px", 0)), mode="erode")
        min_ratio = float(getattr(cfg, "min_core_area_ratio", 0.0))
        if core.sum().float() < foreground.sum().float() * min_ratio:
            core = foreground

        dilated = self._pool_mask(foreground, int(getattr(cfg, "ignore_dilate_px", 0)), mode="dilate")
        background = ~dilated
        return core, background, dilated

    @staticmethod
    def _bbox(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
        ys, xs = mask.nonzero(as_tuple=True)
        if ys.numel() == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        gap_x = max(0, bx0 - ax1, ax0 - bx1)
        gap_y = max(0, by0 - ay1, ay0 - by1)
        return max(gap_x, gap_y)

    def _find_competitors(
        self,
        view_masks: dict[int, torch.Tensor],
        trimaps: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        view_idx: int,
        cfg: Any,
    ) -> list[tuple[int, int]]:
        if len(view_masks) < 2:
            return []
        dilate_px = int(getattr(cfg, "exclusion_dilate_px", 0))
        min_overlap = float(getattr(cfg, "exclusion_min_overlap", 0.0))
        max_gap = int(getattr(cfg, "exclusion_max_bbox_gap_px", -1))
        obj_ids = sorted(view_masks)
        dilated = {idx: self._pool_mask(view_masks[idx], dilate_px, mode="dilate") for idx in obj_ids}
        bboxes = {idx: self._bbox(view_masks[idx]) for idx in obj_ids}

        pairs: list[tuple[int, int]] = []
        for offset, obj_i in enumerate(obj_ids):
            mask_i = view_masks[obj_i]
            area_i = mask_i.sum().float()
            if area_i <= 0:
                continue
            for obj_j in obj_ids[offset + 1 :]:
                mask_j = view_masks[obj_j]
                area_j = mask_j.sum().float()
                if area_j <= 0:
                    continue
                inter = (dilated[obj_i] & dilated[obj_j]).sum().float()
                overlap = inter / torch.minimum(area_i, area_j).clamp_min(1.0)
                bbox_i = bboxes[obj_i]
                bbox_j = bboxes[obj_j]
                gap = self._bbox_gap(bbox_i, bbox_j) if bbox_i is not None and bbox_j is not None else max_gap + 1
                if overlap.item() >= min_overlap or (max_gap >= 0 and gap <= max_gap):
                    core_i = trimaps[(obj_i, view_idx)][0]
                    core_j = trimaps[(obj_j, view_idx)][0]
                    if core_i.any() and core_j.any():
                        pairs.append((obj_i, obj_j))
        return pairs

    def _prepare_instance_separation(
        self,
        masks: MaskSet,
        views: list[int],
    ) -> tuple[
        dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        dict[int, list[tuple[int, int]]],
    ]:
        cfg = self._instance_cfg()
        if cfg is None:
            return {}, {}

        trimaps: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        view_masks: dict[int, dict[int, torch.Tensor]] = {view_idx: {} for view_idx in views}
        for obj_idx, obj_mask in enumerate(masks.object_masks):
            for view_idx, mask in obj_mask.masks.items():
                mask = (mask.to(self._assignment_device()).float() > 0.5)
                trimaps[(obj_idx, view_idx)] = self._build_trimap(mask, cfg)
                view_masks.setdefault(view_idx, {})[obj_idx] = mask

        competitors = {
            view_idx: self._find_competitors(view_obj_masks, trimaps, view_idx, cfg)
            for view_idx, view_obj_masks in view_masks.items()
        }
        return trimaps, competitors

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.device).float()
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def _valid_pixel_mask(self, scene: GaussianScene, view_idx: int) -> torch.Tensor | None:
        if not bool(getattr(self.cfg, "depth_gate_enabled", False)):
            return None
        rendered = scene.render(scene.camera_trajectory[view_idx])
        depth = rendered.depth.to(self._assignment_device())
        alpha = rendered.alpha.to(self._assignment_device())
        valid = depth > 1e-6
        min_alpha = float(getattr(self.cfg, "min_alpha", 0.05))
        if min_alpha > 0.0:
            valid = valid & (alpha >= min_alpha)
        return valid

    def _mask_weight(self, masks: MaskSet, obj_mask: Any, view_idx: int) -> float:
        if not bool(getattr(self.cfg, "use_mask_reliability_weights", True)):
            return 1.0
        weights = masks.metadata.get("mask_weights", {}) if masks.metadata else {}
        obj_weights = weights.get(str(obj_mask.object_id), {})
        return float(obj_weights.get(str(view_idx), 1.0))

    def optimize(self, scene: GaussianScene, masks: MaskSet) -> torch.Tensor:
        opt = torch.optim.Adam([self.logits], lr=float(self.cfg.lr))
        views = masks.all_view_indices()
        instance_cfg = self._instance_cfg()
        trimaps, competitors = self._prepare_instance_separation(masks, views)
        valid_pixels = {view_idx: self._valid_pixel_mask(scene, view_idx) for view_idx in views}
        for it in range(int(self.cfg.num_iters)):
            total = torch.zeros((), device=self.logits.device)
            count = 0
            rendered_cache = {}
            for view_idx in views:
                cam = scene.camera_trajectory[view_idx]
                rendered_cache[view_idx] = render_scores(scene, cam, self.scores, chunk=int(self.cfg.render_chunk))
            for obj_idx, obj_mask in enumerate(masks.object_masks):
                for view_idx, gt in obj_mask.masks.items():
                    pred = rendered_cache[view_idx][..., obj_idx]
                    gt = gt.to(pred.device)
                    mask_weight = self._mask_weight(masks, obj_mask, view_idx)
                    valid = valid_pixels.get(view_idx)
                    if instance_cfg is not None and (obj_idx, view_idx) in trimaps:
                        core, background, _ = trimaps[(obj_idx, view_idx)]
                        core = core.to(pred.device)
                        background = background.to(pred.device)
                        if valid is not None:
                            valid = valid.to(pred.device)
                            core = core & valid
                            background = background & valid
                        total = total + mask_weight * (
                            -float(getattr(instance_cfg, "core_weight", 1.0)) * self._masked_mean(pred, core)
                            + float(self.cfg.lambda_fp) * self._masked_mean(pred, background)
                        )
                    else:
                        if valid is not None:
                            valid = valid.to(pred.device).float()
                            gt = gt * valid
                            bg = (1.0 - gt).clamp(0.0, 1.0) * valid
                        else:
                            bg = 1.0 - gt
                        total = total + mask_weight * (-(gt * pred).mean() + float(self.cfg.lambda_fp) * (bg * pred).mean())
                    count += 1
            if instance_cfg is not None and float(getattr(instance_cfg, "exclusion_weight", 0.0)) > 0.0:
                exclusion_weight = float(getattr(instance_cfg, "exclusion_weight", 0.0))
                for view_idx, pairs in competitors.items():
                    rendered = rendered_cache[view_idx]
                    for obj_i, obj_j in pairs:
                        core_i = trimaps[(obj_i, view_idx)][0].to(rendered.device).float()
                        core_j = trimaps[(obj_j, view_idx)][0].to(rendered.device).float()
                        total = total + exclusion_weight * self._masked_mean(rendered[..., obj_j], core_i)
                        total = total + exclusion_weight * self._masked_mean(rendered[..., obj_i], core_j)
                        count += 2
            if count == 0:
                break
            opt.zero_grad()
            (total / count).backward()
            opt.step()
            if (it + 1) % int(self.cfg.log_interval) == 0:
                log.info("SoftAssignment iter %d/%d loss=%.6f", it + 1, self.cfg.num_iters, (total / count).item())
        return self.scores.detach()
