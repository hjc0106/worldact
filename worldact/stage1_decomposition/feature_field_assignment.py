from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import MaskSet
from worldact.rendering.rasterizer import render_features
from worldact.stage1_decomposition.soft_assignment import SoftAssignment
from worldact.utils.logging import get_logger

log = get_logger(__name__)


class FeatureFieldAssignment(SoftAssignment):
    def __init__(self, num_gaussians: int, num_objects: int, cfg: Any, instance_cfg: Any | None = None) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg
        self.instance_cfg = instance_cfg
        self.num_objects = num_objects
        feature_dim = int(cfg.feature_dim)
        self.features = nn.Parameter(torch.randn(num_gaussians, feature_dim) * 0.01)
        num_queries = num_objects + (1 if bool(getattr(cfg, "use_background_query", True)) else 0)
        self.object_queries = nn.Parameter(torch.randn(num_queries, feature_dim) * 0.01)
        self.loss_history: list[float] = []

    def _assignment_device(self) -> torch.device:
        return self.features.device

    def _instance_cfg(self) -> Any | None:
        cfg = self.instance_cfg
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return None
        return cfg

    @property
    def scores(self) -> torch.Tensor:
        features = F.normalize(self.features, dim=-1)
        queries = F.normalize(self.object_queries, dim=-1)
        temperature = max(float(self.cfg.temperature), 1e-6)
        logits = (features @ queries.T) / temperature
        if bool(getattr(self.cfg, "use_background_query", True)):
            return torch.softmax(logits, dim=-1)[:, : self.num_objects]
        return torch.sigmoid(logits)

    def _render_score_map(self, scene: GaussianScene, view_idx: int) -> torch.Tensor:
        features = F.normalize(self.features, dim=-1)
        queries = F.normalize(self.object_queries, dim=-1)
        feature_map, alpha = render_features(
            scene,
            scene.camera_trajectory[view_idx],
            features,
            chunk=int(self.cfg.render_chunk),
            return_alpha=True,
        )
        logits = torch.einsum("hwc,kc->hwk", feature_map, queries) / max(float(self.cfg.temperature), 1e-6)
        if bool(getattr(self.cfg, "use_background_query", True)):
            return torch.softmax(logits, dim=-1)[..., : self.num_objects] * alpha.unsqueeze(-1)
        return torch.sigmoid(logits) * alpha.unsqueeze(-1)

    def optimize(self, scene: GaussianScene, masks: MaskSet) -> torch.Tensor:
        opt = torch.optim.Adam([self.features, self.object_queries], lr=float(self.cfg.lr))
        views = masks.all_view_indices()
        instance_cfg = self._instance_cfg()
        trimaps, competitors = self._prepare_instance_separation(masks, views)

        for it in range(int(self.cfg.num_iters)):
            total = torch.zeros((), device=self.features.device)
            count = 0
            rendered_cache = {view_idx: self._render_score_map(scene, view_idx) for view_idx in views}

            for obj_idx, obj_mask in enumerate(masks.object_masks):
                for view_idx, gt in obj_mask.masks.items():
                    pred = rendered_cache[view_idx][..., obj_idx]
                    if instance_cfg is not None and (obj_idx, view_idx) in trimaps:
                        core, background, _ = trimaps[(obj_idx, view_idx)]
                        total = total + (
                            -float(self.cfg.lambda_mask) * self._masked_mean(pred, core)
                            + float(self.cfg.lambda_fp) * self._masked_mean(pred, background)
                        )
                    else:
                        gt = gt.to(pred.device)
                        total = total + (
                            -float(self.cfg.lambda_mask) * self._masked_mean(pred, gt > 0.5)
                            + float(self.cfg.lambda_fp) * self._masked_mean(pred, gt <= 0.5)
                        )
                    count += 1

            exclusion_weight = float(self.cfg.lambda_exclusion)
            if instance_cfg is not None and exclusion_weight > 0.0:
                for view_idx, pairs in competitors.items():
                    rendered = rendered_cache[view_idx]
                    for obj_i, obj_j in pairs:
                        core_i = trimaps[(obj_i, view_idx)][0].to(rendered.device)
                        core_j = trimaps[(obj_j, view_idx)][0].to(rendered.device)
                        total = total + exclusion_weight * self._masked_mean(rendered[..., obj_j], core_i)
                        total = total + exclusion_weight * self._masked_mean(rendered[..., obj_i], core_j)
                        count += 2

            if count == 0:
                break
            loss = total / count
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.loss_history.append(float(loss.detach().item()))
            if (it + 1) % int(self.cfg.log_interval) == 0:
                log.info("FeatureField iter %d/%d loss=%.6f", it + 1, self.cfg.num_iters, loss.item())

        return self.scores.detach()

    def export_state(self) -> dict[str, object]:
        return {
            "features": self.features.detach().cpu(),
            "object_queries": self.object_queries.detach().cpu(),
            "loss_history": list(self.loss_history),
            "feature_dim": int(self.cfg.feature_dim),
            "temperature": float(self.cfg.temperature),
        }
