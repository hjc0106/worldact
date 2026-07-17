from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from worldact.data.gaussians import GaussianParams, GaussianScene
from worldact.data.masks import ObjectMask
from worldact.stage2_object_refinement.common import ObjectRefinementConfig


@dataclass
class LocalRefineResult:
    scene: GaussianScene
    stats: dict[str, Any] = field(default_factory=dict)


def refine_local_object(
    scene: GaussianScene,
    work_dir: Path,
    cfg: ObjectRefinementConfig,
    *,
    obj_mask: ObjectMask | None = None,
) -> LocalRefineResult:
    pseudo_dir = work_dir / "multiview_generate"
    pseudo_color, pseudo_stats = _estimate_pseudo_foreground_color(pseudo_dir)
    if pseudo_color is None:
        stats = {
            "mode": "local_refine_skipped",
            "reason": "missing_pseudo_views",
            "num_gaussians": scene.num_gaussians,
        }
        _write_metrics(work_dir, stats)
        return LocalRefineResult(scene=scene, stats=stats)

    strength = max(0.0, min(1.0, float(cfg.local_refine_lr)))
    params = scene.params
    colors = params.colors.clone()
    color_dim = colors.shape[1]
    target = pseudo_color.to(colors.device, dtype=colors.dtype)
    if color_dim >= 3:
        colors[:, :3] = colors[:, :3] * (1.0 - strength) + target.unsqueeze(0) * strength

    opacities = params.opacities.clone()
    if opacities.numel() > 0:
        opacity = torch.sigmoid(opacities)
        boosted = (opacity * (1.0 + 0.05 * strength)).clamp(max=0.98)
        opacities = torch.logit(boosted, eps=1e-6)

    refined = GaussianScene(
        params=GaussianParams(
            means=params.means.clone(),
            scales=params.scales.clone(),
            rotations=params.rotations.clone(),
            opacities=opacities,
            colors=colors,
        ),
        camera_trajectory=list(scene.camera_trajectory),
        metadata=dict(scene.metadata),
    )
    stats = {
        "mode": "local_color_paint",
        "num_gaussians": refined.num_gaussians,
        "optimized_gaussians": refined.num_gaussians,
        "frozen_geometry": True,
        "frozen_background": True,
        "pseudo_color": [float(v) for v in pseudo_color.detach().cpu().tolist()],
        "paint_strength": strength,
        "num_object_mask_views": len(obj_mask.masks) if obj_mask is not None else 0,
    }
    stats.update(pseudo_stats)
    _write_metrics(work_dir, stats)
    return LocalRefineResult(scene=refined, stats=stats)


def _estimate_pseudo_foreground_color(pseudo_dir: Path) -> tuple[torch.Tensor | None, dict[str, Any]]:
    image_paths = sorted(pseudo_dir.glob("view_*.png"))
    if not image_paths:
        return None, {"num_pseudo_views": 0}

    import numpy as np
    import PIL.Image

    colors = []
    used_pixels = 0
    for path in image_paths:
        img = PIL.Image.open(str(path)).convert("RGB")
        arr = np.asarray(img).astype("float32") / 255.0
        # SV3D inputs/outputs use white or light gray backgrounds after foregroundization.
        non_background = (arr < 0.92).any(axis=-1)
        if int(non_background.sum()) == 0:
            continue
        pixels = arr[non_background]
        colors.append(torch.from_numpy(pixels.mean(axis=0)).float())
        used_pixels += int(non_background.sum())
    if not colors:
        return None, {"num_pseudo_views": len(image_paths), "pseudo_foreground_pixels": 0}
    return torch.stack(colors).mean(dim=0).clamp(0.0, 1.0), {
        "num_pseudo_views": len(image_paths),
        "pseudo_foreground_pixels": used_pixels,
    }


def _write_metrics(work_dir: Path, stats: dict[str, Any]) -> None:
    with open(work_dir / "paint_metrics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
