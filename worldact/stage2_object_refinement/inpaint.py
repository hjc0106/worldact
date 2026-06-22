from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import ObjectMask
from worldact.stage2_object_refinement.common import ObjectRefinementConfig, StageObjectRecord
from worldact.stage2_object_refinement.utils import project_points
from worldact.utils.visualization import save_image


@dataclass
class MultiviewInpaintResult:
    view_dir: Path
    views: list[dict[str, Any]] = field(default_factory=list)
    num_views: int = 0


def render_multiview_inpaint(
    refined_scene: GaussianScene,
    source_scene: GaussianScene,
    obj_mask: ObjectMask,
    record: StageObjectRecord,
    cfg: ObjectRefinementConfig,
    work_dir: Path,
    *,
    num_views: int = 6,
    resolution: int = 512,
    inpaint_model: str = "lama",
    device: str = "cuda",
) -> MultiviewInpaintResult:
    view_dir = work_dir / "multiview_inpaint"
    view_dir.mkdir(parents=True, exist_ok=True)
    if not refined_scene.camera_trajectory:
        return MultiviewInpaintResult(view_dir=view_dir, views=[], num_views=0)
    view_indices = _select_views(refined_scene, obj_mask, record, num_views)
    result = MultiviewInpaintResult(view_dir=view_dir, num_views=len(view_indices))
    for i, view_idx in enumerate(view_indices):
        view_info = _render_and_inpaint_view(
            refined_scene=refined_scene,
            source_scene=source_scene,
            obj_mask=obj_mask,
            view_idx=view_idx,
            view_i=i,
            view_dir=view_dir,
            resolution=resolution,
            inpaint_model=inpaint_model,
            device=device,
        )
        result.views.append(view_info)
    return result


def _select_views(
    scene: GaussianScene,
    obj_mask: ObjectMask,
    record: StageObjectRecord,
    num_views: int,
) -> list[int]:
    available = sorted(obj_mask.masks.keys())
    available = [v for v in available if v < len(scene.camera_trajectory)]
    if not available:
        return []
    best = record.best_view_index if record.best_view_index in available else -1
    if best < 0:
        best = available[0]
    if len(available) <= num_views:
        return available
    import random as _rng

    rng = _rng.Random(42)
    rest = [v for v in available if v != best]
    sampled = rng.sample(rest, min(num_views - 1, len(rest)))
    return [best] + sorted(sampled)


def _render_and_inpaint_view(
    refined_scene: GaussianScene,
    source_scene: GaussianScene,
    obj_mask: ObjectMask,
    view_idx: int,
    view_i: int,
    view_dir: Path,
    resolution: int,
    inpaint_model: str,
    device: str,
) -> dict[str, Any]:
    mask = obj_mask.masks[view_idx].to(device).float()
    refined_render = refined_scene.render(refined_scene.camera_trajectory[view_idx])
    source_render = source_scene.render(source_scene.camera_trajectory[view_idx])

    refined_rgb = refined_render.rgb
    refined_alpha = refined_render.alpha
    source_rgb = source_render.rgb
    mask_binary = (mask > 0.5).unsqueeze(-1).float()

    h, w = source_rgb.shape[:2]
    scale_h = resolution / max(h, 1)
    scale_w = resolution / max(w, 1)

    composed = source_rgb * (1.0 - mask_binary) + refined_rgb * mask_binary
    mask_3ch = mask_binary.expand_as(composed)
    inpainted = _inpaint_image(composed, mask_3ch, inpaint_model)

    rgb_path = view_dir / f"view_{view_i:02d}_rgb.png"
    mask_path = view_dir / f"view_{view_i:02d}_mask.png"
    inpaint_path = view_dir / f"view_{view_i:02d}_inpaint.png"

    save_image(source_rgb, rgb_path)
    save_image(mask_3ch, mask_path)
    save_image(inpainted, inpaint_path)

    return {
        "view_index": int(view_idx),
        "rgb_path": str(rgb_path),
        "mask_path": str(mask_path),
        "inpaint_path": str(inpaint_path),
        "resolution": resolution,
        "original_height": h,
        "original_width": w,
        "scale_h": float(scale_h),
        "scale_w": float(scale_w),
    }


def _inpaint_image(
    composed: torch.Tensor,
    mask_3ch: torch.Tensor,
    model: str,
) -> torch.Tensor:
    if model == "none":
        return composed
    try:
        return _inpaint_lama(composed, mask_3ch)
    except ImportError:
        return composed


def _inpaint_lama(composed: torch.Tensor, mask_3ch: torch.Tensor) -> torch.Tensor:
    from lama_inpaint import inpaint

    rgb_np = (composed.permute(2, 0, 1).cpu().clamp(0, 1).numpy() * 255).astype("uint8")
    mask_np = (mask_3ch.permute(2, 0, 1).cpu()[:, :, :, 0] > 0.5).astype("uint8") * 255
    result = inpaint(rgb_np, mask_np)
    return torch.from_numpy(result).float().permute(1, 2, 0) / 255.0