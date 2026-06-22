from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import MaskSet, ObjectMask
from worldact.stage2_object_refinement.common import ObjectRefinementConfig, StageObjectRecord, slugify_name
from worldact.utils.visualization import save_image


@dataclass
class StageRefinementInput:
    scene: GaussianScene
    masks: MaskSet | None
    objects: list[StageObjectRecord]
    assignment_scores: torch.Tensor | None


def load_stage_input(
    ply_path: Path,
    stage_dir: Path,
    *,
    cameras_path: Path | None = None,
    device: str = "cuda",
) -> StageRefinementInput:
    scene = GaussianScene.from_ply(ply_path, cameras_path=cameras_path)
    scene.params = scene.params.to(device)
    scene.camera_trajectory = [cam.to(device) for cam in scene.camera_trajectory]
    with open(stage_dir / "objects.json", encoding="utf-8") as f:
        objects = [StageObjectRecord.from_json(item) for item in json.load(f)]
    masks_path = stage_dir / "masks.pt"
    masks = _torch_load(masks_path, map_location="cpu", weights_only=False) if masks_path.exists() else None
    scores_path = stage_dir / "assignment_scores.pt"
    scores = _torch_load(scores_path, map_location=device, weights_only=True) if scores_path.exists() else None
    return StageRefinementInput(scene=scene, masks=masks, objects=objects, assignment_scores=scores)


def find_object_mask(masks: MaskSet | None, record: StageObjectRecord, fallback_index: int) -> ObjectMask | None:
    if masks is None:
        return None
    for obj_mask in masks.object_masks:
        if str(obj_mask.object_id) == str(record.object_id):
            return obj_mask
    if 0 <= fallback_index < len(masks.object_masks):
        return masks.object_masks[fallback_index]
    return None


def object_dir(output_dir: Path, record: StageObjectRecord) -> Path:
    return output_dir / f"object_{record.object_id}_{slugify_name(record.name)}"


def write_object_package(
    *,
    source_scene: GaussianScene,
    refined_scene: GaussianScene,
    record: StageObjectRecord,
    obj_mask: ObjectMask | None,
    cfg: ObjectRefinementConfig,
    work_dir: Path,
    metrics: dict[str, Any],
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    seed_path = work_dir / "object_refined_seed.ply"
    refined_scene.to_ply(seed_path)
    crop_info = _write_crop_assets(source_scene, record, obj_mask, cfg, work_dir)
    multiview_info = _write_multiview_assets(source_scene, refined_scene, record, obj_mask, cfg, work_dir)
    request = {
        "object_id": record.object_id,
        "name": record.name,
        "seed_ply": str(seed_path),
        "work_dir": str(work_dir),
        "best_view_index": int(record.best_view_index),
        "crop": crop_info,
        "multiview": multiview_info,
        "metrics": metrics,
        "expected_outputs": {
            "generated_ply": str(work_dir / "generated.ply"),
            "metadata_json": str(work_dir / "metadata.json"),
        },
    }
    request_path = work_dir / "object_request.json"
    with open(request_path, "w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)
    return request_path


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_crop_assets(
    source_scene: GaussianScene,
    record: StageObjectRecord,
    obj_mask: ObjectMask | None,
    cfg: ObjectRefinementConfig,
    work_dir: Path,
) -> dict[str, Any]:
    if obj_mask is None or not obj_mask.masks or not source_scene.camera_trajectory:
        return {"status": "missing_mask_or_camera"}
    view_idx = record.best_view_index if record.best_view_index in obj_mask.masks else next(iter(obj_mask.masks))
    if view_idx >= len(source_scene.camera_trajectory):
        return {"status": "view_out_of_range", "view_index": int(view_idx)}
    mask = obj_mask.masks[view_idx].float()
    bbox = _mask_bbox(mask, padding=int(cfg.crop_padding))
    if bbox is None:
        return {"status": "empty_mask", "view_index": int(view_idx)}
    try:
        rendered = source_scene.render(source_scene.camera_trajectory[view_idx])
    except Exception as exc:
        return {"status": "render_failed", "view_index": int(view_idx), "error": str(exc)}
    x0, y0, x1, y1 = bbox
    rgb_crop = rendered.rgb[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1].to(rendered.rgb.device).unsqueeze(-1).repeat(1, 1, 3)
    crop_path = work_dir / "crop.png"
    mask_path = work_dir / "mask.png"
    save_image(rgb_crop, crop_path)
    save_image(mask_crop, mask_path)
    return {
        "status": "ok",
        "view_index": int(view_idx),
        "crop_png": str(crop_path),
        "mask_png": str(mask_path),
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "image_width": int(rendered.width),
        "image_height": int(rendered.height),
    }


def _mask_bbox(mask: torch.Tensor, padding: int) -> tuple[int, int, int, int] | None:
    ys, xs = (mask > 0.5).nonzero(as_tuple=True)
    if ys.numel() == 0:
        return None
    h, w = mask.shape[:2]
    x0 = max(0, int(xs.min().item()) - padding)
    y0 = max(0, int(ys.min().item()) - padding)
    x1 = min(w, int(xs.max().item()) + padding + 1)
    y1 = min(h, int(ys.max().item()) + padding + 1)
    return x0, y0, x1, y1


def _write_multiview_assets(
    source_scene: GaussianScene,
    refined_scene: GaussianScene,
    record: StageObjectRecord,
    obj_mask: ObjectMask | None,
    cfg: ObjectRefinementConfig,
    work_dir: Path,
    num_views: int = 6,
) -> dict[str, Any]:
    if obj_mask is None or not obj_mask.masks or not source_scene.camera_trajectory:
        return {"status": "missing_mask_or_camera"}

    mv_dir = work_dir / "multiview"
    mv_dir.mkdir(parents=True, exist_ok=True)

    available = sorted(obj_mask.masks.keys())
    available = [v for v in available if v < len(source_scene.camera_trajectory)]
    if not available:
        return {"status": "no_valid_views"}

    best = record.best_view_index if record.best_view_index in available else available[0]
    view_indices = _select_multiview_indices(available, best, num_views)

    views_info = []
    for i, view_idx in enumerate(view_indices):
        mask = obj_mask.masks[view_idx].to(source_scene.device).float()
        try:
            rendered = source_scene.render(source_scene.camera_trajectory[view_idx])
        except Exception:
            continue

        mask_3ch = (mask > 0.5).unsqueeze(-1).float().expand_as(rendered.rgb)
        rgb_path = mv_dir / f"view_{i:02d}_rgb.png"
        mask_path = mv_dir / f"view_{i:02d}_mask.png"

        save_image(rendered.rgb, rgb_path)
        save_image(mask_3ch, mask_path)

        cam = source_scene.camera_trajectory[view_idx]
        views_info.append({
            "view_index": int(view_idx),
            "rgb_path": str(rgb_path),
            "mask_path": str(mask_path),
            "extrinsic": cam.extrinsic.detach().cpu().tolist(),
            "intrinsic": cam.intrinsic.detach().cpu().tolist(),
            "width": cam.width,
            "height": cam.height,
        })

    return {
        "status": "ok",
        "num_views": len(views_info),
        "views": views_info,
    }


def _select_multiview_indices(available: list[int], best: int, num_views: int) -> list[int]:
    if len(available) <= num_views:
        return available
    import random as _rng
    rng = _rng.Random(42)
    rest = [v for v in available if v != best]
    sampled = rng.sample(rest, min(num_views - 1, len(rest)))
    return sorted([best] + sampled)


def _torch_load(path: Path, *, map_location: str, weights_only: bool):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)
