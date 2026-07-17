from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch.nn.functional as F
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
    object_scenes: dict[str, GaussianScene] = field(default_factory=dict)
    stage_dir: Path | None = None
    source_ply_path: Path | None = None
    cameras_path: Path | None = None


def load_stage_input(
    ply_path: Path,
    stage_dir: Path,
    *,
    cameras_path: Path | None = None,
    object_ply_dir: Path | None = None,
    object_manifest: Path | None = None,
    device: str = "cuda",
) -> StageRefinementInput:
    scene = GaussianScene.from_ply(ply_path, cameras_path=cameras_path)
    scene.params = scene.params.to(device)
    scene.camera_trajectory = [cam.to(device) for cam in scene.camera_trajectory]
    scene.metadata = dict(scene.metadata)
    scene.metadata["source_ply_path"] = str(ply_path)
    if cameras_path is not None:
        scene.metadata["cameras_path"] = str(cameras_path)
    objects_path = stage_dir / "objects.json"
    if objects_path.exists():
        with open(objects_path, encoding="utf-8") as f:
            objects = [StageObjectRecord.from_json(item) for item in json.load(f)]
    else:
        objects = []
    masks_path = stage_dir / "masks.pt"
    masks = _torch_load(masks_path, map_location="cpu", weights_only=False) if masks_path.exists() else None
    scores_path = stage_dir / "assignment_scores.pt"
    scores = _torch_load(scores_path, map_location=device, weights_only=True) if scores_path.exists() else None
    object_scenes, objects = _load_object_ply_inputs(
        objects,
        object_ply_dir=object_ply_dir,
        object_manifest=object_manifest,
        cameras_path=cameras_path,
        device=device,
    )
    if not objects:
        raise FileNotFoundError(f"No objects found in {objects_path}, {object_ply_dir}, or {object_manifest}")
    return StageRefinementInput(
        scene=scene,
        masks=masks,
        objects=objects,
        assignment_scores=scores,
        object_scenes=object_scenes,
        stage_dir=stage_dir,
        source_ply_path=ply_path,
        cameras_path=cameras_path,
    )


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
    crop_info = _write_crop_assets(source_scene, refined_scene, record, obj_mask, cfg, work_dir)
    multiview_info = _write_multiview_assets(source_scene, refined_scene, record, obj_mask, cfg, work_dir)
    request = {
        "object_id": record.object_id,
        "name": record.name,
        "seed_ply": str(seed_path),
        "source_ply": str(source_scene.metadata.get("source_ply_path", "")),
        "cameras_path": str(source_scene.metadata.get("cameras_path", "")),
        "object_ply": str(record.object_ply_path or seed_path),
        "work_dir": str(work_dir),
        "best_view_index": int(record.best_view_index),
        "memory_state": record.memory_state or {},
        "metadata": record.metadata or {},
        "crop": crop_info,
        "multiview": multiview_info,
        "config": dict(cfg.__dict__),
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


def _load_object_ply_inputs(
    records: list[StageObjectRecord],
    *,
    object_ply_dir: Path | None,
    object_manifest: Path | None,
    cameras_path: Path | None,
    device: str,
) -> tuple[dict[str, GaussianScene], list[StageObjectRecord]]:
    ply_items = _discover_object_plys(object_ply_dir, object_manifest)
    if not ply_items:
        return {}, records

    records_by_id = {str(record.object_id): record for record in records}
    updated_records = list(records)
    object_scenes: dict[str, GaussianScene] = {}
    for item in ply_items:
        object_id = str(item.get("object_id") or _infer_object_id(Path(item["ply_path"])))
        name = str(item.get("name") or _infer_object_name(Path(item["ply_path"]), object_id))
        ply_path = Path(item["ply_path"])
        scene = GaussianScene.from_ply(ply_path, cameras_path=cameras_path)
        scene.params = scene.params.to(device)
        scene.camera_trajectory = [cam.to(device) for cam in scene.camera_trajectory]
        object_scenes[object_id] = scene

        if object_id in records_by_id:
            record = records_by_id[object_id]
            record.object_ply_path = str(ply_path)
            record.metadata = dict(record.metadata or {})
            record.metadata["object_ply_path"] = str(ply_path)
            continue

        metadata = dict(item.get("metadata") or {})
        metadata["object_ply_path"] = str(ply_path)
        updated_records.append(
            StageObjectRecord(
                object_id=object_id,
                name=name,
                source_gaussian_indices=torch.arange(scene.num_gaussians, dtype=torch.long),
                best_view_index=int(item.get("best_view_index", -1) or -1),
                best_view_score=float(item.get("best_view_score", 0.0) or 0.0),
                metadata=metadata,
                memory_state=dict(item.get("memory_state") or {}),
                object_ply_path=str(ply_path),
            )
        )
    return object_scenes, updated_records


def _discover_object_plys(object_ply_dir: Path | None, object_manifest: Path | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if object_manifest is not None and object_manifest.exists():
        with open(object_manifest, encoding="utf-8") as f:
            raw = json.load(f)
        raw_items = raw.get("objects", raw) if isinstance(raw, dict) else raw
        for entry in raw_items:
            if isinstance(entry, str):
                items.append({"ply_path": str(Path(entry))})
            elif isinstance(entry, dict) and entry.get("ply_path"):
                item = dict(entry)
                item["ply_path"] = str(Path(item["ply_path"]))
                items.append(item)
    if object_ply_dir is not None and object_ply_dir.exists():
        candidates = sorted(object_ply_dir.glob("object_*.ply")) or sorted(object_ply_dir.glob("*.ply"))
        seen = {str(Path(item["ply_path"]).resolve()) for item in items if Path(item["ply_path"]).exists()}
        for path in candidates:
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            items.append({"ply_path": str(path)})
    return items


def _infer_object_id(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 2 and parts[0] == "object":
        return parts[1]
    return path.stem


def _infer_object_name(path: Path, object_id: str) -> str:
    prefix = f"object_{object_id}_"
    if path.stem.startswith(prefix):
        return path.stem[len(prefix):] or "object"
    return path.stem


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_crop_assets(
    source_scene: GaussianScene,
    refined_scene: GaussianScene,
    record: StageObjectRecord,
    obj_mask: ObjectMask | None,
    cfg: ObjectRefinementConfig,
    work_dir: Path,
) -> dict[str, Any]:
    if obj_mask is None or not obj_mask.masks or not source_scene.camera_trajectory:
        return {"status": "missing_mask_or_camera"}
    scores = _score_clean_views(source_scene, refined_scene, obj_mask, cfg)
    write_json(scores, work_dir / "view_scores.json")
    if record.best_view_index in obj_mask.masks:
        view_idx = record.best_view_index
    elif scores:
        view_idx = int(scores[0]["view_index"])
    else:
        view_idx = next(iter(obj_mask.masks))
    if view_idx >= len(source_scene.camera_trajectory):
        return {"status": "view_out_of_range", "view_index": int(view_idx)}
    mask = obj_mask.masks[view_idx].float()
    bbox = _mask_bbox(mask, padding=int(cfg.crop_padding))
    if bbox is None:
        return {"status": "empty_mask", "view_index": int(view_idx)}
    bbox = _expand_bbox_for_target_fill(mask, bbox, target_fill=float(cfg.crop_target_fill_ratio))
    try:
        rendered = refined_scene.render(refined_scene.camera_trajectory[view_idx])
    except Exception as exc:
        return {"status": "render_failed", "view_index": int(view_idx), "error": str(exc)}
    x0, y0, x1, y1 = bbox
    rgb_crop, mask_crop = _foregroundized_crop(rendered.rgb, mask.to(rendered.rgb.device), bbox, int(cfg.generation_resolution))
    crop_path = work_dir / "crop.png"
    mask_path = work_dir / "mask.png"
    clean_path = work_dir / "clean_input.png"
    save_image(rgb_crop, crop_path)
    save_image(mask_crop, mask_path)
    save_image(rgb_crop, clean_path)
    return {
        "status": "ok",
        "view_index": int(view_idx),
        "crop_png": str(crop_path),
        "mask_png": str(mask_path),
        "clean_input_png": str(clean_path),
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "image_width": int(rendered.width),
        "image_height": int(rendered.height),
        "view_scores_json": str(work_dir / "view_scores.json"),
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


def _expand_bbox_for_target_fill(
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    *,
    target_fill: float = 0.72,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    h, w = mask.shape[:2]
    region = (mask[y0:y1, x0:x1] > 0.5)
    if not bool(region.any()):
        return bbox
    ys, xs = region.nonzero(as_tuple=True)
    m_y0 = int(y0 + ys.min().item())
    m_y1 = int(y0 + ys.max().item()) + 1
    m_x0 = int(x0 + xs.min().item())
    m_x1 = int(x0 + xs.max().item()) + 1
    mask_h = m_y1 - m_y0
    mask_w = m_x1 - m_x0
    side = max(mask_h, mask_w)
    if side <= 0:
        return bbox
    square = side / max(float(target_fill), 0.1)
    cy = (m_y0 + m_y1) / 2.0
    cx = (m_x0 + m_x1) / 2.0
    half = square / 2.0
    new_x0 = int(round(cx - half))
    new_y0 = int(round(cy - half))
    new_x1 = int(round(cx + half))
    new_y1 = int(round(cy + half))
    if new_x1 <= new_x0 or new_y1 <= new_y0:
        return bbox
    return new_x0, new_y0, new_x1, new_y1


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

    scores = _score_clean_views(source_scene, refined_scene, obj_mask, cfg)
    if record.best_view_index in available:
        best = record.best_view_index
        view_indices = [best] + [int(item["view_index"]) for item in scores if int(item["view_index"]) != best]
    elif scores:
        view_indices = [int(item["view_index"]) for item in scores]
    else:
        view_indices = _select_multiview_indices(available, available[0], num_views)
    view_indices = view_indices[:num_views]

    views_info = []
    for i, view_idx in enumerate(view_indices):
        mask = obj_mask.masks[view_idx].to(source_scene.device).float()
        try:
            rendered = refined_scene.render(refined_scene.camera_trajectory[view_idx])
        except Exception:
            continue

        mask_3ch = (mask > 0.5).unsqueeze(-1).float().expand_as(rendered.rgb)
        rgb_path = mv_dir / f"view_{i:02d}_rgb.png"
        mask_path = mv_dir / f"view_{i:02d}_mask.png"
        depth_path = mv_dir / f"view_{i:02d}_depth.pt"
        alpha_path = mv_dir / f"view_{i:02d}_alpha.pt"

        clean_rgb = rendered.rgb * mask_3ch + (1.0 - mask_3ch)
        save_image(clean_rgb, rgb_path)
        save_image(mask_3ch, mask_path)
        torch.save(rendered.depth.detach().cpu(), depth_path)
        torch.save(rendered.alpha.detach().cpu(), alpha_path)

        cam = source_scene.camera_trajectory[view_idx]
        views_info.append({
            "view_index": int(view_idx),
            "rgb_path": str(rgb_path),
            "mask_path": str(mask_path),
            "depth_path": str(depth_path),
            "alpha_path": str(alpha_path),
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


def _score_clean_views(
    source_scene: GaussianScene,
    refined_scene: GaussianScene,
    obj_mask: ObjectMask,
    cfg: ObjectRefinementConfig,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for view_idx, raw_mask in obj_mask.masks.items():
        if view_idx >= len(refined_scene.camera_trajectory) or view_idx >= len(source_scene.camera_trajectory):
            continue
        mask = raw_mask.to(refined_scene.device).float() > 0.5
        if not bool(mask.any()):
            continue
        try:
            refined = refined_scene.render(refined_scene.camera_trajectory[view_idx])
        except Exception:
            continue
        alpha = refined.alpha >= float(cfg.render_alpha_threshold)
        area = float(mask.float().mean().item())
        bbox = _mask_bbox(raw_mask.float(), padding=0)
        compactness = 0.0
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            bbox_area = max(1, (x1 - x0) * (y1 - y0))
            compactness = float(mask.sum().item() / bbox_area)
        coverage = float((alpha & mask).sum().float().item() / max(1.0, float(mask.sum().item())))
        leakage = float((alpha & ~mask).sum().float().item() / max(1.0, float(alpha.sum().item())))
        sharpness = _edge_strength(refined.rgb, mask)
        score = area * 2.0 + compactness + coverage + sharpness - leakage * 2.0
        scores.append({
            "view_index": int(view_idx),
            "score": float(score),
            "mask_area_ratio": area,
            "compactness": compactness,
            "object_alpha_coverage": coverage,
            "background_leakage": leakage,
            "edge_strength": sharpness,
        })
    return sorted(scores, key=lambda item: item["score"], reverse=True)


def _edge_strength(rgb: torch.Tensor, mask: torch.Tensor) -> float:
    gray = rgb.mean(dim=-1)
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
    gy[1:, :] = (gray[1:, :] - gray[:-1, :]).abs()
    edge = gx + gy
    if bool(mask.any()):
        return float(edge[mask].mean().item())
    return 0.0


def _foregroundized_crop(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0, y0, x1, y1 = bbox
    h, w = rgb.shape[:2]
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        rgb = F.pad(
            rgb.permute(2, 0, 1),
            (pad_left, pad_right, pad_top, pad_bottom),
            value=1.0,
        ).permute(1, 2, 0)
        mask = F.pad(
            mask.unsqueeze(0),
            (pad_left, pad_right, pad_top, pad_bottom),
            value=0.0,
        )[0]
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_mask = (mask[y0:y1, x0:x1] > 0.5).float()
    foreground = crop_rgb * crop_mask.unsqueeze(-1) + (1.0 - crop_mask.unsqueeze(-1))
    foreground, crop_mask = _pad_square(foreground, crop_mask)
    foreground = _resize_image_tensor(foreground, resolution)
    crop_mask = _resize_image_tensor(crop_mask.unsqueeze(-1), resolution).expand(-1, -1, 3)
    return foreground, crop_mask


def _pad_square(rgb: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = rgb.shape[:2]
    size = max(h, w)
    pad_y = size - h
    pad_x = size - w
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left
    rgb_chw = rgb.permute(2, 0, 1).unsqueeze(0)
    mask_chw = mask.unsqueeze(0).unsqueeze(0)
    rgb_pad = F.pad(rgb_chw, (left, right, top, bottom), value=1.0)[0].permute(1, 2, 0)
    mask_pad = F.pad(mask_chw, (left, right, top, bottom), value=0.0)[0, 0]
    return rgb_pad, mask_pad


def _resize_image_tensor(image: torch.Tensor, resolution: int) -> torch.Tensor:
    if resolution <= 0:
        return image
    chw = image.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(chw, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).clamp(0.0, 1.0)


def _torch_load(path: Path, *, map_location: str, weights_only: bool):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)
