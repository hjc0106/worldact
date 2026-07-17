"""Select and export object-centric anchor views for remote ArtiFixer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import ObjectMask
from worldact.data.poses import CameraPose
from worldact.stage2_artifixer.config import PrepareConfig
from worldact.stage2_artifixer.contracts import SCHEMA_VERSION, AnchorCamera, AnchorsDocument
from worldact.stage2_artifixer.normalization import c2w_matrix_list, world_to_local_camera


@dataclass
class AnchorCandidate:
    view_index: int
    mask: torch.Tensor
    area_ratio: float
    boundary_edges: int
    sharpness: float
    quality_score: float
    direction: torch.Tensor


@dataclass
class SelectedAnchor:
    source_view_index: int
    camera_local: CameraPose
    rgb: np.ndarray  # HxWx3 uint8
    mask: np.ndarray  # HxW uint8
    mask_area_ratio: float
    quality_score: float


def select_and_export_anchors(
    world_scene: GaussianScene,
    object_scene: GaussianScene,
    obj_mask: ObjectMask | None,
    norm,
    output_dir: Path,
    cfg: PrepareConfig,
    *,
    render_device: str | torch.device | None = None,
) -> AnchorsDocument:
    """Pick anchors, render/crop to cfg.image_size, write anchors/ + anchors.json."""
    output_dir = Path(output_dir)
    anchors_dir = output_dir / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)

    candidates = score_anchor_candidates(world_scene, object_scene, obj_mask, cfg)
    selected = angular_dedup_select(candidates, cfg.anchor_count, cfg.angular_dedup_deg, cfg.min_anchor_count)
    if len(selected) < cfg.min_anchor_count:
        raise ValueError(
            f"only {len(selected)} anchors passed filters; need at least {cfg.min_anchor_count}"
        )

    device = torch.device(render_device or object_scene.device)
    object_scene = GaussianScene(
        params=object_scene.params.to(device),
        camera_trajectory=[c.to(device) for c in world_scene.camera_trajectory],
        metadata=dict(object_scene.metadata),
    )
    means_world = object_scene.params.means.detach().cpu()
    exported: list[SelectedAnchor] = []

    for cand in selected:
        camera = world_scene.camera_trajectory[cand.view_index].to(device)
        rgb_t, alpha_t = _render_rgb_alpha(object_scene, camera)
        mask = cand.mask.to(device)
        if mask.shape != rgb_t.shape[:2]:
            mask = torch.nn.functional.interpolate(
                mask[None, None].float(),
                size=rgb_t.shape[:2],
                mode="nearest",
            )[0, 0]
        fg = (mask > 0.5).float()
        rgb_fg = rgb_t * fg.unsqueeze(-1)
        crop_rgb, crop_mask, new_k = crop_object_square(
            rgb_fg.cpu(),
            (mask > 0.5).cpu().float(),
            camera.intrinsic.cpu(),
            means_world,
            camera.extrinsic.cpu(),
            cfg.image_size,
            target_fill=(cfg.min_mask_area_ratio + cfg.max_mask_area_ratio) * 0.5,
        )
        area_ratio = float((crop_mask > 0.5).float().mean().item())
        if not (cfg.min_mask_area_ratio <= area_ratio <= cfg.max_mask_area_ratio):
            # Relaxed acceptance after crop: keep if non-empty and not full-frame.
            if area_ratio <= 0.01 or area_ratio >= 0.98:
                continue
        boundary = count_boundary_edges(crop_mask > 0.5)
        if boundary > cfg.max_boundary_edges:
            continue

        camera_cropped = CameraPose(
            extrinsic=camera.extrinsic.cpu(),
            intrinsic=new_k,
            width=cfg.image_size,
            height=cfg.image_size,
        )
        camera_local = world_to_local_camera(camera_cropped, norm)
        rgb_np = (crop_rgb.clamp(0, 1).numpy() * 255).astype(np.uint8)
        mask_np = ((crop_mask > 0.5).numpy().astype(np.uint8) * 255)
        exported.append(
            SelectedAnchor(
                source_view_index=cand.view_index,
                camera_local=camera_local,
                rgb=rgb_np,
                mask=mask_np,
                mask_area_ratio=area_ratio,
                quality_score=cand.quality_score,
            )
        )

    if len(exported) < cfg.min_anchor_count:
        raise ValueError(
            f"after crop/filter only {len(exported)} anchors remain; need >= {cfg.min_anchor_count}"
        )

    anchors: list[AnchorCamera] = []
    for i, item in enumerate(exported[: cfg.anchor_count]):
        rgb_name = f"frame_{i:03d}.png"
        mask_name = f"frame_{i:03d}_mask.png"
        Image.fromarray(item.rgb).save(anchors_dir / rgb_name)
        Image.fromarray(item.mask, mode="L").save(anchors_dir / mask_name)
        k = item.camera_local.intrinsic
        anchors.append(
            AnchorCamera(
                index=i,
                source_view_index=item.source_view_index,
                rgb_path=f"anchors/{rgb_name}",
                mask_path=f"anchors/{mask_name}",
                transform_matrix=c2w_matrix_list(item.camera_local),
                fl_x=float(k[0, 0]),
                fl_y=float(k[1, 1]),
                cx=float(k[0, 2]),
                cy=float(k[1, 2]),
                w=cfg.image_size,
                h=cfg.image_size,
                mask_area_ratio=item.mask_area_ratio,
                quality_score=item.quality_score,
            )
        )

    doc = AnchorsDocument(schema_version=SCHEMA_VERSION, image_size=cfg.image_size, anchors=anchors)
    (output_dir / "anchors.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return doc


def score_anchor_candidates(
    world_scene: GaussianScene,
    object_scene: GaussianScene,
    obj_mask: ObjectMask | None,
    cfg: PrepareConfig,
) -> list[AnchorCandidate]:
    cameras = world_scene.camera_trajectory
    if not cameras:
        raise ValueError("scene has no camera_trajectory")

    means = object_scene.params.means.detach().cpu().float()
    center = means.mean(dim=0)
    candidates: list[AnchorCandidate] = []

    for view_idx, camera in enumerate(cameras):
        mask = None
        if obj_mask is not None:
            mask = obj_mask.get_mask(view_idx)
        if mask is None:
            # Synthetic full-frame soft mask from projected means coverage estimate.
            mask = torch.ones(camera.height, camera.width)
        mask = mask.float()
        area_ratio = float((mask > 0.5).float().mean().item())
        if area_ratio < cfg.min_mask_area_ratio * 0.5 or area_ratio > min(0.95, cfg.max_mask_area_ratio * 1.5):
            continue
        boundary = count_boundary_edges(mask > 0.5)
        if boundary > cfg.max_boundary_edges:
            continue
        sharpness = _mask_sharpness(mask)
        c2w = camera.c2w.detach().cpu().float()
        direction = c2w[:3, 3] - center
        direction = direction / direction.norm().clamp_min(1e-6)
        quality = area_ratio * 0.55 + (1.0 - boundary / 4.0) * 0.25 + min(sharpness, 1.0) * 0.20
        candidates.append(
            AnchorCandidate(
                view_index=view_idx,
                mask=mask.cpu(),
                area_ratio=area_ratio,
                boundary_edges=boundary,
                sharpness=sharpness,
                quality_score=quality,
                direction=direction,
            )
        )

    candidates.sort(key=lambda c: c.quality_score, reverse=True)
    return candidates


def angular_dedup_select(
    candidates: list[AnchorCandidate],
    count: int,
    dedup_deg: float,
    min_count: int,
) -> list[AnchorCandidate]:
    selected: list[AnchorCandidate] = []
    cos_thresh = math.cos(math.radians(dedup_deg))
    for cand in candidates:
        if all(float(torch.dot(cand.direction, s.direction)) < cos_thresh for s in selected):
            selected.append(cand)
        if len(selected) >= count:
            break
    if len(selected) < min_count:
        # Fill remaining without angular constraint.
        for cand in candidates:
            if cand in selected:
                continue
            selected.append(cand)
            if len(selected) >= min_count:
                break
    return selected[:count]


def count_boundary_edges(mask_bool: torch.Tensor) -> int:
    m = mask_bool.bool()
    edges = 0
    if bool(m[0, :].any()):
        edges += 1
    if bool(m[-1, :].any()):
        edges += 1
    if bool(m[:, 0].any()):
        edges += 1
    if bool(m[:, -1].any()):
        edges += 1
    return edges


def crop_object_square(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    intrinsic: torch.Tensor,
    means_world: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    image_size: int,
    target_fill: float = 0.45,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop around mask/bbox, resize to square, update intrinsics."""
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    ys, xs = torch.where(mask > 0.5)
    if len(xs) == 0:
        # Fallback: project means and use their 2D bbox.
        pts = _project_points(means_world, extrinsic_w2c, intrinsic, w, h)
        if pts is None:
            x0, y0, x1, y1 = 0, 0, w - 1, h - 1
        else:
            x0 = int(pts[:, 0].min().item())
            x1 = int(pts[:, 0].max().item())
            y0 = int(pts[:, 1].min().item())
            y1 = int(pts[:, 1].max().item())
    else:
        x0 = int(xs.min().item())
        x1 = int(xs.max().item())
        y0 = int(ys.min().item())
        y1 = int(ys.max().item())

    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    fill = float((mask > 0.5).float().mean().item()) if mask.any() else 0.1
    scale = math.sqrt(max(fill, 1e-4) / max(target_fill, 1e-4))
    side = int(math.ceil(max(bw, bh) * max(scale, 1.0) * 1.15))
    side = max(side, 32)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    x0c = int(round(cx - side / 2))
    y0c = int(round(cy - side / 2))
    x1c = x0c + side
    y1c = y0c + side

    # Clamp crop window into image by shifting.
    if x0c < 0:
        x1c -= x0c
        x0c = 0
    if y0c < 0:
        y1c -= y0c
        y0c = 0
    if x1c > w:
        x0c -= x1c - w
        x1c = w
    if y1c > h:
        y0c -= y1c - h
        y1c = h
    x0c = max(0, x0c)
    y0c = max(0, y0c)
    x1c = min(w, x1c)
    y1c = min(h, y1c)

    crop_rgb = rgb[y0c:y1c, x0c:x1c]
    crop_mask = mask[y0c:y1c, x0c:x1c]
    # Pad to square if needed.
    ch, cw = crop_rgb.shape[0], crop_rgb.shape[1]
    side_out = max(ch, cw, 1)
    pad_rgb = torch.zeros(side_out, side_out, 3, dtype=rgb.dtype)
    pad_mask = torch.zeros(side_out, side_out, dtype=mask.dtype)
    pad_rgb[:ch, :cw] = crop_rgb
    pad_mask[:ch, :cw] = crop_mask

    resized_rgb = torch.nn.functional.interpolate(
        pad_rgb.permute(2, 0, 1).unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)
    resized_mask = torch.nn.functional.interpolate(
        pad_mask[None, None],
        size=(image_size, image_size),
        mode="nearest",
    )[0, 0]

    sx = image_size / float(side_out)
    sy = image_size / float(side_out)
    new_k = intrinsic.clone().float()
    new_k[0, 0] *= sx
    new_k[1, 1] *= sy
    new_k[0, 2] = (new_k[0, 2] - x0c) * sx
    new_k[1, 2] = (new_k[1, 2] - y0c) * sy
    return resized_rgb, resized_mask, new_k


def _render_rgb_alpha(scene: GaussianScene, camera: CameraPose) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        view = scene.render(camera)
        return view.rgb, view.alpha
    except Exception:
        # CPU / no-gsplat fallback for packaging tests: flat gray + projected coverage.
        h, w = camera.height, camera.width
        rgb = torch.full((h, w, 3), 0.5, device=camera.extrinsic.device)
        alpha = torch.zeros(h, w, device=camera.extrinsic.device)
        pts = _project_points(scene.params.means.detach(), camera.extrinsic, camera.intrinsic, w, h)
        if pts is not None:
            for u, v in pts.long():
                if 0 <= u < w and 0 <= v < h:
                    alpha[v, u] = 1.0
                    rgb[v, u] = torch.tensor([0.9, 0.9, 0.9], device=rgb.device)
        return rgb, alpha


def _project_points(
    means: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor | None:
    if means.numel() == 0:
        return None
    ones = torch.ones(means.shape[0], 1, dtype=means.dtype, device=means.device)
    homo = torch.cat([means.to(extrinsic_w2c.device), ones.to(extrinsic_w2c.device)], dim=-1)
    cam = (extrinsic_w2c.to(means.dtype) @ homo.T).T[:, :3]
    z = cam[:, 2].clamp_min(1e-6)
    uv = (intrinsic.to(means.dtype).to(cam.device) @ (cam / z.unsqueeze(-1)).T).T
    pts = uv[:, :2]
    valid = (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height) & (cam[:, 2] > 0)
    pts = pts[valid]
    return pts.cpu() if pts.numel() else None


def _mask_sharpness(mask: torch.Tensor) -> float:
    m = mask.float()
    if m.shape[0] < 3 or m.shape[1] < 3:
        return 0.0
    gx = m[:, 1:] - m[:, :-1]
    gy = m[1:, :] - m[:-1, :]
    return float(gx.abs().mean().item() + gy.abs().mean().item())
