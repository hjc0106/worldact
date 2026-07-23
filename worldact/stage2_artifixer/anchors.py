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
from worldact.stage2_artifixer.normalization import (
    c2w_matrix_list,
    world_to_local_camera,
    world_to_local_params,
)


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
        n_mask_views = 0 if obj_mask is None else len(obj_mask.view_indices)
        raise ValueError(
            f"only {len(selected)} anchors passed filters; need at least {cfg.min_anchor_count}. "
            f"scored_candidates={len(candidates)}, stage_mask_views={n_mask_views}, "
            f"cameras={len(world_scene.camera_trajectory)}. "
            "Wide-scene Stage-1 masks are often <<1% of the frame; if this persists, "
            "set prepare.min_mask_area_ratio lower (e.g. 0.0005) or check masks.pt."
        )

    mode = str(cfg.anchor_camera_mode).strip().lower()
    if mode not in {"object_centered", "source_crop"}:
        raise ValueError(
            f"anchor_camera_mode={cfg.anchor_camera_mode!r} is not supported; "
            "expected 'object_centered' or 'source_crop'"
        )

    device = torch.device(render_device or object_scene.device)
    object_scene = GaussianScene(
        params=object_scene.params.to(device),
        camera_trajectory=[c.to(device) for c in world_scene.camera_trajectory],
        metadata=dict(object_scene.metadata),
    )
    means_world = object_scene.params.means.detach().cpu()
    local_scene = GaussianScene(
        params=world_to_local_params(object_scene.params, norm),
        camera_trajectory=[],
        metadata={**object_scene.metadata, "frame": "local"},
    )
    exported: list[SelectedAnchor] = []

    for cand in selected:
        if mode == "object_centered":
            source_camera = world_scene.camera_trajectory[cand.view_index]
            camera_local = object_centered_anchor_camera(source_camera, cand.direction, cfg).to(device)
            rgb_t, alpha_t = _render_rgb_alpha(local_scene, camera_local)
            alpha = alpha_t.float().clamp(0.0, 1.0)
            crop_mask = alpha > float(cfg.anchor_alpha_threshold)
            area_ratio = float(crop_mask.float().mean().item())
            if area_ratio <= 1e-5 or area_ratio >= 0.98:
                continue
            if count_boundary_edges(crop_mask) > cfg.max_boundary_edges:
                continue

            # The shared rasterizer renders over white. Remove that background
            # contribution to obtain a black-premultiplied object image while
            # retaining soft alpha edges.
            crop_rgb = (rgb_t - (1.0 - alpha).unsqueeze(-1)).clamp(0.0, 1.0)
            exported.append(
                SelectedAnchor(
                    source_view_index=cand.view_index,
                    camera_local=camera_local.to("cpu"),
                    rgb=(crop_rgb.detach().cpu().numpy() * 255).astype(np.uint8),
                    mask=(crop_mask.detach().cpu().numpy().astype(np.uint8) * 255),
                    mask_area_ratio=area_ratio,
                    quality_score=cand.quality_score,
                )
            )
            continue

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
        k = item.camera_local.intrinsic
        if mode == "source_crop":
            rgb_np, mask_np, fl_x, fl_y, cx, cy = _reframe_crop_to_principal_point(
                item.rgb,
                item.mask,
                fl_x=float(k[0, 0]),
                fl_y=float(k[1, 1]),
                cx=float(k[0, 2]),
                cy=float(k[1, 2]),
                image_size=cfg.image_size,
            )
        else:
            rgb_np, mask_np = item.rgb, item.mask
            fl_x, fl_y = float(k[0, 0]), float(k[1, 1])
            cx, cy = float(k[0, 2]), float(k[1, 2])
        Image.fromarray(rgb_np).save(anchors_dir / rgb_name)
        Image.fromarray(mask_np, mode="L").save(anchors_dir / mask_name)
        if not (0.0 < cx < float(cfg.image_size) and 0.0 < cy < float(cfg.image_size)):
            raise ValueError(
                f"anchor {i} principal point still out of bounds after reframe: cx={cx}, cy={cy}"
            )
        anchors.append(
            AnchorCamera(
                index=i,
                source_view_index=item.source_view_index,
                rgb_path=f"anchors/{rgb_name}",
                mask_path=f"anchors/{mask_name}",
                transform_matrix=c2w_matrix_list(item.camera_local),
                fl_x=fl_x,
                fl_y=fl_y,
                cx=cx,
                cy=cy,
                w=cfg.image_size,
                h=cfg.image_size,
                mask_area_ratio=item.mask_area_ratio,
                quality_score=item.quality_score,
            )
        )

    doc = AnchorsDocument(schema_version=SCHEMA_VERSION, image_size=cfg.image_size, anchors=anchors)
    (output_dir / "anchors.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return doc


def object_centered_anchor_camera(
    source_camera: CameraPose,
    direction: torch.Tensor,
    cfg: PrepareConfig,
) -> CameraPose:
    """Build a square-pixel local camera aimed at the normalized object origin."""
    size = int(cfg.image_size)
    fov = math.radians(float(cfg.fov_deg))
    focal = size / (2.0 * math.tan(fov * 0.5))
    principal = (size - 1) * 0.5
    half_extent = math.sqrt(3.0)
    distance = max(
        float(cfg.camera_distance_scale),
        half_extent / max(math.tan(fov * 0.5), 1e-4) * 1.15,
        1.5,
    )

    view_dir = direction.detach().cpu().float()
    view_dir = view_dir / view_dir.norm().clamp_min(1e-6)
    eye = view_dir * distance
    forward = -view_dir

    source_c2w = source_camera.c2w.detach().cpu().float()
    up_hint = -source_c2w[:3, 1]
    up_hint = up_hint / up_hint.norm().clamp_min(1e-6)
    right = torch.linalg.cross(forward, up_hint)
    if float(right.norm()) < 1e-5:
        fallback_up = torch.tensor([0.0, 0.0, 1.0])
        if abs(float(torch.dot(forward, fallback_up))) > 0.98:
            fallback_up = torch.tensor([0.0, 1.0, 0.0])
        right = torch.linalg.cross(forward, fallback_up)
    right = right / right.norm().clamp_min(1e-6)
    true_up = torch.linalg.cross(right, forward)
    true_up = true_up / true_up.norm().clamp_min(1e-6)

    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = -true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    intrinsic = torch.tensor(
        [[focal, 0.0, principal], [0.0, focal, principal], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return CameraPose(
        extrinsic=torch.linalg.inv(c2w),
        intrinsic=intrinsic,
        width=size,
        height=size,
    )


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
    # Stage-1 SAM masks on wide outdoor frames are often <<1% of the image.
    # ``min_mask_area_ratio`` is primarily a post-crop fill target (~0.2), so
    # applying it directly to full-frame Stage-1 masks rejects every view.
    stage_min_area = min(float(cfg.min_mask_area_ratio) * 0.5, 1e-4)
    stage_max_area = min(0.95, float(cfg.max_mask_area_ratio) * 1.5)
    min_mask_pixels = 32

    for view_idx, camera in enumerate(cameras):
        mask = None
        has_stage_mask = False
        if obj_mask is not None:
            mask = obj_mask.get_mask(view_idx)
            has_stage_mask = mask is not None
        if mask is None:
            # Soft coverage from projected object means (not a full-frame ones mask).
            mask = _projected_coverage_mask(means, camera)
            if mask is None:
                continue
        mask = mask.float()
        fg = mask > 0.5
        area_ratio = float(fg.float().mean().item())
        pixel_count = int(fg.sum().item())
        if has_stage_mask:
            if pixel_count < min_mask_pixels or area_ratio > stage_max_area:
                continue
        else:
            if area_ratio < stage_min_area or area_ratio > stage_max_area or pixel_count < min_mask_pixels:
                continue
        boundary = count_boundary_edges(fg)
        if boundary > cfg.max_boundary_edges:
            continue
        sharpness = _mask_sharpness(mask)
        c2w = camera.c2w.detach().cpu().float()
        direction = c2w[:3, 3] - center
        direction = direction / direction.norm().clamp_min(1e-6)
        # Prefer larger Stage-1 masks, but don't let absolute area dominate quality
        # when all candidates are tiny outdoor crops.
        area_term = min(area_ratio / max(stage_min_area, 1e-6), 1.0) if has_stage_mask else area_ratio
        quality = area_term * 0.55 + (1.0 - boundary / 4.0) * 0.25 + min(sharpness, 1.0) * 0.20
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


def _projected_coverage_mask(means: torch.Tensor, camera: CameraPose) -> torch.Tensor | None:
    """Binary mask with projected Gaussian means marked as foreground."""
    h, w = int(camera.height), int(camera.width)
    pts = _project_points(means, camera.extrinsic.cpu(), camera.intrinsic.cpu(), w, h)
    if pts is None or pts.numel() == 0:
        return None
    mask = torch.zeros(h, w, dtype=torch.float32)
    uv = pts.long()
    for u, v in uv:
        if 0 <= int(u) < w and 0 <= int(v) < h:
            mask[int(v), int(u)] = 1.0
    # Dilate slightly so tiny point splat masks pass pixel-count checks.
    if mask.any():
        k = torch.ones(1, 1, 5, 5, dtype=torch.float32)
        mask = torch.nn.functional.conv2d(mask[None, None], k, padding=2).clamp(0, 1)[0, 0]
    return mask


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


def _reframe_crop_to_principal_point(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    image_size: int,
    margin: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Pad-then-resize so the principal point lies inside the square image."""
    w = h = int(image_size)
    if 0.0 < cx < float(w) and 0.0 < cy < float(h):
        return rgb, mask, fl_x, fl_y, cx, cy

    pad_l = max(0, int(math.ceil(margin - cx)))
    pad_t = max(0, int(math.ceil(margin - cy)))
    pad_r = max(0, int(math.ceil(cx + margin - float(w))))
    pad_b = max(0, int(math.ceil(cy + margin - float(h))))
    canvas_w = w + pad_l + pad_r
    canvas_h = h + pad_t + pad_b
    rgb_img = Image.fromarray(rgb)
    mask_img = Image.fromarray(mask, mode="L")
    if rgb_img.size != (w, h):
        rgb_img = rgb_img.resize((w, h), Image.Resampling.BILINEAR)
        mask_img = mask_img.resize((w, h), Image.Resampling.NEAREST)
    rgb_canvas = Image.new("RGB", (canvas_w, canvas_h), color=(0, 0, 0))
    mask_canvas = Image.new("L", (canvas_w, canvas_h), color=0)
    rgb_canvas.paste(rgb_img, (pad_l, pad_t))
    mask_canvas.paste(mask_img, (pad_l, pad_t))
    rgb_out = np.asarray(rgb_canvas.resize((w, h), Image.Resampling.BILINEAR))
    mask_out = np.asarray(mask_canvas.resize((w, h), Image.Resampling.NEAREST))
    sx = float(w) / float(canvas_w)
    sy = float(h) / float(canvas_h)
    return (
        rgb_out,
        mask_out,
        fl_x * sx,
        fl_y * sy,
        (cx + pad_l) * sx,
        (cy + pad_t) * sy,
    )


def _render_rgb_alpha(scene: GaussianScene, camera: CameraPose) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        view = scene.render(camera)
        return view.rgb, view.alpha
    except Exception as exc:
        if scene.device.type != "cpu":
            raise RuntimeError(
                f"anchor rendering failed on {scene.device}; refusing to generate gray fallback images"
            ) from exc
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
