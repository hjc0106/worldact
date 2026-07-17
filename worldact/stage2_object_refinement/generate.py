from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.poses import CameraPose
from worldact.stage2_object_refinement.common import ObjectRefinementConfig

log = logging.getLogger(__name__)


def _ensure_diffusers_sv3d_path() -> None:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "third_party" / "sv3d-diffusers",
        Path.cwd() / "third_party" / "sv3d-diffusers",
    ]
    for candidate in candidates:
        if (candidate / "diffusers_sv3d" / "__init__.py").exists():
            path = str(candidate.resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
            return


def _wonder3d_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "third_party" / "Wonder3D",
        Path.cwd() / "third_party" / "Wonder3D",
    ]
    for candidate in candidates:
        if (candidate / "mvdiffusion").exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _ensure_wonder3d_path() -> Path:
    import sys

    root = _wonder3d_root()
    if not root.exists():
        raise FileNotFoundError(f"Wonder3D repository not found: {root}")
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return root


@dataclass
class MultiviewGenerateResult:
    image_dir: Path
    cameras_path: Path | None
    images: list[Path] = field(default_factory=list)
    num_views: int = 0
    elevation_deg: float = 0.0


def generate_multiview_images(
    crop_rgb_path: Path,
    crop_mask_path: Path,
    work_dir: Path,
    cfg: ObjectRefinementConfig,
    *,
    num_views: int | None = None,
    elevation_deg: float = 0.0,
    model: str = "sv3d",
    model_path: str | None = None,
    resolution: int = 512,
    seed: int = 42,
    device: str = "cuda",
) -> MultiviewGenerateResult:
    n_views = num_views or cfg.num_generation_views
    image_dir = work_dir / "multiview_generate"
    image_dir.mkdir(parents=True, exist_ok=True)

    if model == "sv3d":
        return _generate_sv3d(crop_rgb_path, crop_mask_path, image_dir, n_views,
                              elevation_deg, resolution, seed, device, model_path)
    elif model == "wonder3d":
        return _generate_wonder3d(crop_rgb_path, crop_mask_path, image_dir, n_views,
                                  resolution, seed, device, model_path, cfg)
    elif model == "zero123plus":
        return _generate_zero123plus(crop_rgb_path, crop_mask_path, image_dir, n_views,
                                     elevation_deg, resolution, seed, device, model_path)
    else:
        return _generate_orbit_template(crop_rgb_path, crop_mask_path, image_dir, n_views,
                                        elevation_deg, resolution)


def _estimate_elevation_from_scene(
    seed_scene: GaussianScene,
    best_view_index: int,
) -> float:
    if not seed_scene.camera_trajectory or best_view_index < 0 or best_view_index >= len(seed_scene.camera_trajectory):
        return 0.0
    cam = seed_scene.camera_trajectory[best_view_index]
    ext = cam.extrinsic
    forward = -ext[2, :3]
    forward = forward / (torch.norm(forward) + 1e-8)
    elevation_deg = float(torch.asin(torch.clamp(forward[2], -1.0, 1.0)).item() * 180.0 / 3.141592653589793)
    return elevation_deg


def resolve_condition_view_index(request: dict[str, Any]) -> int:
    best = int(request.get("best_view_index", -1) or -1)
    if best >= 0:
        return best
    crop = request.get("crop", {})
    crop_view = int(crop.get("view_index", -1) or -1)
    if crop_view >= 0:
        return crop_view
    multiview = request.get("multiview", {}).get("views", [])
    if multiview:
        return int(multiview[0].get("view_index", 0) or 0)
    return 0


def resolve_sv3d_elevation(estimated_deg: float, cfg: ObjectRefinementConfig) -> tuple[float, bool]:
    max_elev = float(cfg.sv3d_max_elevation_deg)
    cond_elev = float(cfg.sv3d_condition_elevation_deg)
    if estimated_deg > max_elev:
        return cond_elev, True
    return float(estimated_deg), False


def _is_blank_condition_image(image_path: Path, *, min_foreground_ratio: float = 0.002) -> bool:
    import PIL.Image

    if not image_path.exists():
        return True
    arr = np.asarray(PIL.Image.open(str(image_path)).convert("RGB")).astype("float32") / 255.0
    foreground = (arr < 0.92).any(axis=-1)
    return float(foreground.mean()) < min_foreground_ratio


def generate_scene_multiview_pseudo(
    request: dict[str, Any],
    work_dir: Path,
    cfg: ObjectRefinementConfig,
    *,
    resolution: int,
) -> MultiviewGenerateResult:
    from worldact.stage2_object_refinement.io import (
        _expand_bbox_for_target_fill,
        _foregroundized_crop,
        _mask_bbox,
    )
    from worldact.utils.visualization import save_image

    views = request.get("multiview", {}).get("views", [])
    image_dir = work_dir / "multiview_generate"
    image_dir.mkdir(parents=True, exist_ok=True)
    if not views:
        raise ValueError("scene_multiview requires request.multiview.views")

    output_paths: list[Path] = []
    cameras_data: list[dict[str, Any]] = []
    for i, view in enumerate(views):
        rgb_path = Path(view["rgb_path"])
        mask_path = Path(view.get("mask_path", ""))
        if not rgb_path.exists():
            continue
        rgb_tensor, mask_tensor = _load_rgb_mask_tensors(rgb_path, mask_path if mask_path.exists() else None)
        bbox = _mask_bbox(mask_tensor, padding=int(cfg.crop_padding))
        if bbox is None:
            continue
        bbox = _expand_bbox_for_target_fill(
            mask_tensor,
            bbox,
            target_fill=float(cfg.crop_target_fill_ratio),
        )
        foreground, _ = _foregroundized_crop(rgb_tensor, mask_tensor, bbox, resolution)
        out_path = image_dir / f"view_{i:02d}.png"
        save_image(foreground, out_path)
        output_paths.append(out_path)
        cameras_data.append(
            {
                "view_index": int(view.get("view_index", i)),
                "source_view_index": int(view.get("view_index", i)),
                "rgb_path": str(rgb_path),
                "mask_path": str(mask_path) if mask_path.exists() else None,
                "extrinsic": view.get("extrinsic"),
                "intrinsic": view.get("intrinsic"),
                "width": int(view.get("width", resolution)),
                "height": int(view.get("height", resolution)),
            }
        )

    if not output_paths:
        raise ValueError("scene_multiview could not export any foregroundized views")

    cameras_path = image_dir / "cameras.json"
    with open(cameras_path, "w", encoding="utf-8") as f:
        json.dump(cameras_data, f, indent=2)

    elevation_deg = 0.0
    return MultiviewGenerateResult(
        image_dir=image_dir,
        cameras_path=cameras_path,
        images=output_paths,
        num_views=len(output_paths),
        elevation_deg=elevation_deg,
    )


def _load_rgb_mask_tensors(rgb_path: Path, mask_path: Path | None) -> tuple[torch.Tensor, torch.Tensor]:
    import PIL.Image

    rgb = np.asarray(PIL.Image.open(str(rgb_path)).convert("RGB")).astype("float32") / 255.0
    if mask_path is not None and mask_path.exists():
        mask_img = PIL.Image.open(str(mask_path)).convert("L")
        if mask_img.size != (rgb.shape[1], rgb.shape[0]):
            mask_img = mask_img.resize((rgb.shape[1], rgb.shape[0]))
        mask = np.asarray(mask_img).astype("float32") / 255.0
    else:
        mask = (rgb < 0.95).any(axis=-1).astype("float32")
    return torch.from_numpy(rgb), torch.from_numpy(mask)


def _estimate_azimuth_from_scene(scene: GaussianScene, view_index: int) -> float:
    if not scene.camera_trajectory or view_index < 0 or view_index >= len(scene.camera_trajectory):
        return 0.0
    cam = scene.camera_trajectory[view_index]
    forward = -cam.extrinsic[2, :3]
    forward = forward / (torch.norm(forward) + 1e-8)
    return float(torch.atan2(forward[0], forward[1]).item() * 180.0 / 3.141592653589793)


def _camera_looking_at(
    target: torch.Tensor,
    distance: float,
    elevation_deg: float,
    azimuth_deg: float,
    resolution: int,
) -> CameraPose:
    elev_rad = np.deg2rad(elevation_deg)
    azim_rad = np.deg2rad(azimuth_deg)
    offset = np.array([
        distance * np.cos(elev_rad) * np.sin(azim_rad),
        distance * np.cos(elev_rad) * np.cos(azim_rad),
        distance * np.sin(elev_rad),
    ], dtype=np.float32)
    eye = target.detach().cpu().numpy() + offset
    center = target.detach().cpu().numpy()
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    fwd = center - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up_corrected = np.cross(right, fwd)
    up_corrected = up_corrected / (np.linalg.norm(up_corrected) + 1e-8)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up_corrected
    c2w[:3, 2] = -fwd
    c2w[:3, 3] = eye
    w2c = np.linalg.inv(c2w)
    fx = resolution * 0.8
    fy = resolution * 0.8
    cx = resolution / 2.0
    cy = resolution / 2.0
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return CameraPose(
        extrinsic=torch.tensor(w2c, dtype=torch.float32),
        intrinsic=torch.tensor(intrinsic, dtype=torch.float32),
        width=resolution,
        height=resolution,
    )


def synthesize_sv3d_condition_image(
    scene: GaussianScene,
    *,
    elevation_deg: float,
    azimuth_deg: float,
    resolution: int,
    output_path: Path,
    device: str,
) -> tuple[Path, Path]:
    from worldact.utils.visualization import save_image

    means = scene.params.means
    center = means.mean(dim=0)
    extent = (means.max(dim=0).values - means.min(dim=0).values).max().item()
    distance = max(extent * 3.5, 0.5)
    camera = _camera_looking_at(center, distance, elevation_deg, azimuth_deg, resolution)
    camera = camera.to(device)
    rendered = scene.render(camera)
    alpha = rendered.alpha.squeeze(-1).clamp(0.0, 1.0)
    mask = (alpha > 0.1).float()
    ys, xs = (mask > 0.5).nonzero(as_tuple=True)
    if ys.numel() == 0:
        foreground = rendered.rgb
        mask_rgb = mask.unsqueeze(-1).expand_as(rendered.rgb)
    else:
        y0 = max(0, int(ys.min().item()) - 8)
        y1 = min(rendered.height, int(ys.max().item()) + 9)
        x0 = max(0, int(xs.min().item()) - 8)
        x1 = min(rendered.width, int(xs.max().item()) + 9)
        bbox = _expand_bbox_for_target_fill_static(
            mask,
            (x0, y0, x1, y1),
            target_fill=0.78,
        )
        foreground, mask_rgb = _foregroundized_crop_static(
            rendered.rgb,
            mask,
            bbox,
            resolution,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path = output_path.parent / "sv3d_condition_mask.png"
    save_image(foreground, output_path)
    save_image(mask_rgb, mask_path)
    return output_path, mask_path


def _expand_bbox_for_target_fill_static(
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    *,
    target_fill: float = 0.72,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    region = (mask[y0:y1, x0:x1] > 0.5)
    if not bool(region.any()):
        return bbox
    ys, xs = region.nonzero(as_tuple=True)
    m_y0 = int(y0 + ys.min().item())
    m_y1 = int(y0 + ys.max().item()) + 1
    m_x0 = int(x0 + xs.min().item())
    m_x1 = int(x0 + xs.max().item()) + 1
    side = max(m_y1 - m_y0, m_x1 - m_x0)
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


def _foregroundized_crop_static(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch.nn.functional as F

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
    size = max(foreground.shape[0], foreground.shape[1])
    pad_y = size - foreground.shape[0]
    pad_x = size - foreground.shape[1]
    top = pad_y // 2
    left = pad_x // 2
    foreground = F.pad(
        foreground.permute(2, 0, 1),
        (left, pad_x - left, top, pad_y - top),
        value=1.0,
    ).permute(1, 2, 0)
    crop_mask = F.pad(
        crop_mask.unsqueeze(0).unsqueeze(0),
        (left, pad_x - left, top, pad_y - top),
        value=0.0,
    )[0, 0]
    chw = foreground.permute(2, 0, 1).unsqueeze(0)
    foreground = F.interpolate(chw, size=(resolution, resolution), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).clamp(0.0, 1.0)
    mask_chw = crop_mask.unsqueeze(0).unsqueeze(0)
    crop_mask = F.interpolate(mask_chw, size=(resolution, resolution), mode="bilinear", align_corners=False)[0, 0]
    mask_rgb = crop_mask.unsqueeze(-1).expand_as(foreground)
    return foreground, mask_rgb


def _generate_orbit_cameras(
    num_views: int,
    elevation_deg: float,
    resolution: int,
    distance: float = 1.5,
) -> tuple[list[dict[str, Any]], list[CameraPose]]:
    cameras_data = []
    camera_poses = []
    elev_rad = np.deg2rad(elevation_deg)
    for i in range(num_views):
        azim_rad = 2.0 * np.pi * i / num_views
        eye = np.array([
            distance * np.cos(elev_rad) * np.sin(azim_rad),
            distance * np.cos(elev_rad) * np.cos(azim_rad),
            distance * np.sin(elev_rad),
        ])
        center = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        fwd = center - eye
        fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(fwd, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up_corrected = np.cross(right, fwd)
        up_corrected = up_corrected / (np.linalg.norm(up_corrected) + 1e-8)
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up_corrected
        c2w[:3, 2] = -fwd
        c2w[:3, 3] = eye
        w2c = np.linalg.inv(c2w)
        fx = resolution * 0.8
        fy = resolution * 0.8
        cx = resolution / 2.0
        cy = resolution / 2.0
        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float32)
        cameras_data.append({
            "view_index": i,
            "azimuth_deg": float(np.rad2deg(azim_rad)),
            "elevation_deg": elevation_deg,
            "distance": distance,
            "extrinsic": w2c.astype(np.float32).tolist(),
            "intrinsic": intrinsic.tolist(),
            "width": resolution,
            "height": resolution,
        })
        camera_poses.append(CameraPose(
            extrinsic=torch.tensor(w2c.astype(np.float32)),
            intrinsic=torch.tensor(intrinsic),
            width=resolution,
            height=resolution,
        ))
    return cameras_data, camera_poses


def _generate_sv3d(
    crop_rgb_path: Path,
    crop_mask_path: Path,
    image_dir: Path,
    num_views: int,
    elevation_deg: float,
    resolution: int,
    seed: int,
    device: str,
    model_path: str | None,
) -> MultiviewGenerateResult:
    try:
        images = _run_sv3d_inference(crop_rgb_path, crop_mask_path, image_dir, num_views, elevation_deg,
                                     resolution, seed, device, model_path)
    except Exception as exc:
        log.warning("SV3D generation failed, falling back to crop copies: %s", exc)
        images = _fallback_copy_crop(crop_rgb_path, image_dir, num_views)

    orbit_cameras, _ = _generate_orbit_cameras(num_views, elevation_deg, resolution)
    cameras_path = image_dir / "cameras.json"
    with open(cameras_path, "w", encoding="utf-8") as f:
        json.dump(orbit_cameras, f, indent=2)

    return MultiviewGenerateResult(
        image_dir=image_dir,
        cameras_path=cameras_path,
        images=images,
        num_views=len(images),
        elevation_deg=elevation_deg,
    )


def _generate_wonder3d(
    crop_rgb_path: Path,
    crop_mask_path: Path,
    image_dir: Path,
    num_views: int,
    resolution: int,
    seed: int,
    device: str,
    model_path: str | None,
    cfg: ObjectRefinementConfig,
) -> MultiviewGenerateResult:
    if num_views not in {4, 5, 6}:
        log.warning("Wonder3D supports 4/5/6 fixed views; using 6 instead of %d", num_views)
        num_views = 6
    try:
        images = _run_wonder3d_inference(
            crop_rgb_path,
            crop_mask_path,
            image_dir,
            num_views,
            resolution,
            seed,
            device,
            model_path,
            cfg,
        )
    except Exception as exc:
        log.warning("Wonder3D generation failed, falling back to crop copies: %s", exc)
        images = _fallback_copy_crop(crop_rgb_path, image_dir, num_views)

    cameras = _generate_wonder3d_cameras(num_views, resolution)
    cameras_path = image_dir / "cameras.json"
    with open(cameras_path, "w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=2)

    return MultiviewGenerateResult(
        image_dir=image_dir,
        cameras_path=cameras_path,
        images=images,
        num_views=len(images),
        elevation_deg=0.0,
    )


def _run_wonder3d_inference(
    crop_rgb_path: Path,
    crop_mask_path: Path,
    image_dir: Path,
    num_views: int,
    resolution: int,
    seed: int,
    device: str,
    model_path: str | None,
    cfg: ObjectRefinementConfig,
) -> list[Path]:
    import os
    import PIL.Image
    from einops import rearrange

    wonder_root = _ensure_wonder3d_path()
    from mvdiffusion.data.single_image_dataset import SingleImageDataset
    from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline

    input_dir = (image_dir / "wonder3d_input").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    rgba_path = _write_wonder3d_rgba_condition(
        crop_rgb_path,
        crop_mask_path,
        input_dir / "condition.png",
    ).resolve()

    pretrained = str(model_path) if model_path else "flamehaze1115/wonder3d-v1.0"
    local_files_only = bool(model_path and Path(model_path).exists())
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    old_cwd = Path.cwd()
    try:
        os.chdir(wonder_root)
        dataset = SingleImageDataset(
            root_dir=str(input_dir),
            num_views=int(num_views),
            bg_color="white",
            img_wh=(256, 256),
            num_validation_samples=1,
            crop_size=int(cfg.wonder3d_crop_size),
            filepaths=[rgba_path.name],
        )
    finally:
        os.chdir(old_cwd)

    if len(dataset) == 0:
        raise ValueError(f"Wonder3D condition image could not be loaded: {rgba_path}")

    pipe = MVDiffusionImagePipeline.from_pretrained(
        pretrained,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    batch = dataset[0]
    imgs_in = torch.cat([batch["imgs_in"].unsqueeze(0)] * 2, dim=0)
    camera_embeddings = torch.cat([batch["camera_embeddings"].unsqueeze(0)] * 2, dim=0)
    task_embeddings = torch.cat(
        [
            batch["normal_task_embeddings"].unsqueeze(0),
            batch["color_task_embeddings"].unsqueeze(0),
        ],
        dim=0,
    )
    camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)
    imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")
    camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

    generator_device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    autocast_device = device.split(":")[0]
    with torch.no_grad():
        with torch.autocast(
            device_type=autocast_device,
            dtype=dtype,
            enabled=device.startswith("cuda"),
        ):
            out = pipe(
                imgs_in,
                camera_embeddings,
                generator=generator,
                guidance_scale=float(cfg.wonder3d_guidance_scale),
                output_type="pt",
                num_images_per_prompt=1,
                eta=1.0,
            ).images

    half = out.shape[0] // 2
    normals = out[:half]
    colors = out[half:]
    view_names = _wonder3d_view_names(num_views)
    normal_dir = image_dir / "normals"
    normal_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for idx, view in enumerate(view_names):
        color = colors[idx].detach().cpu().float().clamp(0.0, 1.0)
        normal = normals[idx].detach().cpu().float().clamp(0.0, 1.0)
        color_path = image_dir / f"view_{idx:02d}.png"
        named_color_path = image_dir / f"rgb_000_{view}.png"
        normal_path = normal_dir / f"normals_000_{view}.png"
        _save_chw_image(color, color_path, resolution)
        _save_chw_image(color, named_color_path, resolution)
        _save_chw_image(normal, normal_path, resolution)
        output_paths.append(color_path)

    PIL.Image.open(str(rgba_path)).save(str(image_dir / "wonder3d_condition.png"))
    return output_paths


def _write_wonder3d_rgba_condition(
    crop_rgb_path: Path,
    crop_mask_path: Path | None,
    output_path: Path,
) -> Path:
    import PIL.Image

    rgb = PIL.Image.open(str(crop_rgb_path)).convert("RGB")
    if crop_mask_path is not None and crop_mask_path.exists():
        alpha = PIL.Image.open(str(crop_mask_path)).convert("L").resize(rgb.size, PIL.Image.BILINEAR)
    else:
        arr = np.asarray(rgb).astype("uint8")
        alpha_np = ((arr < 245).any(axis=-1).astype("uint8") * 255)
        alpha = PIL.Image.fromarray(alpha_np, mode="L")
    if max(alpha.getextrema()) <= 0:
        raise ValueError(f"Wonder3D requires a non-empty alpha mask: {crop_mask_path}")
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(str(output_path))
    return output_path


def _save_chw_image(tensor: torch.Tensor, path: Path, resolution: int) -> None:
    import PIL.Image

    arr = tensor.mul(255.0).add_(0.5).clamp_(0.0, 255.0).permute(1, 2, 0).to(torch.uint8).numpy()
    image = PIL.Image.fromarray(arr)
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), PIL.Image.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))


def _wonder3d_view_names(num_views: int) -> list[str]:
    if num_views == 4:
        return ["front", "right", "back", "left"]
    if num_views == 5:
        return ["front", "front_right", "right", "back", "left"]
    return ["front", "front_right", "right", "back", "left", "front_left"]


def _wonder3d_view_azimuths(num_views: int) -> list[float]:
    if num_views == 4:
        return [0.0, 90.0, 180.0, -90.0]
    if num_views == 5:
        return [0.0, 45.0, 90.0, 180.0, -90.0]
    return [0.0, 45.0, 90.0, 180.0, -90.0, -45.0]


def _generate_wonder3d_cameras(num_views: int, resolution: int) -> list[dict[str, Any]]:
    view_names = _wonder3d_view_names(num_views)
    azimuths = _wonder3d_view_azimuths(num_views)
    cameras_data = []
    for i, (view, azimuth) in enumerate(zip(view_names, azimuths)):
        cam = _camera_looking_at(
            target=torch.zeros(3),
            distance=1.5,
            elevation_deg=0.0,
            azimuth_deg=float(azimuth),
            resolution=resolution,
        )
        cameras_data.append(
            {
                "view_index": i,
                "view_name": view,
                "azimuth_deg": float(azimuth),
                "elevation_deg": 0.0,
                "camera_model": "orthographic_relative_input",
                "extrinsic": cam.extrinsic.detach().cpu().tolist(),
                "intrinsic": cam.intrinsic.detach().cpu().tolist(),
                "width": resolution,
                "height": resolution,
            }
        )
    return cameras_data


def _is_sv3d_diffusers_model(model_path: str | None) -> bool:
    if not model_path:
        return False
    index_path = Path(model_path) / "model_index.json"
    if not index_path.exists():
        return False
    try:
        with open(index_path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    return str(payload.get("_class_name", "")) == "StableVideo3DDiffusionPipeline"


def _load_sv3d_rgb_image(crop_rgb_path: Path, crop_mask_path: Path | None = None):
    import PIL.Image

    image = PIL.Image.open(str(crop_rgb_path))
    image.load()
    if crop_mask_path is not None and crop_mask_path.exists():
        mask = PIL.Image.open(str(crop_mask_path)).convert("L").resize(image.size)
        rgb = image.convert("RGB")
        canvas = PIL.Image.new("RGB", image.size, (255, 255, 255))
        canvas.paste(rgb, mask=mask)
        return canvas
    if len(image.split()) == 4:
        rgb = PIL.Image.new("RGB", image.size, (255, 255, 255))
        rgb.paste(image, mask=image.split()[3])
        return rgb
    return image.convert("RGB")


def _run_sv3d_diffusers_inference(
    crop_rgb_path: Path,
    crop_mask_path: Path,
    image_dir: Path,
    num_views: int,
    elevation_deg: float,
    resolution: int,
    seed: int,
    device: str,
    model_path: str,
) -> list[Path]:
    _ensure_diffusers_sv3d_path()
    from diffusers_sv3d import StableVideo3DDiffusionPipeline

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = StableVideo3DDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    num_frames = max(int(num_views), 1)
    sv3d_res = 576 if resolution >= 512 else max(256, int(resolution))
    elevations_deg = [float(elevation_deg)] * num_frames
    polars_rad = [np.deg2rad(90.0 - e) for e in elevations_deg]
    azimuths_deg = np.linspace(0.0, 360.0, num_frames + 1)[1:] % 360.0
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360.0) for a in azimuths_deg]
    azimuths_rad[:-1].sort()

    image = _load_sv3d_rgb_image(crop_rgb_path, crop_mask_path).resize((sv3d_res, sv3d_res))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=device.startswith("cuda")):
            frames = pipe(
                image,
                height=sv3d_res,
                width=sv3d_res,
                num_frames=num_frames,
                decode_chunk_size=8,
                polars_rad=polars_rad,
                azimuths_rad=azimuths_rad,
                generator=generator,
            ).frames[0]

    output_paths = []
    for i, frame in enumerate(frames[:num_views]):
        path = image_dir / f"view_{i:02d}.png"
        if sv3d_res != resolution:
            frame = frame.resize((resolution, resolution))
        frame.save(str(path))
        output_paths.append(path)
    return output_paths


def _run_sv3d_inference(
    crop_rgb_path: Path,
    crop_mask_path: Path,
    image_dir: Path,
    num_views: int,
    elevation_deg: float,
    resolution: int,
    seed: int,
    device: str,
    model_path: str | None,
) -> list[Path]:
    if model_path and _is_sv3d_diffusers_model(model_path):
        return _run_sv3d_diffusers_inference(
            crop_rgb_path,
            crop_mask_path,
            image_dir,
            num_views,
            elevation_deg,
            resolution,
            seed,
            device,
            model_path,
        )

    from diffusers import StableVideoDiffusionPipeline

    pretrained = str(model_path) if model_path else "chenguolin/sv3d-diffusers"
    local_files_only = bool(model_path and Path(model_path).exists())
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        pretrained,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        variant=None if local_files_only else ("fp16" if device.startswith("cuda") else None),
        local_files_only=local_files_only,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    image = _load_sv3d_rgb_image(crop_rgb_path, crop_mask_path).resize((resolution, resolution))
    frames = pipe(
        image,
        num_frames=num_views,
        motion_bucket_id=127,
        noise_aug_strength=0.02,
        decode_chunk_size=8,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).frames[0]

    output_paths = []
    for i, frame in enumerate(frames):
        path = image_dir / f"view_{i:02d}.png"
        frame.save(str(path))
        output_paths.append(path)
    return output_paths


def _generate_zero123plus(
    crop_rgb_path: Path,
    crop_mask_path: Path,
    image_dir: Path,
    num_views: int,
    elevation_deg: float,
    resolution: int,
    seed: int,
    device: str,
    model_path: str | None,
) -> MultiviewGenerateResult:
    try:
        images = _run_zero123plus_inference(crop_rgb_path, crop_mask_path, image_dir,
                                            num_views, elevation_deg, resolution, seed, device, model_path)
    except Exception as exc:
        log.warning("Zero123++ generation failed, falling back to crop copies: %s", exc)
        images = _fallback_copy_crop(crop_rgb_path, image_dir, num_views)

    orbit_cameras, _ = _generate_orbit_cameras(num_views, elevation_deg, resolution)
    cameras_path = image_dir / "cameras.json"
    with open(cameras_path, "w", encoding="utf-8") as f:
        json.dump(orbit_cameras, f, indent=2)

    return MultiviewGenerateResult(
        image_dir=image_dir,
        cameras_path=cameras_path,
        images=images,
        num_views=len(images),
        elevation_deg=elevation_deg,
    )


def _run_zero123plus_inference(
    crop_rgb_path: Path,
    crop_mask_path: Path,
    image_dir: Path,
    num_views: int,
    elevation_deg: float,
    resolution: int,
    seed: int,
    device: str,
    model_path: str | None,
) -> list[Path]:
    from diffusers import StableDiffusionPipeline
    import PIL.Image

    pretrained = str(model_path) if model_path else "stabilityai/zero123plus-v2"
    local_files_only = bool(model_path and Path(model_path).exists())
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        variant=None if local_files_only else ("fp16" if device.startswith("cuda") else None),
        local_files_only=local_files_only,
    )
    pipe = pipe.to(device)

    image = PIL.Image.open(str(crop_rgb_path)).convert("RGB")
    image = image.resize((resolution, resolution), PIL.Image.LANCZOS)
    result = pipe(
        image,
        num_inference_steps=30,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]

    grid_w, grid_h = 6, 2
    tile_w = result.width // grid_w
    tile_h = result.height // grid_h
    n_tiles = grid_w * grid_h

    output_paths = []
    for idx in range(min(num_views, n_tiles)):
        row = idx // grid_w
        col = idx % grid_w
        tile = result.crop((col * tile_w, row * tile_h, (col + 1) * tile_w, (row + 1) * tile_h))
        path = image_dir / f"view_{idx:02d}.png"
        tile.save(str(path))
        output_paths.append(path)
    return output_paths


def _fallback_copy_crop(crop_rgb_path: Path, image_dir: Path, num_views: int) -> list[Path]:
    import shutil

    output_paths = []
    for i in range(num_views):
        path = image_dir / f"view_{i:02d}.png"
        shutil.copy2(str(crop_rgb_path), str(path))
        output_paths.append(path)
    return output_paths
