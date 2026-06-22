from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.poses import CameraPose
from worldact.stage2_object_refinement.common import ObjectRefinementConfig


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
    resolution: int = 512,
    seed: int = 42,
    device: str = "cuda",
) -> MultiviewGenerateResult:
    n_views = num_views or cfg.num_generation_views
    image_dir = work_dir / "multiview_generate"
    image_dir.mkdir(parents=True, exist_ok=True)

    if model == "sv3d":
        return _generate_sv3d(crop_rgb_path, crop_mask_path, image_dir, n_views,
                              elevation_deg, resolution, seed, device)
    elif model == "zero123plus":
        return _generate_zero123plus(crop_rgb_path, crop_mask_path, image_dir, n_views,
                                     elevation_deg, resolution, seed, device)
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
    elevation_deg = float(torch.asin(forward[2]).item() * 180.0 / 3.141592653589793)
    return elevation_deg


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
) -> MultiviewGenerateResult:
    try:
        images = _run_sv3d_inference(crop_rgb_path, image_dir, num_views, elevation_deg,
                                     resolution, seed, device)
    except ImportError:
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


def _run_sv3d_inference(
    crop_rgb_path: Path,
    image_dir: Path,
    num_views: int,
    elevation_deg: float,
    resolution: int,
    seed: int,
    device: str,
) -> list[Path]:
    from diffusers import StableVideoDiffusionPipeline
    import PIL.Image

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/sv3d",
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        variant="fp16" if device.startswith("cuda") else None,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    image = PIL.Image.open(str(crop_rgb_path)).convert("RGB")
    image = image.resize((resolution, resolution), PIL.Image.LANCZOS)
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
) -> MultiviewGenerateResult:
    try:
        images = _run_zero123plus_inference(crop_rgb_path, crop_mask_path, image_dir,
                                            num_views, elevation_deg, resolution, seed, device)
    except ImportError:
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
) -> list[Path]:
    from diffusers import StableDiffusionPipeline
    import PIL.Image

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/zero123plus-v2",
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        variant="fp16" if device.startswith("cuda") else None,
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