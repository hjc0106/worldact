from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldact.data.gaussians import GaussianParams, GaussianScene
from worldact.data.poses import CameraPose
from worldact.stage2_object_refinement.common import ObjectRefinementConfig


@dataclass
class ReconstructResult:
    ply_path: Path | None
    metadata_path: Path | None
    stats: dict[str, Any]


def reconstruct_3dgs(
    multiview_images: list[Path],
    cameras_json_path: Path | None,
    work_dir: Path,
    cfg: ObjectRefinementConfig,
    *,
    model: str = "gs_lrm",
    resolution: int = 512,
    device: str = "cuda",
) -> ReconstructResult:
    output_ply = work_dir / "generated.ply"
    metadata_path = work_dir / "metadata.json"

    if model == "gs_lrm":
        return _reconstruct_gs_lrm(multiview_images, cameras_json_path, output_ply,
                                   metadata_path, work_dir, resolution, device)
    elif model == "instant_splat":
        return _reconstruct_instant_splat(multiview_images, cameras_json_path, output_ply,
                                         metadata_path, work_dir, resolution, device)
    else:
        return ReconstructResult(
            ply_path=None,
            metadata_path=None,
            stats={"model": "unknown", "num_gaussians": 0, "skipped": True},
        )


def _reconstruct_gs_lrm(
    multiview_images: list[Path],
    cameras_json_path: Path | None,
    output_ply: Path,
    metadata_path: Path,
    work_dir: Path,
    resolution: int,
    device: str,
) -> ReconstructResult:
    try:
        return _run_gs_lrm_inference(multiview_images, cameras_json_path, output_ply,
                                     metadata_path, work_dir, resolution, device)
    except ImportError:
        pass
    return ReconstructResult(
        ply_path=None,
        metadata_path=None,
        stats={"model": "none", "num_gaussians": 0, "fallback": "gs_lrm_not_installed"},
    )


def _run_gs_lrm_inference(
    multiview_images: list[Path],
    cameras_json_path: Path | None,
    output_ply: Path,
    metadata_path: Path,
    work_dir: Path,
    resolution: int,
    device: str,
) -> ReconstructResult:
    import PIL.Image
    from gs_lrm import GSLRMModel

    model = GSLRMModel.from_pretrained("VoyagerX/GS-LRM")
    model = model.to(device)
    model.eval()

    images = []
    cameras_list = []
    for img_path in multiview_images:
        img = PIL.Image.open(str(img_path)).convert("RGB").resize((resolution, resolution))
        images.append(np.array(img).astype(np.float32) / 255.0)

    if cameras_json_path and cameras_json_path.exists():
        with open(cameras_json_path, encoding="utf-8") as f:
            cameras_data = json.load(f)
        for cam_data in cameras_data[:len(images)]:
            cameras_list.append(CameraPose(
                extrinsic=torch.tensor(cam_data["extrinsic"], dtype=torch.float32),
                intrinsic=torch.tensor(cam_data["intrinsic"], dtype=torch.float32),
                width=int(cam_data.get("width", resolution)),
                height=int(cam_data.get("height", resolution)),
            ))

    with torch.no_grad():
        images_tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device)
        result = model.reconstruct(images_tensor, cameras=cameras_list if cameras_list else None)

    means = result["means"].cpu()
    scales = result["scales"].cpu()
    rotations = result["rotations"].cpu()
    opacities = result["opacities"].cpu()
    colors = result["colors"].cpu()

    params = GaussianParams(means=means, scales=scales, rotations=rotations,
                           opacities=opacities, colors=colors)
    scene = GaussianScene(params=params)
    scene.to_ply(output_ply)

    metadata = {"transform": torch.eye(4).tolist(), "source": "gs_lrm"}
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return ReconstructResult(
        ply_path=output_ply,
        metadata_path=metadata_path,
        stats={"model": "gs_lrm", "num_gaussians": scene.num_gaussians},
    )


def _reconstruct_instant_splat(
    multiview_images: list[Path],
    cameras_json_path: Path | None,
    output_ply: Path,
    metadata_path: Path,
    work_dir: Path,
    resolution: int,
    device: str,
) -> ReconstructResult:
    try:
        result = _run_instant_splat_inference(
            multiview_images,
            cameras_json_path,
            output_ply,
            metadata_path,
            work_dir,
            resolution,
            device,
        )
        if result.ply_path is not None:
            return result
    except ImportError:
        pass
    if cameras_json_path and cameras_json_path.exists() and multiview_images:
        return _fallback_point_cloud(
            multiview_images,
            cameras_json_path,
            output_ply,
            metadata_path,
            resolution,
            device,
        )
    return ReconstructResult(
        ply_path=None,
        metadata_path=None,
        stats={"model": "none", "num_gaussians": 0, "fallback": "instant_splat_not_installed"},
    )


def _instant_splat_python() -> str:
    import os
    import sys
    from pathlib import Path

    env_python = os.environ.get("INSTANT_SPLAT_PYTHON")
    if env_python:
        return env_python
    default = Path.home() / "miniconda3/envs/instantsplat/bin/python"
    if default.exists():
        return str(default)
    return sys.executable


def _instant_splat_env() -> dict[str, str]:
    import os

    env = os.environ.copy()
    if "INSTANT_SPLAT_ROOT" not in env:
        repo_root = Path(__file__).resolve().parents[2]
        instantsplat_root = repo_root / "third_party" / "InstantSplat"
        if instantsplat_root.exists():
            env["INSTANT_SPLAT_ROOT"] = str(instantsplat_root)
    return env


def _run_instant_splat_inference(
    multiview_images: list[Path],
    cameras_json_path: Path | None,
    output_ply: Path,
    metadata_path: Path,
    work_dir: Path,
    resolution: int,
    device: str,
) -> ReconstructResult:
    import subprocess

    python = _instant_splat_python()
    cmd = [
        python, "-m", "instant_splat",
        "--images", str(work_dir / "multiview_generate"),
        "--output", str(work_dir / "instant_splat_out"),
        "--resolution", str(resolution),
        "--device", device,
    ]
    if cameras_json_path:
        cmd.extend(["--cameras", str(cameras_json_path)])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,
        env=_instant_splat_env(),
    )

    inst_out = work_dir / "instant_splat_out" / "point_cloud.ply"
    if inst_out.exists() and inst_out.stat().st_size > 0:
        import shutil
        shutil.copy2(str(inst_out), str(output_ply))
        metadata = {"transform": torch.eye(4).tolist(), "source": "instant_splat"}
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return ReconstructResult(
            ply_path=output_ply,
            metadata_path=metadata_path,
            stats={"model": "instant_splat", "returncode": result.returncode},
        )

    return ReconstructResult(
        ply_path=None,
        metadata_path=None,
        stats={
            "model": "instant_splat",
            "returncode": result.returncode,
            "fallback": "instant_splat_no_output",
            "stderr": result.stderr[-500:] if result.stderr else "",
        },
    )


def _fallback_point_cloud(
    multiview_images: list[Path],
    cameras_json_path: Path,
    output_ply: Path,
    metadata_path: Path,
    resolution: int,
    device: str,
) -> ReconstructResult:
    import PIL.Image

    with open(cameras_json_path, encoding="utf-8") as f:
        cameras_data = json.load(f)

    all_points = []
    all_colors = []
    depth_scale = 0.01

    for img_path, cam_data in zip(multiview_images[:len(cameras_data)], cameras_data):
        img = PIL.Image.open(str(img_path)).convert("RGB").resize((resolution, resolution))
        img_np = np.array(img).astype(np.float32) / 255.0

        extrinsic = np.array(cam_data["extrinsic"], dtype=np.float32)
        intrinsic = np.array(cam_data["intrinsic"], dtype=np.float32)
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        h, w = img_np.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        xs_f = xs.astype(np.float32)
        ys_f = ys.astype(np.float32)

        step = 4
        xs_s = xs_f[::step, ::step].flatten()
        ys_s = ys_f[::step, ::step].flatten()
        colors_s = img_np[::step, ::step, :].reshape(-1, 3)

        z = depth_scale * np.ones_like(xs_s)
        x_camera = (xs_s - cx) * z / fx
        y_camera = (ys_s - cy) * z / fy
        pts_camera = np.stack([x_camera, y_camera, z], axis=-1)

        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        pts_world = (R.T @ (pts_camera - t).T).T

        all_points.append(pts_world)
        all_colors.append(colors_s)

    if all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
    else:
        points = np.zeros((1, 3), dtype=np.float32)
        colors = np.zeros((1, 3), dtype=np.float32)

    means = torch.from_numpy(points).float()
    n = means.shape[0]
    params = GaussianParams(
        means=means,
        scales=torch.full((n, 3), -5.0),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacities=torch.full((n, 1), 2.0),
        colors=torch.from_numpy(colors).float(),
    )
    scene = GaussianScene(params=params)
    scene.to_ply(output_ply)

    metadata = {"transform": torch.eye(4).tolist(), "source": "fallback_pointcloud"}
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return ReconstructResult(
        ply_path=output_ply,
        metadata_path=metadata_path,
        stats={"model": "fallback_pointcloud", "num_gaussians": n},
    )


def _reconstruct_dummy(
    multiview_images: list[Path],
    cameras_json_path: Path | None,
    output_ply: Path,
    metadata_path: Path,
    work_dir: Path,
) -> ReconstructResult:
    params = GaussianParams(
        means=torch.zeros(1, 3),
        scales=torch.full((1, 3), -5.0),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        opacities=torch.full((1, 1), -5.0),
        colors=torch.zeros(1, 3),
    )
    scene = GaussianScene(params=params)
    scene.to_ply(output_ply)

    metadata = {"transform": torch.eye(4).tolist(), "source": "dummy"}
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return ReconstructResult(
        ply_path=output_ply,
        metadata_path=metadata_path,
        stats={"model": "dummy", "num_gaussians": 1},
    )