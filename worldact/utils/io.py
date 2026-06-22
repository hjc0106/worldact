from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from worldact.data.gaussians import GaussianParams
from worldact.data.poses import CameraPose


def load_gaussians_ply(path: Path) -> GaussianParams:
    from plyfile import PlyData

    plydata = PlyData.read(str(path))
    el = plydata["vertex"]
    props = {p.name for p in el.properties}
    if "packed_position" in props:
        return _load_compressed_gaussians_ply(plydata)

    def require(names: list[str]) -> torch.Tensor:
        missing = [n for n in names if n not in props]
        if missing:
            raise ValueError(f"{path} missing PLY properties: {missing}")
        arr = np.stack([np.asarray(el[n], dtype=np.float32) for n in names], axis=-1)
        return torch.from_numpy(arr)

    means = require(["x", "y", "z"])
    scales = require(["scale_0", "scale_1", "scale_2"])
    rotations = require(["rot_0", "rot_1", "rot_2", "rot_3"])
    opacities = torch.from_numpy(np.asarray(el["opacity"], dtype=np.float32)).unsqueeze(-1)
    sh_dc = require(["f_dc_0", "f_dc_1", "f_dc_2"])
    rest_names = sorted(
        [p for p in props if p.startswith("f_rest_")],
        key=lambda s: int(s.rsplit("_", 1)[-1]),
    )
    if rest_names:
        sh_rest = torch.from_numpy(np.stack([np.asarray(el[n], dtype=np.float32) for n in rest_names], axis=-1))
        colors = torch.cat([sh_dc, sh_rest], dim=-1)
    else:
        colors = sh_dc
    return GaussianParams(means=means, scales=scales, rotations=rotations, opacities=opacities, colors=colors)


def _load_compressed_gaussians_ply(plydata) -> GaussianParams:
    chunks = plydata["chunk"]
    verts = plydata["vertex"]
    n = verts.count
    chunk_size = 256

    def chunk_xyz(prefix_min: str, prefix_max: str) -> tuple[np.ndarray, np.ndarray]:
        mins = np.stack(
            [
                np.asarray(chunks[f"min_{prefix_min}"], dtype=np.float32),
                np.asarray(chunks[f"min_{prefix_max}"], dtype=np.float32),
                np.asarray(chunks["min_z" if prefix_min == "x" else "min_scale_z"], dtype=np.float32),
            ],
            axis=-1,
        )
        maxs = np.stack(
            [
                np.asarray(chunks[f"max_{prefix_min}"], dtype=np.float32),
                np.asarray(chunks[f"max_{prefix_max}"], dtype=np.float32),
                np.asarray(chunks["max_z" if prefix_min == "x" else "max_scale_z"], dtype=np.float32),
            ],
            axis=-1,
        )
        return mins, maxs

    pos_min, pos_max = chunk_xyz("x", "y")
    scale_min, scale_max = chunk_xyz("scale_x", "scale_y")
    color_min = np.stack(
        [np.asarray(chunks[f"min_{c}"], dtype=np.float32) for c in ("r", "g", "b")],
        axis=-1,
    )
    color_max = np.stack(
        [np.asarray(chunks[f"max_{c}"], dtype=np.float32) for c in ("r", "g", "b")],
        axis=-1,
    )
    chunk_ids = np.minimum(np.arange(n) // chunk_size, chunks.count - 1)

    packed_pos = np.asarray(verts["packed_position"], dtype=np.uint32)
    px = ((packed_pos >> 21) & 0x7FF).astype(np.float32) / 2047.0
    py = ((packed_pos >> 11) & 0x3FF).astype(np.float32) / 1023.0
    pz = (packed_pos & 0x7FF).astype(np.float32) / 2047.0
    pos_norm = np.stack([px, py, pz], axis=-1)
    means = pos_min[chunk_ids] + pos_norm * (pos_max[chunk_ids] - pos_min[chunk_ids])

    packed_scale = np.asarray(verts["packed_scale"], dtype=np.uint32)
    sx = ((packed_scale >> 21) & 0x7FF).astype(np.float32) / 2047.0
    sy = ((packed_scale >> 11) & 0x3FF).astype(np.float32) / 1023.0
    sz = (packed_scale & 0x7FF).astype(np.float32) / 2047.0
    scale_norm = np.stack([sx, sy, sz], axis=-1)
    scales = scale_min[chunk_ids] + scale_norm * (scale_max[chunk_ids] - scale_min[chunk_ids])

    packed_rot = np.asarray(verts["packed_rotation"], dtype=np.uint32)
    largest = ((packed_rot >> 30) & 0x3).astype(np.int32)
    norm = np.float32(np.sqrt(0.5))
    a = ((packed_rot >> 20) & 0x3FF).astype(np.float32) / 1023.0 * 2.0 * norm - norm
    b = ((packed_rot >> 10) & 0x3FF).astype(np.float32) / 1023.0 * 2.0 * norm - norm
    c = (packed_rot & 0x3FF).astype(np.float32) / 1023.0 * 2.0 * norm - norm
    d = np.sqrt(np.maximum(0.0, 1.0 - a * a - b * b - c * c))
    rotations = np.zeros((n, 4), dtype=np.float32)
    for idx in range(4):
        mask = largest == idx
        if not np.any(mask):
            continue
        if idx == 0:
            rotations[mask] = np.stack([d[mask], a[mask], b[mask], c[mask]], axis=-1)
        elif idx == 1:
            rotations[mask] = np.stack([a[mask], d[mask], b[mask], c[mask]], axis=-1)
        elif idx == 2:
            rotations[mask] = np.stack([a[mask], b[mask], d[mask], c[mask]], axis=-1)
        else:
            rotations[mask] = np.stack([a[mask], b[mask], c[mask], d[mask]], axis=-1)

    packed_color = np.asarray(verts["packed_color"], dtype=np.uint32)
    cr = ((packed_color >> 24) & 0xFF).astype(np.float32) / 255.0
    cg = ((packed_color >> 16) & 0xFF).astype(np.float32) / 255.0
    cb = ((packed_color >> 8) & 0xFF).astype(np.float32) / 255.0
    ca = (packed_color & 0xFF).astype(np.float32) / 255.0
    color_norm = np.stack([cr, cg, cb], axis=-1)
    colors_dc = color_min[chunk_ids] + color_norm * (color_max[chunk_ids] - color_min[chunk_ids])
    alpha = np.clip(ca, 1e-6, 1.0 - 1e-6)
    opacities = np.log(alpha / (1.0 - alpha)).astype(np.float32)

    sh_names: list[str] = []
    if "sh" in plydata:
        sh_names = sorted(
            [prop.name for prop in plydata["sh"].properties],
            key=lambda s: int(s.rsplit("_", 1)[-1]),
        )
    if sh_names:
        sh_el = plydata["sh"]
        sh_rest = np.stack([np.asarray(sh_el[name], dtype=np.float32) for name in sh_names], axis=-1)
        colors = np.concatenate([colors_dc, sh_rest], axis=-1)
    else:
        colors = colors_dc

    return GaussianParams(
        means=torch.from_numpy(means),
        scales=torch.from_numpy(scales),
        rotations=torch.from_numpy(rotations),
        opacities=torch.from_numpy(opacities).unsqueeze(-1),
        colors=torch.from_numpy(colors),
    )


def save_gaussians_ply(params: GaussianParams, path: Path) -> None:
    from plyfile import PlyData, PlyElement

    path.parent.mkdir(parents=True, exist_ok=True)
    means = params.means.detach().cpu().numpy()
    scales = params.scales.detach().cpu().numpy()
    rotations = params.rotations.detach().cpu().numpy()
    opacities = params.opacities.detach().cpu().numpy().reshape(-1)
    colors = params.colors.detach().cpu().numpy()
    sh_dim = colors.shape[1]
    dtype = (
        [("x", "f4"), ("y", "f4"), ("z", "f4")]
        + [(f"scale_{i}", "f4") for i in range(3)]
        + [(f"rot_{i}", "f4") for i in range(4)]
        + [("opacity", "f4")]
        + [(f"f_dc_{i}", "f4") for i in range(min(3, sh_dim))]
        + [(f"f_rest_{i}", "f4") for i in range(max(0, sh_dim - 3))]
    )
    data = np.zeros(params.num_gaussians, dtype=dtype)
    data["x"], data["y"], data["z"] = means[:, 0], means[:, 1], means[:, 2]
    for i in range(3):
        data[f"scale_{i}"] = scales[:, i]
    for i in range(4):
        data[f"rot_{i}"] = rotations[:, i]
    data["opacity"] = opacities
    for i in range(min(3, sh_dim)):
        data[f"f_dc_{i}"] = colors[:, i]
    for i in range(max(0, sh_dim - 3)):
        data[f"f_rest_{i}"] = colors[:, 3 + i]
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


def save_cameras_json(cameras: list[CameraPose], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "extrinsic": cam.extrinsic.detach().cpu().tolist(),
            "intrinsic": cam.intrinsic.detach().cpu().tolist(),
            "width": cam.width,
            "height": cam.height,
        }
        for cam in cameras
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_cameras_json(path: Path) -> list[CameraPose]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        CameraPose(
            extrinsic=torch.tensor(item["extrinsic"], dtype=torch.float32),
            intrinsic=torch.tensor(item["intrinsic"], dtype=torch.float32),
            width=int(item["width"]),
            height=int(item["height"]),
        )
        for item in data
    ]
