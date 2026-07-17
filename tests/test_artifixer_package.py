from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from worldact.data.gaussians import GaussianParams, GaussianScene
from worldact.data.masks import MaskSet, ObjectMask
from worldact.data.poses import CameraPose
from worldact.stage2_artifixer.config import PrepareConfig
from worldact.stage2_artifixer.inputs import load_stage_artifixer_input
from worldact.stage2_artifixer.package import archive_package, prepare_object_package
from worldact.stage2_artifixer.validation import validate_package_dir


def _look_at(eye, target=None, up=None) -> torch.Tensor:
    target = torch.tensor([0.0, 0.0, 0.0] if target is None else target, dtype=torch.float32)
    up = torch.tensor([0.0, 0.0, 1.0] if up is None else up, dtype=torch.float32)
    eye = torch.tensor(eye, dtype=torch.float32)
    forward = target - eye
    forward = forward / forward.norm().clamp_min(1e-6)
    right = torch.linalg.cross(forward, up)
    right = right / right.norm().clamp_min(1e-6)
    true_up = torch.linalg.cross(right, forward)
    c2w = torch.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = -true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return torch.linalg.inv(c2w)


def make_synthetic_stage(tmp_path: Path) -> tuple[Path, Path]:
    stage = tmp_path / "stage1"
    stage.mkdir(parents=True)
    (stage / "background").mkdir()

    # Compact object blob around origin.
    means = torch.randn(40, 3) * 0.15
    params = GaussianParams(
        means=means,
        scales=torch.full((40, 3), -4.0),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(40, 1),
        opacities=torch.full((40, 1), 4.0),
        colors=torch.zeros(40, 3),
    )
    cameras = []
    masks = {}
    for i in range(8):
        az = 2 * math.pi * i / 8
        eye = [2.5 * math.cos(az), 2.5 * math.sin(az), 0.4]
        w2c = _look_at(eye)
        k = torch.tensor([[180.0, 0.0, 128.0], [0.0, 180.0, 128.0], [0.0, 0.0, 1.0]])
        cameras.append(CameraPose(extrinsic=w2c, intrinsic=k, width=256, height=256))
        m = torch.zeros(256, 256)
        m[80:176, 80:176] = 1.0
        masks[i] = m

    scene = GaussianScene(params=params, camera_trajectory=cameras)
    chunk = tmp_path / "chunk.ply"
    scene.to_ply(chunk)
    scene.save(stage / "background")

    obj_scene = GaussianScene(params=params, camera_trajectory=cameras)
    obj_ply = stage / "object_1_box.ply"
    obj_scene.to_ply(obj_ply)

    objects = [
        {
            "object_id": "1",
            "name": "box",
            "source_gaussian_indices": list(range(40)),
            "memory_state": {"name": "box", "text_prompt": "a complete white box"},
            "best_view_index": 0,
        }
    ]
    (stage / "objects.json").write_text(json.dumps(objects, indent=2), encoding="utf-8")
    mask_set = MaskSet(
        object_masks=[ObjectMask(object_id="1", object_name="box", masks=masks)],
        fused_masks={},
    )
    torch.save(mask_set, stage / "masks.pt")
    return chunk, stage


def test_prepare_object_package(tmp_path: Path) -> None:
    chunk, stage = make_synthetic_stage(tmp_path)
    stage_input = load_stage_artifixer_input(chunk, stage, device="cpu")
    cfg = PrepareConfig(
        anchor_count=4,
        min_anchor_count=3,
        target_count=8,
        horizontal_targets=8,
        high_elevation_targets=0,
        low_elevation_targets=0,
        image_size=128,
    )
    package_dir, manifest = prepare_object_package(stage_input, "1", tmp_path / "out", cfg)
    assert (package_dir / "object_local.ply").exists()
    assert (package_dir / "normalization.json").exists()
    assert (package_dir / "anchors.json").exists()
    assert (package_dir / "trajectory.json").exists()
    assert manifest.object_id == "1"
    assert manifest.anchor_count >= 3
    assert manifest.target_count >= 4
    result = validate_package_dir(package_dir)
    assert result.ok, result.errors
    archive = archive_package(package_dir)
    assert archive.exists() and archive.stat().st_size > 0
