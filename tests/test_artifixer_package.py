from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch

from worldact.data.gaussians import GaussianParams, GaussianScene
from worldact.data.masks import MaskSet, ObjectMask
from worldact.data.poses import CameraPose
from worldact.rendering.rasterizer import _ensure_ninja_on_path
from worldact.stage2_artifixer.anchors import _render_rgb_alpha, object_centered_anchor_camera
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
    anchors = json.loads((package_dir / "anchors.json").read_text(encoding="utf-8"))["anchors"]
    for anchor in anchors:
        assert anchor["fl_x"] == pytest.approx(anchor["fl_y"])
        assert anchor["cx"] == pytest.approx((cfg.image_size - 1) * 0.5)
        assert anchor["cy"] == pytest.approx((cfg.image_size - 1) * 0.5)
    result = validate_package_dir(package_dir)
    assert result.ok, result.errors
    archive = archive_package(package_dir)
    assert archive.exists() and archive.stat().st_size > 0


def test_ninja_beside_active_python_is_added_to_path(monkeypatch) -> None:
    ninja = Path(sys.executable).resolve().parent / "ninja"
    if not ninja.is_file():
        pytest.skip("active Python environment does not provide Ninja")
    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))

    _ensure_ninja_on_path()

    assert os.environ["PATH"].split(os.pathsep)[0] == str(ninja.parent)


def test_cuda_anchor_render_failure_does_not_fallback() -> None:
    class BrokenCudaScene:
        device = torch.device("cuda")

        def render(self, _camera):
            raise RuntimeError("synthetic CUDA failure")

    camera = CameraPose(
        extrinsic=torch.eye(4),
        intrinsic=torch.eye(3),
        width=8,
        height=8,
    )

    with pytest.raises(RuntimeError, match="refusing to generate gray fallback"):
        _render_rgb_alpha(BrokenCudaScene(), camera)  # type: ignore[arg-type]


def test_object_centered_anchor_camera_is_square_and_looks_at_origin() -> None:
    source = CameraPose(
        extrinsic=_look_at([10.0, -5.0, 3.0]),
        intrinsic=torch.tensor([[900.0, 0.0, 12.0], [0.0, 450.0, 900.0], [0.0, 0.0, 1.0]]),
        width=1920,
        height=1080,
    )
    cfg = PrepareConfig(image_size=512, fov_deg=60.0)
    direction = torch.tensor([1.0, -0.5, 0.3])

    camera = object_centered_anchor_camera(source, direction, cfg)

    assert camera.width == camera.height == 512
    assert float(camera.intrinsic[0, 0]) == pytest.approx(float(camera.intrinsic[1, 1]))
    assert float(camera.intrinsic[0, 2]) == pytest.approx(255.5)
    assert float(camera.intrinsic[1, 2]) == pytest.approx(255.5)
    c2w = camera.c2w
    expected_forward = -c2w[:3, 3] / c2w[:3, 3].norm()
    assert torch.allclose(c2w[:3, 2], expected_forward, atol=1e-5)
