from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from worldact.data.gaussians import GaussianParams, GaussianScene
from worldact.data.gaussians import RenderedView
from worldact.data.masks import ObjectMask
from worldact.data.poses import CameraPose
from worldact.stage2_object_refinement.backend import ExternalCommandBackend, NoneBackend, make_backend
from worldact.stage2_object_refinement.artifixer_backend import ArtiFixerBackend, prepare_artifixer_scene
from worldact.stage2_object_refinement.common import ObjectRefinementConfig
from worldact.stage2_object_refinement.confidence import foreground_confidence_keep
from worldact.stage2_object_refinement.generate import (
    _estimate_elevation_from_scene,
    _generate_wonder3d_cameras,
    _write_wonder3d_rgba_condition,
    resolve_condition_view_index,
    resolve_sv3d_elevation,
)
import numpy as np
from worldact.stage2_object_refinement.io import (
    StageRefinementInput,
    _expand_bbox_for_target_fill,
    _score_clean_views,
    load_stage_input,
    write_object_package,
)
from worldact.stage2_object_refinement.merge import merge_generated_object
from worldact.stage2_object_refinement.paint import refine_local_object
from worldact.stage2_object_refinement.prune import prune_object
from worldact.stage2_object_refinement.runner import refine_stage_objects


def make_scene(means: torch.Tensor) -> GaussianScene:
    n = int(means.shape[0])
    params = GaussianParams(
        means=means.float(),
        scales=torch.full((n, 3), -4.0),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacities=torch.full((n, 1), 4.0),
        colors=torch.zeros(n, 3),
    )
    return GaussianScene(params=params)


def make_camera() -> CameraPose:
    return CameraPose(
        extrinsic=torch.eye(4),
        intrinsic=torch.tensor([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]]),
        width=20,
        height=20,
    )


def test_prune_keeps_largest_component_and_filters_low_opacity() -> None:
    scene = make_scene(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [5.0, 5.0, 5.0],
            ]
        )
    )
    scene.params.opacities[1] = -8.0
    cfg = ObjectRefinementConfig(opacity_min=0.1, component_radius=0.5, use_depth_gate=False)

    result = prune_object(scene, None, cfg)

    assert result.kept_local_indices.tolist() == [0, 2]
    assert result.scene.num_gaussians == 2
    assert result.stats["after_opacity"] == 3
    assert result.stats["after_component"] == 2


def test_prune_reports_core_uncertain_and_rejected_indices() -> None:
    scene = make_scene(torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.9, 0.9, 1.0]]))
    scene.camera_trajectory = [make_camera()]
    mask = torch.zeros(20, 20)
    mask[9:12, 9:12] = 1.0
    obj_mask = ObjectMask(object_id="x", object_name="generic", masks={0: mask})
    cfg = ObjectRefinementConfig(use_depth_gate=False, component_radius=0.0, min_mask_support=1)

    result = prune_object(scene, obj_mask, cfg)

    assert result.object_core_indices is not None
    assert result.background_reject_indices is not None
    assert result.background_reject_indices.tolist() == [2]
    assert result.stats["separation"]["background_reject"] == 1


def test_prune_rejects_high_off_mask_support() -> None:
    scene = make_scene(torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]))
    scene.camera_trajectory = [make_camera(), make_camera()]
    mask_hit = torch.zeros(20, 20)
    mask_hit[9:12, 9:12] = 1.0
    mask_miss = torch.zeros(20, 20)
    obj_mask = ObjectMask(object_id="x", object_name="generic", masks={0: mask_hit, 1: mask_miss})
    cfg = ObjectRefinementConfig(
        use_depth_gate=False,
        component_radius=0.0,
        min_mask_support=1,
        max_off_mask_ratio=0.4,
    )

    result = prune_object(scene, obj_mask, cfg)

    assert result.scene.num_gaussians == 0 or result.stats["mask_support"]["off_mask_rejected"] == 2
    assert result.stats["mask_support"]["off_mask_rejected"] == 2


def test_load_stage_input_accepts_object_ply_dir(tmp_path) -> None:
    region = make_scene(torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]))
    region_path = tmp_path / "chunk.ply"
    region.to_ply(region_path)
    stage_dir = tmp_path / "stage1"
    stage_dir.mkdir()
    object_dir = tmp_path / "objects"
    object_dir.mkdir()
    object_path = object_dir / "object_7_test_box.ply"
    make_scene(torch.tensor([[0.0, 0.0, 1.0]])).to_ply(object_path)

    loaded = load_stage_input(region_path, stage_dir, object_ply_dir=object_dir, device="cpu")

    assert [record.object_id for record in loaded.objects] == ["7"]
    assert loaded.objects[0].name == "test_box"
    assert "7" in loaded.object_scenes
    assert loaded.object_scenes["7"].num_gaussians == 1


def test_foreground_confidence_keep_is_class_agnostic_mask_vote() -> None:
    scene = make_scene(torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.8, 0.8, 1.0]]))
    scene.camera_trajectory = [make_camera()]
    mask = torch.zeros(20, 20)
    mask[9:12, 9:12] = 1.0
    obj_mask = ObjectMask(object_id="x", object_name="generic", masks={0: mask})
    cfg = ObjectRefinementConfig(
        use_depth_gate=False,
        component_radius=0.0,
        min_foreground_confidence=0.35,
        background_confidence_threshold=0.65,
    )

    keep, stats = foreground_confidence_keep(scene, obj_mask, cfg)

    assert keep.tolist() == [True, True, False]
    assert stats["applied"] is True
    assert stats["valid_views"] == 1


def test_write_object_package_exports_explicit_backend_contract(tmp_path) -> None:
    scene = make_scene(torch.tensor([[0.0, 0.0, 1.0]]))
    scene.metadata["source_ply_path"] = str(tmp_path / "chunk.ply")
    scene.metadata["cameras_path"] = str(tmp_path / "cameras.json")
    record = type(
        "Record",
        (),
        {
            "object_id": "1",
            "name": "generic",
            "best_view_index": -1,
            "memory_state": {"text_prompt": "generic object"},
            "metadata": {"quality": "test"},
            "object_ply_path": str(tmp_path / "object.ply"),
        },
    )()

    request_path = write_object_package(
        source_scene=scene,
        refined_scene=scene,
        record=record,
        obj_mask=None,
        cfg=ObjectRefinementConfig(),
        work_dir=tmp_path / "work",
        metrics={"before": {}, "prune": {}, "after_prune": {}},
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert request["source_ply"].endswith("chunk.ply")
    assert request["cameras_path"].endswith("cameras.json")
    assert request["object_ply"].endswith("object.ply")
    assert request["memory_state"]["text_prompt"] == "generic object"
    assert "config" in request


def test_clean_view_scoring_prefers_covered_low_leakage_view() -> None:
    scene = make_scene(torch.tensor([[0.0, 0.0, 1.0]]))
    scene.camera_trajectory = [make_camera(), make_camera()]
    mask_good = torch.zeros(20, 20)
    mask_good[8:13, 8:13] = 1.0
    mask_bad = torch.zeros(20, 20)
    mask_bad[1:3, 1:3] = 1.0
    obj_mask = ObjectMask(object_id="x", object_name="generic", masks={0: mask_good, 1: mask_bad})

    def fake_render(camera):
        alpha = torch.zeros(20, 20)
        if camera is scene.camera_trajectory[0]:
            alpha[8:13, 8:13] = 1.0
        else:
            alpha[:, :] = 1.0
        rgb = torch.zeros(20, 20, 3)
        return RenderedView(rgb=rgb, depth=torch.ones(20, 20), alpha=alpha)

    scene.render = fake_render  # type: ignore[method-assign]

    scores = _score_clean_views(scene, scene, obj_mask, ObjectRefinementConfig())

    assert scores[0]["view_index"] == 0


def test_none_backend_reports_skipped(tmp_path) -> None:
    request = tmp_path / "object_request.json"
    request.write_text("{}", encoding="utf-8")

    result = NoneBackend().run(request, tmp_path)

    assert result.status == "skipped"
    assert result.generated_ply is None


def test_wonder3d_backend_alias_selects_wonder3d_generation() -> None:
    backend = make_backend("wonder3d", device="cpu")

    assert backend.__class__.__name__ == "SV3DBackend"
    assert getattr(backend, "generation_model") == "wonder3d"


def _write_test_multiview_request(tmp_path, num_views: int = 3) -> tuple[dict, object]:
    from PIL import Image

    mv_dir = tmp_path / "multiview"
    mv_dir.mkdir()
    views = []
    for idx in range(num_views):
        rgb_path = mv_dir / f"view_{idx:02d}_rgb.png"
        mask_path = mv_dir / f"view_{idx:02d}_mask.png"
        alpha_path = mv_dir / f"view_{idx:02d}_alpha.pt"
        Image.new("RGB", (16, 16), (20 + idx, 40, 60)).save(rgb_path)
        Image.new("L", (16, 16), 255).save(mask_path)
        torch.save(torch.ones(16, 16), alpha_path)
        ext = torch.eye(4)
        ext[0, 3] = float(idx)
        views.append(
            {
                "view_index": idx,
                "rgb_path": str(rgb_path),
                "mask_path": str(mask_path),
                "alpha_path": str(alpha_path),
                "extrinsic": ext.tolist(),
                "intrinsic": [[12.0, 0.0, 8.0], [0.0, 12.0, 8.0], [0.0, 0.0, 1.0]],
                "width": 16,
                "height": 16,
            }
        )
    request = {
        "object_id": "9",
        "name": "box",
        "best_view_index": 0,
        "memory_state": {"text_prompt": "small box"},
        "multiview": {"views": views},
    }
    request_path = tmp_path / "object_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return request, request_path


def test_artifixer_exporter_writes_reconstructed_colmap_split(tmp_path) -> None:
    request, _ = _write_test_multiview_request(tmp_path)
    prompt_path = tmp_path / "caption.h5"
    prompt_path.write_bytes(b"placeholder")

    prepared = prepare_artifixer_scene(request, tmp_path, prompt_path=prompt_path)
    split = json.loads(prepared.split_path.read_text(encoding="utf-8"))
    scene_meta = split["test"]["object_9"]
    transforms = json.loads((prepared.scene_root / "transforms.json").read_text(encoding="utf-8"))

    assert prepared.target_indices == [1, 2]
    assert scene_meta["has_gt"] is False
    assert Path(scene_meta["render_dir"]).joinpath("00001.png").exists()
    assert Path(scene_meta["opacity_dir"]).joinpath("00001.png").exists()
    assert Path(scene_meta["prompt_path"]).exists()
    assert transforms["camera_model"] == "OPENCV"
    assert len(transforms["frames"]) == 3
    assert transforms["frames"][0]["fl_x"] == 12.0


def test_artifixer_backend_collects_fake_predictions(tmp_path) -> None:
    from PIL import Image

    _, request_path = _write_test_multiview_request(tmp_path)
    fake_script = tmp_path / "fake_artifixer.py"
    fake_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "from PIL import Image\n"
        "save_dir=Path(sys.argv[1]); scene_id=sys.argv[2]\n"
        "pred=save_dir/'ckpt'/'run'/scene_id/'frames'/'batch_0000'/'pred'\n"
        "pred.mkdir(parents=True, exist_ok=True)\n"
        "Image.new('RGB',(16,16),(1,2,3)).save(pred/'00001.png')\n"
        "Image.new('RGB',(16,16),(4,5,6)).save(pred/'00002.png')\n",
        encoding="utf-8",
    )
    backend = ArtiFixerBackend(
        repo=None,
        checkpoint=None,
        python=sys.executable,
        command=f'"{sys.executable}" "{fake_script}" "{{save_dir}}" "{{scene_id}}"',
        prompt_path=str(tmp_path / "caption.h5"),
        timeout=10.0,
    )
    (tmp_path / "caption.h5").write_bytes(b"placeholder")

    result = backend.run(request_path, tmp_path)

    assert result.status == "generated"
    assert (tmp_path / "multiview_generate" / "view_00.png").exists()
    assert (tmp_path / "multiview_generate" / "view_01.png").exists()
    assert (tmp_path / "multiview_generate" / "cameras.json").exists()
    assert (tmp_path / "generation_meta.json").exists()


def test_wonder3d_condition_is_rgba_with_mask_alpha(tmp_path) -> None:
    from PIL import Image

    rgb = Image.new("RGB", (8, 8), (255, 255, 255))
    for y in range(2, 6):
        for x in range(3, 7):
            rgb.putpixel((x, y), (20, 40, 60))
    mask = Image.new("L", (8, 8), 0)
    for y in range(2, 6):
        for x in range(3, 7):
            mask.putpixel((x, y), 255)
    rgb_path = tmp_path / "crop.png"
    mask_path = tmp_path / "mask.png"
    out_path = tmp_path / "condition.png"
    rgb.save(rgb_path)
    mask.save(mask_path)

    _write_wonder3d_rgba_condition(rgb_path, mask_path, out_path)

    out = Image.open(out_path)
    alpha = np.asarray(out.getchannel("A"))
    assert out.mode == "RGBA"
    assert alpha[0, 0] == 0
    assert alpha[3, 4] == 255


def test_wonder3d_camera_metadata_uses_fixed_view_order() -> None:
    cameras = _generate_wonder3d_cameras(6, 256)

    assert [cam["view_name"] for cam in cameras] == [
        "front",
        "front_right",
        "right",
        "back",
        "left",
        "front_left",
    ]
    assert [cam["azimuth_deg"] for cam in cameras] == [0.0, 45.0, 90.0, 180.0, -90.0, -45.0]


def test_cleanup_only_stage_skips_backend_and_writes_refined_object(tmp_path) -> None:
    scene = make_scene(torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    record = type(
        "Record",
        (),
        {
            "object_id": "1",
            "name": "generic",
            "source_gaussian_indices": torch.arange(2),
            "best_view_index": -1,
            "best_view_score": 0.0,
            "memory_state": {},
            "metadata": {},
            "object_ply_path": None,
        },
    )()
    stage_input = StageRefinementInput(scene=scene, masks=None, objects=[record], assignment_scores=None)
    cfg = ObjectRefinementConfig(refinement_stage="cleanup_only")

    result = refine_stage_objects(stage_input, tmp_path / "out", cfg, NoneBackend())

    metrics = result.object_metrics[0]
    assert metrics["backend"]["status"] == "skipped"
    assert metrics["outputs"]["request_json"] is None
    assert (tmp_path / "out/object_1_generic/object_1_generic_final.ply").exists()


def test_external_backend_contract_writes_generated_ply(tmp_path) -> None:
    request = tmp_path / "object_request.json"
    request.write_text("{}", encoding="utf-8")
    command = f'"{sys.executable}" -c "from pathlib import Path; Path(\'generated.ply\').write_text(\'ok\')"'

    result = ExternalCommandBackend(command, timeout=10.0).run(request, tmp_path)

    assert result.status == "ok"
    assert result.generated_ply is not None
    assert (tmp_path / "generated.ply").exists()
    assert (tmp_path / "backend.log").exists()


def test_merge_preserves_refined_object_when_no_generated_output() -> None:
    refined = make_scene(torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    cfg = ObjectRefinementConfig()

    result = merge_generated_object(refined, None, None, cfg)

    assert result.scene.num_gaussians == 2
    assert result.stats["mode"] == "refined_only"


def test_merge_rejects_duplicate_generated_gaussians(tmp_path) -> None:
    refined = make_scene(torch.tensor([[0.0, 0.0, 0.0]]))
    generated = make_scene(torch.tensor([[0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    generated_path = tmp_path / "generated.ply"
    generated.to_ply(generated_path)
    cfg = ObjectRefinementConfig(duplicate_distance=0.05)

    result = merge_generated_object(refined, generated_path, None, cfg)

    assert result.scene.num_gaussians == 2
    assert result.stats["kept_generated_gaussians"] == 1


def test_merge_rejects_generated_gaussians_outside_expanded_bbox(tmp_path) -> None:
    refined = make_scene(torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]))
    generated = make_scene(torch.tensor([[10.0, 10.0, 10.0]]))
    generated_path = tmp_path / "generated.ply"
    generated.to_ply(generated_path)
    cfg = ObjectRefinementConfig(duplicate_distance=0.0, generated_bbox_expand=0.25)

    result = merge_generated_object(refined, generated_path, None, cfg)

    assert result.scene.num_gaussians == refined.num_gaussians
    assert result.stats["kept_generated_gaussians"] == 0


def test_local_refine_paints_colors_without_changing_geometry(tmp_path) -> None:
    scene = make_scene(torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    pseudo_dir = tmp_path / "multiview_generate"
    pseudo_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (8, 8), (20, 40, 60)).save(pseudo_dir / "view_00.png")

    result = refine_local_object(scene, tmp_path, ObjectRefinementConfig(refinement_stage="local_refine"))

    assert result.scene.num_gaussians == scene.num_gaussians
    assert torch.allclose(result.scene.params.means, scene.params.means)
    assert result.stats["frozen_geometry"] is True
    assert (tmp_path / "paint_metrics.json").exists()


def test_resolve_condition_view_index_uses_crop_when_best_view_missing() -> None:
    request = {
        "best_view_index": -1,
        "crop": {"view_index": 22},
        "multiview": {"views": [{"view_index": 40}]},
    }
    assert resolve_condition_view_index(request) == 22


def test_estimate_elevation_from_aerial_camera() -> None:
    ext = torch.tensor(
        [
            [-0.923879, 0.0, 0.382683, 0.0],
            [0.131652, 0.938961, 0.317836, 0.0],
            [-0.359325, 0.344023, -0.867486, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    cam = CameraPose(
        extrinsic=ext,
        intrinsic=torch.tensor([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]]),
        width=20,
        height=20,
    )
    scene = GaussianScene(params=make_scene(torch.zeros(1, 3)).params, camera_trajectory=[cam])
    elevation = _estimate_elevation_from_scene(scene, 0)
    assert elevation > 45.0


def test_expand_bbox_for_target_fill_zooms_small_object() -> None:
    mask = torch.zeros(100, 100)
    mask[45:55, 45:55] = 1.0
    bbox = (40, 40, 60, 60)
    expanded = _expand_bbox_for_target_fill(mask, bbox, target_fill=0.72)
    x0, y0, x1, y1 = expanded
    crop_mask = mask[max(0, y0):min(100, y1), max(0, x0):min(100, x1)]
    fill = float(crop_mask.sum().item() / max(1, crop_mask.numel()))
    assert fill >= 0.5
    assert (x1 - x0) >= 12
    assert (y1 - y0) >= 12


def test_resolve_sv3d_elevation_clamps_aerial_views() -> None:
    cfg = ObjectRefinementConfig(sv3d_max_elevation_deg=25.0, sv3d_condition_elevation_deg=15.0)
    elev, scene_mv = resolve_sv3d_elevation(60.0, cfg)
    assert elev == 15.0
    assert scene_mv is True
    elev2, scene_mv2 = resolve_sv3d_elevation(20.0, cfg)
    assert elev2 == 20.0
    assert scene_mv2 is False


def test_generate_scene_multiview_pseudo_exports_foregroundized_views(tmp_path) -> None:
    from PIL import Image
    from worldact.stage2_object_refinement.generate import generate_scene_multiview_pseudo

    mv_dir = tmp_path / "multiview"
    mv_dir.mkdir()
    rgb = Image.new("RGB", (64, 64), (180, 40, 40))
    for y in range(20, 44):
        for x in range(20, 44):
            rgb.putpixel((x, y), (20, 40, 60))
    mask = Image.new("L", (64, 64), 0)
    for y in range(20, 44):
        for x in range(20, 44):
            mask.putpixel((x, y), 255)
    rgb_path = mv_dir / "view_00_rgb.png"
    mask_path = mv_dir / "view_00_mask.png"
    rgb.save(rgb_path)
    mask.save(mask_path)
    request = {
        "multiview": {
            "views": [
                {
                    "view_index": 0,
                    "rgb_path": str(rgb_path),
                    "mask_path": str(mask_path),
                    "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "intrinsic": [[10, 0, 32], [0, 10, 32], [0, 0, 1]],
                    "width": 64,
                    "height": 64,
                }
            ]
        }
    }
    result = generate_scene_multiview_pseudo(
        request,
        tmp_path,
        ObjectRefinementConfig(),
        resolution=32,
    )
    assert result.num_views == 1
    assert (tmp_path / "multiview_generate" / "view_00.png").exists()
    arr = np.asarray(Image.open(tmp_path / "multiview_generate" / "view_00.png").convert("RGB"))
    assert float((arr < 250).any(axis=-1).mean()) > 0.05
