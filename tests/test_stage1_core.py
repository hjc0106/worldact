from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from worldact.config import load_config
from worldact.data.gaussians import GaussianParams, GaussianScene
from worldact.data.objects import ObjectAsset, ObjectMemoryState
from worldact.stage1_decomposition.binarize import binarize_and_partition
from worldact.stage1_decomposition.aerial_roi import RegionCameraEstimate, write_region_camera_config
from worldact.stage1_decomposition.chunking import build_xy_chunks, filter_chunks_by_roi, make_chunk_scene, merge_chunk_objects
from worldact.stage1_decomposition.sam3_segmenter import SAM3Candidate, SAM3CandidateSet
from worldact.stage1_decomposition.view_graph_propagation import (
    MaskTrack,
    _enforce_unique_candidate_assignments,
    _fuse_candidate_mask,
    _match_target_candidate,
)
from worldact.stage1_decomposition.view_graph_propagation import ViewGraphEdge
from worldact.stage1_decomposition.vlm import VLMClient


def make_scene(n: int = 4, means: torch.Tensor | None = None) -> GaussianScene:
    if means is None:
        means = torch.zeros(n, 3)
    else:
        n = int(means.shape[0])
    params = GaussianParams(
        means=means,
        scales=torch.zeros(n, 3),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacities=torch.zeros(n, 1),
        colors=torch.zeros(n, 3),
    )
    return GaussianScene(params=params)


def test_binarize_conflicts_assign_highest_score() -> None:
    scene = make_scene(4)
    scores = torch.tensor(
        [
            [0.9, 0.2],
            [0.4, 0.8],
            [0.7, 0.9],
            [0.1, 0.2],
        ]
    )
    states = [ObjectMemoryState(name="cup"), ObjectMemoryState(name="book")]
    bg, objects, _ = binarize_and_partition(scene, scores, states, threshold=0.5)
    assert len(objects) == 2
    assert objects[0].source_gaussian_indices.tolist() == [0]
    assert objects[1].source_gaussian_indices.tolist() == [1, 2]
    assert bg.num_gaussians == 1


def test_build_xy_chunks_uses_expanded_bounds_and_local_mapping() -> None:
    xs, ys = torch.meshgrid(torch.linspace(0, 3, 4), torch.linspace(0, 3, 4), indexing="ij")
    means = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.zeros(16)], dim=-1)
    scene = make_scene(means=means)
    cfg = SimpleNamespace(
        bounds_quantile=[0.0, 1.0],
        target_gaussians_per_chunk=4,
        min_chunk_size_xy=0.0,
        max_chunk_size_xy=2.0,
        overlap_xy=0.25,
        min_gaussians_per_chunk=1,
        max_chunks=4,
    )

    chunks = build_xy_chunks(scene, cfg)
    assert len(chunks) == 4
    assert all(len(chunk.indices) >= int(chunk.core_mask.sum().item()) for chunk in chunks)
    assert sum(int(chunk.core_mask.sum().item()) for chunk in chunks) == scene.num_gaussians

    chunk_scene = make_chunk_scene(scene, chunks[0])
    assert chunk_scene.num_gaussians == len(chunks[0].indices)
    assert chunk_scene.metadata["global_indices"].tolist() == chunks[0].indices.tolist()
    assert chunk_scene.metadata["chunk_id"] == "chunk_000"


def test_build_xy_chunks_respects_up_axis() -> None:
    xs, zs = torch.meshgrid(torch.linspace(0, 3, 4), torch.linspace(0, 3, 4), indexing="ij")
    means = torch.stack([xs.reshape(-1), torch.zeros(16), zs.reshape(-1)], dim=-1)
    scene = make_scene(means=means)
    cfg = SimpleNamespace(
        bounds_quantile=[0.0, 1.0],
        target_gaussians_per_chunk=4,
        min_chunk_size_xy=0.0,
        max_chunk_size_xy=2.0,
        overlap_xy=0.0,
        min_gaussians_per_chunk=1,
        max_chunks=4,
    )

    chunks = build_xy_chunks(scene, cfg, up_axis="y")

    assert len(chunks) == 4
    assert sum(int(chunk.core_mask.sum().item()) for chunk in chunks) == scene.num_gaussians


def test_filter_chunks_by_roi_selects_roi_chunks_and_falls_back_when_needed() -> None:
    xs, ys = torch.meshgrid(torch.linspace(0, 3, 4), torch.linspace(0, 3, 4), indexing="ij")
    means = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.zeros(16)], dim=-1)
    scene = make_scene(means=means)
    cfg = SimpleNamespace(
        bounds_quantile=[0.0, 1.0],
        target_gaussians_per_chunk=4,
        min_chunk_size_xy=0.0,
        max_chunk_size_xy=2.0,
        overlap_xy=0.25,
        min_gaussians_per_chunk=1,
        max_chunks=4,
    )

    chunks = build_xy_chunks(scene, cfg)
    selected, manifest, summary = filter_chunks_by_roi(
        scene,
        chunks,
        roi_center=[0.5, 0.5, 0.0],
        roi_radius=0.9,
        min_roi_overlap=0.5,
        keep_top_k=2,
        prompt="corner table",
        prompt_guided=True,
    )

    assert [chunk.chunk_id for chunk in selected] == ["chunk_000"]
    assert summary["fallback_used"] is False
    assert summary["selected_chunk_ids"] == ["chunk_000"]
    assert manifest[0]["selected_by_prompt"] is True
    assert manifest[0]["prompt_guided"] is True
    assert manifest[1]["selected_by_prompt"] is False

    fallback_selected, fallback_manifest, fallback_summary = filter_chunks_by_roi(
        scene,
        chunks,
        roi_center=[10.0, 10.0, 0.0],
        roi_radius=0.1,
        min_roi_overlap=1.1,
        keep_top_k=2,
        prompt="far corner",
        prompt_guided=True,
    )

    assert len(fallback_selected) == 2
    assert fallback_summary["fallback_used"] is True
    assert fallback_summary["selected_chunk_ids"] == ["chunk_000", "chunk_001"]
    assert fallback_manifest[0]["selection_reason"] in {"fallback_top_k", "rejected"}


def test_merge_chunk_objects_merges_same_name_overlap_and_dedupes() -> None:
    means = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    scene = make_scene(means=means)
    candidates = [
        ObjectAsset("0", "white cup", torch.tensor([0, 1, 2]), metadata={"chunk_ids": ["chunk_000"]}),
        ObjectAsset("0", "white cup", torch.tensor([1, 2, 3]), metadata={"chunk_ids": ["chunk_001"]}),
        ObjectAsset("0", "book", torch.tensor([4]), metadata={"chunk_ids": ["chunk_001"]}),
    ]
    cfg = SimpleNamespace(merge_iou_threshold=0.5, merge_centroid_distance=0.1)

    merged = merge_chunk_objects(scene, candidates, cfg)
    by_name = {obj.name: obj for obj in merged}
    assert sorted(by_name["white cup"].source_gaussian_indices.tolist()) == [0, 1, 2, 3]
    assert by_name["white cup"].metadata["chunk_ids"] == ["chunk_000", "chunk_001"]
    assert by_name["book"].source_gaussian_indices.tolist() == [4]


def test_outdoor_config_drives_vlm_prompts() -> None:
    cfg = load_config("configs/scenes/outdoor.yaml", ["device.device=cpu"])
    client = object.__new__(VLMClient)
    client.cfg = cfg.stage1.vlm

    system, user = client._build_prompts()

    assert "outdoor open-air" in system
    assert "garden, park" in system
    assert "{max_objects}" not in user
    assert "Return at most 28 items" in user


def test_config_override_updates_camera_ring_list_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output_dir: outputs/test
stage1:
  camera_gen:
    trajectory_mode: hybrid
    rings:
      - name: inner
        num_cameras: 8
      - name: outer
        num_cameras: 16
      - name: high
        num_cameras: 8
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(
        config_path,
        [
            "stage1.camera_gen.rings.0.num_cameras=12",
            "stage1.camera_gen.rings.1.num_cameras=24",
            "stage1.camera_gen.rings.2.num_cameras=12",
            "stage1.vlm.num_keyframes=32",
        ],
    )

    assert [ring.num_cameras for ring in cfg.stage1.camera_gen.rings] == [12, 24, 12]
    assert cfg.stage1.vlm.num_keyframes == 32


def test_region_camera_config_injects_region_assetization_prompt(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        """
output_dir: outputs/test
stage1:
  vlm:
    max_objects: 20
    prompts:
      system: base system
      user: base user
  camera_gen:
    num_cameras: 8
    image_width: 64
    image_height: 64
    fov_deg: 80.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    estimate = RegionCameraEstimate(
        center=torch.tensor([1.0, 2.0, 3.0]),
        bbox_center=torch.tensor([1.0, 2.0, 3.0]),
        robust_center=torch.tensor([1.0, 2.0, 3.0]),
        orbit_radius=4.0,
        camera_purpose="coverage",
        coverage_mode="cylinder",
        coverage_radius=2.0,
        coverage_standoff=1.0,
        aspect_ratio=1.0,
        horizontal_span=(2.0, 2.0),
        horizontal_diagonal=2.8,
        point_radius_q90=1.5,
        center_source="coverage_bbox",
        center_height_above_floor=1.0,
        region_floor_height=0.0,
        floor_quantile=0.02,
        height_mode="geometric",
        height_scale=None,
        height_quantile=0.98,
        height_min=None,
        height_max=None,
        region_height_quantile=2.0,
        region_height_extent=2.0,
        vertical_axis="y",
    )

    config_path = write_region_camera_config(
        tmp_path / "region_000",
        base_config_path=base_config,
        config_name="region_camera_config.yaml",
        up_axis="-y",
        estimate=estimate,
        source_prompts=["parking lot", "black car", "white car"],
        region_vlm_prompt=True,
        region_vlm_max_objects=12,
    )

    assert config_path is not None
    cfg = load_config(config_path)
    assert cfg.stage1.vlm.max_objects == 12
    assert "assetizable object instances" in cfg.stage1.vlm.prompts.system
    assert "parking lot, black car, white car" in cfg.stage1.vlm.prompts.user
    assert "{num_keyframes}" in cfg.stage1.vlm.prompts.user
    assert "{max_objects}" in cfg.stage1.vlm.prompts.user


def test_view_propagation_intersection_fusion_rejects_extra_local_region() -> None:
    local = torch.zeros(8, 8)
    local[1:3, 1:3] = 1.0
    local[5:7, 5:7] = 1.0
    propagated = torch.zeros(8, 8)
    propagated[1:3, 1:3] = 1.0
    cfg = SimpleNamespace(fusion_mode="intersection", local_mask_dilate_px=0, min_fused_area_ratio=0.25)

    fused, mode, rejected = _fuse_candidate_mask(local, propagated, cfg)

    assert mode == "intersection"
    assert rejected is None
    assert int(fused.sum().item()) == 4
    assert bool(fused[1:3, 1:3].all())
    assert not bool(fused[5:7, 5:7].any())


def test_view_propagation_fallback_disabled_rejects_low_identity_candidate() -> None:
    candidate_mask = torch.zeros(8, 8)
    candidate_mask[5:7, 5:7] = 1.0
    propagated = torch.zeros(8, 8)
    propagated[1:3, 1:3] = 1.0
    candidate = SAM3Candidate(
        candidate_id=0,
        mask=candidate_mask,
        score=0.9,
        box=torch.tensor([5.0, 5.0, 6.0, 6.0]),
        quality=0.9,
        area=4.0,
        bbox_area=4.0,
    )
    candidates = SAM3CandidateSet(
        prompt="chair",
        image_size=(8, 8),
        masks=candidate_mask.unsqueeze(0),
        scores=torch.tensor([0.9]),
        boxes=torch.tensor([[5.0, 5.0, 6.0, 6.0]]),
        qualities=torch.tensor([0.9]),
        candidates=[candidate],
    )
    cfg = SimpleNamespace(
        association_enabled=True,
        local_sam_weight=1.0,
        propagated_weight=0.75,
        geometry_weight=0.75,
        min_assignment_iou=0.6,
        min_assignment_visibility=0.25,
        min_view_overlap=0.15,
        anchor_min_score=0.5,
        fallback_to_sam3=False,
    )
    edge = ViewGraphEdge(0, 1, 0.2, 0.5, 20.0, 0.2)

    decision = _match_target_candidate(
        source_view=0,
        target_view=1,
        warped_mask=propagated,
        warp_stats={"visibility_score": 0.1, "depth_consistency": 0.8},
        target_candidates=candidates,
        edge=edge,
        cfg=cfg,
    )

    assert decision is None


def test_view_propagation_unique_candidate_assignment_keeps_highest_score() -> None:
    mask = torch.ones(4, 4)
    tracks = [
        MaskTrack(
            track_id="0",
            object_id="0",
            object_name="chair left",
            prompt="chair left",
            source_view=0,
            source_candidate_id=0,
            masks_by_view={0: mask.clone()},
            scores_by_view={0: 0.8},
            visibility_by_view={0: 1.0},
            metadata={"associations": {}, "candidate_conflicts": [], "dropped_views": []},
        ),
        MaskTrack(
            track_id="1",
            object_id="1",
            object_name="chair right",
            prompt="chair right",
            source_view=0,
            source_candidate_id=0,
            masks_by_view={0: mask.clone()},
            scores_by_view={0: 0.4},
            visibility_by_view={0: 1.0},
            metadata={"associations": {}, "candidate_conflicts": [], "dropped_views": []},
        ),
    ]

    _enforce_unique_candidate_assignments(tracks)

    assert 0 in tracks[0].masks_by_view
    assert 0 not in tracks[1].masks_by_view
    assert tracks[1].metadata["candidate_conflicts"][0]["winner_track_id"] == "0"
    assert tracks[1].metadata["dropped_views"][0]["reason"] == "candidate_conflict"
