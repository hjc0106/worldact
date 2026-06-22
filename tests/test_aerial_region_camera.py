from __future__ import annotations

import torch

from worldact.data.gaussians import GaussianScene
from worldact.stage1_decomposition.aerial_roi import RegionProposal, estimate_region_camera


def test_volume_center_uses_scene_frame_not_point_median() -> None:
    # Dense ground band plus a sparse elevated object: median stays near ground.
    ground = torch.stack(
        [
            torch.zeros(4000),
            torch.full((4000,), 1.6),
            torch.zeros(4000),
        ],
        dim=1,
    )
    top = torch.stack(
        [
            torch.zeros(200),
            torch.full((200,), -0.2),
            torch.zeros(200),
        ],
        dim=1,
    )
    points = torch.cat([ground, top], dim=0)
    proposal = RegionProposal(
        region_id="region_test",
        cell_bbox=(0, 0, 1, 1),
        bounds={"x": [-1.0, 1.0], "z": [-1.0, 1.0]},
        score=1.0,
        mean_score=1.0,
        area_cells=1,
    )

    volume = estimate_region_camera(
        proposal,
        points,
        up_axis="-y",
        center_source="volume",
        frame_quantile=(0.05, 0.95),
    )
    robust = estimate_region_camera(
        proposal,
        points,
        up_axis="-y",
        center_source="robust",
        height_mode="fixed",
        center_height=1.0,
    )
    assert volume is not None and robust is not None
    assert float(volume.center[1]) < float(robust.center[1])
    assert float(volume.center[1]) < 1.0


def test_volume_center_on_garden_chunk_if_present() -> None:
    ply_path = "outputs/garden/aerial_regions_table_bev/regions/region_000/chunk.ply"
    try:
        scene = GaussianScene.from_ply(ply_path)
    except FileNotFoundError:
        return
    points = scene.params.means.detach().cpu().float()
    lo = torch.quantile(points[:, 1], 0.05)
    hi = torch.quantile(points[:, 1], 0.95)
    proposal = RegionProposal(
        region_id="region_000",
        cell_bbox=(0, 0, 1, 1),
        bounds={
            "x": [float(points[:, 0].min()), float(points[:, 0].max())],
            "z": [float(points[:, 2].min()), float(points[:, 2].max())],
        },
        score=1.0,
        mean_score=1.0,
        area_cells=1,
    )
    estimate = estimate_region_camera(
        proposal,
        points,
        up_axis="-y",
        center_source="volume",
        frame_quantile=(0.05, 0.95),
    )
    assert estimate is not None
    assert lo <= estimate.center[1] <= hi
    assert float(estimate.center[1]) < 1.0
