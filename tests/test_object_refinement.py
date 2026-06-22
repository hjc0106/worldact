from __future__ import annotations

import sys

import torch

from worldact.data.gaussians import GaussianParams, GaussianScene
from worldact.stage2_object_refinement.backend import ExternalCommandBackend, NoneBackend
from worldact.stage2_object_refinement.common import ObjectRefinementConfig
from worldact.stage2_object_refinement.merge import merge_generated_object
from worldact.stage2_object_refinement.prune import prune_object


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


def test_none_backend_reports_skipped(tmp_path) -> None:
    request = tmp_path / "object_request.json"
    request.write_text("{}", encoding="utf-8")

    result = NoneBackend().run(request, tmp_path)

    assert result.status == "skipped"
    assert result.generated_ply is None


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
