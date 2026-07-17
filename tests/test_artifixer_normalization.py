from __future__ import annotations

import math

import torch

from worldact.data.gaussians import GaussianParams
from worldact.stage2_artifixer.normalization import (
    build_normalization,
    local_to_world_params,
    position_roundtrip_error,
    world_to_local_params,
)


def _params(means: torch.Tensor) -> GaussianParams:
    n = means.shape[0]
    return GaussianParams(
        means=means.float(),
        scales=torch.full((n, 3), -3.0),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacities=torch.full((n, 1), 2.0),
        colors=torch.zeros(n, 3),
    )


def test_position_roundtrip_within_tolerance() -> None:
    means = torch.tensor(
        [
            [10.0, -3.0, 5.0],
            [11.0, -2.5, 5.5],
            [9.5, -3.5, 4.5],
            [10.5, -2.0, 6.0],
        ]
    )
    params = _params(means)
    norm = build_normalization(params)
    err = position_roundtrip_error(params, norm)
    assert err < 1e-5 * norm.radius


def test_log_scale_shifts_by_log_radius() -> None:
    means = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    params = _params(means)
    norm = build_normalization(params, quantile=(0.0, 1.0))
    local = world_to_local_params(params, norm)
    world = local_to_world_params(local, norm)
    assert torch.allclose(world.scales, params.scales, atol=1e-5)
    assert torch.allclose(local.scales, params.scales - math.log(norm.radius), atol=1e-5)
