from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from worldact.data.poses import CameraPose


@dataclass
class GaussianParams:
    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    @property
    def device(self) -> torch.device:
        return self.means.device

    def to(self, device: torch.device | str) -> "GaussianParams":
        return GaussianParams(
            means=self.means.to(device),
            scales=self.scales.to(device),
            rotations=self.rotations.to(device),
            opacities=self.opacities.to(device),
            colors=self.colors.to(device),
        )

    def subset(self, indices: torch.Tensor) -> "GaussianParams":
        indices = indices.to(self.means.device)
        return GaussianParams(
            means=self.means[indices],
            scales=self.scales[indices],
            rotations=self.rotations[indices],
            opacities=self.opacities[indices],
            colors=self.colors[indices],
        )


@dataclass
class RenderedView:
    rgb: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    weights: Optional[torch.Tensor] = None

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


@dataclass
class GaussianScene:
    params: GaussianParams
    camera_trajectory: list[CameraPose] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def num_gaussians(self) -> int:
        return self.params.num_gaussians

    @property
    def device(self) -> torch.device:
        return self.params.device

    @classmethod
    def from_ply(cls, ply_path: Path | str, cameras_path: Path | str | None = None) -> "GaussianScene":
        from worldact.utils.io import load_cameras_json, load_gaussians_ply

        cameras = load_cameras_json(Path(cameras_path)) if cameras_path else []
        return cls(params=load_gaussians_ply(Path(ply_path)), camera_trajectory=cameras)

    def to_ply(self, ply_path: Path | str) -> None:
        from worldact.utils.io import save_gaussians_ply

        save_gaussians_ply(self.params, Path(ply_path))

    def save(self, output_dir: Path | str) -> None:
        from worldact.utils.io import save_cameras_json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.to_ply(output_dir / "gaussians.ply")
        if self.camera_trajectory:
            save_cameras_json(self.camera_trajectory, output_dir / "cameras.json")

    def subset(self, indices: torch.Tensor) -> "GaussianScene":
        return GaussianScene(
            params=self.params.subset(indices),
            camera_trajectory=list(self.camera_trajectory),
            metadata=dict(self.metadata),
        )

    def render(self, camera: CameraPose) -> RenderedView:
        from worldact.rendering.rasterizer import rasterize

        return rasterize(self, camera)
