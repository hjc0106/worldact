from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CameraPose:
    extrinsic: torch.Tensor
    intrinsic: torch.Tensor
    width: int
    height: int

    @property
    def c2w(self) -> torch.Tensor:
        return self.extrinsic.inverse()

    def to(self, device: torch.device | str) -> "CameraPose":
        return CameraPose(
            extrinsic=self.extrinsic.to(device),
            intrinsic=self.intrinsic.to(device),
            width=self.width,
            height=self.height,
        )
