from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ObjectMask:
    object_id: str
    object_name: str
    masks: dict[int, torch.Tensor] = field(default_factory=dict)
    track_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def view_indices(self) -> list[int]:
        return sorted(self.masks)

    def get_mask(self, view_idx: int) -> Optional[torch.Tensor]:
        return self.masks.get(view_idx)


@dataclass
class MaskSet:
    object_masks: list[ObjectMask] = field(default_factory=list)
    fused_masks: dict[int, torch.Tensor] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def fuse(self) -> None:
        fused: dict[int, torch.Tensor] = {}
        for obj_mask in self.object_masks:
            for view_idx, mask in obj_mask.masks.items():
                fused[view_idx] = torch.clamp(fused.get(view_idx, 0) + mask, 0.0, 1.0)
        self.fused_masks = fused

    def all_view_indices(self) -> list[int]:
        view_ids: set[int] = set()
        for obj_mask in self.object_masks:
            view_ids.update(obj_mask.masks)
        return sorted(view_ids)
