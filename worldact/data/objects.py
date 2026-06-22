from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import torch


@dataclass
class ObjectMemoryState:
    name: str
    category: str = ""
    count: int = 1
    recognizability: str = "precise"
    is_portable: bool = True
    text_prompt: str = ""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class ObjectAsset:
    object_id: str
    name: str
    source_gaussian_indices: Optional[torch.Tensor] = None
    memory_state: Optional[ObjectMemoryState] = None
    best_view_index: int = -1
    best_view_score: float = 0.0
    metadata: dict = field(default_factory=dict)
