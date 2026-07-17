"""Load WorldAct stage1 artifacts for a single-object ArtiFixer package."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import MaskSet, ObjectMask
from worldact.data.poses import CameraPose
from worldact.utils.io import load_cameras_json


@dataclass
class ObjectRecord:
    object_id: str
    name: str
    text_prompt: str
    source_gaussian_indices: list[int] = field(default_factory=list)
    best_view_index: int = -1
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, item: dict[str, Any]) -> "ObjectRecord":
        memory = item.get("memory_state") or {}
        return cls(
            object_id=str(item.get("object_id", "")),
            name=str(item.get("name") or memory.get("name") or f"object_{item.get('object_id', '0')}"),
            text_prompt=str(memory.get("text_prompt") or item.get("prompt") or item.get("name") or "object"),
            source_gaussian_indices=[int(x) for x in (item.get("source_gaussian_indices") or [])],
            best_view_index=int(item.get("best_view_index", -1) or -1),
            raw=dict(item),
        )


@dataclass
class StageArtifixerInput:
    stage_dir: Path
    scene: GaussianScene
    objects: list[ObjectRecord]
    masks: MaskSet | None
    object_ply_paths: dict[str, Path]
    cameras_path: Path | None


def resolve_cameras_path(stage_dir: Path, explicit: Path | str | None = None) -> Path | None:
    if explicit is not None:
        path = Path(explicit)
        return path if path.exists() else None
    candidate = Path(stage_dir) / "background" / "cameras.json"
    return candidate if candidate.exists() else None


def load_stage_artifixer_input(
    ply_path: Path | str,
    stage_dir: Path | str,
    *,
    cameras_path: Path | str | None = None,
    object_ply_dir: Path | str | None = None,
    device: str | torch.device = "cpu",
) -> StageArtifixerInput:
    stage_dir = Path(stage_dir)
    ply_path = Path(ply_path)
    cam_path = resolve_cameras_path(stage_dir, cameras_path)
    scene = GaussianScene.from_ply(ply_path, cameras_path=cam_path)
    scene.params = scene.params.to(device)
    scene.camera_trajectory = [cam.to(device) for cam in scene.camera_trajectory]

    objects_path = stage_dir / "objects.json"
    if not objects_path.exists():
        raise FileNotFoundError(f"missing objects.json under {stage_dir}")
    objects_raw = json.loads(objects_path.read_text(encoding="utf-8"))
    objects = [ObjectRecord.from_json(item) for item in objects_raw]

    masks: MaskSet | None = None
    masks_path = stage_dir / "masks.pt"
    if masks_path.exists():
        masks = torch.load(masks_path, map_location="cpu", weights_only=False)

    search_dir = Path(object_ply_dir) if object_ply_dir is not None else stage_dir
    object_ply_paths = discover_object_plys(search_dir)
    return StageArtifixerInput(
        stage_dir=stage_dir,
        scene=scene,
        objects=objects,
        masks=masks,
        object_ply_paths=object_ply_paths,
        cameras_path=cam_path,
    )


def discover_object_plys(directory: Path) -> dict[str, Path]:
    directory = Path(directory)
    found: dict[str, Path] = {}
    for path in sorted(directory.glob("object_*.ply")):
        object_id = _infer_object_id(path.stem)
        if object_id is not None:
            found[object_id] = path
    return found


def _infer_object_id(stem: str) -> str | None:
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "object":
        return parts[1]
    match = re.match(r"object[_-](.+?)(?:_|$)", stem)
    return match.group(1) if match else None


def find_object_record(objects: list[ObjectRecord], object_id: str) -> ObjectRecord:
    for record in objects:
        if str(record.object_id) == str(object_id):
            return record
    raise KeyError(f"object_id={object_id!r} not found in objects.json")


def find_object_mask(masks: MaskSet | None, object_id: str, fallback_index: int = -1) -> ObjectMask | None:
    if masks is None:
        return None
    for obj_mask in masks.object_masks:
        if str(obj_mask.object_id) == str(object_id):
            return obj_mask
    if 0 <= fallback_index < len(masks.object_masks):
        return masks.object_masks[fallback_index]
    return None


def load_object_scene(
    stage_input: StageArtifixerInput,
    object_id: str,
    *,
    device: str | torch.device | None = None,
) -> GaussianScene:
    record = find_object_record(stage_input.objects, object_id)
    device = device or stage_input.scene.device
    ply_path = stage_input.object_ply_paths.get(str(object_id))
    if ply_path is not None and ply_path.exists():
        scene = GaussianScene.from_ply(ply_path)
        scene.params = scene.params.to(device)
        scene.camera_trajectory = [cam.to(device) for cam in stage_input.scene.camera_trajectory]
        scene.metadata = {"object_id": record.object_id, "name": record.name, "source_ply": str(ply_path)}
        return scene

    if record.source_gaussian_indices:
        indices = torch.tensor(record.source_gaussian_indices, dtype=torch.long, device=stage_input.scene.device)
        scene = stage_input.scene.subset(indices)
        scene.params = scene.params.to(device)
        scene.camera_trajectory = [cam.to(device) for cam in scene.camera_trajectory]
        scene.metadata = {"object_id": record.object_id, "name": record.name, "source": "indices"}
        return scene

    raise FileNotFoundError(
        f"no object_{object_id}_*.ply under {stage_input.stage_dir} and no source_gaussian_indices"
    )


def load_world_cameras(path: Path | str) -> list[CameraPose]:
    return load_cameras_json(Path(path))
