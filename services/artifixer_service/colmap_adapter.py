"""COLMAP-ish scene scaffolding from object_package anchors + trajectory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def build_colmap_scene(package_dir: Path, workspace: Path) -> Path:
    """Materialize a reconstructed-colmap style directory for ArtiFixer split.json."""
    package_dir = Path(package_dir)
    workspace = Path(workspace)
    scene_root = workspace / "colmap_scene"
    images = scene_root / "images"
    masks = scene_root / "masks"
    renders = scene_root / "renders"
    opacity = scene_root / "opacity"
    for d in (images, masks, renders, opacity):
        d.mkdir(parents=True, exist_ok=True)

    anchors = json.loads((package_dir / "anchors.json").read_text(encoding="utf-8"))
    trajectory = json.loads((package_dir / "trajectory.json").read_text(encoding="utf-8"))

    frames = []
    for i, anchor in enumerate(anchors["anchors"]):
        name = f"{i:05d}.png"
        src = package_dir / anchor["rgb_path"]
        shutil.copy2(src, images / name)
        shutil.copy2(src, renders / name)
        mask_src = package_dir / anchor["mask_path"]
        shutil.copy2(mask_src, masks / name)
        shutil.copy2(mask_src, opacity / name)
        frames.append(
            {
                "file_path": name,
                "transform_matrix": anchor["transform_matrix"],
                "fl_x": anchor["fl_x"],
                "fl_y": anchor["fl_y"],
                "cx": anchor["cx"],
                "cy": anchor["cy"],
                "w": anchor["w"],
                "h": anchor["h"],
                "is_anchor": True,
            }
        )

    offset = len(frames)
    for j, frame in enumerate(trajectory["frames"]):
        name = f"{offset + j:05d}.png"
        # Placeholder target images; real pipeline renders from 3DGRUT.
        if frames:
            shutil.copy2(images / frames[0]["file_path"], images / name)
            shutil.copy2(opacity / frames[0]["file_path"], opacity / name)
            shutil.copy2(renders / frames[0]["file_path"], renders / name)
        frames.append(
            {
                "file_path": name,
                "transform_matrix": frame["transform_matrix"],
                "fl_x": frame["fl_x"],
                "fl_y": frame["fl_y"],
                "cx": frame["cx"],
                "cy": frame["cy"],
                "w": frame["w"],
                "h": frame["h"],
                "is_anchor": False,
            }
        )

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": trajectory["fl_x"],
        "fl_y": trajectory["fl_y"],
        "cx": trajectory["cx"],
        "cy": trajectory["cy"],
        "w": trajectory["w"],
        "h": trajectory["h"],
        "frames": frames,
    }
    (scene_root / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    selected = list(range(len(anchors["anchors"])))
    targets = list(range(offset, offset + len(trajectory["frames"])))
    (scene_root / "selected_indices.json").write_text(json.dumps(selected), encoding="utf-8")
    (scene_root / "target_indices.json").write_text(json.dumps(targets), encoding="utf-8")
    shutil.copy2(package_dir / "object_local.ply", scene_root / "object_local.ply")
    return scene_root
