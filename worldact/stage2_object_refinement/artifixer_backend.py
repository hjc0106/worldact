from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from worldact.stage2_object_refinement.backend import FeedForwardBackend
from worldact.stage2_object_refinement.common import BackendRunResult


@dataclass
class ArtiFixerPreparedScene:
    scene_id: str
    scene_root: Path
    split_path: Path
    target_indices: list[int]
    frame_to_view: dict[int, int]


class ArtiFixerBackend(FeedForwardBackend):
    """External-adapter backend for NVIDIA ArtiFixer direct frame generation."""

    def __init__(
        self,
        *,
        repo: str | None,
        checkpoint: str | None,
        python: str = "python",
        save_dir: str | None = None,
        command: str | None = None,
        render_trajectory: str = "trajectory",
        timeout: float = 3600.0,
        prompt_path: str | None = None,
        num_views: int | None = None,
    ) -> None:
        self.repo = Path(repo).expanduser() if repo else None
        self.checkpoint = Path(checkpoint).expanduser() if checkpoint else None
        self.python = python
        self.save_dir = Path(save_dir).expanduser() if save_dir else None
        self.command = command
        self.render_trajectory = render_trajectory
        self.timeout = float(timeout)
        self.prompt_path = Path(prompt_path).expanduser() if prompt_path else None
        self.num_views = num_views

    def run(self, request_path: Path, work_dir: Path) -> BackendRunResult:
        try:
            request = _load_json(request_path)
            prepared = prepare_artifixer_scene(
                request,
                work_dir,
                prompt_path=self.prompt_path,
                max_views=self.num_views,
            )
        except Exception as exc:
            return BackendRunResult(None, None, "failed", f"ArtiFixer prepare failed: {exc}")

        log_path = work_dir / "artifixer_backend.log"
        if not prepared.target_indices:
            copied = _copy_fallback_views(prepared, work_dir)
            _write_generation_meta(
                work_dir,
                prepared,
                command=None,
                status="fallback_no_target_frames",
                copied_views=copied,
                log_path=log_path,
            )
            return BackendRunResult(
                generated_ply=None,
                metadata_path=str(work_dir / "metadata.json") if (work_dir / "metadata.json").exists() else None,
                status="generated",
                message="ArtiFixer skipped because only one usable view was available; copied fallback view.",
            )

        if self.command is None:
            if self.repo is None or not self.repo.exists():
                return BackendRunResult(None, None, "failed", "--artifixer-repo must point to the official ArtiFixer repo")
            if self.checkpoint is None or not self.checkpoint.exists():
                return BackendRunResult(None, None, "failed", "--artifixer-checkpoint must point to an ArtiFixer checkpoint")

        save_dir = self.save_dir or (work_dir / "artifixer_output")
        command = self._format_command(prepared, work_dir, save_dir)
        completed = None
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(self.repo) if self.repo is not None and self.repo.exists() else str(work_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            log_path.write_text(f"$ {command}\n\nTIMEOUT after {self.timeout}s\n{exc}\n", encoding="utf-8")
            return BackendRunResult(None, None, "timeout", f"ArtiFixer timed out after {self.timeout}s")

        log_path.write_text(
            f"$ {command}\n\n[stdout]\n{completed.stdout}\n\n[stderr]\n{completed.stderr}\n",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            return BackendRunResult(None, None, "failed", f"ArtiFixer exited with code {completed.returncode}; see {log_path}")

        try:
            copied = collect_artifixer_predictions(prepared, save_dir, work_dir)
        except Exception as exc:
            return BackendRunResult(None, None, "missing_output", f"Could not collect ArtiFixer outputs: {exc}")

        _write_generation_meta(
            work_dir,
            prepared,
            command=command,
            status="generated",
            copied_views=copied,
            log_path=log_path,
        )
        _write_identity_metadata(work_dir, "artifixer_pseudo_views")
        return BackendRunResult(
            generated_ply=None,
            metadata_path=str(work_dir / "metadata.json"),
            status="generated",
            message=f"Generated {copied} ArtiFixer pseudo views",
        )

    def _format_command(self, prepared: ArtiFixerPreparedScene, work_dir: Path, save_dir: Path) -> str:
        if self.command:
            return self.command.format(
                split=str(prepared.split_path),
                save_dir=str(save_dir),
                checkpoint=str(self.checkpoint or ""),
                repo=str(self.repo or ""),
                workdir=str(work_dir),
                scene_id=prepared.scene_id,
            )
        return (
            f'"{self.python}" -m model_eval.run_inference '
            f"--evalset reconstructed_colmap "
            f'--checkpoint_pt "{self.checkpoint}" '
            f'--save_dir "{save_dir}" '
            f'--split_path "{prepared.split_path}" '
            f"--render_trajectory {self.render_trajectory} "
            f"--save_frame_outputs_only"
        )


def prepare_artifixer_scene(
    request: dict[str, Any],
    work_dir: Path,
    *,
    prompt_path: Path | None = None,
    max_views: int | None = None,
) -> ArtiFixerPreparedScene:
    views = list((request.get("multiview") or {}).get("views") or [])
    if max_views is not None:
        views = views[: max(1, int(max_views))]
    if not views:
        raise ValueError("object_request.multiview.views is required")

    scene_id = f"object_{request.get('object_id', '0')}"
    root = work_dir / "artifixer_scene"
    image_root = root / "images"
    render_dir = root / "renders"
    opacity_dir = root / "opacity"
    captions_dir = root / "captions"
    for path in (image_root, render_dir, opacity_dir, captions_dir):
        path.mkdir(parents=True, exist_ok=True)

    best_source_view = int(request.get("best_view_index", -1) or -1)
    selected_frame = _selected_frame_index(views, best_source_view)
    target_indices = [idx for idx in range(len(views)) if idx != selected_frame]
    frame_to_view: dict[int, int] = {}
    frames: list[dict[str, Any]] = []
    top_intrinsics: dict[str, Any] | None = None

    for frame_idx, view in enumerate(views):
        frame_to_view[frame_idx] = int(view.get("view_index", frame_idx))
        rgb_path = Path(view["rgb_path"])
        if not rgb_path.exists():
            raise FileNotFoundError(f"Missing multiview rgb_path: {rgb_path}")
        name = f"{frame_idx:05d}.png"
        shutil.copy2(rgb_path, image_root / name)
        shutil.copy2(rgb_path, render_dir / name)
        _write_opacity_png(view, opacity_dir / name)

        intrinsic = view.get("intrinsic")
        if intrinsic is None:
            raise ValueError(f"View {frame_idx} is missing intrinsic")
        intr = _intrinsics_dict(intrinsic, int(view.get("width", 0)), int(view.get("height", 0)))
        top_intrinsics = top_intrinsics or intr
        frame = {
            "file_path": name,
            "transform_matrix": _c2w_from_view(view),
            "fl_x": intr["fl_x"],
            "fl_y": intr["fl_y"],
            "cx": intr["cx"],
            "cy": intr["cy"],
            "w": intr["w"],
            "h": intr["h"],
        }
        frames.append(frame)

    if top_intrinsics is None:
        raise ValueError("No camera intrinsics found")
    transforms = {
        "camera_model": "OPENCV",
        **top_intrinsics,
        "frames": frames,
    }
    transforms_path = root / "transforms.json"
    _write_json(transforms_path, transforms)
    selected_path = root / "selected_indices.json"
    target_path = root / "target_indices.json"
    _write_json(selected_path, [selected_frame])
    _write_json(target_path, target_indices)

    prompt_out = captions_dir / "caption.h5"
    if prompt_path is not None:
        shutil.copy2(prompt_path, prompt_out)
    else:
        _write_minimal_caption_h5(prompt_out, str((request.get("memory_state") or {}).get("text_prompt", "")))

    split = {
        "test": {
            scene_id: {
                "transforms_path": str(transforms_path),
                "image_root": str(image_root),
                "render_dir": str(render_dir),
                "opacity_dir": str(opacity_dir),
                "selected_indices_path": str(selected_path),
                "target_indices_path": str(target_path),
                "prompt_path": str(prompt_out),
                "camera_scale": 1.0,
                "has_gt": False,
            }
        }
    }
    split_path = root / "split.json"
    _write_json(split_path, split)
    _write_json(
        root / "worldact_artifixer_export.json",
        {
            "scene_id": scene_id,
            "selected_frame": selected_frame,
            "target_indices": target_indices,
            "frame_to_view": frame_to_view,
        },
    )
    return ArtiFixerPreparedScene(scene_id, root, split_path, target_indices, frame_to_view)


def collect_artifixer_predictions(prepared: ArtiFixerPreparedScene, save_dir: Path, work_dir: Path) -> int:
    out_dir = work_dir / "multiview_generate"
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    cameras = []
    transforms = _load_json(prepared.scene_root / "transforms.json")
    for out_idx, frame_idx in enumerate(prepared.target_indices):
        pred = _find_prediction(save_dir, prepared.scene_id, frame_idx)
        if pred is None:
            raise FileNotFoundError(f"Missing prediction for frame {frame_idx:05d} under {save_dir}")
        shutil.copy2(pred, out_dir / f"view_{out_idx:02d}.png")
        frame = transforms["frames"][frame_idx]
        cameras.append(
            {
                "view_index": int(prepared.frame_to_view.get(frame_idx, frame_idx)),
                "source_view_index": int(prepared.frame_to_view.get(frame_idx, frame_idx)),
                "rgb_path": str(out_dir / f"view_{out_idx:02d}.png"),
                "extrinsic": _w2c_from_c2w(frame["transform_matrix"]),
                "intrinsic": [
                    [frame["fl_x"], 0.0, frame["cx"]],
                    [0.0, frame["fl_y"], frame["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                "width": int(frame["w"]),
                "height": int(frame["h"]),
            }
        )
        copied += 1
    _write_json(out_dir / "cameras.json", cameras)
    _write_pseudo_view_metrics(out_dir)
    return copied


def _copy_fallback_views(prepared: ArtiFixerPreparedScene, work_dir: Path) -> int:
    out_dir = work_dir / "multiview_generate"
    out_dir.mkdir(parents=True, exist_ok=True)
    source = prepared.scene_root / "renders" / "00000.png"
    if not source.exists():
        return 0
    shutil.copy2(source, out_dir / "view_00.png")
    _write_pseudo_view_metrics(out_dir)
    _write_identity_metadata(work_dir, "artifixer_fallback_no_target_frames")
    return 1


def _selected_frame_index(views: list[dict[str, Any]], best_source_view: int) -> int:
    for idx, view in enumerate(views):
        if int(view.get("view_index", -999999)) == best_source_view:
            return idx
    return 0


def _write_opacity_png(view: dict[str, Any], output_path: Path) -> None:
    from PIL import Image
    import numpy as np

    alpha_path = view.get("alpha_path")
    mask_path = view.get("mask_path")
    if alpha_path and Path(alpha_path).exists():
        alpha = torch.load(alpha_path, map_location="cpu", weights_only=False)
        arr = alpha.detach().float().clamp(0.0, 1.0).numpy()
        if arr.ndim == 3:
            arr = arr[..., 0]
    elif mask_path and Path(mask_path).exists():
        arr = np.asarray(Image.open(mask_path).convert("L")).astype("float32") / 255.0
    else:
        rgb = Image.open(view["rgb_path"]).convert("RGB")
        arr_rgb = np.asarray(rgb).astype("float32") / 255.0
        arr = (arr_rgb < 0.98).any(axis=-1).astype("float32")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr * 255.0).clip(0, 255).astype("uint8"), mode="L").save(output_path)


def _intrinsics_dict(intrinsic: Any, width: int, height: int) -> dict[str, Any]:
    return {
        "w": int(width),
        "h": int(height),
        "fl_x": float(intrinsic[0][0]),
        "fl_y": float(intrinsic[1][1]),
        "cx": float(intrinsic[0][2]),
        "cy": float(intrinsic[1][2]),
    }


def _c2w_from_view(view: dict[str, Any]) -> list[list[float]]:
    extrinsic = torch.tensor(view["extrinsic"], dtype=torch.float32)
    c2w = torch.linalg.inv(extrinsic)
    return [[float(v) for v in row] for row in c2w.tolist()]


def _w2c_from_c2w(c2w_raw: Any) -> list[list[float]]:
    c2w = torch.tensor(c2w_raw, dtype=torch.float32)
    w2c = torch.linalg.inv(c2w)
    return [[float(v) for v in row] for row in w2c.tolist()]


def _write_minimal_caption_h5(path: Path, caption: str) -> None:
    import numpy as np

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to auto-create an ArtiFixer caption HDF5; "
            "install h5py in this environment or pass --artifixer-prompt-path"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("0", data=np.zeros((1, 4096), dtype=np.uint16))
        ds.attrs["caption"] = caption


def _find_prediction(save_dir: Path, scene_id: str, frame_idx: int) -> Path | None:
    name = f"{frame_idx:05d}.png"
    candidates = list(save_dir.glob(f"**/{scene_id}/frames/batch_0000/pred/{name}"))
    candidates.extend(save_dir.glob(f"**/frames/batch_0000/pred/{name}"))
    candidates.extend(save_dir.glob(f"**/pred/{name}"))
    for path in candidates:
        if path.is_file():
            return path
    return None


def _write_generation_meta(
    work_dir: Path,
    prepared: ArtiFixerPreparedScene,
    *,
    command: str | None,
    status: str,
    copied_views: int,
    log_path: Path,
) -> None:
    _write_json(
        work_dir / "generation_meta.json",
        {
            "generation_model": "artifixer",
            "status": status,
            "scene_root": str(prepared.scene_root),
            "split_path": str(prepared.split_path),
            "target_indices": prepared.target_indices,
            "frame_to_view": prepared.frame_to_view,
            "copied_views": int(copied_views),
            "command": command,
            "log_path": str(log_path),
        },
    )


def _write_pseudo_view_metrics(pseudo_dir: Path) -> None:
    from worldact.stage2_object_refinement.sv3d_backend import _write_pseudo_view_metrics as write_metrics

    write_metrics(sorted(pseudo_dir.glob("view_*.png")), pseudo_dir.parent / "pseudo_view_metrics.json")


def _write_identity_metadata(work_dir: Path, source: str) -> None:
    _write_json(work_dir / "metadata.json", {"transform": torch.eye(4).tolist(), "source": source})


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
