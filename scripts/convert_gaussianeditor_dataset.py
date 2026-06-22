#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldact.utils.io import load_gaussians_ply, save_gaussians_ply


DEFAULT_DATASET_ROOT = "/home/hongjiacheng/Codes/gs_lab/GaussianEditor/dataset"
REQUIRED_GAUSSIAN_PROPS = {
    "x",
    "y",
    "z",
    "scale_0",
    "scale_1",
    "scale_2",
    "opacity",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
}


def _read_ply_header(path: Path) -> tuple[int | None, list[str]]:
    vertex_count: int | None = None
    props: list[str] = []
    in_vertex = False
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="replace").strip()
            if line == "end_header":
                break
            if line.startswith("element "):
                parts = line.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) >= 3:
                    props.append(parts[-1])
    return vertex_count, props


def _is_gaussian_ply(path: Path) -> tuple[bool, int | None, list[str]]:
    vertex_count, props = _read_ply_header(path)
    return REQUIRED_GAUSSIAN_PROPS.issubset(set(props)), vertex_count, props


def _iteration_number(path: Path) -> int:
    match = re.search(r"iteration_(\d+)", str(path))
    return int(match.group(1)) if match else -1


def _scene_dirs(dataset_root: Path, scene_names: list[str] | None) -> list[Path]:
    if scene_names:
        return [dataset_root / name for name in scene_names]
    return sorted(path for path in dataset_root.iterdir() if path.is_dir())


def _select_scene_ply(scene_dir: Path, iteration: str) -> Path | None:
    candidates = sorted(scene_dir.glob("point_cloud/iteration_*/point_cloud.ply"), key=_iteration_number)
    if not candidates:
        return None
    if iteration == "latest":
        return candidates[-1]
    target_name = f"iteration_{int(iteration)}"
    for candidate in candidates:
        if candidate.parent.name == target_name:
            return candidate
    return None


def _write_output(source: Path, output: Path, mode: str, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        if not overwrite:
            return
        output.unlink()
    if mode == "symlink":
        output.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, output)
    elif mode == "rewrite":
        params = load_gaussians_ply(source)
        save_gaussians_ply(params, output)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export GaussianEditor trained Gaussian PLYs into a WorldAct-friendly location."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", default="demo/gaussianeditor")
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene names; default exports all scenes")
    parser.add_argument("--iteration", default="latest", help="'latest' or a number such as 30000")
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy", "rewrite"],
        default="symlink",
        help="symlink avoids copying GB files; rewrite strips extra properties but is slow and memory-heavy",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    manifest: list[dict[str, object]] = []
    for scene_dir in _scene_dirs(dataset_root, args.scenes):
        record: dict[str, object] = {"scene": scene_dir.name}
        if not scene_dir.exists():
            record.update({"status": "missing_scene_dir", "scene_dir": str(scene_dir)})
            manifest.append(record)
            continue

        source = _select_scene_ply(scene_dir, str(args.iteration))
        if source is None:
            input_ply = scene_dir / "input.ply"
            record.update(
                {
                    "status": "missing_trained_gaussian_ply",
                    "note": "input.ply is an initial COLMAP point cloud and is not enough for WorldAct 3DGS rendering",
                    "input_ply": str(input_ply) if input_ply.exists() else None,
                }
            )
            manifest.append(record)
            continue

        is_gaussian, vertex_count, props = _is_gaussian_ply(source)
        output = output_dir / f"{scene_dir.name}.ply"
        record.update(
            {
                "status": "ok" if is_gaussian else "not_gaussian_ply",
                "source": str(source),
                "output": str(output),
                "mode": args.mode,
                "iteration": _iteration_number(source),
                "vertex_count": vertex_count,
                "properties": props,
            }
        )
        if is_gaussian and not args.dry_run:
            _write_output(source, output, args.mode, args.overwrite)
        manifest.append(record)

    manifest_path = output_dir / "manifest.json"
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        print(f"Dry run only; no files written. Manifest would be: {manifest_path}")
    else:
        print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
