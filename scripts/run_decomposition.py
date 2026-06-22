#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from worldact.config import default_config, load_config
from worldact.data.gaussians import GaussianScene
from worldact.data.masks import MaskSet
from worldact.rendering.camera_planning import (
    generate_camera_trajectory,
    select_keyframes,
    trajectory_center_for_keyframes,
)
from worldact.stage1_decomposition import DecompositionRunner
from worldact.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def _ensure_scene_cameras(scene: GaussianScene, cfg, *, ignore_config_center: bool = False) -> None:
    if scene.camera_trajectory:
        scene.camera_trajectory = [cam.to(scene.device) for cam in scene.camera_trajectory]
        return

    cam_cfg = cfg.stage1.camera_gen
    scene.camera_trajectory = [
        cam.to(scene.device)
        for cam in generate_camera_trajectory(
            scene.params.means,
            cam_cfg,
            ignore_config_center=ignore_config_center,
        )
    ]
    log.info("Generated %d cameras (mode=%s)", len(scene.camera_trajectory), getattr(cam_cfg, "trajectory_mode", "single_ring"))


def _keyframe_indices(scene: GaussianScene, cfg, *, ignore_config_center: bool = False) -> list[int]:
    cam_cfg = cfg.stage1.camera_gen
    center = trajectory_center_for_keyframes(
        scene.params.means,
        cam_cfg,
        ignore_config_center=ignore_config_center,
    )
    return select_keyframes(
        scene.camera_trajectory,
        int(cfg.stage1.vlm.num_keyframes),
        cam_cfg,
        center=center,
    )


def _save_keyframes(scene: GaussianScene, output_dir: Path, keyframe_indices: list[int]) -> None:
    from worldact.utils.visualization import save_image

    keyframe_dir = output_dir / "stage1" / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for order, view_idx in enumerate(keyframe_indices):
        view = scene.render(scene.camera_trajectory[view_idx])
        filename = f"keyframe_{order:02d}_view_{view_idx:02d}.png"
        save_image(view.rgb, keyframe_dir / filename)
        records.append({"order": order, "view_index": view_idx, "filename": filename})

    with open(keyframe_dir / "keyframes.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    log.info("Saved %d keyframes to %s", len(keyframe_indices), keyframe_dir)


def _slugify_name(name: str) -> str:
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip().lower())
    return slug.strip("_") or "object"


def _save_sam3_masks(
    scene: GaussianScene,
    masks: MaskSet | None,
    output_dir: Path,
    keyframe_indices: list[int],
) -> None:
    if masks is None:
        log.warning("No SAM3 masks available to save")
        return

    from worldact.utils.visualization import COLORS, save_image

    keyframe_indices = list(keyframe_indices)
    keyframe_dir = output_dir / "stage1" / "keyframes"
    masks_dir = keyframe_dir / "sam3_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for order, view_idx in enumerate(keyframe_indices):
        view = scene.render(scene.camera_trajectory[view_idx])
        base_name = f"keyframe_{order:02d}_view_{view_idx:02d}"
        overlay = view.rgb.clone()
        fused = torch.zeros((*view.rgb.shape[:2], 1), dtype=overlay.dtype, device=overlay.device)
        object_files: list[dict[str, object]] = []

        for obj_idx, obj_mask in enumerate(masks.object_masks):
            mask = obj_mask.get_mask(view_idx)
            if mask is None:
                continue
            mask = mask.to(overlay.device).float().clamp(0.0, 1.0).unsqueeze(-1)
            color = torch.tensor(COLORS[obj_idx % len(COLORS)], dtype=overlay.dtype, device=overlay.device)
            overlay = overlay * (1.0 - 0.45 * mask) + color.view(1, 1, 3) * 0.45 * mask
            fused = torch.maximum(fused, mask)

            obj_slug = _slugify_name(obj_mask.object_name)
            obj_filename = f"{base_name}_obj_{obj_idx:02d}_{obj_slug}.png"
            save_image(mask.repeat(1, 1, 3), masks_dir / obj_filename)
            object_files.append(
                {
                    "object_id": obj_mask.object_id,
                    "object_name": obj_mask.object_name,
                    "filename": obj_filename,
                }
            )

        save_image(overlay, keyframe_dir / f"{base_name}_sam3_overlay.png")
        save_image(fused.repeat(1, 1, 3), keyframe_dir / f"{base_name}_sam3_union.png")
        records.append(
            {
                "order": order,
                "view_index": view_idx,
                "overlay": f"{base_name}_sam3_overlay.png",
                "union_mask": f"{base_name}_sam3_union.png",
                "objects": object_files,
            }
        )

    with open(keyframe_dir / "sam3_masks.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    log.info("Saved SAM3 masks for %d keyframes to %s", len(keyframe_indices), keyframe_dir)


def _save_object_memory(states, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [state.to_json() for state in states]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info("Saved %d object memory entries to %s", len(data), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="WorldAct Stage 1: scene decomposition with SAM3")
    parser.add_argument("--ply", required=True, help="Input standard 3DGS PLY")
    parser.add_argument("--cameras", default=None, help="Optional cameras.json")
    parser.add_argument("--config", default="configs/scenes/indoor.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--objects", nargs="+", default=None, help="Manual object prompts; bypass VLM")
    parser.add_argument("--use-vlm", action="store_true", help="Use OpenAI-compatible GPT-5.5 object discovery")
    parser.add_argument("--vlm-api-base", default=None)
    parser.add_argument("--vlm-api-key", default=None)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--sam3-source-dir", default=None)
    parser.add_argument("--sam3-checkpoint", default=None)
    parser.add_argument("--chunked", action="store_true", help="Enable XY chunked scene decomposition")
    parser.add_argument("--chunk-target-gaussians", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=float, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--keyframe-num", type=int, default=None, help="Number of keyframes sampled for VLM and saving")
    parser.add_argument(
        "--save-object-memory",
        action="store_true",
        help="Save discovered ObjectMemory to JSON and exit before segmentation",
    )
    parser.add_argument(
        "--object-memory-json",
        default=None,
        help="Output path for ObjectMemory JSON (defaults to <output_dir>/stage1/object_memory.json)",
    )
    parser.add_argument(
        "--save-sam3-masks",
        action="store_true",
        help="Save SAM3 masks/overlays for sampled keyframes",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save-keyframes", action="store_true", help="Save sampled keyframe renders for inspection")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    setup_logging("INFO")
    overrides = list(args.overrides)
    if args.output_dir:
        overrides.append(f"output_dir={args.output_dir}")
    if args.device:
        overrides.append(f"device.device={args.device}")
        overrides.append(f"stage1.sam3.device={args.device}")
    if args.objects:
        quoted = ",".join(args.objects)
        overrides.append(f"stage1.manual_prompts=[{quoted}]")
    elif not args.use_vlm:
        raise SystemExit("Provide --objects or --use-vlm.")
    if args.vlm_api_base:
        overrides.append(f"stage1.vlm.api_base={args.vlm_api_base}")
    if args.vlm_api_key:
        overrides.append(f"stage1.vlm.api_key={args.vlm_api_key}")
    if args.vlm_model:
        overrides.append(f"stage1.vlm.model={args.vlm_model}")
    if args.sam3_source_dir:
        overrides.append(f"stage1.sam3.source_dir={args.sam3_source_dir}")
    if args.sam3_checkpoint:
        overrides.append(f"stage1.sam3.checkpoint_path={args.sam3_checkpoint}")
    if args.chunked:
        overrides.append("stage1.chunking.enabled=true")
    if args.chunk_target_gaussians is not None:
        overrides.append(f"stage1.chunking.target_gaussians_per_chunk={args.chunk_target_gaussians}")
    if args.chunk_overlap is not None:
        overrides.append(f"stage1.chunking.overlap_xy={args.chunk_overlap}")
    if args.max_chunks is not None:
        overrides.append(f"stage1.chunking.max_chunks={args.max_chunks}")
    if args.keyframe_num is not None:
        overrides.append(f"stage1.vlm.num_keyframes={args.keyframe_num}")

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path, overrides) if cfg_path.exists() else default_config(overrides)
    if "???" in OmegaConf.to_yaml(cfg):
        raise SystemExit("Config has missing required values. Set --output-dir or output_dir=...")

    scene = GaussianScene.from_ply(args.ply, cameras_path=args.cameras)
    scene.params = scene.params.to(cfg.device.device)
    log.info("Loaded %d Gaussians from %s", scene.num_gaussians, args.ply)

    runner = DecompositionRunner(cfg)
    _ensure_scene_cameras(scene, cfg)
    output_dir = Path(cfg.output_dir)
    keyframe_indices = _keyframe_indices(scene, cfg)
    if args.save_keyframes:
        _save_keyframes(scene, output_dir, keyframe_indices)

    if args.save_object_memory:
        manual = list(cfg.stage1.manual_prompts) if cfg.stage1.manual_prompts else None
        from worldact.stage1_decomposition.agent import MultimodalAgent

        object_agent = MultimodalAgent(cfg.stage1)
        keyframe_views = [scene.render(scene.camera_trajectory[i]) for i in keyframe_indices]
        states = object_agent.discover_objects(keyframe_views, manual_prompts=manual)
        memory_path = Path(args.object_memory_json) if args.object_memory_json else output_dir / "stage1" / "object_memory.json"
        _save_object_memory(states, memory_path)
        return

    result = runner.run(scene)
    runner.save_checkpoint(result, output_dir, scene)
    log.info("Saved decomposition to %s", output_dir / "stage1")
    log.info("Objects: %d; background Gaussians: %d", result.num_objects, result.background.num_gaussians)

    if args.save_sam3_masks:
        _save_sam3_masks(scene, result.masks, output_dir, keyframe_indices)

    if args.visualize:
        from worldact.utils.visualization import visualize_decomposition

        visualize_decomposition(scene, result, output_dir / "stage1" / "vis", camera_cfg=cfg.stage1.camera_gen)


if __name__ == "__main__":
    main()
