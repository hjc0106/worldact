#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldact.config import default_config, load_config
from worldact.data.gaussians import GaussianScene
from worldact.data.poses import CameraPose
from worldact.rendering.camera_planning import (
    compute_scene_frame,
    generate_camera_trajectory,
    parse_up_axis,
    select_keyframes,
    trajectory_center_for_keyframes,
)
from worldact.stage1_decomposition.chunking import build_xy_chunks, chunk_manifest, filter_chunks_by_roi, make_chunk_scene
from worldact.stage1_decomposition.roi_guidance import localize_prompt_roi, save_roi_selection_artifacts
from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
from worldact.utils.logging import get_logger, setup_logging
from worldact.utils.visualization import save_image

log = get_logger(__name__)


def _make_local_cameras(scene: GaussianScene, cfg) -> None:
    scene.camera_trajectory = [
        cam.to(scene.device)
        for cam in generate_camera_trajectory(scene.params.means, cfg, ignore_config_center=True)
    ]


def _image_to_uint8(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _save_contact_sheet(image_paths: list[Path], output_path: Path, columns: int) -> None:
    if not image_paths:
        return
    from PIL import Image, ImageDraw

    thumbs = [_image_to_uint8(path) for path in image_paths]
    thumb_w = min(256, thumbs[0].width)
    thumb_h = int(thumbs[0].height * (thumb_w / thumbs[0].width))
    label_h = 24
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (path, image) in enumerate(zip(image_paths, thumbs)):
        row, col = divmod(idx, columns)
        image = image.resize((thumb_w, thumb_h))
        x, y = col * thumb_w, row * (thumb_h + label_h)
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 6, y + 5), path.parent.name, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _save_selected_contact_sheet(image_paths: list[Path], output_path: Path, columns: int) -> None:
    if not image_paths:
        return
    _save_contact_sheet(image_paths, output_path, columns)


def _axis_vector(axis: int, sign: float = 1.0) -> torch.Tensor:
    vec = torch.zeros(3, dtype=torch.float32)
    vec[int(axis)] = float(sign)
    return vec


def _horizontal_axes(up_axis: str) -> tuple[int, int, int, float]:
    up_idx, up_sign = parse_up_axis(up_axis)
    axes = [0, 1, 2]
    axes.remove(up_idx)
    return axes[0], axes[1], up_idx, up_sign


def _make_intrinsic(image_width: int, image_height: int, fov_deg: float) -> torch.Tensor:
    fov = math.radians(float(fov_deg))
    fy = image_height / (2.0 * math.tan(fov / 2.0))
    fx = fy
    return torch.tensor(
        [[fx, 0.0, image_width / 2.0], [0.0, fy, image_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _topdown_camera_pose(
    means: torch.Tensor,
    up_axis: str,
    *,
    image_width: int,
    image_height: int,
    fov_deg: float,
) -> CameraPose:
    frame = compute_scene_frame(means, up_axis=up_axis)
    h0, h1, up_idx, up_sign = _horizontal_axes(up_axis)
    center = frame.center.detach().cpu().float()

    means_cpu = means.detach().cpu().float()
    lo = torch.quantile(means_cpu, 0.01, dim=0)
    hi = torch.quantile(means_cpu, 0.99, dim=0)
    span_h0 = float(max(1e-4, (hi[h0] - lo[h0]).item()))
    span_h1 = float(max(1e-4, (hi[h1] - lo[h1]).item()))
    span_up = float(max(1e-4, ((hi[up_idx] - lo[up_idx]).abs()).item()))
    half_extent = 0.5 * max(span_h0, span_h1)
    fov = math.radians(float(fov_deg))
    fit_distance = half_extent / max(1e-4, math.tan(fov / 2.0))
    distance = max(fit_distance * 1.35, span_up + 0.5, 2.0)

    cam_pos = center.clone()
    cam_pos[up_idx] += up_sign * distance
    forward = F.normalize(center - cam_pos, dim=0)

    ref = _axis_vector(h0)
    right = torch.cross(ref, forward, dim=0)
    if float(torch.linalg.norm(right).item()) < 1e-6:
        ref = _axis_vector(h1)
        right = torch.cross(ref, forward, dim=0)
    right = F.normalize(right, dim=0)
    up = torch.cross(right, forward, dim=0)

    R = torch.stack([right, -up, forward], dim=0)
    t = -R @ cam_pos
    extrinsic = torch.eye(4, dtype=torch.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return CameraPose(
        extrinsic=extrinsic,
        intrinsic=_make_intrinsic(image_width, image_height, fov_deg),
        width=image_width,
        height=image_height,
    )


def _save_chunk_topdown(
    chunk_scene: GaussianScene,
    up_axis: str,
    output_path: Path,
    *,
    image_width: int,
    image_height: int,
    fov_deg: float,
) -> Path:
    camera = _topdown_camera_pose(
        chunk_scene.params.means,
        up_axis,
        image_width=image_width,
        image_height=image_height,
        fov_deg=fov_deg,
    ).to(chunk_scene.device)
    view = chunk_scene.render(camera)
    save_image(view.rgb, output_path)
    return output_path


def _write_selected_chunk_plys(
    scene: GaussianScene,
    chunks: list,
    manifest_by_id: dict[str, dict],
    selected_ids: set[str],
    output_dir: Path,
    *,
    ply_name: str = "chunk.ply",
) -> list[dict[str, object]]:
    selected_dir = output_dir / "selected_chunks"
    selected_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for chunk in chunks:
        if chunk.chunk_id not in selected_ids:
            continue
        chunk_dir = selected_dir / chunk.chunk_id
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_scene = make_chunk_scene(scene, chunk)
        chunk_scene.params = chunk_scene.params.to(scene.device)
        ply_path = chunk_dir / ply_name
        chunk_scene.to_ply(ply_path)
        item = dict(manifest_by_id[chunk.chunk_id])
        item["ply"] = str(ply_path)
        with open(chunk_dir / "chunk_meta.json", "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)
        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "ply": str(ply_path),
                "num_gaussians": int(len(chunk.indices)),
                "num_core_gaussians": int(chunk.core_mask.sum().item()),
                "roi_overlap": item.get("roi_overlap"),
                "selection_reason": item.get("selection_reason"),
            }
        )
    with open(selected_dir / "selected_chunks.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Render XY chunk previews for a 3DGS PLY scene")
    parser.add_argument("--ply", required=True, help="Input 3DGS PLY")
    parser.add_argument("--config", default="configs/scenes/indoor.yaml")
    parser.add_argument("--output-dir", default="outputs/chunk_preview")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--save-ply", action="store_true", help="Also save each expanded chunk as a PLY")
    parser.add_argument(
        "--export-selected-ply",
        action="store_true",
        help="Export only prompt/ROI-selected chunks as selected_chunks/<chunk_id>/chunk.ply",
    )
    parser.add_argument(
        "--selected-ply-name",
        default="chunk.ply",
        help="Filename used under each selected chunk directory",
    )
    parser.add_argument(
        "--render-selected-only",
        action="store_true",
        help="Render only selected chunks when prompt/ROI guidance is active",
    )
    parser.add_argument("--chunk-target-gaussians", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=float, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument(
        "--save-topdown",
        dest="save_topdown",
        action="store_true",
        default=True,
        help="Save one top-down render per rendered chunk and a topdown_contact_sheet.png summary",
    )
    parser.add_argument("--no-topdown", dest="save_topdown", action="store_false", help="Disable top-down chunk renders")
    parser.add_argument("--topdown-width", type=int, default=None)
    parser.add_argument("--topdown-height", type=int, default=None)
    parser.add_argument("--topdown-fov-deg", type=float, default=None)
    parser.add_argument(
        "--save-probe-views",
        action="store_true",
        help="Save prompt-guided ROI probe renders to probe_views/ even when SAM3 produces no ROI mask",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    setup_logging("INFO")
    overrides = list(args.overrides)
    overrides.append(f"output_dir={args.output_dir}")
    overrides.append(f"device.device={args.device}")
    overrides.append("stage1.chunking.enabled=true")
    if args.chunk_target_gaussians is not None:
        overrides.append(f"stage1.chunking.target_gaussians_per_chunk={args.chunk_target_gaussians}")
    if args.chunk_overlap is not None:
        overrides.append(f"stage1.chunking.overlap_xy={args.chunk_overlap}")
    if args.max_chunks is not None:
        overrides.append(f"stage1.chunking.max_chunks={args.max_chunks}")
    if args.camera_width is not None:
        overrides.append(f"stage1.camera_gen.image_width={args.camera_width}")
    if args.camera_height is not None:
        overrides.append(f"stage1.camera_gen.image_height={args.camera_height}")

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path, overrides) if cfg_path.exists() else default_config(overrides)
    OmegaConf.resolve(cfg)

    scene = GaussianScene.from_ply(args.ply)
    scene.params = scene.params.to(cfg.device.device)
    chunk_cfg = cfg.stage1.chunking
    cam_cfg = cfg.stage1.camera_gen
    up_axis = str(getattr(chunk_cfg, "up_axis", None) or getattr(cam_cfg, "up_axis", "z"))
    chunks = build_xy_chunks(scene, chunk_cfg, up_axis=up_axis)
    if not chunks:
        raise SystemExit("No valid chunks were generated. Lower stage1.chunking.min_gaussians_per_chunk or max chunk size.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = chunk_manifest(chunks)

    prompt = str(getattr(chunk_cfg, "prompt", None) or "").strip() or None
    explicit_center = getattr(chunk_cfg, "roi_center", None)
    explicit_radius = getattr(chunk_cfg, "roi_radius", None)
    roi_margin = float(getattr(chunk_cfg, "roi_margin", 0.0))
    selected_ids: set[str] = set()
    guidance_summary: dict[str, object] = {
        "enabled": False,
        "prompt": prompt,
        "up_axis": up_axis,
        "save_probe_views": bool(args.save_probe_views),
    }
    if prompt and (explicit_center is None or explicit_radius is None):
        shared_segmenter = SAM3Segmenter(cfg.stage1.sam3)
        probe_output_dir = output_dir if args.save_probe_views else None
        try:
            selection = localize_prompt_roi(
                scene,
                shared_segmenter,
                prompt,
                up_axis=up_axis,
                probe_output_dir=probe_output_dir,
            )
        except Exception as exc:
            log.warning("Prompt-guided ROI localization failed for %r: %s; rendering full chunk grid", prompt, exc)
            for item in manifest:
                item.update(
                    {
                        "roi_guided": False,
                        "prompt_guided": True,
                        "prompt": prompt,
                        "selected_by_prompt": False,
                        "selection_reason": "prompt_localization_failed",
                    }
                )
            guidance_summary.update(
                {
                    "enabled": True,
                    "source": "prompt_failed",
                    "prompt_guided": True,
                    "roi_guided": False,
                    "error": str(exc),
                }
            )
            if args.save_probe_views:
                guidance_summary["probe_artifacts"] = {
                    "probe_views_dir": str(output_dir / "probe_views"),
                    "probe_views_manifest": str(output_dir / "probe_views" / "probe_views.json"),
                    "probe_cameras": str(output_dir / "probe_cameras.json"),
                    "probe_contact_sheet": str(output_dir / "probe_contact_sheet.png"),
                }
        else:
            roi_artifacts = save_roi_selection_artifacts(selection, output_dir)
            probe_artifacts = selection.probe_artifacts or {}
            roi_center = (
                torch.tensor(list(explicit_center), dtype=torch.float32, device=scene.device)
                if explicit_center is not None
                else selection.center.to(scene.device)
            )
            roi_radius = float(explicit_radius) if explicit_radius is not None else float(selection.radius)
            _, manifest, roi_summary = filter_chunks_by_roi(
                scene,
                chunks,
                roi_center,
                float(roi_radius) + roi_margin,
                up_axis=up_axis,
                min_roi_overlap=float(getattr(chunk_cfg, "min_roi_overlap", 0.15)),
                keep_top_k=int(getattr(chunk_cfg, "keep_top_k", 2)),
                prompt=prompt,
                prompt_guided=True,
                roi_margin=roi_margin,
            )
            selected_ids = set(roi_summary["selected_chunk_ids"])
            guidance_summary.update(
                {
                    "enabled": True,
                    "source": "prompt",
                    "prompt_guided": True,
                    "roi_guided": True,
                    "prompt": prompt,
                    "roi_center": [float(v) for v in roi_center.detach().cpu().tolist()],
                    "roi_radius": float(roi_radius),
                    "roi_margin": roi_margin,
                    "best_view_index": int(selection.best_view_index),
                    "best_candidate_id": int(selection.best_candidate.candidate_id),
                    "best_candidate_score": float(selection.best_score),
                    "best_candidate_area": float(selection.best_candidate.area),
                    "probe_artifacts": probe_artifacts,
                    **roi_artifacts,
                    **roi_summary,
                }
            )
    elif explicit_center is not None and explicit_radius is not None:
        roi_center = torch.tensor(list(explicit_center), dtype=torch.float32, device=scene.device)
        roi_radius = float(explicit_radius)
        _, manifest, roi_summary = filter_chunks_by_roi(
            scene,
            chunks,
            roi_center,
            float(roi_radius) + roi_margin,
            up_axis=up_axis,
            min_roi_overlap=float(getattr(chunk_cfg, "min_roi_overlap", 0.15)),
            keep_top_k=int(getattr(chunk_cfg, "keep_top_k", 2)),
            prompt=prompt,
            prompt_guided=False,
            roi_margin=roi_margin,
        )
        selected_ids = set(roi_summary["selected_chunk_ids"])
        guidance_summary.update(
            {
                "enabled": True,
                "source": "explicit",
                "prompt_guided": False,
                "roi_guided": True,
                "prompt": prompt,
                "roi_center": [float(v) for v in roi_center.detach().cpu().tolist()],
                "roi_radius": float(roi_radius),
                "roi_margin": roi_margin,
                **roi_summary,
            }
        )
    else:
        for item in manifest:
            item.update(
                {
                    "roi_guided": False,
                    "prompt_guided": False,
                    "prompt": None,
                    "selected_by_prompt": False,
                    "selection_reason": "unfiltered",
                }
            )
        if args.export_selected_ply:
            selected_ids = {chunk.chunk_id for chunk in chunks}
            guidance_summary.update(
                {
                    "enabled": False,
                    "source": "unfiltered",
                    "roi_guided": False,
                    "prompt_guided": False,
                    "selected_chunk_ids": sorted(selected_ids),
                    "num_selected": len(selected_ids),
                }
            )

    first_view_paths: list[Path] = []
    selected_first_view_paths: list[Path] = []
    topdown_paths: list[Path] = []
    selected_topdown_paths: list[Path] = []
    manifest_by_id = {item["chunk_id"]: item for item in manifest}
    chunks_to_render = [chunk for chunk in chunks if chunk.chunk_id in selected_ids] if args.render_selected_only and selected_ids else chunks
    log.info("Rendering %d chunks to %s", len(chunks_to_render), output_dir)
    for chunk in chunks_to_render:
        item = manifest_by_id[chunk.chunk_id]
        chunk_dir = output_dir / chunk.chunk_id
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_scene = make_chunk_scene(scene, chunk)
        chunk_scene.params = chunk_scene.params.to(scene.device)
        _make_local_cameras(chunk_scene, cfg.stage1.camera_gen)
        if args.save_ply:
            chunk_scene.to_ply(chunk_dir / "expanded_chunk.ply")

        cam_cfg = cfg.stage1.camera_gen
        center = trajectory_center_for_keyframes(chunk_scene.params.means, cam_cfg, ignore_config_center=True)
        view_indices = select_keyframes(
            chunk_scene.camera_trajectory,
            int(args.num_views),
            cam_cfg,
            center=center,
        )
        item["rendered_views"] = []
        for out_idx, view_idx in enumerate(view_indices):
            view = chunk_scene.render(chunk_scene.camera_trajectory[view_idx])
            image_path = chunk_dir / f"view_{out_idx:02d}.png"
            save_image(view.rgb, image_path)
            item["rendered_views"].append(str(image_path))
            if out_idx == 0:
                first_view_paths.append(image_path)
                if chunk.chunk_id in selected_ids:
                    selected_first_view_paths.append(image_path)
        if args.save_topdown:
            topdown_width = int(args.topdown_width or getattr(cfg.stage1.camera_gen, "image_width", 512))
            topdown_height = int(args.topdown_height or getattr(cfg.stage1.camera_gen, "image_height", 512))
            topdown_fov = float(args.topdown_fov_deg or getattr(cfg.stage1.camera_gen, "fov_deg", 90.0))
            topdown_path = _save_chunk_topdown(
                chunk_scene,
                up_axis,
                chunk_dir / "topdown.png",
                image_width=topdown_width,
                image_height=topdown_height,
                fov_deg=topdown_fov,
            )
            item["topdown_view"] = str(topdown_path)
            topdown_paths.append(topdown_path)
            if chunk.chunk_id in selected_ids:
                selected_topdown_paths.append(topdown_path)
        log.info(
            "Rendered %s: %d Gaussians, %d core",
            chunk.chunk_id,
            len(chunk.indices),
            int(chunk.core_mask.sum().item()),
        )

    with open(output_dir / "chunk_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(output_dir / "chunk_guidance.json", "w", encoding="utf-8") as f:
        json.dump(guidance_summary, f, indent=2)
    _save_contact_sheet(first_view_paths, output_dir / "contact_sheet.png", columns=4)
    _save_contact_sheet(topdown_paths, output_dir / "topdown_contact_sheet.png", columns=4)
    if selected_first_view_paths:
        _save_selected_contact_sheet(selected_first_view_paths, output_dir / "selected_contact_sheet.png", columns=4)
    if selected_topdown_paths:
        _save_selected_contact_sheet(selected_topdown_paths, output_dir / "selected_topdown_contact_sheet.png", columns=4)
    if args.export_selected_ply:
        if not selected_ids:
            raise SystemExit("No selected chunks to export. Check prompt/ROI guidance or lower stage1.chunking.min_roi_overlap.")
        selected_records = _write_selected_chunk_plys(
            scene,
            chunks,
            manifest_by_id,
            selected_ids,
            output_dir,
            ply_name=str(args.selected_ply_name),
        )
        log.info("Exported %d selected chunk PLY files to %s", len(selected_records), output_dir / "selected_chunks")
    log.info("Saved chunk preview to %s", output_dir)


if __name__ == "__main__":
    main()
