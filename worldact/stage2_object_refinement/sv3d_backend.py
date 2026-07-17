from __future__ import annotations

import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import ObjectMask
from worldact.stage2_object_refinement.align import AlignmentResult, compute_alignment_transform, write_alignment_metadata
from worldact.stage2_object_refinement.backend import FeedForwardBackend
from worldact.stage2_object_refinement.common import BackendRunResult, ObjectRefinementConfig
from worldact.stage2_object_refinement.generate import (
    generate_multiview_images,
    generate_scene_multiview_pseudo,
    resolve_condition_view_index,
    resolve_sv3d_elevation,
    _is_blank_condition_image,
)
from worldact.stage2_object_refinement.inpaint import render_multiview_inpaint
from worldact.stage2_object_refinement.reconstruct import reconstruct_3dgs

log = logging.getLogger(__name__)


class SV3DBackend(FeedForwardBackend):
    def __init__(
        self,
        *,
        generation_model: str = "sv3d",
        generation_model_path: str | None = None,
        reconstruction_model: str = "gs_lrm",
        inpaint_model: str = "lama",
        num_generation_views: int = 6,
        generation_resolution: int = 512,
        icp_iterations: int = 50,
        device: str = "cuda",
    ) -> None:
        self.generation_model = generation_model
        self.generation_model_path = generation_model_path
        self.reconstruction_model = reconstruction_model
        self.inpaint_model = inpaint_model
        self.num_generation_views = num_generation_views
        self.generation_resolution = generation_resolution
        self.icp_iterations = icp_iterations
        self.device = device

    def run(self, request_path: Path, work_dir: Path) -> BackendRunResult:
        request = self._load_request(request_path)
        if request is None:
            return BackendRunResult(
                generated_ply=None,
                metadata_path=None,
                status="failed",
                message=f"Could not load request from {request_path}",
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            seed_scene = GaussianScene.from_ply(request["seed_ply"])
            seed_scene.params = seed_scene.params.to(self.device)
            seed_scene.camera_trajectory = [cam.to(self.device) for cam in seed_scene.camera_trajectory]
        except Exception as exc:
            return BackendRunResult(
                generated_ply=None,
                metadata_path=None,
                status="failed",
                message=f"Could not load seed PLY: {exc}",
            )

        source_scene = self._load_source_scene(request, seed_scene)
        obj_mask = self._load_obj_mask(request)
        cfg = self._make_config(request)

        inpaint_result = None
        if cfg.refinement_stage == "legacy":
            try:
                log.info("SV3DBackend: Step 1 - Multi-view inpaint rendering")
                inpaint_result = render_multiview_inpaint(
                    refined_scene=seed_scene,
                    source_scene=source_scene if source_scene is not None else seed_scene,
                    obj_mask=obj_mask,
                    record=self._make_record(request),
                    cfg=cfg,
                    work_dir=work_dir,
                    num_views=self.num_generation_views,
                    resolution=self.generation_resolution,
                    inpaint_model=self.inpaint_model,
                    device=self.device,
                )
            except Exception as exc:
                log.warning("SV3DBackend: Inpaint step failed, using fallback: %s", exc)
                inpaint_result = None

        try:
            condition_view_idx = resolve_condition_view_index(request)
            if cfg.refinement_stage != "legacy":
                crop = request.get("crop", {})
                crop_rgb_path = Path(crop.get("clean_input_png") or crop.get("crop_png") or work_dir / "crop.png")
                crop_mask_path = Path(crop.get("mask_png") or work_dir / "mask.png")
            elif inpaint_result and inpaint_result.views:
                crop_rgb_path = Path(inpaint_result.views[0].get("inpaint_path",
                                    inpaint_result.views[0].get("rgb_path", "")))
                crop_mask_path = Path(inpaint_result.views[0].get("mask_path", ""))
            else:
                crop_rgb_path = work_dir / "crop.png"
                crop_mask_path = work_dir / "mask.png"

            elevation_scene = source_scene if source_scene is not None else seed_scene
            estimated_elevation = self._estimate_elevation(elevation_scene, condition_view_idx)
            elevation_deg, sv3d_scene_multiview = resolve_sv3d_elevation(estimated_elevation, cfg)
            use_scene_multiview = (
                cfg.generation_model == "scene_multiview"
                or (self.generation_model == "sv3d" and sv3d_scene_multiview)
            )

            if use_scene_multiview:
                log.info(
                    "SV3DBackend: aerial/stage input (%.1f deg) -> scene_multiview pseudo export",
                    estimated_elevation,
                )
                gen_result = generate_scene_multiview_pseudo(
                    request,
                    work_dir,
                    cfg,
                    resolution=self.generation_resolution,
                )
                generation_model_used = "scene_multiview"
            else:
                if not crop_rgb_path.exists():
                    fallback_rgb = work_dir / "crop.png"
                    if fallback_rgb.exists():
                        crop_rgb_path = fallback_rgb
                if _is_blank_condition_image(crop_rgb_path):
                    log.warning(
                        "SV3DBackend: blank conditioning image %s; falling back to scene_multiview",
                        crop_rgb_path,
                    )
                    gen_result = generate_scene_multiview_pseudo(
                        request,
                        work_dir,
                        cfg,
                        resolution=self.generation_resolution,
                    )
                    generation_model_used = "scene_multiview"
                else:
                    log.info(
                        "SV3DBackend: condition view=%d estimated_elev=%.1f sv3d_elev=%.1f",
                        condition_view_idx,
                        estimated_elevation,
                        elevation_deg,
                    )
                    log.info("SV3DBackend: Step 2 - Multi-view generation (%s)", self.generation_model)
                    gen_result = generate_multiview_images(
                        crop_rgb_path=crop_rgb_path,
                        crop_mask_path=crop_mask_path,
                        work_dir=work_dir,
                        cfg=cfg,
                        num_views=self.num_generation_views,
                        elevation_deg=elevation_deg,
                        model=self.generation_model,
                        model_path=self.generation_model_path or cfg.generation_model_path,
                        resolution=self.generation_resolution,
                        device=self.device,
                    )
                    generation_model_used = self.generation_model

            generation_meta = {
                "condition_view_index": int(condition_view_idx),
                "estimated_elevation_deg": float(estimated_elevation),
                "elevation_deg": float(elevation_deg),
                "scene_multiview_fallback": bool(use_scene_multiview),
                "generation_model": generation_model_used,
                "num_views": int(gen_result.num_views),
                "crop_rgb_path": str(crop_rgb_path),
                "crop_mask_path": str(crop_mask_path),
            }
            with open(work_dir / "generation_meta.json", "w", encoding="utf-8") as f:
                json.dump(generation_meta, f, indent=2)
            _write_pseudo_view_metrics(gen_result.images, work_dir / "pseudo_view_metrics.json")
            if cfg.refinement_stage in {"generate_only", "local_refine"}:
                metadata_path = work_dir / "metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump({"transform": torch.eye(4).tolist(), "source": cfg.refinement_stage}, f, indent=2)
                return BackendRunResult(
                    generated_ply=None,
                    metadata_path=str(metadata_path),
                    status="generated",
                    message=f"Generated pseudo views for refinement_stage={cfg.refinement_stage}",
                )
        except Exception as exc:
            log.error("SV3DBackend: Generation step failed: %s", exc)
            return BackendRunResult(
                generated_ply=None,
                metadata_path=None,
                status="failed",
                message=f"Generation failed: {exc}",
            )

        try:
            log.info("SV3DBackend: Step 3 - 3DGS reconstruction (%s)", self.reconstruction_model)
            recon_result = reconstruct_3dgs(
                multiview_images=gen_result.images,
                cameras_json_path=gen_result.cameras_path,
                work_dir=work_dir,
                cfg=cfg,
                model=self.reconstruction_model,
                resolution=self.generation_resolution,
                device=self.device,
            )
        except Exception as exc:
            log.error("SV3DBackend: Reconstruction step failed: %s", exc)
            return BackendRunResult(
                generated_ply=None,
                metadata_path=None,
                status="failed",
                message=f"Reconstruction failed: {exc}",
            )

        generated_scene = None
        metadata_path = work_dir / "metadata.json"
        try:
            if recon_result.ply_path is not None and recon_result.ply_path.exists():
                try:
                    generated_scene = GaussianScene.from_ply(recon_result.ply_path)
                    generated_scene.params = generated_scene.params.to(self.device)
                    generated_scene.camera_trajectory = [cam.to(self.device) for cam in generated_scene.camera_trajectory]
                except Exception as exc:
                    log.warning("SV3DBackend: Could not load generated PLY: %s", exc)
                    generated_scene = None
            else:
                log.warning(
                    "SV3DBackend: No reconstruction output (%s); skipping generation",
                    recon_result.stats.get("fallback", recon_result.stats.get("model", "unknown")),
                )
                generated_scene = None

            if generated_scene is not None:
                log.info("SV3DBackend: Step 4 - Coordinate alignment (ICP + Procrustes)")
                alignment = compute_alignment_transform(
                    seed_scene=seed_scene,
                    generated_scene=generated_scene,
                    obj_mask=obj_mask,
                    record=self._make_record(request),
                    source_scene=source_scene if source_scene is not None else seed_scene,
                    icp_iterations=self.icp_iterations,
                    device=self.device,
                )

                metadata_path = work_dir / "metadata.json"
                write_alignment_metadata(alignment, str(metadata_path))

                log.info(
                    "SV3DBackend: completed. Scale=%.4f, ICP_iter=%d",
                    alignment.scale,
                    alignment.stats.get("icp_iterations", 0),
                )
            else:
                metadata_path = work_dir / "metadata.json"
                metadata = {"transform": torch.eye(4).tolist(), "source": "no_generation"}
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
        except Exception as exc:
            log.warning("SV3DBackend: Alignment step failed, using identity transform: %s", exc)
            metadata_path = work_dir / "metadata.json"
            metadata = {"transform": torch.eye(4).tolist(), "source": "identity_fallback"}
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        generated_ply_path = work_dir / "generated.ply"
        if generated_scene is not None and recon_result.ply_path is not None and recon_result.ply_path.exists():
            if not generated_ply_path.exists():
                import shutil
                shutil.copy2(str(recon_result.ply_path), str(generated_ply_path))
            generated_ply_str = str(generated_ply_path)
        else:
            generated_ply_str = None

        return BackendRunResult(
            generated_ply=generated_ply_str,
            metadata_path=str(metadata_path) if metadata_path.exists() else None,
            status="ok" if generated_ply_str else "skipped",
            message=f"SV3D backend: model={self.generation_model}, recon={self.reconstruction_model}"
                    + (", generation_skipped" if generated_ply_str is None else ""),
        )

    def _load_request(self, request_path: Path) -> dict[str, Any] | None:
        if not request_path.exists():
            return None
        try:
            with open(request_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _load_source_scene(self, request: dict[str, Any], seed_scene: GaussianScene) -> GaussianScene | None:
        source_ply = request.get("source_ply")
        if not source_ply:
            return seed_scene
        candidate = Path(source_ply)
        if not candidate.exists():
            return seed_scene
        cameras_path = request.get("cameras_path") or None
        try:
            scene = GaussianScene.from_ply(candidate, cameras_path=cameras_path)
            scene.params = scene.params.to(self.device)
            scene.camera_trajectory = [cam.to(self.device) for cam in scene.camera_trajectory]
            if not scene.camera_trajectory:
                scene.camera_trajectory = list(seed_scene.camera_trajectory)
            return scene
        except Exception:
            return seed_scene

    def _load_obj_mask(self, request: dict[str, Any]) -> ObjectMask | None:
        masks: dict[int, torch.Tensor] = {}
        for view in request.get("multiview", {}).get("views", []):
            mask_path = view.get("mask_path")
            if mask_path:
                mask = self._load_mask_png(Path(mask_path))
                if mask is not None:
                    masks[int(view.get("view_index", len(masks)))] = mask
        crop = request.get("crop", {})
        if not masks and crop.get("mask_png"):
            mask = self._load_mask_png(Path(crop["mask_png"]))
            if mask is not None:
                masks[int(crop.get("view_index", 0))] = mask
        if not masks:
            return None
        return ObjectMask(
            object_id=str(request.get("object_id", "0")),
            object_name=str(request.get("name", "object")),
            masks=masks,
            metadata={"source": "object_request"},
        )

    def _load_mask_png(self, path: Path) -> torch.Tensor | None:
        if not path.exists():
            return None
        try:
            import PIL.Image
            import numpy as np

            img = PIL.Image.open(str(path)).convert("L")
            arr = np.array(img).astype("float32") / 255.0
            return torch.from_numpy(arr)
        except Exception:
            return None

    def _make_config(self, request: dict[str, Any]) -> ObjectRefinementConfig:
        raw = dict(request.get("config") or {})
        valid = {field.name for field in fields(ObjectRefinementConfig)}
        return ObjectRefinementConfig(**{key: value for key, value in raw.items() if key in valid})

    def _make_record(self, request: dict[str, Any]):
        from worldact.stage2_object_refinement.common import StageObjectRecord
        return StageObjectRecord(
            object_id=str(request.get("object_id", "0")),
            name=str(request.get("name", "object")),
            source_gaussian_indices=torch.tensor([], dtype=torch.long),
            best_view_index=int(request.get("best_view_index", -1) or -1),
            memory_state=dict(request.get("memory_state") or {}),
            metadata=dict(request.get("metadata") or {}),
            object_ply_path=str(request.get("object_ply") or request.get("seed_ply") or ""),
        )

    def _estimate_elevation(self, scene: GaussianScene, best_view_index: int) -> float:
        from worldact.stage2_object_refinement.generate import _estimate_elevation_from_scene
        return _estimate_elevation_from_scene(scene, best_view_index)


def _write_pseudo_view_metrics(image_paths: list[Path], output_path: Path) -> None:
    import numpy as np
    import PIL.Image

    metrics: dict[str, Any] = {"num_views": len(image_paths), "views": []}
    areas = []
    for path in image_paths:
        try:
            arr = np.asarray(PIL.Image.open(str(path)).convert("RGB")).astype("float32") / 255.0
        except Exception:
            continue
        foreground = (arr < 0.92).any(axis=-1)
        area = float(foreground.mean())
        areas.append(area)
        metrics["views"].append({
            "path": str(path),
            "foreground_area_ratio": area,
            "mean_rgb": [float(v) for v in arr.mean(axis=(0, 1)).tolist()],
        })
    if areas:
        metrics["mean_foreground_area_ratio"] = float(sum(areas) / len(areas))
        metrics["foreground_area_std"] = float(np.std(np.asarray(areas)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
