from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.stage2_object_refinement.align import AlignmentResult, compute_alignment_transform, write_alignment_metadata
from worldact.stage2_object_refinement.backend import FeedForwardBackend
from worldact.stage2_object_refinement.common import BackendRunResult, ObjectRefinementConfig
from worldact.stage2_object_refinement.generate import generate_multiview_images
from worldact.stage2_object_refinement.inpaint import render_multiview_inpaint
from worldact.stage2_object_refinement.reconstruct import reconstruct_3dgs

log = logging.getLogger(__name__)


class SV3DBackend(FeedForwardBackend):
    def __init__(
        self,
        *,
        generation_model: str = "sv3d",
        reconstruction_model: str = "gs_lrm",
        inpaint_model: str = "lama",
        num_generation_views: int = 6,
        generation_resolution: int = 512,
        icp_iterations: int = 50,
        device: str = "cuda",
    ) -> None:
        self.generation_model = generation_model
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

        source_scene = self._load_source_scene(work_dir)
        obj_mask = self._load_obj_mask(work_dir)
        cfg = self._make_config(request)

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
            best_view_idx = request.get("best_view_index", 0)
            if inpaint_result and inpaint_result.views:
                crop_rgb_path = Path(inpaint_result.views[0].get("inpaint_path",
                                    inpaint_result.views[0].get("rgb_path", "")))
                crop_mask_path = Path(inpaint_result.views[0].get("mask_path", ""))
            else:
                crop_rgb_path = work_dir / "crop.png"
                crop_mask_path = work_dir / "mask.png"

            if not crop_rgb_path.exists():
                fallback_rgb = work_dir / "crop.png"
                if fallback_rgb.exists():
                    crop_rgb_path = fallback_rgb

            elevation_deg = self._estimate_elevation(seed_scene, best_view_idx)

            log.info("SV3DBackend: Step 2 - Multi-view generation (%s)", self.generation_model)
            gen_result = generate_multiview_images(
                crop_rgb_path=crop_rgb_path,
                crop_mask_path=crop_mask_path,
                work_dir=work_dir,
                cfg=cfg,
                num_views=self.num_generation_views,
                elevation_deg=elevation_deg,
                model=self.generation_model,
                resolution=self.generation_resolution,
                device=self.device,
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

        try:
            if recon_result.ply_path is not None and recon_result.ply_path.exists():
                generated_scene = GaussianScene.from_ply(recon_result.ply_path)
                generated_scene.params = generated_scene.params.to(self.device)
                generated_scene.camera_trajectory = [cam.to(self.device) for cam in generated_scene.camera_trajectory]
            elif recon_result.stats.get("skipped") or recon_result.stats.get("fallback"):
                log.warning("SV3DBackend: No real reconstruction available (model not installed?), skipping generation")
                generated_scene = None
            else:
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

    def _load_source_scene(self, work_dir: Path) -> GaussianScene | None:
        for candidate in [work_dir.parent.parent / "gaussians.ply", work_dir / "source.ply"]:
            if candidate.exists():
                try:
                    scene = GaussianScene.from_ply(candidate)
                    scene.params = scene.params.to(self.device)
                    scene.camera_trajectory = [cam.to(self.device) for cam in scene.camera_trajectory]
                    return scene
                except Exception:
                    continue
        return None

    def _load_obj_mask(self, work_dir: Path) -> Any:
        masks_path = work_dir.parent.parent / "masks.pt"
        if masks_path.exists():
            try:
                masks = torch.load(masks_path, map_location="cpu", weights_only=False)
                if hasattr(masks, "object_masks") and masks.object_masks:
                    return masks.object_masks[0]
            except Exception:
                pass
        return None

    def _make_config(self, request: dict[str, Any]) -> ObjectRefinementConfig:
        return ObjectRefinementConfig()

    def _make_record(self, request: dict[str, Any]):
        from worldact.stage2_object_refinement.common import StageObjectRecord
        return StageObjectRecord(
            object_id=str(request.get("object_id", "0")),
            name=str(request.get("name", "object")),
            source_gaussian_indices=torch.tensor([], dtype=torch.long),
            best_view_index=int(request.get("best_view_index", -1) or -1),
        )

    def _estimate_elevation(self, scene: GaussianScene, best_view_index: int) -> float:
        from worldact.stage2_object_refinement.generate import _estimate_elevation_from_scene
        return _estimate_elevation_from_scene(scene, best_view_index)