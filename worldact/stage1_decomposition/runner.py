from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.scene import DecomposedScene
from worldact.rendering.camera_planning import (
    generate_camera_trajectory,
    select_keyframes,
    trajectory_center_for_keyframes,
)
from worldact.stage1_decomposition.chunking import (
    build_xy_chunks,
    chunk_manifest,
    filter_chunks_by_roi,
    make_chunk_scene,
    merge_chunk_objects,
)
from worldact.stage1_decomposition.agent import MultimodalAgent
from worldact.stage1_decomposition.binarize import (
    binarize_and_partition,
    cleanup_object_components,
    deduplicate_objects_3d,
)
from worldact.stage1_decomposition.roi_guidance import localize_prompt_roi, save_roi_selection_artifacts
from worldact.stage1_decomposition.feature_field_assignment import FeatureFieldAssignment
from worldact.stage1_decomposition.prefilter import GaussianMaskPrefilter
from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
from worldact.stage1_decomposition.soft_assignment import SoftAssignment
from worldact.utils.logging import get_logger

log = get_logger(__name__)


def _background_from_objects(scene: GaussianScene, objects: list[Any]) -> GaussianScene:
    used = torch.zeros(scene.num_gaussians, dtype=torch.bool, device=scene.device)
    for obj in objects:
        indices = getattr(obj, "source_gaussian_indices", None)
        if indices is not None and len(indices) > 0:
            used[indices.to(scene.device)] = True
    return scene.subset((~used).nonzero(as_tuple=True)[0])


class DecompositionRunner:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def run(self, scene: GaussianScene) -> DecomposedScene:
        if bool(getattr(self.cfg.stage1.chunking, "enabled", False)):
            return self._run_chunked(scene)
        return self._run_single(scene)

    def _run_single(
        self,
        scene: GaussianScene,
        agent: MultimodalAgent | None = None,
        *,
        ignore_config_center: bool = False,
    ) -> DecomposedScene:
        self._ensure_cameras(scene, ignore_config_center=ignore_config_center)
        cam_cfg = self.cfg.stage1.camera_gen
        traj_center = trajectory_center_for_keyframes(
            scene.params.means,
            cam_cfg,
            ignore_config_center=ignore_config_center,
        )
        keyframe_indices = select_keyframes(
            scene.camera_trajectory,
            int(self.cfg.stage1.vlm.num_keyframes),
            cam_cfg,
            center=traj_center,
        )
        keyframe_views = [scene.render(scene.camera_trajectory[i]) for i in keyframe_indices]
        agent = agent or MultimodalAgent(self.cfg.stage1)
        manual = list(self.cfg.stage1.manual_prompts) if self.cfg.stage1.manual_prompts else None
        states = agent.discover_objects(keyframe_views, manual_prompts=manual)
        log.info("Discovered %d object prompts: %s", len(states), [s.text_prompt or s.name for s in states])
        if not states:
            empty = torch.zeros(scene.num_gaussians, 0, device=scene.device)
            return DecomposedScene(background=scene, objects=[], masks=None, assignment_scores=empty)
        masks, valid_states = agent.segment_objects(states, scene, keyframe_indices, keyframe_views=keyframe_views)
        if not valid_states:
            empty = torch.zeros(scene.num_gaussians, 0, device=scene.device)
            return DecomposedScene(background=scene, objects=[], masks=masks, assignment_scores=empty)
        prefilter_cfg = getattr(self.cfg.stage1, "prefilter", None)
        prefilter_result = GaussianMaskPrefilter(prefilter_cfg).apply(scene, masks) if prefilter_cfg is not None else None
        assign_scene = prefilter_result.scene if prefilter_result is not None else scene
        assignment_method = str(getattr(self.cfg.stage1.assignment, "method", "soft"))
        feature_state = None
        if assignment_method == "feature_field":
            opt = FeatureFieldAssignment(
                assign_scene.num_gaussians,
                len(valid_states),
                self.cfg.stage1.assignment.feature_field,
                self.cfg.stage1.soft_assignment.instance_separation,
            ).to(assign_scene.device)
        elif assignment_method == "soft":
            opt = SoftAssignment(assign_scene.num_gaussians, len(valid_states), self.cfg.stage1.soft_assignment).to(assign_scene.device)
        else:
            raise ValueError(f"Unknown assignment method: {assignment_method}")
        scores = opt.optimize(assign_scene, masks)
        if assignment_method == "feature_field" and bool(self.cfg.stage1.assignment.feature_field.save_feature_field):
            feature_state = opt.export_state()
        threshold = (
            float(self.cfg.stage1.assignment.feature_field.threshold)
            if assignment_method == "feature_field"
            else float(self.cfg.stage1.soft_assignment.threshold)
        )
        background, objects, assignment_scores = binarize_and_partition(
            assign_scene,
            scores,
            valid_states,
            threshold=threshold,
            cfg=self.cfg.stage1.soft_assignment,
        )
        num_objects_after_binarize = len(objects)
        cleanup_meta = None
        if prefilter_result is None:
            objects, cleanup_meta = cleanup_object_components(assign_scene, objects, getattr(self.cfg.stage1, "object_cleanup", None))
        if prefilter_result is not None:
            keep_indices = prefilter_result.keep_indices.to(scene.device)
            expanded_scores = torch.zeros(scene.num_gaussians, assignment_scores.shape[1], device=scene.device)
            expanded_scores[keep_indices] = assignment_scores
            assignment_scores = expanded_scores
            for obj in objects:
                if obj.source_gaussian_indices is not None:
                    obj.source_gaussian_indices = keep_indices[obj.source_gaussian_indices.to(scene.device)].detach().cpu()
            objects, cleanup_meta = cleanup_object_components(scene, objects, getattr(self.cfg.stage1, "object_cleanup", None))
        num_objects_after_cleanup = len(objects)
        dedup_meta = None
        objects, dedup_meta = deduplicate_objects_3d(scene, objects, getattr(self.cfg.stage1, "object_dedup", None))
        num_objects_after_dedup = len(objects)
        if (
            isinstance(dedup_meta, dict)
            and bool(dedup_meta.get("enabled", {}).get("value", False))
            and bool(dedup_meta.get("enabled", {}).get("suppress", False))
        ):
            background = _background_from_objects(scene, objects)
        metadata = dict(getattr(masks, "metadata", {}))
        metadata["assignment_method"] = assignment_method
        if prefilter_result is not None:
            metadata["prefilter"] = prefilter_result.stats.to_dict()
            metadata["prefilter_keep_indices"] = prefilter_result.keep_indices.detach().cpu()
        if feature_state is not None:
            metadata["feature_assignment"] = feature_state
        if cleanup_meta is not None:
            metadata["object_cleanup"] = cleanup_meta
        if dedup_meta is not None:
            metadata["object_dedup"] = dedup_meta
        candidate_support_meta = getattr(masks, "metadata", {}).get("candidate_support", {}) if masks is not None else {}
        track_identity_meta = getattr(masks, "metadata", {}).get("track_identity", {}) if masks is not None else {}
        metadata["mapping_diagnostics"] = {
            "num_discovered_prompts": len(states),
            "num_valid_objects": len(valid_states),
            "num_mask_objects": len(masks.object_masks) if masks is not None else 0,
            "num_objects_after_binarize": num_objects_after_binarize,
            "num_objects_after_cleanup": num_objects_after_cleanup,
            "num_objects_after_dedup": num_objects_after_dedup,
            "num_exported_objects": len(objects),
            "assignment_threshold": threshold,
            "assignment_method": assignment_method,
            "prefilter_enabled": prefilter_result is not None and bool(getattr(getattr(self.cfg.stage1, "prefilter", None), "enabled", False)),
            "prefilter_keep_ratio": float(prefilter_result.stats.keep_ratio) if prefilter_result is not None else 1.0,
            "object_cleanup_enabled": cleanup_meta is not None and bool(cleanup_meta.get("enabled", {}).get("value", False))
            if isinstance(cleanup_meta, dict)
            else False,
            "object_dedup_enabled": dedup_meta is not None and bool(dedup_meta.get("enabled", {}).get("value", False))
            if isinstance(dedup_meta, dict)
            else False,
            "object_dedup_num_duplicates": len(dedup_meta.get("duplicates", [])) if isinstance(dedup_meta, dict) else 0,
            "candidate_support_enabled": bool(candidate_support_meta),
            "candidate_support_num_objects": len(candidate_support_meta) if isinstance(candidate_support_meta, dict) else 0,
            "track_identity_num_conflicts": int(track_identity_meta.get("num_conflicts", 0)) if isinstance(track_identity_meta, dict) else 0,
        }
        return DecomposedScene(
            background=background,
            objects=objects,
            masks=masks,
            assignment_scores=assignment_scores,
            metadata=metadata,
        )

    def _run_chunked(self, scene: GaussianScene) -> DecomposedScene:
        chunk_cfg = self.cfg.stage1.chunking
        cam_cfg = self.cfg.stage1.camera_gen
        up_axis = str(getattr(chunk_cfg, "up_axis", None) or getattr(cam_cfg, "up_axis", "z"))
        chunks = build_xy_chunks(scene, chunk_cfg, up_axis=up_axis)
        log.info("Generated %d chunks for decomposition on %s plane", len(chunks), up_axis)
        if not chunks:
            log.warning("No valid chunks were generated; falling back to single-scene decomposition")
            return self._run_single(scene)
        candidates = []
        assignment_records: dict[str, dict] = {}
        shared_segmenter = SAM3Segmenter(self.cfg.stage1.sam3)
        manifest = chunk_manifest(chunks)
        guidance_summary: dict[str, object] = {
            "enabled": False,
            "prompt": getattr(chunk_cfg, "prompt", None),
            "up_axis": up_axis,
        }

        roi_center = None
        roi_radius = None
        prompt = str(getattr(chunk_cfg, "prompt", None) or "").strip() or None
        explicit_center = getattr(chunk_cfg, "roi_center", None)
        explicit_radius = getattr(chunk_cfg, "roi_radius", None)
        roi_margin = float(getattr(chunk_cfg, "roi_margin", 0.0))
        prompt_guided = False
        if prompt and (explicit_center is None or explicit_radius is None):
            try:
                selection = localize_prompt_roi(scene, shared_segmenter, prompt, up_axis=up_axis)
            except Exception as exc:
                log.warning("Prompt-guided ROI localization failed for %r: %s; processing full chunk grid", prompt, exc)
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
            else:
                roi_artifacts: dict[str, object] = {}
                output_dir = getattr(self.cfg, "output_dir", None)
                if output_dir is not None:
                    roi_artifacts = save_roi_selection_artifacts(selection, Path(str(output_dir)) / "stage1")
                roi_center = (
                    torch.tensor(list(explicit_center), dtype=torch.float32, device=scene.device)
                    if explicit_center is not None
                    else selection.center.to(scene.device)
                )
                roi_radius = float(explicit_radius) if explicit_radius is not None else float(selection.radius)
                prompt_guided = True
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
                        **roi_artifacts,
                    }
                )
        elif explicit_center is not None and explicit_radius is not None:
            roi_center = torch.tensor(list(explicit_center), dtype=torch.float32, device=scene.device)
            roi_radius = float(explicit_radius)
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
                }
            )

        if roi_center is not None and roi_radius is not None:
            selected_chunks, manifest, roi_summary = filter_chunks_by_roi(
                scene,
                chunks,
                roi_center,
                float(roi_radius) + roi_margin,
                up_axis=up_axis,
                min_roi_overlap=float(getattr(chunk_cfg, "min_roi_overlap", 0.15)),
                keep_top_k=int(getattr(chunk_cfg, "keep_top_k", 2)),
                prompt=prompt,
                prompt_guided=prompt_guided,
                roi_margin=roi_margin,
            )
            chunks = selected_chunks
            guidance_summary.update(roi_summary)
            log.info(
                "Prompt/ROI filter kept %d/%d chunks (fallback=%s)",
                len(chunks),
                len(manifest),
                bool(roi_summary.get("fallback_used", False)),
            )
        else:
            for item in manifest:
                item.update(
                    {
                        "roi_guided": bool(explicit_center is not None or explicit_radius is not None),
                        "prompt_guided": bool(prompt_guided),
                        "prompt": prompt,
                        "selected_by_prompt": False,
                        "selection_reason": "unfiltered",
                    }
                )

        manifest_by_id = {item["chunk_id"]: item for item in manifest}

        for chunk in chunks:
            manifest_item = manifest_by_id[chunk.chunk_id]
            log.info(
                "Processing %s: %d Gaussians (%d core)",
                chunk.chunk_id,
                len(chunk.indices),
                int(chunk.core_mask.sum().item()),
            )
            chunk_scene = make_chunk_scene(scene, chunk)
            chunk_scene.params = chunk_scene.params.to(scene.device)
            chunk_scene.camera_trajectory = []
            chunk_agent = MultimodalAgent(self.cfg.stage1, segmenter=shared_segmenter)
            chunk_result = self._run_single(chunk_scene, agent=chunk_agent, ignore_config_center=True)
            global_indices = chunk.indices.cpu()
            local_scores = chunk_result.assignment_scores.detach().cpu()
            manifest_item["num_objects"] = chunk_result.num_objects
            manifest_item["status"] = "ok"

            for obj in chunk_result.objects:
                if obj.source_gaussian_indices is None or len(obj.source_gaussian_indices) == 0:
                    continue
                local_idx = obj.source_gaussian_indices.long()
                global_core = chunk.core_mask[local_idx]
                if not bool(global_core.any()):
                    continue
                core_local_idx = local_idx[global_core]
                global_obj_idx = global_indices[core_local_idx]
                obj.source_gaussian_indices = global_obj_idx
                obj.metadata = {
                    **obj.metadata,
                    "chunk_ids": [chunk.chunk_id],
                    "core_fraction": float(global_core.float().mean().item()),
                    "num_local_gaussians": int(len(local_idx)),
                    "num_core_gaussians": int(len(core_local_idx)),
                }
                candidates.append(obj)
                if local_scores.numel() > 0:
                    obj_col = int(obj.object_id)
                    score_vals = (
                        local_scores[core_local_idx, obj_col]
                        if obj_col < local_scores.shape[1]
                        else torch.ones(len(core_local_idx))
                    )
                    assignment_records[f"{chunk.chunk_id}:{obj.object_id}"] = {
                        "name": obj.name,
                        "global_indices": global_obj_idx.cpu(),
                        "scores": score_vals.cpu(),
                        "core_mask": torch.ones(len(core_local_idx), dtype=torch.bool),
                    }

        objects = merge_chunk_objects(scene, candidates, self.cfg.stage1.chunking)
        used = torch.zeros(scene.num_gaussians, dtype=torch.bool, device=scene.device)
        for obj in objects:
            if obj.source_gaussian_indices is not None and len(obj.source_gaussian_indices) > 0:
                used[obj.source_gaussian_indices.to(scene.device)] = True
        bg_indices = (~used).nonzero(as_tuple=True)[0]
        background = scene.subset(bg_indices)
        empty_scores = torch.zeros(scene.num_gaussians, 0, device=scene.device)
        log.info("Chunked merge produced %d objects; background has %d Gaussians", len(objects), background.num_gaussians)
        return DecomposedScene(
            background=background,
            objects=objects,
            masks=None,
            assignment_scores=empty_scores,
            assignment_records=assignment_records,
            metadata={"chunk_manifest": manifest, "chunk_guidance": guidance_summary},
        )

    def save_checkpoint(self, result: DecomposedScene, output_dir: Path, source_scene: GaussianScene) -> None:
        result.save(output_dir / "stage1", source_scene=source_scene)

    def _ensure_cameras(self, scene: GaussianScene, *, ignore_config_center: bool = False) -> None:
        if scene.camera_trajectory:
            scene.camera_trajectory = [cam.to(scene.device) for cam in scene.camera_trajectory]
            return
        cfg = self.cfg.stage1.camera_gen
        scene.camera_trajectory = [
            cam.to(scene.device)
            for cam in generate_camera_trajectory(
                scene.params.means,
                cfg,
                ignore_config_center=ignore_config_center,
            )
        ]
        log.info("Generated %d cameras (mode=%s)", len(scene.camera_trajectory), getattr(cfg, "trajectory_mode", "single_ring"))
