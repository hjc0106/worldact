from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.stage2_object_refinement.backend import FeedForwardBackend
from worldact.stage2_object_refinement.common import ObjectRefinementConfig, StageObjectRecord, slugify_name
from worldact.stage2_object_refinement.diagnose import diagnose_object
from worldact.stage2_object_refinement.io import (
    StageRefinementInput,
    find_object_mask,
    object_dir,
    write_json,
    write_object_package,
)
from worldact.stage2_object_refinement.merge import merge_generated_object
from worldact.stage2_object_refinement.paint import refine_local_object
from worldact.stage2_object_refinement.prune import prune_object


@dataclass
class ObjectRefinementRunResult:
    summary: dict[str, Any]
    object_metrics: list[dict[str, Any]]


def refine_stage_objects(
    stage_input: StageRefinementInput,
    output_dir: Path,
    cfg: ObjectRefinementConfig,
    backend: FeedForwardBackend,
    *,
    only_object_id: str | None = None,
) -> ObjectRefinementRunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    object_metrics: list[dict[str, Any]] = []
    for obj_idx, record in enumerate(stage_input.objects):
        if only_object_id is not None and str(record.object_id) != str(only_object_id):
            continue
        obj_mask = find_object_mask(stage_input.masks, record, obj_idx)
        object_scene = _object_scene(stage_input, record)
        local_record = _local_record(record, object_scene)
        work_dir = object_dir(output_dir, record)
        before = diagnose_object(object_scene, local_record, obj_mask, cfg)
        prune = prune_object(object_scene, obj_mask, cfg, source_scene=stage_input.scene)
        _write_separation_artifacts(object_scene, prune, work_dir)
        refined_record = _local_record(record, prune.scene)
        after_prune = diagnose_object(prune.scene, refined_record, obj_mask, cfg)
        metrics = {
            "object_id": record.object_id,
            "name": record.name,
            "before": before.to_dict(),
            "prune": prune.stats,
            "after_prune": after_prune.to_dict(),
        }
        request_path = None
        backend_result = None
        local_refine_stats: dict[str, Any] | None = None
        if cfg.refinement_stage != "cleanup_only":
            request_path = write_object_package(
                source_scene=stage_input.scene,
                refined_scene=prune.scene,
                record=record,
                obj_mask=obj_mask,
                cfg=cfg,
                work_dir=work_dir,
                metrics=metrics,
            )
            backend_result = backend.run(request_path, work_dir)

        if cfg.refinement_stage == "local_refine":
            local_refine = refine_local_object(prune.scene, work_dir, cfg, obj_mask=obj_mask)
            merge_scene = local_refine.scene
            local_refine_stats = local_refine.stats
            merge_stats = {
                "mode": "local_refine",
                "refined_gaussians": prune.scene.num_gaussians,
                "total_gaussians": merge_scene.num_gaussians,
            }
            merge_stats.update(local_refine_stats)
        elif cfg.refinement_stage in {"cleanup_only", "generate_only"}:
            merge_scene = prune.scene
            merge_stats = {
                "mode": f"{cfg.refinement_stage}_refined_only",
                "refined_gaussians": prune.scene.num_gaussians,
                "generated_gaussians": 0,
                "kept_generated_gaussians": 0,
            }
        else:
            generated_ply = Path(backend_result.generated_ply) if backend_result and backend_result.generated_ply else None
            metadata_path = Path(backend_result.metadata_path) if backend_result and backend_result.metadata_path else None
            merge = merge_generated_object(
                prune.scene,
                generated_ply,
                obj_mask,
                cfg,
                metadata_path=metadata_path,
                best_view_index=record.best_view_index,
            )
            merge_scene = merge.scene
            merge_stats = merge.stats
        final_path = work_dir / f"object_{record.object_id}_{slugify_name(record.name)}_final.ply"
        merge_scene.to_ply(final_path)
        final_record = StageObjectRecord(
            object_id=record.object_id,
            name=record.name,
            source_gaussian_indices=torch.arange(merge_scene.num_gaussians, dtype=torch.long),
            best_view_index=record.best_view_index,
            best_view_score=record.best_view_score,
            metadata=record.metadata,
            memory_state=record.memory_state,
            object_ply_path=record.object_ply_path,
        )
        final_diag = diagnose_object(merge_scene, final_record, obj_mask, cfg)
        metrics.update(
            {
                "backend": backend_result.__dict__ if backend_result is not None else {
                    "generated_ply": None,
                    "metadata_path": None,
                    "status": "skipped",
                    "message": f"Skipped by refinement_stage={cfg.refinement_stage}",
                },
                "merge": merge_stats,
                "local_refine": local_refine_stats,
                "final": final_diag.to_dict(),
                "outputs": {
                    "work_dir": str(work_dir),
                    "request_json": str(request_path) if request_path is not None else None,
                    "final_ply": str(final_path),
                },
            }
        )
        write_json(metrics, work_dir / "metrics.json")
        object_metrics.append(metrics)

    summary = {
        "num_input_objects": len(stage_input.objects),
        "num_processed_objects": len(object_metrics),
        "output_dir": str(output_dir),
    }
    write_json({"summary": summary, "objects": object_metrics}, output_dir / "object_metrics.json")
    return ObjectRefinementRunResult(summary=summary, object_metrics=object_metrics)


def _object_scene(stage_input: StageRefinementInput, record: StageObjectRecord) -> GaussianScene:
    object_scene = stage_input.object_scenes.get(str(record.object_id))
    if object_scene is not None:
        if not object_scene.camera_trajectory:
            object_scene.camera_trajectory = list(stage_input.scene.camera_trajectory)
        return object_scene
    source_scene = stage_input.scene
    indices = record.source_gaussian_indices.to(source_scene.device)
    return source_scene.subset(indices)


def _local_record(record: StageObjectRecord, scene: GaussianScene) -> StageObjectRecord:
    return StageObjectRecord(
        object_id=record.object_id,
        name=record.name,
        source_gaussian_indices=torch.arange(scene.num_gaussians, dtype=torch.long, device=scene.device),
        best_view_index=record.best_view_index,
        best_view_score=record.best_view_score,
        metadata=record.metadata,
        memory_state=record.memory_state,
        object_ply_path=record.object_ply_path,
    )


def _write_separation_artifacts(object_scene: GaussianScene, prune, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    prune.scene.to_ply(work_dir / "object_refined_seed.ply")
    if prune.object_core_indices is not None:
        object_scene.subset(prune.object_core_indices).to_ply(work_dir / "object_core.ply")
    if prune.uncertain_shell_indices is not None:
        object_scene.subset(prune.uncertain_shell_indices).to_ply(work_dir / "uncertain_shell.ply")
    if prune.background_reject_indices is not None:
        object_scene.subset(prune.background_reject_indices).to_ply(work_dir / "background_reject.ply")
    write_json(prune.stats.get("separation", {}), work_dir / "separation_metrics.json")
