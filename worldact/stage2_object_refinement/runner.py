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
        object_scene = _object_scene(stage_input.scene, record)
        before = diagnose_object(stage_input.scene, record, obj_mask, cfg)
        prune = prune_object(object_scene, obj_mask, cfg, source_scene=stage_input.scene)
        refined_record = StageObjectRecord(
            object_id=record.object_id,
            name=record.name,
            source_gaussian_indices=torch.arange(prune.scene.num_gaussians, dtype=torch.long),
            best_view_index=record.best_view_index,
            best_view_score=record.best_view_score,
            metadata=record.metadata,
        )
        after_prune = diagnose_object(prune.scene, refined_record, obj_mask, cfg)
        work_dir = object_dir(output_dir, record)
        metrics = {
            "object_id": record.object_id,
            "name": record.name,
            "before": before.to_dict(),
            "prune": prune.stats,
            "after_prune": after_prune.to_dict(),
        }
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
        generated_ply = Path(backend_result.generated_ply) if backend_result.generated_ply else None
        metadata_path = Path(backend_result.metadata_path) if backend_result.metadata_path else None
        merge = merge_generated_object(
            prune.scene,
            generated_ply,
            obj_mask,
            cfg,
            metadata_path=metadata_path,
            best_view_index=record.best_view_index,
        )
        final_path = work_dir / f"object_{record.object_id}_{slugify_name(record.name)}_final.ply"
        merge.scene.to_ply(final_path)
        final_record = StageObjectRecord(
            object_id=record.object_id,
            name=record.name,
            source_gaussian_indices=torch.arange(merge.scene.num_gaussians, dtype=torch.long),
            best_view_index=record.best_view_index,
            best_view_score=record.best_view_score,
            metadata=record.metadata,
        )
        final_diag = diagnose_object(merge.scene, final_record, obj_mask, cfg)
        metrics.update(
            {
                "backend": backend_result.__dict__,
                "merge": merge.stats,
                "final": final_diag.to_dict(),
                "outputs": {
                    "work_dir": str(work_dir),
                    "request_json": str(request_path),
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


def _object_scene(source_scene: GaussianScene, record: StageObjectRecord) -> GaussianScene:
    indices = record.source_gaussian_indices.to(source_scene.device)
    return source_scene.subset(indices)
