from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.masks import MaskSet
from worldact.data.objects import ObjectAsset
from worldact.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class DecomposedScene:
    background: GaussianScene
    objects: list[ObjectAsset]
    masks: MaskSet | None
    assignment_scores: torch.Tensor
    assignment_records: dict | None = None
    metadata: dict | None = None

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    def save(self, output_dir: Path | str, source_scene: GaussianScene | None = None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.background.save(output_dir / "background")
        torch.save(self.assignment_scores.cpu(), output_dir / "assignment_scores.pt")
        if self.assignment_records is not None:
            torch.save(self.assignment_records, output_dir / "assignment_records.pt")
        if self.metadata and "chunk_manifest" in self.metadata:
            with open(output_dir / "chunk_manifest.json", "w", encoding="utf-8") as f:
                json.dump(self.metadata["chunk_manifest"], f, indent=2)
        if self.metadata and "chunk_guidance" in self.metadata:
            with open(output_dir / "chunk_guidance.json", "w", encoding="utf-8") as f:
                json.dump(self.metadata["chunk_guidance"], f, indent=2)
        if self.metadata and "prefilter" in self.metadata:
            with open(output_dir / "prefilter_meta.json", "w", encoding="utf-8") as f:
                json.dump(self.metadata["prefilter"], f, indent=2)
            if "prefilter_keep_indices" in self.metadata:
                torch.save(self.metadata["prefilter_keep_indices"], output_dir / "prefilter_keep_indices.pt")
        if self.metadata and "object_cleanup" in self.metadata:
            with open(output_dir / "object_cleanup_meta.json", "w", encoding="utf-8") as f:
                json.dump(self.metadata["object_cleanup"], f, indent=2)
        if self.metadata and "object_dedup" in self.metadata:
            with open(output_dir / "object_dedup_meta.json", "w", encoding="utf-8") as f:
                json.dump(self.metadata["object_dedup"], f, indent=2)
        if self.masks is not None and self.masks.metadata:
            if "candidate_support" in self.masks.metadata:
                with open(output_dir / "candidate_support_meta.json", "w", encoding="utf-8") as f:
                    json.dump(self.masks.metadata["candidate_support"], f, indent=2)
            if "track_identity" in self.masks.metadata:
                with open(output_dir / "track_identity_meta.json", "w", encoding="utf-8") as f:
                    json.dump(self.masks.metadata["track_identity"], f, indent=2)
            if "mask_weights" in self.masks.metadata:
                with open(output_dir / "mask_reliability_meta.json", "w", encoding="utf-8") as f:
                    json.dump(self.masks.metadata["mask_weights"], f, indent=2)
        if self.metadata and "mapping_diagnostics" in self.metadata:
            with open(output_dir / "mapping_diagnostics.json", "w", encoding="utf-8") as f:
                json.dump(self.metadata["mapping_diagnostics"], f, indent=2)
        if self.metadata and "feature_assignment" in self.metadata:
            feature_assignment = self.metadata["feature_assignment"]
            if "features" in feature_assignment:
                torch.save(feature_assignment["features"], output_dir / "feature_field.pt")
            if "object_queries" in feature_assignment:
                torch.save(feature_assignment["object_queries"], output_dir / "object_queries.pt")
            feature_meta = {
                key: value
                for key, value in feature_assignment.items()
                if key not in {"features", "object_queries"}
            }
            feature_meta["assignment_method"] = self.metadata.get("assignment_method", "feature_field")
            with open(output_dir / "feature_assignment_meta.json", "w", encoding="utf-8") as f:
                json.dump(feature_meta, f, indent=2)
        if self.metadata and "view_propagation" in self.metadata:
            view_prop = self.metadata["view_propagation"]
            with open(output_dir / "view_propagation_meta.json", "w", encoding="utf-8") as f:
                json.dump(view_prop, f, indent=2)
            if isinstance(view_prop, dict):
                if "view_graph" in view_prop:
                    with open(output_dir / "view_graph.json", "w", encoding="utf-8") as f:
                        json.dump(view_prop["view_graph"], f, indent=2)
                if "tracks" in view_prop:
                    with open(output_dir / "tracks.json", "w", encoding="utf-8") as f:
                        json.dump(view_prop["tracks"], f, indent=2)
                if "stats" in view_prop:
                    with open(output_dir / "propagation_stats.json", "w", encoding="utf-8") as f:
                        json.dump(view_prop["stats"], f, indent=2)
                keyframe_indices = view_prop.get("keyframe_indices", [])
            else:
                keyframe_indices = []
            raw_masks = self.metadata.get("view_propagation_raw_masks")
            if raw_masks is None and self.masks is not None:
                raw_masks = self.masks.metadata.get("view_propagation_raw_masks")
            if raw_masks is not None:
                torch.save(raw_masks, output_dir / "view_propagation_raw_masks.pt")
            if source_scene is not None and raw_masks is not None and bool(view_prop.get("stats", {}).get("save_debug", True)):
                from worldact.stage1_decomposition.view_graph_propagation import save_view_propagation_artifacts

                refined_masks = self.masks if self.masks is not None else raw_masks
                if keyframe_indices:
                    log.info("Saving view propagation artifacts to %s", output_dir / "view_propagation")
                    save_view_propagation_artifacts(
                        source_scene,
                        raw_masks,
                        refined_masks,
                        output_dir / "view_propagation",
                        keyframe_indices=list(keyframe_indices),
                    )
        if self.masks is not None:
            torch.save(self.masks, output_dir / "masks.pt")

        for old_object_ply in output_dir.glob("object_*.ply"):
            old_object_ply.unlink()
        records = []
        for obj in self.objects:
            indices = obj.source_gaussian_indices
            records.append(
                {
                    "object_id": obj.object_id,
                    "name": obj.name,
                    "source_gaussian_indices": indices.cpu().tolist() if indices is not None else [],
                    "memory_state": obj.memory_state.to_json() if obj.memory_state else None,
                    "best_view_index": obj.best_view_index,
                    "best_view_score": obj.best_view_score,
                    "metadata": obj.metadata,
                }
            )
            if source_scene is not None and indices is not None and len(indices) > 0:
                safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in obj.name)
                source_scene.subset(indices).to_ply(output_dir / f"object_{obj.object_id}_{safe_name}.ply")
        with open(output_dir / "objects.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
