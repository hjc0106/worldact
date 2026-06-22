from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from worldact.data.gaussians import GaussianScene, RenderedView
from worldact.data.masks import MaskSet, ObjectMask
from worldact.data.poses import CameraPose
from worldact.stage1_decomposition.sam3_segmenter import SAM3Candidate, SAM3CandidateSet
from worldact.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PromptMaskCandidates:
    object_id: str
    object_name: str
    prompt: str
    view_candidates: dict[int, SAM3CandidateSet] = field(default_factory=dict)
    memory_state: Any | None = None


@dataclass
class ViewGraphEdge:
    source_view: int
    target_view: int
    weight: float
    pose_distance: float
    angle_deg: float
    estimated_overlap: float

    def to_json(self) -> dict[str, float | int]:
        return {
            "source_view": self.source_view,
            "target_view": self.target_view,
            "weight": self.weight,
            "pose_distance": self.pose_distance,
            "angle_deg": self.angle_deg,
            "estimated_overlap": self.estimated_overlap,
        }


@dataclass
class PropagationDecision:
    source_view: int
    target_view: int
    candidate_id: int | None
    mode: str
    association_score: float
    reprojection_iou: float
    visibility_score: float
    depth_consistency: float
    local_sam_score: float
    edge_weight: float
    mask: torch.Tensor | None = None
    fusion_mode: str | None = None
    pre_fusion_area: float = 0.0
    post_fusion_area: float = 0.0
    rejected_reason: str | None = None

    def to_json(self) -> dict[str, float | int | str | None]:
        return {
            "source_view": self.source_view,
            "target_view": self.target_view,
            "candidate_id": self.candidate_id,
            "mode": self.mode,
            "association_score": self.association_score,
            "reprojection_iou": self.reprojection_iou,
            "visibility_score": self.visibility_score,
            "depth_consistency": self.depth_consistency,
            "local_sam_score": self.local_sam_score,
            "edge_weight": self.edge_weight,
            "fusion_mode": self.fusion_mode,
            "pre_fusion_area": self.pre_fusion_area,
            "post_fusion_area": self.post_fusion_area,
            "rejected_reason": self.rejected_reason,
        }


@dataclass
class MaskTrack:
    track_id: str
    object_id: str
    object_name: str
    prompt: str
    source_view: int
    source_candidate_id: int
    masks_by_view: dict[int, torch.Tensor] = field(default_factory=dict)
    scores_by_view: dict[int, float] = field(default_factory=dict)
    visibility_by_view: dict[int, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_object_mask(self) -> ObjectMask:
        return ObjectMask(
            object_id=self.object_id,
            object_name=self.object_name,
            masks={view_idx: mask.detach().cpu() for view_idx, mask in sorted(self.masks_by_view.items())},
            track_id=self.track_id,
            metadata=dict(self.metadata),
        )

    def summary(self) -> dict[str, Any]:
        scores = list(self.scores_by_view.values())
        visibilities = list(self.visibility_by_view.values())
        return {
            "track_id": self.track_id,
            "object_id": self.object_id,
            "object_name": self.object_name,
            "prompt": self.prompt,
            "source_view": self.source_view,
            "source_candidate_id": self.source_candidate_id,
            "coverage_views": len(self.masks_by_view),
            "mean_score": float(sum(scores) / max(1, len(scores))),
            "mean_visibility": float(sum(visibilities) / max(1, len(visibilities))),
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class ViewGraphPropagationResult:
    raw_mask_set: MaskSet
    refined_mask_set: MaskSet
    tracks: list[MaskTrack]
    view_graph: list[ViewGraphEdge]
    stats: dict[str, Any]
    keyframe_indices: list[int]

    def to_summary(self) -> dict[str, Any]:
        return {
            "keyframe_indices": list(self.keyframe_indices),
            "view_graph": [edge.to_json() for edge in self.view_graph],
            "tracks": [track.summary() for track in self.tracks],
            "stats": _json_safe(self.stats),
        }


def build_view_graph(scene: GaussianScene, view_indices: list[int], cfg: Any) -> dict[int, list[ViewGraphEdge]]:
    if len(view_indices) <= 1:
        return {view_idx: [] for view_idx in view_indices}
    cameras = {view_idx: scene.camera_trajectory[view_idx] for view_idx in view_indices}
    centers = {view_idx: _camera_center(cameras[view_idx]) for view_idx in view_indices}
    forwards = {view_idx: _camera_forward(cameras[view_idx]) for view_idx in view_indices}

    max_pose_distance = float(getattr(cfg, "max_pose_distance", 0.0))
    max_angle_deg = float(getattr(cfg, "max_angle_deg", 180.0))
    min_view_overlap = float(getattr(cfg, "min_view_overlap", 0.0))
    neighbor_top_k = max(1, int(getattr(cfg, "neighbor_top_k", 4)))

    edges: dict[int, list[ViewGraphEdge]] = {view_idx: [] for view_idx in view_indices}
    for source in view_indices:
        candidates: list[ViewGraphEdge] = []
        for target in view_indices:
            if target == source:
                continue
            pose_distance = float(torch.norm(centers[source] - centers[target]).item())
            angle_deg = _angle_deg(forwards[source], forwards[target])
            if max_pose_distance > 0.0 and pose_distance > max_pose_distance:
                continue
            if max_angle_deg > 0.0 and angle_deg > max_angle_deg:
                continue
            overlap = _estimate_view_overlap(pose_distance, angle_deg, max_pose_distance, max_angle_deg, source, target)
            if overlap < min_view_overlap:
                continue
            weight = overlap
            candidates.append(
                ViewGraphEdge(
                    source_view=source,
                    target_view=target,
                    weight=weight,
                    pose_distance=pose_distance,
                    angle_deg=angle_deg,
                    estimated_overlap=overlap,
                )
            )
        candidates.sort(key=lambda e: e.weight, reverse=True)
        edges[source] = candidates[:neighbor_top_k]
    return edges


def propagate_candidates(
    scene: GaussianScene,
    keyframe_indices: list[int],
    keyframe_views: list[RenderedView],
    prompt_candidates: list[PromptMaskCandidates],
    cfg: Any,
    *,
    raw_mask_set: MaskSet | None = None,
) -> ViewGraphPropagationResult:
    view_graph = build_view_graph(scene, keyframe_indices, cfg)
    view_map = {view_idx: view for view_idx, view in zip(keyframe_indices, keyframe_views)}

    raw_mask_set = raw_mask_set or _build_raw_mask_set(prompt_candidates, cfg)
    if not bool(getattr(cfg, "enabled", False)):
        result = ViewGraphPropagationResult(
            raw_mask_set=raw_mask_set,
            refined_mask_set=raw_mask_set,
            tracks=[_mask_track_from_object_mask(obj) for obj in raw_mask_set.object_masks],
            view_graph=_flatten_graph(view_graph),
            stats={
                "enabled": False,
                "num_tracks": len(raw_mask_set.object_masks),
                "num_views": len(keyframe_indices),
                "num_edges": sum(len(edges) for edges in view_graph.values()),
            },
            keyframe_indices=list(keyframe_indices),
        )
        result.refined_mask_set.metadata["view_propagation"] = result.to_summary()
        return result

    num_edges = sum(len(edges) for edges in view_graph.values())
    if num_edges == 0 and bool(getattr(cfg, "fallback_to_raw_on_empty_graph", True)):
        log.warning("View propagation graph has no valid edges; falling back to raw multi-view SAM3 masks")
        result = ViewGraphPropagationResult(
            raw_mask_set=raw_mask_set,
            refined_mask_set=raw_mask_set,
            tracks=[_mask_track_from_object_mask(obj) for obj in raw_mask_set.object_masks],
            view_graph=[],
            stats={
                "enabled": True,
                "fallback_reason": "empty_view_graph",
                "used_raw_masks": True,
                "num_tracks": len(raw_mask_set.object_masks),
                "num_views": len(keyframe_indices),
                "num_edges": 0,
                "candidate_unique_assignment": bool(getattr(cfg, "candidate_unique_assignment", True)),
                "fusion_mode": str(getattr(cfg, "fusion_mode", "intersection")),
                "fallback_to_sam3": bool(getattr(cfg, "fallback_to_sam3", False)),
                "save_debug": bool(getattr(cfg, "save_debug", False)),
            },
            keyframe_indices=list(keyframe_indices),
        )
        result.refined_mask_set.metadata["view_propagation"] = result.to_summary()
        result.refined_mask_set.metadata["view_propagation_raw_masks"] = raw_mask_set
        return result

    tracks: list[MaskTrack] = []
    for prompt_idx, prompt_group in enumerate(prompt_candidates):
        track = _build_track(
            prompt_idx=prompt_idx,
            prompt_group=prompt_group,
            scene=scene,
            view_graph=view_graph,
            view_map=view_map,
            cfg=cfg,
        )
        if track is not None:
            tracks.append(track)

    if bool(getattr(cfg, "candidate_unique_assignment", True)):
        tracks = _enforce_unique_candidate_assignments(tracks)
    refined_tracks = _suppress_conflicts(tracks, cfg) if bool(getattr(cfg, "conflict_suppression", True)) else tracks

    if bool(getattr(cfg, "fallback_to_raw_on_no_association", True)):
        propagated_views = sum(
            1
            for track in refined_tracks
            for assoc in track.metadata.get("associations", {}).values()
            if assoc.get("mode") in {"fused", "propagated", "propagated_only"}
        )
        covered_tracks = sum(1 for track in refined_tracks if len(track.masks_by_view) > 0)
        if num_edges > 0 and propagated_views == 0 and covered_tracks < len(raw_mask_set.object_masks):
            log.warning(
                "View propagation produced no accepted associations and dropped %d/%d raw tracks; falling back to raw masks",
                len(raw_mask_set.object_masks) - covered_tracks,
                len(raw_mask_set.object_masks),
            )
            result = ViewGraphPropagationResult(
                raw_mask_set=raw_mask_set,
                refined_mask_set=raw_mask_set,
                tracks=[_mask_track_from_object_mask(obj) for obj in raw_mask_set.object_masks],
                view_graph=_flatten_graph(view_graph),
                stats={
                    "enabled": True,
                    "fallback_reason": "no_valid_association",
                    "used_raw_masks": True,
                    "num_tracks": len(raw_mask_set.object_masks),
                    "num_views": len(keyframe_indices),
                    "num_edges": num_edges,
                    "candidate_unique_assignment": bool(getattr(cfg, "candidate_unique_assignment", True)),
                    "fusion_mode": str(getattr(cfg, "fusion_mode", "intersection")),
                    "fallback_to_sam3": bool(getattr(cfg, "fallback_to_sam3", False)),
                    "save_debug": bool(getattr(cfg, "save_debug", False)),
                },
                keyframe_indices=list(keyframe_indices),
            )
            result.refined_mask_set.metadata["view_propagation"] = result.to_summary()
            result.refined_mask_set.metadata["view_propagation_raw_masks"] = raw_mask_set
            return result

    refined_mask_set = MaskSet(
        object_masks=[track.to_object_mask() for track in refined_tracks],
        metadata=dict(raw_mask_set.metadata),
    )
    refined_mask_set.fuse()

    stats = _compute_stats(refined_tracks, view_graph, keyframe_indices, cfg)
    result = ViewGraphPropagationResult(
        raw_mask_set=raw_mask_set,
        refined_mask_set=refined_mask_set,
        tracks=refined_tracks,
        view_graph=_flatten_graph(view_graph),
        stats=stats,
        keyframe_indices=list(keyframe_indices),
    )
    refined_mask_set.metadata["view_propagation"] = result.to_summary()
    refined_mask_set.metadata["view_propagation_raw_masks"] = raw_mask_set
    return result


def save_view_propagation_artifacts(
    scene: GaussianScene,
    raw_mask_set: MaskSet,
    refined_mask_set: MaskSet,
    output_dir: Path | str,
    *,
    keyframe_indices: list[int],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Writing view propagation overlays for %d keyframes (scene has %d cameras)",
        len(keyframe_indices),
        len(scene.camera_trajectory),
    )
    for out_idx, view_idx in enumerate(keyframe_indices):
        if view_idx >= len(scene.camera_trajectory):
            continue
        view = scene.render(scene.camera_trajectory[view_idx])
        raw_overlay = _overlay_masks(view.rgb, raw_mask_set, view_idx, alpha=0.40)
        refined_overlay = _overlay_masks(view.rgb, refined_mask_set, view_idx, alpha=0.45)
        track_overlay = _overlay_masks(view.rgb, refined_mask_set, view_idx, alpha=0.60, use_track_colors=True)
        _save_image(raw_overlay, output_dir / f"keyframe_{out_idx:02d}_view_{view_idx:02d}_sam3_raw.png")
        _save_image(refined_overlay, output_dir / f"keyframe_{out_idx:02d}_view_{view_idx:02d}_propagated.png")
        _save_image(track_overlay, output_dir / f"keyframe_{out_idx:02d}_view_{view_idx:02d}_track_overlay.png")


def _build_track(
    *,
    prompt_idx: int,
    prompt_group: PromptMaskCandidates,
    scene: GaussianScene,
    view_graph: dict[int, list[ViewGraphEdge]],
    view_map: dict[int, RenderedView],
    cfg: Any,
) -> MaskTrack | None:
    anchor = _select_anchor_candidate(prompt_group, cfg)
    if anchor is None:
        return None

    anchor_view_idx, anchor_candidate = anchor
    track_id = str(prompt_idx)
    track = MaskTrack(
        track_id=track_id,
        object_id=track_id,
        object_name=prompt_group.object_name,
        prompt=prompt_group.prompt,
        source_view=anchor_view_idx,
        source_candidate_id=anchor_candidate.candidate_id,
        masks_by_view={anchor_view_idx: anchor_candidate.mask.detach().cpu().float()},
        scores_by_view={anchor_view_idx: float(anchor_candidate.quality)},
        visibility_by_view={anchor_view_idx: 1.0},
        metadata={
            "anchor_view": anchor_view_idx,
            "anchor_candidate_id": anchor_candidate.candidate_id,
            "anchor_quality": float(anchor_candidate.quality),
            "anchor_score": float(anchor_candidate.score),
            "associations": {},
            "fallback_views": [],
            "fallback_disabled_views": [],
            "dropped_views": [],
            "suppressed_views": [],
            "candidate_conflicts": [],
        },
    )

    best_score_by_view = {anchor_view_idx: float(anchor_candidate.quality)}
    queue: deque[tuple[int, int]] = deque([(anchor_view_idx, 0)])
    visited_hops = {anchor_view_idx: 0}

    while queue:
        source_view, hop = queue.popleft()
        if hop >= int(getattr(cfg, "max_hops", 1)):
            continue
        source_mask = track.masks_by_view[source_view]
        source_render = view_map[source_view]
        source_camera = scene.camera_trajectory[source_view]
        for edge in view_graph.get(source_view, []):
            target_view = edge.target_view
            if target_view not in view_map:
                continue
            target_render = view_map[target_view]
            target_camera = scene.camera_trajectory[target_view]
            warped_mask, warp_stats = _warp_mask(source_render, target_render, source_camera, target_camera, source_mask, cfg)
            target_candidates = prompt_group.view_candidates.get(target_view)
            decision = _match_target_candidate(
                source_view=source_view,
                target_view=target_view,
                warped_mask=warped_mask,
                warp_stats=warp_stats,
                target_candidates=target_candidates,
                edge=edge,
                cfg=cfg,
            )
            if not decision:
                track.metadata.setdefault("dropped_views", []).append(
                    {"view": target_view, "reason": "no_valid_association", "source_view": source_view}
                )
                continue
            current_score = best_score_by_view.get(target_view, float("-inf"))
            if decision.association_score <= current_score:
                track.metadata.setdefault("dropped_views", []).append(
                    {
                        "view": target_view,
                        "reason": "lower_association_score",
                        "source_view": source_view,
                        "score": float(decision.association_score),
                        "current_score": float(current_score),
                    }
                )
                continue
            if decision.mask is None:
                track.metadata.setdefault("dropped_views", []).append(
                    {"view": target_view, "reason": "empty_decision_mask", "source_view": source_view}
                )
                continue
            track.masks_by_view[target_view] = decision.mask.detach().cpu().float()
            track.scores_by_view[target_view] = float(decision.association_score)
            track.visibility_by_view[target_view] = float(decision.visibility_score)
            track.metadata.setdefault("associations", {})[str(target_view)] = decision.to_json()
            best_score_by_view[target_view] = float(decision.association_score)
            visited_hops[target_view] = hop + 1
            queue.append((target_view, hop + 1))

    # Fallback to local SAM3 candidates is intentionally opt-in. In repeated
    # instance scenes, unconstrained fallback can attach background instances to
    # a foreground track.
    fallback_to_sam3 = bool(getattr(cfg, "fallback_to_sam3", False))
    for view_idx, candidate_set in prompt_group.view_candidates.items():
        if view_idx in track.masks_by_view:
            continue
        if not fallback_to_sam3:
            track.metadata.setdefault("fallback_disabled_views", []).append(view_idx)
            continue
        candidate = _choose_raw_candidate(candidate_set)
        if candidate is None or candidate.mask.numel() == 0:
            continue
        track.masks_by_view[view_idx] = candidate.mask.detach().cpu().float()
        track.scores_by_view[view_idx] = float(candidate.quality)
        track.visibility_by_view[view_idx] = 0.0
        track.metadata.setdefault("fallback_views", []).append(view_idx)

    if not track.masks_by_view:
        return None

    track.metadata["mean_score"] = float(sum(track.scores_by_view.values()) / max(1, len(track.scores_by_view)))
    track.metadata["mean_visibility"] = float(sum(track.visibility_by_view.values()) / max(1, len(track.visibility_by_view)))
    track.metadata["coverage_views"] = len(track.masks_by_view)
    track.metadata["visited_hops"] = visited_hops
    return track


def _match_target_candidate(
    *,
    source_view: int,
    target_view: int,
    warped_mask: torch.Tensor,
    warp_stats: dict[str, float],
    target_candidates: SAM3CandidateSet | None,
    edge: ViewGraphEdge,
    cfg: Any,
) -> PropagationDecision | None:
    if warped_mask.numel() == 0 or not bool((warped_mask > 0.5).any()):
        return None

    propagated_mask = warped_mask > 0.5
    propagated_visibility = float(warp_stats.get("visibility_score", 0.0))
    propagated_depth = float(warp_stats.get("depth_consistency", 0.0))
    propagated_score = (
        float(getattr(cfg, "propagated_weight", 0.0)) * propagated_visibility
        + float(getattr(cfg, "geometry_weight", 0.0)) * edge.weight
        + 0.25 * propagated_depth
    )
    association_enabled = bool(getattr(cfg, "association_enabled", True))

    if not association_enabled:
        if target_candidates is None or not target_candidates.candidates:
            if propagated_visibility < float(getattr(cfg, "min_view_overlap", 0.0)) and not bool(getattr(cfg, "fallback_to_sam3", False)):
                return None
            return PropagationDecision(
                source_view=source_view,
                target_view=target_view,
                candidate_id=None,
                mode="propagated",
                association_score=propagated_score,
                reprojection_iou=1.0,
                visibility_score=propagated_visibility,
                depth_consistency=propagated_depth,
                local_sam_score=0.0,
                edge_weight=edge.weight,
                mask=propagated_mask.float(),
            )
        best_candidate = max(target_candidates.candidates, key=lambda cand: cand.quality)
        best_mask = best_candidate.mask > 0.5
        fused_mask = best_mask.float() if best_candidate.quality >= propagated_score else propagated_mask.float()
        return PropagationDecision(
            source_view=source_view,
            target_view=target_view,
            candidate_id=best_candidate.candidate_id,
            mode="sam3" if best_candidate.quality >= propagated_score else "propagated",
            association_score=max(best_candidate.quality, propagated_score),
            reprojection_iou=_mask_iou(propagated_mask, best_mask),
            visibility_score=propagated_visibility,
            depth_consistency=propagated_depth,
            local_sam_score=float(best_candidate.quality),
            edge_weight=edge.weight,
            mask=fused_mask,
        )

    if target_candidates is None or not target_candidates.candidates:
        if propagated_visibility < float(getattr(cfg, "min_assignment_visibility", getattr(cfg, "min_view_overlap", 0.0))):
            return None
        return PropagationDecision(
            source_view=source_view,
            target_view=target_view,
            candidate_id=None,
            mode="propagated",
            association_score=propagated_score,
            reprojection_iou=1.0,
            visibility_score=propagated_visibility,
            depth_consistency=propagated_depth,
            local_sam_score=0.0,
            edge_weight=edge.weight,
            mask=propagated_mask.float(),
            fusion_mode="propagated",
            pre_fusion_area=_mask_area(propagated_mask.float()),
            post_fusion_area=_mask_area(propagated_mask.float()),
        )

    best_candidate: SAM3Candidate | None = None
    best_iou = 0.0
    best_score = float("-inf")
    for candidate in target_candidates.candidates:
        candidate_mask = candidate.mask > 0.5
        iou = _mask_iou(propagated_mask, candidate_mask)
        local_score = float(candidate.quality)
        assoc = (
            float(getattr(cfg, "local_sam_weight", 1.0)) * local_score
            + float(getattr(cfg, "propagated_weight", 0.0)) * iou
            + float(getattr(cfg, "geometry_weight", 0.0)) * edge.weight
            + 0.25 * propagated_visibility
            + 0.15 * propagated_depth
        )
        if assoc > best_score:
            best_score = assoc
            best_candidate = candidate
            best_iou = iou

    if best_candidate is None:
        return None

    min_reproj_iou = max(
        float(getattr(cfg, "min_assignment_iou", 0.0)),
        float(getattr(cfg, "min_reprojection_iou", 0.0)),
    )
    min_visibility = max(
        float(getattr(cfg, "min_assignment_visibility", 0.0)),
        float(getattr(cfg, "min_view_overlap", 0.0)),
    )
    anchor_min_score = float(getattr(cfg, "anchor_min_score", 0.0))
    best_mask = best_candidate.mask > 0.5
    if best_iou >= min_reproj_iou and propagated_visibility >= min_visibility:
        fuse_mask, fusion_mode, rejected_reason = _fuse_candidate_mask(best_mask, propagated_mask, cfg)
        mode = "fused" if rejected_reason is None else "propagated_only"
    elif propagated_visibility >= min_visibility:
        fuse_mask = propagated_mask.float()
        fusion_mode = "propagated"
        rejected_reason = "candidate_failed_identity_gate"
        mode = "propagated"
    elif best_candidate.quality < anchor_min_score and not bool(getattr(cfg, "fallback_to_sam3", False)):
        return None
    elif not bool(getattr(cfg, "fallback_to_sam3", False)):
        return None
    else:
        fuse_mask = best_mask.float()
        fusion_mode = "sam3"
        rejected_reason = "fallback_to_sam3"
        mode = "sam3"

    return PropagationDecision(
        source_view=source_view,
        target_view=target_view,
        candidate_id=best_candidate.candidate_id,
        mode=mode,
        association_score=best_score,
        reprojection_iou=best_iou,
        visibility_score=propagated_visibility,
        depth_consistency=propagated_depth,
        local_sam_score=float(best_candidate.quality),
        edge_weight=edge.weight,
        mask=fuse_mask,
        fusion_mode=fusion_mode,
        pre_fusion_area=float(best_mask.float().sum().item()),
        post_fusion_area=float((fuse_mask > 0.5).float().sum().item()),
        rejected_reason=rejected_reason,
    )


def _fuse_candidate_mask(
    local_mask: torch.Tensor,
    propagated_mask: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, str, str | None]:
    fusion_mode = str(getattr(cfg, "fusion_mode", "intersection"))
    local_bool = local_mask > 0.5
    propagated_bool = propagated_mask > 0.5

    if fusion_mode == "union":
        fused = local_bool | propagated_bool
        return fused.float(), "union", None
    if fusion_mode == "propagated":
        return propagated_bool.float(), "propagated", None
    if fusion_mode != "intersection":
        raise ValueError(f"Unknown view propagation fusion_mode: {fusion_mode}")

    dilate_px = int(getattr(cfg, "local_mask_dilate_px", 0))
    gated = local_bool & _dilate_mask(propagated_bool, dilate_px)
    local_area = float(local_bool.sum().item())
    propagated_area = float(propagated_bool.sum().item())
    min_area = max(1.0, min(local_area, propagated_area))
    retained_ratio = float(gated.sum().item()) / min_area
    min_retained = float(getattr(cfg, "min_fused_area_ratio", 0.25))
    if gated.any() and retained_ratio >= min_retained:
        return gated.float(), "intersection", None
    return propagated_bool.float(), "propagated", "intersection_too_small"


def _enforce_unique_candidate_assignments(tracks: list[MaskTrack]) -> list[MaskTrack]:
    claims: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for track_idx, track in enumerate(tracks):
        if track.source_view in track.masks_by_view:
            claims.setdefault((track.source_view, track.source_candidate_id), []).append(
                (track_idx, float(track.scores_by_view.get(track.source_view, 0.0)))
            )
        for view_key, assoc in track.metadata.get("associations", {}).items():
            candidate_id = assoc.get("candidate_id")
            if candidate_id is None:
                continue
            view_idx = int(view_key)
            claims.setdefault((view_idx, int(candidate_id)), []).append(
                (track_idx, float(assoc.get("association_score", track.scores_by_view.get(view_idx, 0.0))))
            )

    num_conflicts = 0
    num_dropped = 0
    for (view_idx, candidate_id), contenders in claims.items():
        by_track: dict[int, float] = {}
        for track_idx, score in contenders:
            by_track[track_idx] = max(score, by_track.get(track_idx, float("-inf")))
        contenders = sorted(by_track.items(), key=lambda item: item[1], reverse=True)
        if len(contenders) <= 1:
            continue
        num_conflicts += 1
        winner_idx, winner_score = contenders[0]
        for loser_idx, loser_score in contenders[1:]:
            loser = tracks[loser_idx]
            if view_idx not in loser.masks_by_view:
                continue
            loser.masks_by_view.pop(view_idx, None)
            loser.scores_by_view.pop(view_idx, None)
            loser.visibility_by_view.pop(view_idx, None)
            loser.metadata.get("associations", {}).pop(str(view_idx), None)
            loser.metadata.setdefault("candidate_conflicts", []).append(
                {
                    "view": view_idx,
                    "candidate_id": candidate_id,
                    "winner_track_id": tracks[winner_idx].track_id,
                    "winner_score": float(winner_score),
                    "dropped_score": float(loser_score),
                }
            )
            loser.metadata.setdefault("dropped_views", []).append(
                {"view": view_idx, "reason": "candidate_conflict", "candidate_id": candidate_id}
            )
            num_dropped += 1

    for track in tracks:
        track.metadata["mean_score"] = float(sum(track.scores_by_view.values()) / max(1, len(track.scores_by_view)))
        track.metadata["mean_visibility"] = float(sum(track.visibility_by_view.values()) / max(1, len(track.visibility_by_view)))
        track.metadata["coverage_views"] = len(track.masks_by_view)
        track.metadata["num_candidate_conflicts"] = len(track.metadata.get("candidate_conflicts", []))
    for track in tracks:
        track.metadata.setdefault("candidate_assignment_stats", {})["num_global_conflicts"] = num_conflicts
        track.metadata.setdefault("candidate_assignment_stats", {})["num_global_dropped"] = num_dropped
    return tracks


def _suppress_conflicts(tracks: list[MaskTrack], cfg: Any) -> list[MaskTrack]:
    if len(tracks) <= 1:
        return tracks
    track_order = sorted(range(len(tracks)), key=lambda idx: tracks[idx].metadata.get("mean_score", 0.0), reverse=True)
    per_view_occupied: dict[int, torch.Tensor] = {}
    track_merge_iou = float(getattr(cfg, "track_merge_iou", 0.25))
    min_track_views = int(getattr(cfg, "min_track_views", 1))
    conflict_min_views = int(getattr(cfg, "conflict_min_views", 1))
    fallback_to_sam3 = bool(getattr(cfg, "fallback_to_sam3", False))
    for idx in track_order:
        track = tracks[idx]
        if len(track.masks_by_view) < conflict_min_views:
            continue
        for view_idx, mask in list(track.masks_by_view.items()):
            bool_mask = mask > 0.5
            occupied = per_view_occupied.get(view_idx)
            if occupied is None:
                per_view_occupied[view_idx] = bool_mask.clone()
                continue
            overlap = _mask_overlap(bool_mask, occupied)
            if overlap < track_merge_iou:
                per_view_occupied[view_idx] = occupied | bool_mask
                continue
            reduced = bool_mask & ~occupied
            retained_ratio = float(reduced.sum().item()) / max(1.0, float(bool_mask.sum().item()))
            if reduced.any() and retained_ratio >= 0.25:
                track.masks_by_view[view_idx] = reduced.float()
                track.metadata.setdefault("suppressed_views", []).append(view_idx)
                per_view_occupied[view_idx] = occupied | reduced
            elif fallback_to_sam3:
                per_view_occupied[view_idx] = occupied | bool_mask
            else:
                track.masks_by_view.pop(view_idx, None)
                track.scores_by_view.pop(view_idx, None)
                track.visibility_by_view.pop(view_idx, None)
                track.metadata.setdefault("dropped_views", []).append(
                    {"view": view_idx, "reason": "track_overlap_conflict", "overlap": float(overlap)}
                )
    for track in tracks:
        track.metadata["mean_score"] = float(sum(track.scores_by_view.values()) / max(1, len(track.scores_by_view)))
        track.metadata["mean_visibility"] = float(sum(track.visibility_by_view.values()) / max(1, len(track.visibility_by_view)))
        track.metadata["coverage_views"] = len(track.masks_by_view)
    return tracks


def _compute_stats(
    tracks: list[MaskTrack],
    view_graph: dict[int, list[ViewGraphEdge]],
    keyframe_indices: list[int],
    cfg: Any,
) -> dict[str, Any]:
    track_scores = [track.metadata.get("mean_score", 0.0) for track in tracks]
    visibilities = [track.metadata.get("mean_visibility", 0.0) for track in tracks]
    candidate_conflicts = sum(len(track.metadata.get("candidate_conflicts", [])) for track in tracks)
    dropped_views = [drop for track in tracks for drop in track.metadata.get("dropped_views", [])]
    dropped_by_assignment = sum(1 for drop in dropped_views if str(drop.get("reason", "")).startswith("candidate_"))
    dropped_by_fusion = sum(1 for drop in dropped_views if "fusion" in str(drop.get("reason", "")))
    return {
        "enabled": bool(getattr(cfg, "enabled", False)),
        "association_enabled": bool(getattr(cfg, "association_enabled", False)),
        "save_debug": bool(getattr(cfg, "save_debug", False)),
        "num_tracks": len(tracks),
        "num_views": len(keyframe_indices),
        "num_edges": sum(len(edges) for edges in view_graph.values()),
        "avg_track_score": float(sum(track_scores) / max(1, len(track_scores))),
        "avg_visibility": float(sum(visibilities) / max(1, len(visibilities))),
        "min_track_views": int(getattr(cfg, "min_track_views", 1)),
        "candidate_unique_assignment": bool(getattr(cfg, "candidate_unique_assignment", True)),
        "fusion_mode": str(getattr(cfg, "fusion_mode", "intersection")),
        "fallback_to_sam3": bool(getattr(cfg, "fallback_to_sam3", False)),
        "num_candidate_conflicts": int(candidate_conflicts),
        "num_dropped_by_assignment": int(dropped_by_assignment),
        "num_dropped_by_fusion": int(dropped_by_fusion),
    }


def _choose_raw_candidate(candidate_set: SAM3CandidateSet) -> SAM3Candidate | None:
    if not candidate_set.candidates:
        return None
    scores = candidate_set.scores
    if scores.numel() == 0:
        return max(candidate_set.candidates, key=lambda c: c.area)
    idx = int(torch.argmax(scores).item())
    return candidate_set.candidates[idx]


def _select_anchor_candidate(prompt_group: PromptMaskCandidates, cfg: Any) -> tuple[int, SAM3Candidate] | None:
    anchor_min_score = float(getattr(cfg, "anchor_min_score", 0.0))
    anchor_min_area = float(getattr(cfg, "anchor_min_area", 0.0))
    anchor_max_area_ratio = float(getattr(cfg, "anchor_max_area_ratio", 1.0))
    best: tuple[int, SAM3Candidate] | None = None
    best_quality = float("-inf")
    fallback: tuple[int, SAM3Candidate] | None = None
    fallback_quality = float("-inf")
    for view_idx, candidate_set in prompt_group.view_candidates.items():
        image_area = max(1.0, float(candidate_set.image_size[0] * candidate_set.image_size[1]))
        for candidate in candidate_set.candidates:
            area_ratio = float(candidate.area / image_area)
            if candidate.quality > fallback_quality:
                fallback = (view_idx, candidate)
                fallback_quality = float(candidate.quality)
            if candidate.score < anchor_min_score:
                continue
            if area_ratio < anchor_min_area or area_ratio > anchor_max_area_ratio:
                continue
            if candidate.quality > best_quality:
                best = (view_idx, candidate)
                best_quality = float(candidate.quality)
    return best or fallback


def _build_raw_mask_set(prompt_candidates: list[PromptMaskCandidates], cfg: Any) -> MaskSet:
    object_masks: list[ObjectMask] = []
    for prompt_idx, prompt_group in enumerate(prompt_candidates):
        masks: dict[int, torch.Tensor] = {}
        for view_idx, candidate_set in prompt_group.view_candidates.items():
            candidate = _choose_raw_candidate(candidate_set)
            if candidate is None or candidate.mask.numel() == 0 or not bool((candidate.mask > 0.5).any()):
                continue
            masks[view_idx] = candidate.mask.detach().cpu().float()
        if masks:
            object_masks.append(
                ObjectMask(
                    object_id=str(prompt_idx),
                    object_name=prompt_group.object_name,
                    masks=masks,
                    track_id=str(prompt_idx),
                    metadata={
                        "prompt": prompt_group.prompt,
                        "source": "raw_sam3",
                    },
                )
            )
    mask_set = MaskSet(object_masks=object_masks, metadata={"enabled": bool(getattr(cfg, "enabled", False))})
    mask_set.fuse()
    return mask_set


def _mask_overlap(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    inter = (mask_a & mask_b).sum().float()
    denom = torch.minimum(mask_a.sum().float(), mask_b.sum().float()).clamp_min(1.0)
    return float((inter / denom).item())


def _mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    inter = (mask_a & mask_b).sum().float()
    union = (mask_a | mask_b).sum().float().clamp_min(1.0)
    return float((inter / union).item())


def _mask_area(mask: torch.Tensor) -> float:
    return float(mask.float().sum().item())


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    x = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius)
    return dilated.squeeze(0).squeeze(0) > 0.5


def _warp_mask(
    source_view: RenderedView,
    target_view: RenderedView,
    source_camera: CameraPose,
    target_camera: CameraPose,
    mask: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    source_mask = mask > 0.5
    if not bool(source_mask.any()):
        return torch.zeros_like(mask, dtype=torch.float32), {"visibility_score": 0.0, "depth_consistency": 0.0}

    ys, xs = source_mask.nonzero(as_tuple=True)
    source_depth = source_view.depth.to(mask.device)
    depth = source_depth[ys, xs]
    valid = depth > 1e-6
    if not bool(valid.any()):
        return torch.zeros_like(mask, dtype=torch.float32), {"visibility_score": 0.0, "depth_consistency": 0.0}

    ys = ys[valid].float()
    xs = xs[valid].float()
    depth = depth[valid].float()
    src_K = source_camera.intrinsic.to(mask.device)
    tgt_K = target_camera.intrinsic.to(mask.device)
    src_c2w = source_camera.c2w.to(mask.device)
    tgt_w2c = target_camera.extrinsic.to(mask.device)

    x_cam = (xs - src_K[0, 2]) / src_K[0, 0] * depth
    y_cam = (ys - src_K[1, 2]) / src_K[1, 1] * depth
    cam_pts = torch.stack([x_cam, y_cam, depth], dim=1)
    world = (src_c2w[:3, :3] @ cam_pts.T).T + src_c2w[:3, 3]
    target_cam = (tgt_w2c[:3, :3] @ world.T).T + tgt_w2c[:3, 3]
    z = target_cam[:, 2]
    valid = z > 1e-6
    if not bool(valid.any()):
        return torch.zeros_like(target_view.depth, dtype=torch.float32), {"visibility_score": 0.0, "depth_consistency": 0.0}

    x_t = tgt_K[0, 0] * target_cam[:, 0] / z + tgt_K[0, 2]
    y_t = tgt_K[1, 1] * target_cam[:, 1] / z + tgt_K[1, 2]
    px = torch.round(x_t).long()
    py = torch.round(y_t).long()
    in_bounds = (
        (px >= 0)
        & (py >= 0)
        & (px < target_view.width)
        & (py < target_view.height)
    )
    valid = valid & in_bounds
    if not bool(valid.any()):
        return torch.zeros_like(target_view.depth, dtype=torch.float32), {"visibility_score": 0.0, "depth_consistency": 0.0}

    px = px[valid]
    py = py[valid]
    z = z[valid]
    target_depth = target_view.depth.to(mask.device)
    depth_at_pixel = target_depth[py, px]
    depth_error = torch.abs(depth_at_pixel - z)
    max_depth_error = float(getattr(cfg, "max_depth_error", 0.0))
    if max_depth_error > 0.0:
        visible = (depth_at_pixel > 1e-6) & (depth_error <= max_depth_error)
        depth_consistency = 0.0
        if bool(visible.any()):
            depth_consistency = float((1.0 - (depth_error[visible].mean() / max_depth_error).clamp(0.0, 1.0)).item())
    else:
        visible = depth_at_pixel > 1e-6
        depth_consistency = 1.0

    target_flat = torch.zeros(target_view.height * target_view.width, dtype=torch.bool, device=mask.device)
    if bool(visible.any()):
        flat_idx = py[visible] * target_view.width + px[visible]
        target_flat[flat_idx] = True
    propagated = target_flat.view(target_view.height, target_view.width)
    propagated = _dilate_mask(propagated, 1)
    source_area = max(1.0, float(source_mask.sum().item()))
    visibility_score = float(visible.sum().item() / source_area)
    return propagated.float(), {"visibility_score": visibility_score, "depth_consistency": depth_consistency}


def _camera_center(camera: CameraPose) -> torch.Tensor:
    return camera.c2w[:3, 3].detach().cpu()


def _camera_forward(camera: CameraPose) -> torch.Tensor:
    forward = camera.c2w[:3, 2].detach().cpu()
    return F.normalize(forward, dim=0)


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    cosine = torch.clamp(torch.dot(a, b), -1.0, 1.0)
    return float(torch.rad2deg(torch.arccos(cosine)).item())


def _estimate_view_overlap(
    pose_distance: float,
    angle_deg: float,
    max_pose_distance: float,
    max_angle_deg: float,
    source_view: int,
    target_view: int,
) -> float:
    dist_term = 1.0 if max_pose_distance <= 0.0 else max(0.0, 1.0 - pose_distance / max_pose_distance)
    angle_term = 1.0 if max_angle_deg <= 0.0 else max(0.0, 1.0 - angle_deg / max_angle_deg)
    order_term = 1.0 / (1.0 + abs(source_view - target_view))
    return float(dist_term * angle_term * order_term)


def _flatten_graph(view_graph: dict[int, list[ViewGraphEdge]]) -> list[ViewGraphEdge]:
    edges: list[ViewGraphEdge] = []
    for items in view_graph.values():
        edges.extend(items)
    return edges


def _mask_track_from_object_mask(obj_mask: ObjectMask) -> MaskTrack:
    masks = {view_idx: mask.detach().cpu().float() for view_idx, mask in obj_mask.masks.items()}
    scores_by_view = {view_idx: float(mask.float().mean().item()) for view_idx, mask in masks.items()}
    visibility_by_view = {view_idx: 1.0 for view_idx in masks}
    return MaskTrack(
        track_id=obj_mask.track_id or obj_mask.object_id,
        object_id=obj_mask.object_id,
        object_name=obj_mask.object_name,
        prompt=str(obj_mask.metadata.get("prompt", obj_mask.object_name)),
        source_view=min(masks) if masks else -1,
        source_candidate_id=0,
        masks_by_view=masks,
        scores_by_view=scores_by_view,
        visibility_by_view=visibility_by_view,
        metadata=dict(obj_mask.metadata),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (tensor.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
    iio.imwrite(path, arr)


def _overlay_masks(
    rgb: torch.Tensor,
    mask_set: MaskSet,
    view_idx: int,
    *,
    alpha: float,
    use_track_colors: bool = False,
) -> torch.Tensor:
    overlay = rgb.clone()
    colors = [
        (1.0, 0.0, 0.0),
        (0.0, 0.8, 0.0),
        (0.0, 0.3, 1.0),
        (1.0, 0.8, 0.0),
        (0.8, 0.0, 0.8),
        (0.0, 0.8, 0.8),
        (1.0, 0.5, 0.2),
        (0.8, 0.8, 0.8),
    ]
    for obj_idx, obj_mask in enumerate(mask_set.object_masks):
        mask = obj_mask.get_mask(view_idx)
        if mask is None:
            continue
        color_idx = obj_idx if not use_track_colors else int((obj_mask.track_id or obj_mask.object_id).__hash__()) % len(colors)
        color = torch.tensor(colors[color_idx % len(colors)], dtype=overlay.dtype, device=overlay.device)
        mask = mask.to(overlay.device).unsqueeze(-1)
        overlay = overlay * (1.0 - alpha * mask) + color.view(1, 1, 3) * alpha * mask
    return overlay


def _build_track_mask_decision(
    *,
    source_view: int,
    target_view: int,
    candidate: SAM3Candidate,
    warped_mask: torch.Tensor,
    warp_stats: dict[str, float],
    edge: ViewGraphEdge,
    cfg: Any,
) -> PropagationDecision:
    candidate_mask = candidate.mask > 0.5
    iou = _mask_iou(warped_mask > 0.5, candidate_mask)
    association_score = (
        float(getattr(cfg, "local_sam_weight", 1.0)) * float(candidate.quality)
        + float(getattr(cfg, "propagated_weight", 0.0)) * iou
        + float(getattr(cfg, "geometry_weight", 0.0)) * edge.weight
        + 0.25 * float(warp_stats.get("visibility_score", 0.0))
        + 0.15 * float(warp_stats.get("depth_consistency", 0.0))
    )
    min_reproj_iou = max(
        float(getattr(cfg, "min_assignment_iou", 0.0)),
        float(getattr(cfg, "min_reprojection_iou", 0.0)),
    )
    min_visibility = max(
        float(getattr(cfg, "min_assignment_visibility", 0.0)),
        float(getattr(cfg, "min_view_overlap", 0.0)),
    )
    propagated_mask = warped_mask > 0.5
    visibility = float(warp_stats.get("visibility_score", 0.0))
    if iou >= min_reproj_iou and visibility >= min_visibility:
        mask, fusion_mode, rejected_reason = _fuse_candidate_mask(candidate_mask, propagated_mask, cfg)
        mode = "fused" if rejected_reason is None else "propagated_only"
    elif visibility >= min_visibility:
        mode = "propagated"
        mask = propagated_mask.float()
        fusion_mode = "propagated"
        rejected_reason = "candidate_failed_identity_gate"
    else:
        mode = "sam3"
        mask = candidate_mask.float()
        fusion_mode = "sam3"
        rejected_reason = "fallback_to_sam3"
    decision = PropagationDecision(
        source_view=source_view,
        target_view=target_view,
        candidate_id=candidate.candidate_id,
        mode=mode,
        association_score=association_score,
        reprojection_iou=iou,
        visibility_score=float(warp_stats.get("visibility_score", 0.0)),
        depth_consistency=float(warp_stats.get("depth_consistency", 0.0)),
        local_sam_score=float(candidate.quality),
        edge_weight=edge.weight,
        mask=mask,
        fusion_mode=fusion_mode,
        pre_fusion_area=float(candidate_mask.float().sum().item()),
        post_fusion_area=float((mask > 0.5).float().sum().item()),
        rejected_reason=rejected_reason,
    )
    return decision
