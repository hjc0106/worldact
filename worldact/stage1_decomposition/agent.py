from __future__ import annotations

from typing import Any

import torch
from worldact.data.gaussians import GaussianScene, RenderedView
from worldact.data.masks import MaskSet, ObjectMask
from worldact.data.objects import ObjectMemoryState
from worldact.stage1_decomposition.candidate_support import (
    build_candidate_supports,
    compute_mask_overlap_penalties,
    summarize_supports,
    support_containment,
    support_iou,
)
from worldact.stage1_decomposition.memory import ObjectMemory
from worldact.stage1_decomposition.view_graph_propagation import (
    PromptMaskCandidates,
    propagate_candidates,
)
from worldact.stage1_decomposition.sam3_segmenter import SAM3Segmenter
from worldact.stage1_decomposition.vlm import VLMClient, manual_memory_states
from worldact.utils.logging import get_logger

log = get_logger(__name__)


class MultimodalAgent:
    def __init__(self, cfg: Any, segmenter: SAM3Segmenter | None = None) -> None:
        self.cfg = cfg
        self.memory = ObjectMemory()
        self.segmenter = segmenter or SAM3Segmenter(cfg.sam3)

    def discover_objects(
        self,
        keyframe_views: list[RenderedView],
        manual_prompts: list[str] | None = None,
    ) -> list[ObjectMemoryState]:
        states = manual_memory_states(manual_prompts) if manual_prompts else VLMClient(self.cfg.vlm).analyze_keyframes(keyframe_views)
        for state in states:
            self.memory.add(state)
        return self.memory.get_all()

    def segment_objects(
        self,
        states: list[ObjectMemoryState],
        scene: GaussianScene,
        keyframe_indices: list[int],
        keyframe_views: list[RenderedView] | None = None,
    ) -> tuple[MaskSet, list[ObjectMemoryState]]:
        keyframe_views = keyframe_views or [scene.render(scene.camera_trajectory[view_idx]) for view_idx in keyframe_indices]
        valid_states: list[ObjectMemoryState] = []
        raw_object_masks: list[ObjectMask] = []
        prompt_candidates: list[PromptMaskCandidates] = []
        support_cfg = getattr(self.cfg, "candidate_support", None)
        support_enabled = support_cfg is not None and bool(getattr(support_cfg, "enabled", True))
        supports_by_object: dict[str, list] = {}
        mask_weights: dict[str, dict[str, float]] = {}
        for state in states:
            prompt = state.text_prompt or state.name
            view_candidates: dict[int, Any] = {}
            raw_masks: dict[int, torch.Tensor] = {}
            pending_supports: list = []
            for view_idx, view in zip(keyframe_indices, keyframe_views):
                candidate_set = self.segmenter.segment_candidates(view.rgb, prompt)
                if candidate_set is None or not candidate_set.candidates:
                    continue
                view_candidates[view_idx] = candidate_set
                view_supports = []
                if support_enabled:
                    view_supports = build_candidate_supports(
                        scene,
                        view_idx,
                        view,
                        object_id=str(len(valid_states)),
                        object_name=state.name,
                        prompt=prompt,
                        candidate_set=candidate_set,
                        cfg=support_cfg,
                    )
                    pending_supports.extend(view_supports)
                selected = _choose_candidate_with_support(self.segmenter, candidate_set, view_supports)
                if selected is None:
                    if str(self.segmenter.cfg.instance_strategy) == "union":
                        union_mask = (candidate_set.masks.sum(dim=0) > 0).to("cpu").float()
                        if bool((union_mask > 0.5).any()):
                            raw_masks[view_idx] = union_mask
                    continue
                if selected.mask.numel() == 0 or not bool((selected.mask > 0.5).any()):
                    continue
                raw_masks[view_idx] = selected.mask.detach().cpu().float()
            if view_candidates:
                object_id = str(len(valid_states))
                valid_states.append(state)
                if pending_supports:
                    supports_by_object[object_id] = pending_supports
                    selected_ids = _selected_candidate_ids(raw_masks, view_candidates)
                    weights = {
                        str(support.view_idx): float(support.reliability)
                        for support in pending_supports
                        if (support.view_idx, support.candidate_id) in selected_ids
                    }
                    if weights:
                        mask_weights[object_id] = weights
                prompt_candidates.append(
                    PromptMaskCandidates(
                        object_id=object_id,
                        object_name=state.name,
                        prompt=prompt,
                        view_candidates=view_candidates,
                        memory_state=state,
                    )
                )
                raw_object_masks.append(
                    ObjectMask(
                        object_id=object_id,
                        object_name=state.name,
                        masks=raw_masks,
                        track_id=object_id,
                        metadata={"prompt": prompt, "selection": str(self.segmenter.cfg.instance_strategy)},
                    )
                )
            else:
                log.warning("Dropping object %r: SAM3 produced no valid masks", state.name)
        if support_enabled and supports_by_object:
            compute_mask_overlap_penalties(supports_by_object)
            mask_weights = {}
            for object_id, supports in supports_by_object.items():
                selected_ids = _selected_candidate_ids(
                    raw_object_masks[int(object_id)].masks,
                    prompt_candidates[int(object_id)].view_candidates,
                )
                weights = {
                    str(support.view_idx): float(support.reliability)
                    for support in supports
                    if (support.view_idx, support.candidate_id) in selected_ids
                }
                if weights:
                    mask_weights[object_id] = weights
            identity_meta = _track_identity_summary(supports_by_object, support_cfg)
            if bool(getattr(support_cfg, "identity_suppression_enabled", True)):
                (
                    valid_states,
                    prompt_candidates,
                    raw_object_masks,
                    supports_by_object,
                    mask_weights,
                ) = _filter_identity_suppressed_objects(
                    valid_states,
                    prompt_candidates,
                    raw_object_masks,
                    supports_by_object,
                    mask_weights,
                    identity_meta,
                )
        else:
            identity_meta = {}
        raw_mask_set = MaskSet(object_masks=raw_object_masks, metadata={})
        if support_enabled and supports_by_object:
            raw_mask_set.metadata["candidate_support"] = summarize_supports(supports_by_object)
            raw_mask_set.metadata["track_identity"] = identity_meta
            if mask_weights:
                raw_mask_set.metadata["mask_weights"] = mask_weights
        raw_mask_set.fuse()
        propagation_cfg = getattr(self.cfg, "view_propagation", None)
        if propagation_cfg is not None and bool(getattr(propagation_cfg, "enabled", False)):
            propagated = propagate_candidates(
                scene,
                keyframe_indices,
                keyframe_views,
                prompt_candidates,
                propagation_cfg,
                raw_mask_set=raw_mask_set,
            )
            mask_set = propagated.refined_mask_set
            mask_set.metadata.setdefault("view_propagation", propagated.to_summary())
            mask_set.metadata["view_propagation_raw_masks"] = propagated.raw_mask_set
        else:
            mask_set = raw_mask_set
        mask_set.fuse()
        return mask_set, valid_states


def _choose_candidate_with_support(segmenter: SAM3Segmenter, candidate_set: Any, supports: list) -> Any | None:
    if not supports:
        return segmenter.choose_candidate(candidate_set)
    support_by_candidate = {support.candidate_id: support for support in supports}
    best = None
    best_score = float("-inf")
    for candidate in candidate_set.candidates:
        support = support_by_candidate.get(candidate.candidate_id)
        if support is None:
            continue
        score = 0.55 * float(candidate.quality) + 0.45 * float(support.reliability)
        if score > best_score:
            best = candidate
            best_score = score
    return best or segmenter.choose_candidate(candidate_set)


def _selected_candidate_ids(raw_masks: dict[int, torch.Tensor], view_candidates: dict[int, Any]) -> set[tuple[int, int]]:
    selected_ids: set[tuple[int, int]] = set()
    for view_idx, mask in raw_masks.items():
        candidate_set = view_candidates.get(view_idx)
        if candidate_set is None:
            continue
        for candidate in candidate_set.candidates:
            if candidate.mask.shape == mask.shape and bool(((candidate.mask > 0.5) == (mask > 0.5)).all()):
                selected_ids.add((int(view_idx), int(candidate.candidate_id)))
                break
    return selected_ids


def _track_identity_summary(supports_by_object: dict[str, list], cfg: Any) -> dict[str, Any]:
    duplicate_iou = float(getattr(cfg, "support_duplicate_iou", 0.45))
    duplicate_containment = float(getattr(cfg, "support_duplicate_containment", 0.70))
    suppression_enabled = bool(getattr(cfg, "identity_suppression_enabled", False))
    conflicts = []
    object_ids = sorted(supports_by_object)
    object_names = {
        object_id: supports[0].object_name if supports else object_id
        for object_id, supports in supports_by_object.items()
    }
    object_scores = {
        object_id: _support_object_score(supports)
        for object_id, supports in supports_by_object.items()
    }
    adjacency: dict[str, set[str]] = {object_id: set() for object_id in object_ids}
    for offset, object_a in enumerate(object_ids):
        supports_a = supports_by_object[object_a]
        indices_a = _union_support_indices(supports_a)
        for object_b in object_ids[offset + 1 :]:
            supports_b = supports_by_object[object_b]
            indices_b = _union_support_indices(supports_b)
            iou = support_iou(indices_a, indices_b)
            containment = support_containment(indices_a, indices_b)
            if iou >= duplicate_iou or containment >= duplicate_containment:
                adjacency[object_a].add(object_b)
                adjacency[object_b].add(object_a)
                conflicts.append(
                    {
                        "object_a": object_a,
                        "object_b": object_b,
                        "object_a_name": object_names.get(object_a, object_a),
                        "object_b_name": object_names.get(object_b, object_b),
                        "support_iou": iou,
                        "support_containment": containment,
                        "type": "duplicate" if iou >= duplicate_iou else "contained_fragment",
                        "score_a": object_scores.get(object_a, 0.0),
                        "score_b": object_scores.get(object_b, 0.0),
                    }
                )
    components = _identity_conflict_components(adjacency)
    component_records = []
    canonical_by_object: dict[str, str] = {}
    suppressed: set[str] = set()
    for component in components:
        if len(component) <= 1:
            continue
        keep = _choose_identity_canonical(component, object_scores)
        dropped = [object_id for object_id in component if object_id != keep]
        suppressed.update(dropped)
        for object_id in component:
            canonical_by_object[object_id] = keep
        component_records.append(
            {
                "keep": keep,
                "keep_name": object_names.get(keep, keep),
                "drop": dropped,
                "drop_names": {object_id: object_names.get(object_id, object_id) for object_id in dropped},
                "members": component,
                "member_names": {object_id: object_names.get(object_id, object_id) for object_id in component},
                "scores": {object_id: object_scores.get(object_id, 0.0) for object_id in component},
            }
        )
    for conflict in conflicts:
        conflict["component_keep"] = canonical_by_object.get(str(conflict["object_a"]))
        pair_suppressed = [
            object_id
            for object_id in (str(conflict["object_a"]), str(conflict["object_b"]))
            if object_id in suppressed
        ]
        conflict["pair_suppressed"] = pair_suppressed
    return {
        "num_objects": len(object_ids),
        "num_conflicts": len(conflicts),
        "conflicts": conflicts,
        "conflict_components": component_records,
        "object_scores": object_scores,
        "object_names": object_names,
        "would_suppress_object_ids": sorted(suppressed, key=lambda item: int(item) if item.isdigit() else item),
        "suppressed_object_ids": sorted(suppressed, key=lambda item: int(item) if item.isdigit() else item)
        if suppression_enabled
        else [],
    }


def _union_support_indices(supports: list) -> torch.Tensor:
    parts = [support.gaussian_indices.long() for support in supports if support.gaussian_indices.numel() > 0]
    if not parts:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(torch.cat(parts, dim=0))


def _support_object_score(supports: list) -> float:
    if not supports:
        return 0.0
    reliabilities = sorted((float(support.reliability) for support in supports), reverse=True)
    top = reliabilities[: min(5, len(reliabilities))]
    reliability_score = sum(top) / max(1, len(top))
    support_count = sum(1 for support in supports if support.gaussian_indices.numel() > 0)
    coverage_score = min(1.0, support_count / 3.0)
    return float(0.75 * reliability_score + 0.25 * coverage_score)


def _identity_conflict_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    components: list[list[str]] = []
    visited: set[str] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component, key=lambda item: int(item) if item.isdigit() else item))
    return components


def _choose_identity_canonical(component: list[str], object_scores: dict[str, float]) -> str:
    return max(
        component,
        key=lambda object_id: (
            float(object_scores.get(object_id, 0.0)),
            -int(object_id) if object_id.isdigit() else 0,
        ),
    )


def _filter_identity_suppressed_objects(
    valid_states: list[ObjectMemoryState],
    prompt_candidates: list[PromptMaskCandidates],
    object_masks: list[ObjectMask],
    supports_by_object: dict[str, list],
    mask_weights: dict[str, dict[str, float]],
    identity_meta: dict[str, Any],
) -> tuple[list[ObjectMemoryState], list[PromptMaskCandidates], list[ObjectMask], dict[str, list], dict[str, dict[str, float]]]:
    suppressed = set(str(item) for item in identity_meta.get("suppressed_object_ids", []))
    if not suppressed:
        identity_meta["identity_suppression_applied"] = False
        identity_meta["id_remap"] = {str(idx): str(idx) for idx in range(len(valid_states))}
        return valid_states, prompt_candidates, object_masks, supports_by_object, mask_weights

    id_remap: dict[str, str | None] = {}
    filtered_states: list[ObjectMemoryState] = []
    filtered_candidates: list[PromptMaskCandidates] = []
    filtered_masks: list[ObjectMask] = []
    filtered_supports: dict[str, list] = {}
    filtered_weights: dict[str, dict[str, float]] = {}
    suppressed_names: dict[str, str] = {}

    for old_idx, state in enumerate(valid_states):
        old_id = str(old_idx)
        if old_id in suppressed:
            id_remap[old_id] = None
            suppressed_names[old_id] = state.name
            continue
        new_id = str(len(filtered_states))
        id_remap[old_id] = new_id
        filtered_states.append(state)

        prompt_group = prompt_candidates[old_idx]
        prompt_group.object_id = new_id
        filtered_candidates.append(prompt_group)

        obj_mask = object_masks[old_idx]
        obj_mask.object_id = new_id
        obj_mask.track_id = new_id
        obj_mask.metadata = dict(obj_mask.metadata)
        obj_mask.metadata["original_object_id"] = old_id
        filtered_masks.append(obj_mask)

        remapped_supports = supports_by_object.get(old_id, [])
        for support in remapped_supports:
            support.object_id = new_id
        if remapped_supports:
            filtered_supports[new_id] = remapped_supports
        if old_id in mask_weights:
            filtered_weights[new_id] = mask_weights[old_id]

    identity_meta["identity_suppression_applied"] = True
    identity_meta["id_remap"] = id_remap
    identity_meta["suppressed_names"] = suppressed_names
    identity_meta["num_active_after_suppression"] = len(filtered_states)
    log.info(
        "Identity suppression removed %d duplicate prompts before view propagation: %s",
        len(suppressed_names),
        suppressed_names,
    )
    return filtered_states, filtered_candidates, filtered_masks, filtered_supports, filtered_weights
