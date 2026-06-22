from __future__ import annotations

from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.objects import ObjectAsset, ObjectMemoryState


def _object_thresholds(scores: torch.Tensor, base_threshold: float, cfg: Any | None) -> torch.Tensor:
    thresholds = torch.full((scores.shape[1],), float(base_threshold), dtype=scores.dtype, device=scores.device)
    sep_cfg = getattr(cfg, "instance_separation", None) if cfg is not None else None
    if sep_cfg is None or not bool(getattr(sep_cfg, "adaptive_threshold", False)):
        return thresholds

    max_delta = float(getattr(sep_cfg, "adaptive_threshold_max_delta", 0.0))
    quantile = float(getattr(sep_cfg, "adaptive_threshold_quantile", 0.75))
    if max_delta <= 0.0:
        return thresholds

    upper = float(base_threshold) + max_delta
    min_active_ratio = float(getattr(sep_cfg, "adaptive_threshold_min_active_ratio", 0.0))
    for obj_idx in range(scores.shape[1]):
        candidates = scores[:, obj_idx]
        active = candidates[candidates > float(base_threshold)]
        if active.numel() == 0:
            continue
        if active.numel() / max(1, scores.shape[0]) < min_active_ratio:
            continue
        adaptive = torch.quantile(active, quantile)
        thresholds[obj_idx] = torch.clamp(adaptive, min=float(base_threshold), max=upper)
    return thresholds


def binarize_and_partition(
    scene: GaussianScene,
    scores: torch.Tensor,
    states: list[ObjectMemoryState],
    threshold: float,
    cfg: Any | None = None,
) -> tuple[GaussianScene, list[ObjectAsset], torch.Tensor]:
    if scores.numel() == 0 or scores.shape[1] == 0:
        all_indices = torch.arange(scene.num_gaussians, device=scene.device)
        return scene.subset(all_indices), [], scores
    max_scores, best_obj = scores.max(dim=1)
    thresholds = _object_thresholds(scores, threshold, cfg)
    assigned = max_scores > thresholds[best_obj]
    used = torch.zeros(scene.num_gaussians, dtype=torch.bool, device=scores.device)
    objects: list[ObjectAsset] = []
    for obj_idx, state in enumerate(states):
        indices = ((best_obj == obj_idx) & assigned).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            continue
        used[indices] = True
        objects.append(
            ObjectAsset(
                object_id=str(obj_idx),
                name=state.name,
                source_gaussian_indices=indices.detach().cpu(),
                memory_state=state,
            )
        )
    bg_indices = (~used).nonzero(as_tuple=True)[0]
    return scene.subset(bg_indices), objects, scores


def cleanup_object_components(
    scene: GaussianScene,
    objects: list[ObjectAsset],
    cfg: Any | None,
) -> tuple[list[ObjectAsset], dict[str, dict[str, int | float | bool]]]:
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return objects, {"enabled": {"value": False}}

    radius = float(getattr(cfg, "radius", 1.2))
    min_count = int(getattr(cfg, "min_component_gaussians", 128))
    min_ratio = float(getattr(cfg, "min_component_ratio", 0.08))
    keep_largest = bool(getattr(cfg, "keep_largest_component", True))
    meta: dict[str, dict[str, int | float | bool]] = {
        "enabled": {
            "value": True,
            "radius": radius,
            "min_component_gaussians": min_count,
            "min_component_ratio": min_ratio,
            "keep_largest_component": keep_largest,
        }
    }
    cleaned: list[ObjectAsset] = []
    for obj in objects:
        indices = obj.source_gaussian_indices
        if indices is None or len(indices) == 0:
            continue
        indices_dev = indices.to(scene.device)
        means = scene.params.means[indices_dev].detach()
        components = _radius_components(means, radius)
        sizes = [int(component.numel()) for component in components]
        if not components:
            kept_local = torch.arange(indices_dev.numel(), device=scene.device)
        elif keep_largest:
            largest_idx = max(range(len(components)), key=lambda i: sizes[i])
            kept_local = components[largest_idx]
        else:
            threshold = max(min_count, int(round(min_ratio * max(1, indices_dev.numel()))))
            kept = [component for component in components if int(component.numel()) >= threshold]
            kept_local = torch.cat(kept) if kept else components[max(range(len(components)), key=lambda i: sizes[i])]
        kept_indices = indices_dev[kept_local].detach().cpu()
        obj.source_gaussian_indices = kept_indices
        obj.metadata = dict(obj.metadata)
        obj.metadata["cleanup"] = {
            "before": int(indices_dev.numel()),
            "after": int(kept_indices.numel()),
            "num_components": int(len(components)),
            "largest_component": int(max(sizes) if sizes else 0),
        }
        meta[str(obj.object_id)] = obj.metadata["cleanup"]
        cleaned.append(obj)
    return cleaned, meta


def deduplicate_objects_3d(
    scene: GaussianScene,
    objects: list[ObjectAsset],
    cfg: Any | None,
) -> tuple[list[ObjectAsset], dict[str, Any]]:
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return objects, {"enabled": {"value": False}}

    up_axis = _axis_index(str(getattr(cfg, "up_axis", None) or "y").lstrip("-"))
    horizontal_axes = [idx for idx in range(3) if idx != up_axis]
    mode = str(getattr(cfg, "mode", "adaptive"))
    duplicate_score_threshold = float(getattr(cfg, "duplicate_score_threshold", 0.62))
    fragment_containment_threshold = float(getattr(cfg, "fragment_containment_threshold", 0.85))
    fragment_max_count_ratio = float(getattr(cfg, "fragment_max_count_ratio", 0.70))
    max_centroid_distance = float(getattr(cfg, "max_centroid_distance", 1.0))
    min_horizontal_iou = float(getattr(cfg, "min_horizontal_iou", 0.45))
    min_3d_iou = float(getattr(cfg, "min_3d_iou", 0.35))
    min_containment = float(getattr(cfg, "min_containment", 0.60))
    suppress = bool(getattr(cfg, "suppress", False))
    prefer_larger = bool(getattr(cfg, "prefer_larger", True))

    meta: dict[str, Any] = {
        "enabled": {
            "value": True,
            "suppress": suppress,
            "mode": mode,
            "duplicate_score_threshold": duplicate_score_threshold,
            "fragment_containment_threshold": fragment_containment_threshold,
            "fragment_max_count_ratio": fragment_max_count_ratio,
            "max_centroid_distance": max_centroid_distance,
            "min_horizontal_iou": min_horizontal_iou,
            "min_3d_iou": min_3d_iou,
            "min_containment": min_containment,
            "up_axis": up_axis,
        },
        "duplicates": [],
        "suppressed_object_ids": [],
    }
    if len(objects) <= 1:
        return objects, meta

    boxes: dict[str, dict[str, Any]] = {}
    for obj in objects:
        indices = obj.source_gaussian_indices
        if indices is None or len(indices) == 0:
            continue
        pts = scene.params.means[indices.to(scene.device)].detach()
        boxes[str(obj.object_id)] = {
            "min": pts.min(dim=0).values.detach().cpu(),
            "max": pts.max(dim=0).values.detach().cpu(),
            "centroid": pts.mean(dim=0).detach().cpu(),
            "count": int(indices.numel()),
            "object": obj,
        }

    suppress_ids: set[str] = set()
    ordered = list(objects)
    for i in range(len(ordered)):
        a = ordered[i]
        aid = str(a.object_id)
        if aid not in boxes:
            continue
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            bid = str(b.object_id)
            if bid not in boxes:
                continue
            ba = boxes[aid]
            bb = boxes[bid]
            centroid_distance = float(torch.norm(ba["centroid"] - bb["centroid"]).item())
            normalized_centroid_distance = _normalized_centroid_distance(ba, bb)
            horizontal_iou = _box_iou(ba["min"][horizontal_axes], ba["max"][horizontal_axes], bb["min"][horizontal_axes], bb["max"][horizontal_axes])
            iou_3d = _box_iou(ba["min"], ba["max"], bb["min"], bb["max"])
            containment = _box_containment(ba["min"], ba["max"], bb["min"], bb["max"])
            count_ratio = min(float(ba["count"]), float(bb["count"])) / max(1.0, max(float(ba["count"]), float(bb["count"])))
            duplicate_score = _duplicate_score(
                horizontal_iou=horizontal_iou,
                iou_3d=iou_3d,
                containment=containment,
                normalized_centroid_distance=normalized_centroid_distance,
            )
            fragment_duplicate = (
                containment >= fragment_containment_threshold
                and count_ratio <= fragment_max_count_ratio
                and normalized_centroid_distance <= 1.0
            )
            if mode == "legacy":
                is_duplicate = (
                    centroid_distance <= max_centroid_distance
                    and (horizontal_iou >= min_horizontal_iou or iou_3d >= min_3d_iou or containment >= min_containment)
                )
            else:
                is_duplicate = duplicate_score >= duplicate_score_threshold or fragment_duplicate
            if not is_duplicate:
                continue
            keep, drop = _choose_duplicate_keep_drop(a, b, ba["count"], bb["count"], prefer_larger)
            record = {
                "object_a": aid,
                "name_a": a.name,
                "count_a": int(ba["count"]),
                "object_b": bid,
                "name_b": b.name,
                "count_b": int(bb["count"]),
                "centroid_distance": centroid_distance,
                "normalized_centroid_distance": normalized_centroid_distance,
                "horizontal_iou": horizontal_iou,
                "iou_3d": iou_3d,
                "containment": containment,
                "count_ratio": count_ratio,
                "duplicate_score": duplicate_score,
                "fragment_duplicate": fragment_duplicate,
                "suggest_keep": str(keep.object_id),
                "suggest_drop": str(drop.object_id),
            }
            meta["duplicates"].append(record)
            if suppress:
                suppress_ids.add(str(drop.object_id))

    if not suppress_ids:
        return objects, meta

    meta["suppressed_object_ids"] = sorted(suppress_ids)
    filtered = []
    for obj in objects:
        if str(obj.object_id) in suppress_ids:
            obj.metadata = dict(obj.metadata)
            obj.metadata["dedup_suppressed"] = True
            continue
        filtered.append(obj)
    return filtered, meta


def _axis_index(axis: str) -> int:
    axis = axis.lower()
    if axis == "x":
        return 0
    if axis == "y":
        return 1
    if axis == "z":
        return 2
    return 1


def _choose_duplicate_keep_drop(
    a: ObjectAsset,
    b: ObjectAsset,
    count_a: int,
    count_b: int,
    prefer_larger: bool,
) -> tuple[ObjectAsset, ObjectAsset]:
    if prefer_larger and count_a != count_b:
        return (a, b) if count_a > count_b else (b, a)
    score_a = float(getattr(a, "best_view_score", 0.0) or 0.0)
    score_b = float(getattr(b, "best_view_score", 0.0) or 0.0)
    if score_a != score_b:
        return (a, b) if score_a > score_b else (b, a)
    return (a, b) if str(a.object_id) <= str(b.object_id) else (b, a)


def _normalized_centroid_distance(box_a: dict[str, Any], box_b: dict[str, Any]) -> float:
    diag_a = torch.norm(torch.clamp(box_a["max"] - box_a["min"], min=0.0))
    diag_b = torch.norm(torch.clamp(box_b["max"] - box_b["min"], min=0.0))
    scale = torch.minimum(diag_a, diag_b).clamp_min(1e-6)
    return float((torch.norm(box_a["centroid"] - box_b["centroid"]) / scale).item())


def _duplicate_score(
    *,
    horizontal_iou: float,
    iou_3d: float,
    containment: float,
    normalized_centroid_distance: float,
) -> float:
    proximity = max(0.0, 1.0 - min(normalized_centroid_distance, 1.0))
    return float(
        0.35 * max(0.0, min(1.0, horizontal_iou))
        + 0.30 * max(0.0, min(1.0, iou_3d))
        + 0.25 * max(0.0, min(1.0, containment))
        + 0.10 * proximity
    )


def _box_iou(min_a: torch.Tensor, max_a: torch.Tensor, min_b: torch.Tensor, max_b: torch.Tensor) -> float:
    inter_min = torch.maximum(min_a, min_b)
    inter_max = torch.minimum(max_a, max_b)
    inter_size = torch.clamp(inter_max - inter_min, min=0.0)
    inter = float(torch.prod(inter_size).item())
    vol_a = float(torch.prod(torch.clamp(max_a - min_a, min=0.0)).item())
    vol_b = float(torch.prod(torch.clamp(max_b - min_b, min=0.0)).item())
    union = vol_a + vol_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _box_containment(min_a: torch.Tensor, max_a: torch.Tensor, min_b: torch.Tensor, max_b: torch.Tensor) -> float:
    inter_min = torch.maximum(min_a, min_b)
    inter_max = torch.minimum(max_a, max_b)
    inter_size = torch.clamp(inter_max - inter_min, min=0.0)
    inter = float(torch.prod(inter_size).item())
    vol_a = float(torch.prod(torch.clamp(max_a - min_a, min=0.0)).item())
    vol_b = float(torch.prod(torch.clamp(max_b - min_b, min=0.0)).item())
    denom = min(vol_a, vol_b)
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _radius_components(points: torch.Tensor, radius: float) -> list[torch.Tensor]:
    n = int(points.shape[0])
    if n == 0:
        return []
    if n == 1 or radius <= 0.0:
        return [torch.arange(n, device=points.device)]
    cell_size = float(radius)
    coords = torch.floor(points / cell_size).to(torch.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for idx, coord in enumerate(coords.detach().cpu().tolist()):
        buckets.setdefault((int(coord[0]), int(coord[1]), int(coord[2])), []).append(idx)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    r2 = float(radius) * float(radius)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for key, ids in buckets.items():
        candidates: list[int] = []
        for dx, dy, dz in offsets:
            candidates.extend(buckets.get((key[0] + dx, key[1] + dy, key[2] + dz), []))
        cand_t = torch.tensor(candidates, dtype=torch.long, device=points.device)
        for idx in ids:
            if cand_t.numel() == 0:
                continue
            d2 = ((points[cand_t] - points[idx]) ** 2).sum(dim=1)
            neighbors = cand_t[d2 <= r2].detach().cpu().tolist()
            adjacency[idx].extend(int(v) for v in neighbors if int(v) != idx)
    visited = [False] * n
    components: list[torch.Tensor] = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            idx = stack.pop()
            component.append(idx)
            for nb in adjacency[idx]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        components.append(torch.tensor(component, dtype=torch.long, device=points.device))
    return components
