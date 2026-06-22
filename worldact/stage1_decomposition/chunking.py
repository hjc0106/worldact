from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.objects import ObjectAsset
from worldact.rendering.camera_planning import parse_up_axis


@dataclass
class SceneChunk:
    chunk_id: str
    core_bounds: tuple[float, float, float, float]
    expanded_bounds: tuple[float, float, float, float]
    indices: torch.Tensor
    core_mask: torch.Tensor

    @property
    def center_xy(self) -> tuple[float, float]:
        x0, x1, y0, y1 = self.core_bounds
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _resolve_plane_axes(up_axis: str | int | None) -> tuple[int, int, int, float]:
    up_idx, up_sign = parse_up_axis(up_axis)
    axes = [0, 1, 2]
    axes.remove(up_idx)
    return axes[0], axes[1], up_idx, up_sign


def _resolve_chunk_up_axis(cfg: Any | None, up_axis: str | int | None = None) -> str:
    if up_axis is not None:
        return str(up_axis)
    if cfg is None:
        return "z"
    cfg_up_axis = getattr(cfg, "up_axis", None)
    return str(cfg_up_axis) if cfg_up_axis is not None else "z"


def build_xy_chunks(scene: GaussianScene, cfg: Any, *, up_axis: str | int | None = None) -> list[SceneChunk]:
    means = scene.params.means.detach().cpu()
    resolved_up_axis = _resolve_chunk_up_axis(cfg, up_axis)
    h0, h1, _, _ = _resolve_plane_axes(resolved_up_axis)
    horizontal = means[:, [h0, h1]]
    q_lo, q_hi = float(cfg.bounds_quantile[0]), float(cfg.bounds_quantile[1])
    lo = torch.quantile(horizontal, q_lo, dim=0)
    hi = torch.quantile(horizontal, q_hi, dim=0)
    x0, y0 = float(lo[0]), float(lo[1])
    x1, y1 = float(hi[0]), float(hi[1])
    x_range = max(x1 - x0, 1e-6)
    y_range = max(y1 - y0, 1e-6)

    desired = max(1, math.ceil(scene.num_gaussians / max(1, int(cfg.target_gaussians_per_chunk))))
    desired = min(desired, max(1, int(cfg.max_chunks)))
    nx = max(1, math.ceil(math.sqrt(desired * x_range / max(y_range, 1e-6))))
    ny = max(1, math.ceil(desired / nx))
    max_size = max(float(cfg.max_chunk_size_xy), 1e-6)
    nx = max(nx, math.ceil(x_range / max_size))
    ny = max(ny, math.ceil(y_range / max_size))
    min_size = float(getattr(cfg, "min_chunk_size_xy", 0.0))
    if 0 < min_size <= max_size:
        nx = min(nx, max(1, math.floor(x_range / min_size)))
        ny = min(ny, max(1, math.floor(y_range / min_size)))
    while nx * ny > int(cfg.max_chunks) and (nx > 1 or ny > 1):
        if nx >= ny and nx > 1:
            nx -= 1
        elif ny > 1:
            ny -= 1
        else:
            break

    step_x = x_range / nx
    step_y = y_range / ny
    overlap = float(cfg.overlap_xy)
    chunks: list[SceneChunk] = []
    chunk_idx = 0
    xs, ys = means[:, h0], means[:, h1]
    for ix in range(nx):
        for iy in range(ny):
            cx0, cx1 = x0 + ix * step_x, x0 + (ix + 1) * step_x
            cy0, cy1 = y0 + iy * step_y, y0 + (iy + 1) * step_y
            ex0, ex1 = cx0 - overlap, cx1 + overlap
            ey0, ey1 = cy0 - overlap, cy1 + overlap
            expanded = (xs >= ex0) & (xs <= ex1) & (ys >= ey0) & (ys <= ey1)
            indices = expanded.nonzero(as_tuple=True)[0]
            if len(indices) < int(cfg.min_gaussians_per_chunk):
                continue
            core = (xs[indices] >= cx0) & (xs[indices] <= cx1) & (ys[indices] >= cy0) & (ys[indices] <= cy1)
            chunks.append(
                SceneChunk(
                    chunk_id=f"chunk_{chunk_idx:03d}",
                    core_bounds=(cx0, cx1, cy0, cy1),
                    expanded_bounds=(ex0, ex1, ey0, ey1),
                    indices=indices,
                    core_mask=core,
                )
            )
            chunk_idx += 1
    return chunks


def chunk_roi_overlap(
    scene: GaussianScene,
    chunk: SceneChunk,
    roi_center: torch.Tensor | list[float],
    roi_radius: float,
    *,
    up_axis: str | int | None = None,
) -> tuple[float, int, int]:
    resolved_up_axis = _resolve_chunk_up_axis(None, up_axis)
    h0, h1, _, _ = _resolve_plane_axes(resolved_up_axis)
    indices = chunk.indices.to(scene.device)
    if bool(chunk.core_mask.any()):
        indices = indices[chunk.core_mask.to(scene.device)]
    if indices.numel() == 0:
        return 0.0, 0, 0
    means = scene.params.means[indices]
    center = torch.as_tensor(roi_center, dtype=means.dtype, device=means.device)
    distances = torch.norm(means[:, [h0, h1]] - center[[h0, h1]], dim=1)
    hits = distances <= float(roi_radius)
    total = int(hits.numel())
    if total == 0:
        return 0.0, 0, 0
    score = float(hits.float().mean().item())
    return score, int(hits.sum().item()), total


def filter_chunks_by_roi(
    scene: GaussianScene,
    chunks: list[SceneChunk],
    roi_center: torch.Tensor | list[float],
    roi_radius: float,
    *,
    up_axis: str | int | None = None,
    min_roi_overlap: float = 0.15,
    keep_top_k: int = 2,
    prompt: str | None = None,
    prompt_guided: bool = False,
    roi_margin: float = 0.0,
) -> tuple[list[SceneChunk], list[dict[str, Any]], dict[str, Any]]:
    manifest = chunk_manifest(chunks)
    scores: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        overlap, hits, total = chunk_roi_overlap(scene, chunk, roi_center, roi_radius, up_axis=up_axis)
        scores.append(
            {
                "index": idx,
                "chunk_id": chunk.chunk_id,
                "roi_overlap": overlap,
                "roi_hits": hits,
                "roi_total": total,
            }
        )

    ranked = sorted(scores, key=lambda item: (item["roi_overlap"], item["roi_hits"], -int(item["index"])), reverse=True)
    selected_indices = [item["index"] for item in ranked if float(item["roi_overlap"]) >= float(min_roi_overlap)]
    fallback_used = False
    fallback_reason = None
    if not selected_indices:
        fallback_used = True
        keep = max(1, int(keep_top_k))
        fallback_reason = "keep_top_k"
        selected_indices = sorted(item["index"] for item in ranked[:keep])

    selected_set = set(selected_indices)
    selected_chunks = [chunks[idx] for idx in selected_indices]
    rank_by_index = {int(item["index"]): rank for rank, item in enumerate(ranked, start=1)}
    roi_center_list = [float(v) for v in torch.as_tensor(roi_center).detach().cpu().tolist()]
    roi_radius_effective = float(roi_radius)
    for idx, item in enumerate(manifest):
        score_item = scores[idx]
        item.update(
            {
                "roi_guided": True,
                "prompt_guided": bool(prompt_guided),
                "prompt": prompt,
                "selected_by_prompt": bool(idx in selected_set),
                "roi_overlap": float(score_item["roi_overlap"]),
                "roi_hits": int(score_item["roi_hits"]),
                "roi_total": int(score_item["roi_total"]),
                "roi_rank": int(rank_by_index.get(idx, len(ranked) + 1)),
                "roi_center": roi_center_list,
                "roi_radius": roi_radius_effective,
                "roi_margin": float(roi_margin),
                "min_roi_overlap": float(min_roi_overlap),
                "keep_top_k": int(keep_top_k),
                "selection_reason": "threshold" if idx in selected_set and not fallback_used else ("fallback_top_k" if idx in selected_set else "rejected"),
            }
        )

    summary = {
        "prompt": prompt,
        "roi_guided": True,
        "prompt_guided": bool(prompt_guided),
        "roi_center": roi_center_list,
        "roi_radius": roi_radius_effective,
        "roi_margin": float(roi_margin),
        "min_roi_overlap": float(min_roi_overlap),
        "keep_top_k": int(keep_top_k),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "num_chunks": len(chunks),
        "num_selected": len(selected_chunks),
        "selected_chunk_ids": [chunk.chunk_id for chunk in selected_chunks],
        "selected_indices": selected_indices,
        "ranked_chunk_ids": [item["chunk_id"] for item in ranked],
        "selected_scores": {
            item["chunk_id"]: {
                "roi_overlap": float(item["roi_overlap"]),
                "roi_hits": int(item["roi_hits"]),
                "roi_total": int(item["roi_total"]),
            }
            for item in ranked
        },
    }
    return selected_chunks, manifest, summary


def make_chunk_scene(scene: GaussianScene, chunk: SceneChunk) -> GaussianScene:
    local = scene.subset(chunk.indices.to(scene.device))
    local.metadata["global_indices"] = chunk.indices.cpu()
    local.metadata["chunk_id"] = chunk.chunk_id
    local.metadata["core_bounds"] = chunk.core_bounds
    local.metadata["expanded_bounds"] = chunk.expanded_bounds
    return local


def chunk_manifest(chunks: list[SceneChunk]) -> list[dict]:
    records = []
    for chunk in chunks:
        records.append(
            {
                "chunk_id": chunk.chunk_id,
                "core_bounds_xy": list(chunk.core_bounds),
                "expanded_bounds_xy": list(chunk.expanded_bounds),
                "num_gaussians": int(len(chunk.indices)),
                "num_core_gaussians": int(chunk.core_mask.sum().item()),
            }
        )
    return records


def normalize_name(name: str) -> str:
    words = re.findall(r"[a-z0-9]+", name.lower())
    return " ".join(words)


def object_centroid(scene: GaussianScene, indices: torch.Tensor) -> torch.Tensor:
    if len(indices) == 0:
        return torch.zeros(3)
    return scene.params.means[indices.to(scene.device)].detach().cpu().mean(dim=0)


def merge_chunk_objects(scene: GaussianScene, candidates: list[ObjectAsset], cfg: Any) -> list[ObjectAsset]:
    merged: list[ObjectAsset] = []
    merged_sets: list[set[int]] = []
    merged_centroids: list[torch.Tensor] = []

    for cand in candidates:
        if cand.source_gaussian_indices is None or len(cand.source_gaussian_indices) == 0:
            continue
        cand_indices = cand.source_gaussian_indices.unique().cpu()
        cand_set = set(int(i) for i in cand_indices.tolist())
        cand_name = normalize_name(cand.name)
        cand_centroid = object_centroid(scene, cand_indices)
        target = -1
        for idx, obj in enumerate(merged):
            if normalize_name(obj.name) != cand_name:
                continue
            overlap = len(cand_set & merged_sets[idx]) / max(1, min(len(cand_set), len(merged_sets[idx])))
            dist = float(torch.linalg.norm(cand_centroid - merged_centroids[idx]).item())
            if overlap >= float(cfg.merge_iou_threshold) or dist <= float(cfg.merge_centroid_distance):
                target = idx
                break
        if target < 0:
            cand.source_gaussian_indices = cand_indices
            cand.metadata.setdefault("chunk_ids", [])
            merged.append(cand)
            merged_sets.append(cand_set)
            merged_centroids.append(cand_centroid)
        else:
            merged_sets[target] |= cand_set
            obj = merged[target]
            obj.source_gaussian_indices = torch.tensor(sorted(merged_sets[target]), dtype=torch.long)
            chunk_ids = obj.metadata.setdefault("chunk_ids", [])
            chunk_ids.extend(cand.metadata.get("chunk_ids", []))
            obj.metadata["chunk_ids"] = sorted(set(chunk_ids))
            merged_centroids[target] = object_centroid(scene, obj.source_gaussian_indices)

    used: set[int] = set()
    final: list[ObjectAsset] = []
    def num_indices(obj: ObjectAsset) -> int:
        return 0 if obj.source_gaussian_indices is None else int(len(obj.source_gaussian_indices))

    for obj in sorted(merged, key=num_indices, reverse=True):
        indices = [int(i) for i in obj.source_gaussian_indices.tolist() if int(i) not in used]
        if not indices:
            continue
        used.update(indices)
        obj.object_id = str(len(final))
        obj.source_gaussian_indices = torch.tensor(indices, dtype=torch.long)
        final.append(obj)
    return final
