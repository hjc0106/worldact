"""Scene-level and object-level upright-axis estimation for Stage2 trajectories.

Priority (most reliable first):
  1. Explicit manual ``up_axis``
  2. Scene ground plane from Stage-1 background (shared gravity for all objects)
  3. Robust PCA + support-surface scoring on the object PLY
  4. Camera / anchor image-up hint
  5. World +Z fallback (low confidence)

Normals transform with ``up_local = normalize(A.T @ up_world)`` where ``A`` is the
linear part of ``local_to_world``. Uniform scale+translation leaves direction unchanged.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class UpAxisEstimate:
    up_axis: list[float]
    source: str
    confidence: float
    eigenvalue_gap: float | None = None
    sign_source: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)

    def as_unit_vector(self) -> np.ndarray:
        v = np.asarray(self.up_axis, dtype=np.float64).reshape(3)
        n = np.linalg.norm(v)
        if n < 1e-12:
            raise ValueError("zero up_axis")
        return v / n

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_up_axis_json(path: Path | str, estimate: UpAxisEstimate) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(estimate.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def world_normal_to_local(up_world: np.ndarray, local_to_world: np.ndarray | None) -> np.ndarray:
    """Convert a world-space plane normal into object-local coordinates.

    For ``x_world = A @ x_local + t``, normals transform as ``n_local ∝ A.T @ n_world``.
    """
    up = np.asarray(up_world, dtype=np.float64).reshape(3)
    if local_to_world is None:
        return _normalize(up)
    A = np.asarray(local_to_world, dtype=np.float64)[:3, :3]
    return _normalize(A.T @ up)


def estimate_up_axis(
    *,
    stage_dir: Path | str | None = None,
    points_local: np.ndarray | None = None,
    opacities: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    camera_centers_world: np.ndarray | None = None,
    camera_centers_local: np.ndarray | None = None,
    local_to_world: np.ndarray | None = None,
    manual_up: np.ndarray | list[float] | None = None,
    camera_up_hint: np.ndarray | None = None,
    min_confidence: float = 0.45,
    prefer_vehicle_pca: bool = True,
) -> UpAxisEstimate:
    """Estimate a unit upright axis in object-local coordinates when possible."""
    if manual_up is not None:
        up = _normalize(np.asarray(manual_up, dtype=np.float64).reshape(3))
        return UpAxisEstimate(
            up_axis=up.tolist(),
            source="manual",
            confidence=1.0,
            sign_source="manual",
            details={},
        )

    # 1) Scene ground (world) → local.
    if stage_dir is not None:
        try:
            ground = estimate_scene_ground_up(Path(stage_dir), camera_centers_world=camera_centers_world)
            up_local = world_normal_to_local(ground.as_unit_vector(), local_to_world)
            # Optional PCA check for elongated objects.
            pca_check: dict[str, Any] = {}
            if points_local is not None and prefer_vehicle_pca:
                pca = estimate_robust_pca_up(
                    points_local,
                    opacities=opacities,
                    scales=scales,
                    camera_centers=camera_centers_local,
                    camera_up_hint=camera_up_hint,
                )
                gap = float(pca.eigenvalue_gap or 0.0)
                align = abs(float(np.dot(up_local, pca.as_unit_vector())))
                pca_check = {
                    "pca_up": pca.up_axis,
                    "pca_confidence": pca.confidence,
                    "pca_eigenvalue_gap": gap,
                    "alignment_with_ground": align,
                }
                # If PCA is very confident and disagrees strongly, lower confidence.
                if gap >= 0.5 and pca.confidence >= 0.7 and align < 0.7:
                    ground = UpAxisEstimate(
                        up_axis=up_local.tolist(),
                        source="scene_ground",
                        confidence=min(ground.confidence, 0.55),
                        eigenvalue_gap=gap,
                        sign_source=ground.sign_source,
                        details={**ground.details, **pca_check, "warning": "pca_disagrees"},
                    )
                    return ground
            return UpAxisEstimate(
                up_axis=up_local.tolist(),
                source="scene_ground",
                confidence=float(ground.confidence),
                eigenvalue_gap=ground.eigenvalue_gap,
                sign_source=ground.sign_source,
                details={**ground.details, **pca_check},
            )
        except Exception as exc:  # noqa: BLE001 — fall through to PCA
            ground_error = str(exc)
    else:
        ground_error = None

    # 2) Robust PCA + support scoring on the object.
    pca_estimate: UpAxisEstimate | None = None
    if points_local is not None:
        pca_estimate = estimate_robust_pca_up(
            points_local,
            opacities=opacities,
            scales=scales,
            camera_centers=camera_centers_local,
            camera_up_hint=camera_up_hint,
        )
        if ground_error:
            pca_estimate.details["scene_ground_error"] = ground_error
        if pca_estimate.confidence >= min_confidence:
            return pca_estimate

    # 3) Camera / anchor hint (when PCA is missing or low-confidence).
    if camera_up_hint is not None and float(np.linalg.norm(camera_up_hint)) > 1e-8:
        up = _normalize(np.asarray(camera_up_hint, dtype=np.float64).reshape(3))
        details: dict[str, Any] = {"scene_ground_error": ground_error}
        if pca_estimate is not None:
            details["pca_fallback"] = pca_estimate.to_dict()
        return UpAxisEstimate(
            up_axis=up.tolist(),
            source="camera_hint",
            confidence=0.4,
            sign_source="camera_hint",
            details=details,
        )

    if pca_estimate is not None:
        # Keep low-confidence PCA; callers may refuse to auto-orbit.
        return pca_estimate

    return UpAxisEstimate(
        up_axis=[0.0, 0.0, 1.0],
        source="fallback_z",
        confidence=0.1,
        sign_source="default",
        details={"scene_ground_error": ground_error},
    )


def estimate_scene_ground_up(
    stage_dir: Path,
    *,
    camera_centers_world: np.ndarray | None = None,
    max_points: int = 200_000,
    ransac_iterations: int = 500,
    inlier_threshold: float | None = None,
    rng: np.random.Generator | None = None,
) -> UpAxisEstimate:
    """Fit a shared gravity direction from Stage-1 background Gaussians."""
    stage_dir = Path(stage_dir)
    ply_path = stage_dir / "background" / "gaussians.ply"
    cams_path = stage_dir / "background" / "cameras.json"
    if not ply_path.is_file():
        raise FileNotFoundError(f"missing background PLY: {ply_path}")

    means, opacities, scales = _load_gaussian_xyz_opacity_scales(ply_path)
    # Prefer large, opaque Gaussians (road/pavement tends to be broad and opaque).
    opacity = 1.0 / (1.0 + np.exp(-opacities)) if opacities is not None else np.ones(len(means))
    max_scale = np.max(np.exp(scales), axis=1) if scales is not None else np.ones(len(means))
    keep = (opacity >= np.quantile(opacity, 0.35)) & (max_scale >= np.quantile(max_scale, 0.40))
    if camera_centers_world is None and cams_path.is_file():
        camera_centers_world = _load_camera_centers(cams_path)
    # Prefer lower points as ground candidates using mean camera image-up.
    coarse_up = _estimate_camera_image_up(cams_path) if cams_path.is_file() else None
    if coarse_up is not None:
        heights = means @ coarse_up
        lo, hi = np.quantile(heights[keep], [0.02, 0.55])
        keep &= (heights >= lo) & (heights <= hi)

    pts = means[keep]
    if len(pts) > max_points:
        generator = rng or np.random.default_rng(0)
        idx = generator.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]

    # Adaptive threshold from scene scale.
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    thr = float(inlier_threshold) if inlier_threshold is not None else max(0.05, 0.002 * extent)
    normal, distance, inliers = fit_plane_ransac(
        pts,
        num_iterations=ransac_iterations,
        inlier_threshold=thr,
        min_inliers=max(50, len(pts) // 50),
        rng=rng,
    )
    normal, distance = refine_plane_huber(pts[inliers], normal, distance)

    sign_source = "plane_default"
    if camera_centers_world is not None and len(camera_centers_world) > 0:
        # Cameras should lie on the +up side of the ground plane.
        side = float(np.mean(camera_centers_world @ normal + distance))
        if side < 0:
            normal = -normal
            distance = -distance
        sign_source = "camera_centers"

    inlier_ratio = float(np.mean(inliers))
    residual = np.abs(pts[inliers] @ normal + distance)
    mad = float(np.median(np.abs(residual - np.median(residual)))) + 1e-9
    # Confidence: inlier ratio + residual tightness.
    confidence = float(
        np.clip(0.35 * inlier_ratio + 0.65 * math.exp(-mad / max(thr, 1e-3)), 0.0, 0.99)
    )
    return UpAxisEstimate(
        up_axis=_normalize(normal).tolist(),
        source="scene_ground",
        confidence=confidence,
        eigenvalue_gap=None,
        sign_source=sign_source,
        details={
            "inlier_ratio": inlier_ratio,
            "plane_distance": float(distance),
            "residual_mad": mad,
            "num_candidate_points": int(len(pts)),
            "inlier_threshold": thr,
            "background_ply": str(ply_path),
        },
    )


def estimate_robust_pca_up(
    points: np.ndarray,
    *,
    opacities: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    camera_centers: np.ndarray | None = None,
    camera_up_hint: np.ndarray | None = None,
) -> UpAxisEstimate:
    """Robust PCA upright estimate with support-surface / camera-hemisphere scoring."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 16:
        raise ValueError("need >=16 points for robust PCA up estimation")

    keep = _filter_object_points(pts, opacities=opacities, scales=scales)
    filtered = pts[keep]
    center, cov = robust_covariance_irls(filtered)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)  # ascending: height?, width?, length?
    evals = evals[order]
    evecs = evecs[:, order]
    gap = float((evals[1] - evals[0]) / max(evals[-1], 1e-9))

    best_score = -1e9
    best_axis = evecs[:, 0].copy()
    best_meta: dict[str, Any] = {}
    for axis_idx in range(3):
        axis = evecs[:, axis_idx]
        for sign in (1.0, -1.0):
            cand = axis * sign
            support = score_support_surface(filtered, cand)
            cam_score = score_camera_hemisphere(camera_centers, cand) if camera_centers is not None else 0.0
            hint_score = 0.0
            if camera_up_hint is not None:
                hint = np.asarray(camera_up_hint, dtype=np.float64).reshape(3)
                hn = np.linalg.norm(hint)
                if hn > 1e-8:
                    hint_score = float(np.dot(cand, hint / hn))
            # Prefer smallest eigenvalue for vehicle-like shapes, but allow others
            # when support evidence is stronger (poles/trees).
            eig_bonus = 0.25 * (1.0 - axis_idx / 2.0)
            score = (
                0.45 * support["score"]
                + 0.25 * cam_score
                + 0.15 * max(0.0, hint_score)
                + 0.15 * eig_bonus
            )
            if score > best_score:
                best_score = score
                best_axis = cand
                best_meta = {
                    "axis_index": int(axis_idx),
                    "eigenvalues": evals.tolist(),
                    "support": support,
                    "camera_hemisphere_score": cam_score,
                    "hint_score": hint_score,
                    "total_score": score,
                    "num_points_filtered": int(len(filtered)),
                }

    # Confidence combines eigenvalue gap (vehicle-like) and support score.
    support_score = float(best_meta.get("support", {}).get("score", 0.0))
    confidence = float(np.clip(0.4 * min(gap / 0.5, 1.0) + 0.6 * support_score, 0.05, 0.95))
    sign_source = "support_surface"
    if camera_centers is not None and float(best_meta.get("camera_hemisphere_score", 0.0)) > 0.2:
        sign_source = "camera_centers"
    elif camera_up_hint is not None and float(best_meta.get("hint_score", 0.0)) > 0.5:
        sign_source = "camera_hint"

    return UpAxisEstimate(
        up_axis=_normalize(best_axis).tolist(),
        source="robust_pca",
        confidence=confidence,
        eigenvalue_gap=gap,
        sign_source=sign_source,
        details=best_meta,
    )


def score_support_surface(points: np.ndarray, up: np.ndarray) -> dict[str, float]:
    """Score whether ``up`` yields a flat support under the object."""
    up = _normalize(np.asarray(up, dtype=np.float64).reshape(3))
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center = pts.mean(axis=0)
    heights = (pts - center) @ up
    # Bottom slice.
    lo = np.quantile(heights, 0.08)
    hi = np.quantile(heights, 0.22)
    bottom = pts[(heights >= lo) & (heights <= hi)]
    if len(bottom) < 12:
        return {"score": 0.0, "mad": 1e3, "centroid_above": 0.0, "num_bottom": float(len(bottom))}

    normal, dist, _ = fit_plane_ransac(
        bottom,
        num_iterations=200,
        inlier_threshold=max(0.01, 0.05 * float(np.std(heights) + 1e-6)),
        min_inliers=max(8, len(bottom) // 5),
    )
    if float(np.dot(normal, up)) < 0:
        normal = -normal
        dist = -dist
    residual = np.abs(bottom @ normal + dist)
    mad = float(np.median(np.abs(residual - np.median(residual)))) + 1e-9
    align = abs(float(np.dot(normal, up)))
    centroid_above = float((center @ normal + dist) > 0)
    # Flat + aligned + centroid above → high score.
    score = float(
        np.clip(
            0.45 * align
            + 0.40 * math.exp(-mad / (0.02 + 0.05 * float(np.std(heights) + 1e-6)))
            + 0.15 * centroid_above,
            0.0,
            1.0,
        )
    )
    return {
        "score": score,
        "mad": mad,
        "align": align,
        "centroid_above": centroid_above,
        "num_bottom": float(len(bottom)),
    }


def score_camera_hemisphere(camera_centers: np.ndarray | None, up: np.ndarray) -> float:
    if camera_centers is None or len(camera_centers) == 0:
        return 0.0
    cams = np.asarray(camera_centers, dtype=np.float64).reshape(-1, 3)
    up = _normalize(np.asarray(up, dtype=np.float64).reshape(3))
    # Relative to object centroid approximation = mean of cameras projected? Use mean of points via cams mean offset.
    # Here we only know camera centers; score fraction with positive up component relative to their mean.
    origin = cams.mean(axis=0)
    dots = (cams - origin) @ up
    return float(np.clip(np.mean(dots > 0.0), 0.0, 1.0))


def fit_plane_ransac(
    points: np.ndarray,
    num_iterations: int = 400,
    inlier_threshold: float = 0.02,
    min_inliers: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    if len(points) < 3:
        raise ValueError("Need at least 3 points for plane fitting")
    generator = rng or np.random.default_rng(0)
    best_inliers: np.ndarray | None = None
    best_normal: np.ndarray | None = None
    best_distance = 0.0
    for _ in range(num_iterations):
        sample_idx = generator.choice(len(points), size=3, replace=False)
        p0, p1, p2 = points[sample_idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        distance = -float(np.dot(normal, p0))
        signed = points @ normal + distance
        inliers = np.abs(signed) < inlier_threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_normal = normal
            best_distance = distance
    if best_inliers is None or best_normal is None or int(best_inliers.sum()) < min(3, min_inliers):
        raise ValueError("RANSAC failed to find a stable ground plane")
    # SVD refine.
    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    refined = vh[-1]
    if float(np.dot(refined, best_normal)) < 0:
        refined = -refined
    refined_distance = -float(np.dot(refined, centroid))
    return refined.astype(np.float64), refined_distance, best_inliers


def refine_plane_huber(
    points: np.ndarray,
    normal: np.ndarray,
    distance: float,
    *,
    delta: float | None = None,
    iters: int = 8,
) -> tuple[np.ndarray, float]:
    """IRLS Huber refinement of a plane (n, d) with n unit and n·x + d = 0."""
    n = _normalize(np.asarray(normal, dtype=np.float64))
    d = float(distance)
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return n, d
    for _ in range(iters):
        residual = pts @ n + d
        scale = float(np.median(np.abs(residual))) + 1e-6
        thr = float(delta) if delta is not None else (1.345 * scale)
        w = np.ones(len(pts), dtype=np.float64)
        mask = np.abs(residual) > thr
        w[mask] = thr / np.abs(residual[mask])
        # Weighted centroid + SVD.
        w_sum = float(w.sum()) + 1e-12
        centroid = (w[:, None] * pts).sum(axis=0) / w_sum
        centered = pts - centroid
        cov = (centered * w[:, None]).T @ centered / w_sum
        evals, evecs = np.linalg.eigh(cov)
        n_new = evecs[:, 0]
        if float(np.dot(n_new, n)) < 0:
            n_new = -n_new
        n = _normalize(n_new)
        d = -float(np.dot(n, centroid))
    return n, d


def robust_covariance_irls(
    points: np.ndarray,
    *,
    iters: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Huber IRLS mean + covariance."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center = np.median(pts, axis=0)
    cov = np.cov(pts.T) + np.eye(3) * 1e-8
    for _ in range(iters):
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(cov)
        diff = pts - center
        mahal = np.sqrt(np.sum(diff @ inv * diff, axis=1) + 1e-12)
        med = float(np.median(mahal)) + 1e-6
        thr = 1.5 * med
        w = np.ones(len(pts), dtype=np.float64)
        mask = mahal > thr
        w[mask] = thr / mahal[mask]
        w_sum = float(w.sum()) + 1e-12
        center = (w[:, None] * pts).sum(axis=0) / w_sum
        centered = pts - center
        cov = (centered * w[:, None]).T @ centered / w_sum + np.eye(3) * 1e-8
    return center, cov


def _filter_object_points(
    points: np.ndarray,
    *,
    opacities: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    center = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    lo, hi = np.quantile(dist, [0.02, 0.98])
    keep = (dist >= lo) & (dist <= hi)
    if opacities is not None:
        opacity = 1.0 / (1.0 + np.exp(-np.asarray(opacities, dtype=np.float64).reshape(-1)))
        keep &= opacity >= max(0.05, float(np.quantile(opacity, 0.15)))
    if scales is not None:
        sc = np.exp(np.asarray(scales, dtype=np.float64).reshape(-1, 3))
        max_s = sc.max(axis=1)
        keep &= max_s <= float(np.quantile(max_s, 0.98))
    # kNN density: drop isolated points via batched pairwise distances on a cap.
    if int(keep.sum()) >= 32:
        idx_all = np.where(keep)[0]
        sub = pts[idx_all]
        if len(sub) > 4000:
            # Subsample for density estimate, then map thresholds back.
            pick = np.linspace(0, len(sub) - 1, 4000, dtype=int)
            dens_ref = _knn_kth_distance(sub[pick], k=8)
            thr = float(np.quantile(dens_ref, 0.92))
            dens = _knn_kth_distance(sub, k=8, ref_points=sub[pick])
        else:
            dens = _knn_kth_distance(sub, k=8)
            thr = float(np.quantile(dens, 0.92))
        dens_keep = dens <= thr
        mask = np.zeros(len(pts), dtype=bool)
        mask[idx_all[dens_keep]] = True
        keep = mask
    if int(keep.sum()) < 16:
        return np.ones(len(pts), dtype=bool)
    return keep


def _knn_kth_distance(
    points: np.ndarray,
    *,
    k: int = 8,
    ref_points: np.ndarray | None = None,
    batch: int = 512,
) -> np.ndarray:
    """Approximate k-th neighbour distance (pure NumPy)."""
    pts = np.asarray(points, dtype=np.float64)
    ref = pts if ref_points is None else np.asarray(ref_points, dtype=np.float64)
    exclude_self = ref_points is None
    k_eff = min(k, max(1, len(ref) - (1 if exclude_self else 0)))
    out = np.empty(len(pts), dtype=np.float64)
    for start in range(0, len(pts), batch):
        chunk = pts[start : start + batch]
        d2 = ((chunk[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
        if exclude_self:
            for i, gi in enumerate(range(start, start + len(chunk))):
                d2[i, gi] = np.inf
        part = np.partition(d2, k_eff - 1, axis=1)[:, k_eff - 1]
        out[start : start + len(chunk)] = np.sqrt(part)
    return out


def _estimate_camera_image_up(cameras_json: Path) -> np.ndarray | None:
    """Mean OpenCV image-up direction in world coords (``-c2w[:3, 1]``)."""
    raw = json.loads(Path(cameras_json).read_text(encoding="utf-8"))
    cams = raw if isinstance(raw, list) else raw.get("cameras") or raw.get("frames") or []
    ups: list[np.ndarray] = []
    for cam in cams:
        ext = np.asarray(cam.get("extrinsic") or cam.get("w2c"), dtype=np.float64)
        if ext.shape == (4, 4):
            c2w = np.linalg.inv(ext)
            ups.append(-c2w[:3, 1])
            continue
        mat = np.asarray(cam.get("transform_matrix") or cam.get("c2w"), dtype=np.float64)
        if mat.shape == (4, 4):
            ups.append(-mat[:3, 1])
    if not ups:
        return None
    return _normalize(np.mean(np.stack(ups, axis=0), axis=0))


def _load_gaussian_xyz_opacity_scales(
    ply_path: Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load means / opacity logits / log-scales from a Gaussian PLY.

    Uses ``plyfile`` when available, otherwise a minimal binary/ascii PLY reader.
    """
    try:
        from plyfile import PlyData

        ply = PlyData.read(str(ply_path))
        vertex = ply["vertex"]
        means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
        opacities = None
        scales = None
        names = set(vertex.data.dtype.names or [])
        if "opacity" in names:
            opacities = np.asarray(vertex["opacity"], dtype=np.float64)
        if {"scale_0", "scale_1", "scale_2"} <= names:
            scales = np.stack(
                [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
            ).astype(np.float64)
        return means, opacities, scales
    except ImportError:
        return _load_gaussian_xyz_opacity_scales_numpy(ply_path)


def _load_gaussian_xyz_opacity_scales_numpy(
    ply_path: Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Minimal PLY loader for 3DGS-style vertex properties."""
    type_map = {
        "char": "i1",
        "uchar": "u1",
        "short": "i2",
        "ushort": "u2",
        "int": "i4",
        "uint": "u4",
        "float": "f4",
        "double": "f8",
        "float32": "f4",
        "float64": "f8",
        "int32": "i4",
        "uint8": "u1",
    }
    fmt = None
    vertex_count = None
    props: list[tuple[str, str]] = []
    in_vertex = False
    header_size = 0
    with open(ply_path, "rb") as f:
        while True:
            raw = f.readline()
            header_size += len(raw)
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element "):
                parts = line.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                    props = []
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) >= 3 and parts[1] != "list":
                    props.append((parts[-1], parts[1]))
            elif line == "end_header":
                break
        if vertex_count is None or not props or fmt is None:
            raise ValueError(f"unsupported PLY header: {ply_path}")
        names = [n for n, _ in props]
        if not {"x", "y", "z"} <= set(names):
            raise ValueError(f"PLY missing xyz: {ply_path}")
        if fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                rows.append(f.readline().decode("ascii", errors="replace").split())
            data = np.asarray(rows, dtype=np.float64)
            col = {n: i for i, n in enumerate(names)}
            means = data[:, [col["x"], col["y"], col["z"]]]
            opacities = data[:, col["opacity"]] if "opacity" in col else None
            if {"scale_0", "scale_1", "scale_2"} <= set(col):
                scales = data[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]
            else:
                scales = None
            return means, opacities, scales

        endian = "<" if "little" in fmt else ">"
        dtype = np.dtype([(n, endian + type_map[t]) for n, t in props])
        body = f.read(vertex_count * dtype.itemsize)
        arr = np.frombuffer(body, dtype=dtype, count=vertex_count)
        means = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
        opacities = np.asarray(arr["opacity"], dtype=np.float64) if "opacity" in arr.dtype.names else None
        if {"scale_0", "scale_1", "scale_2"} <= set(arr.dtype.names or []):
            scales = np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], axis=1).astype(np.float64)
        else:
            scales = None
        return means, opacities, scales


def _load_camera_centers(cameras_json: Path) -> np.ndarray:
    raw = json.loads(Path(cameras_json).read_text(encoding="utf-8"))
    cams = raw if isinstance(raw, list) else raw.get("cameras") or raw.get("frames") or []
    centers = []
    for cam in cams:
        ext = np.asarray(cam.get("extrinsic") or cam.get("w2c"), dtype=np.float64)
        if ext.shape == (4, 4):
            c2w = np.linalg.inv(ext)
            centers.append(c2w[:3, 3])
            continue
        # Already c2w?
        mat = np.asarray(cam.get("transform_matrix") or cam.get("c2w"), dtype=np.float64)
        if mat.shape == (4, 4):
            centers.append(mat[:3, 3])
    if not centers:
        raise ValueError(f"no camera centers in {cameras_json}")
    return np.asarray(centers, dtype=np.float64)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalize near-zero vector")
    return v / n
