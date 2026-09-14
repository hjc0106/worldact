from __future__ import annotations

import math
import os
import shutil
import sys
from pathlib import Path

import torch

from worldact.data.gaussians import GaussianScene, RenderedView
from worldact.data.poses import CameraPose

SH_C0 = 0.28209479177387814


def _ensure_ninja_on_path() -> None:
    """Expose Ninja installed beside the active Python to PyTorch JIT builds."""
    if shutil.which("ninja") is not None:
        return
    python_bin = Path(sys.executable).resolve().parent
    ninja = python_bin / "ninja"
    if not ninja.is_file():
        return
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(python_bin) + (os.pathsep + current_path if current_path else "")


def _prepared_params(scene: GaussianScene) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
    p = scene.params
    scales = torch.exp(p.scales)
    quats = torch.nn.functional.normalize(p.rotations, dim=-1)
    opacities = torch.sigmoid(p.opacities.squeeze(-1))
    colors = p.colors
    num_channels = colors.shape[1]
    sh_degree = None

    if num_channels < 3:
        colors = torch.nn.functional.pad(colors, (0, 3 - colors.shape[1]))
    else:
        num_bases = num_channels // 3
        degree = int(math.isqrt(num_bases)) - 1
        uses_full_sh = num_channels % 3 == 0 and degree >= 0 and (degree + 1) ** 2 == num_bases
        if uses_full_sh:
            if num_bases == 1:
                if colors.min() >= 0.0 and colors.max() <= 1.0:
                    colors = colors[:, :3]
                else:
                    sh_degree = degree
                    colors = colors.reshape(-1, 1, 3)
            else:
                sh_degree = degree
                # Standard 3DGS PLY stores f_dc as RGB, then f_rest channel-major:
                # R coefficients, then G coefficients, then B coefficients. gsplat
                # expects SH coefficients grouped per basis as [N, basis, RGB].
                sh_colors = colors.new_empty(colors.shape[0], num_bases, 3)
                sh_colors[:, 0, :] = colors[:, :3]
                sh_colors[:, 1:, :] = colors[:, 3:].reshape(colors.shape[0], 3, num_bases - 1).transpose(1, 2)
                colors = sh_colors
        else:
            # If the SH layout is incomplete, use the DC term as an RGB approximation
            # instead of treating raw SH coefficients as display colors.
            colors = colors[:, :3]
            if colors.min() < 0.0 or colors.max() > 1.0:
                colors = (colors * SH_C0 + 0.5).clamp(0.0, 1.0)
    return p.means, quats, scales, opacities, colors, sh_degree


def hits_from_alpha(alpha: torch.Tensor, eps: float) -> torch.Tensor:
    """Invert front-to-back alpha compositing at uniform opacity eps to recover a hit count.

    If N Gaussians of opacity eps each contribute to a pixel, standard alpha
    compositing gives accumulated alpha = 1-(1-eps)^N. Solving for N:
    N = log1p(-alpha) / log1p(-eps).
    """
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    alpha = alpha.clamp(0.0, 1.0 - 1e-6)
    hits = torch.log1p(-alpha) / math.log1p(-eps)
    return hits.clamp_min(0.0)


def rasterize_hit_count(scene: GaussianScene, camera: CameraPose, *, eps: float = 0.01) -> torch.Tensor:
    """Approximate per-pixel count of Gaussians composited into this pixel.

    Re-renders with opacities forced to a tiny uniform eps and colors forced
    to 1, so accumulated alpha ~= 1-(1-eps)^N; inverting (see hits_from_alpha)
    recovers N, giving a gsplat-native analogue of 3DGRUT's hits_count without
    touching any CUDA kernel.

    eps must stay above gsplat's internal per-Gaussian alpha-contribution cutoff
    (1/255 ~= 0.00392, hardcoded in its CUDA rasterize_to_pixels kernels) or every
    Gaussian is culled at every pixel and the whole render comes back exactly zero.
    """
    _ensure_ninja_on_path()
    try:
        from gsplat import rasterization
    except ImportError as exc:
        raise ImportError("gsplat is required for rendering in scene decomposition") from exc

    means, quats, scales, _, colors, _ = _prepared_params(scene)
    device = means.device
    n = means.shape[0]
    opacities_eps = torch.full((n,), eps, device=device, dtype=means.dtype)
    colors_ones = torch.ones(n, 3, device=device, dtype=means.dtype)
    _, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities_eps,
        colors=colors_ones,
        viewmats=camera.extrinsic.to(device).unsqueeze(0),
        Ks=camera.intrinsic.to(device).unsqueeze(0),
        width=camera.width,
        height=camera.height,
        backgrounds=torch.zeros(1, 3, device=device, dtype=means.dtype),
        render_mode="RGB",
        packed=False,
        sh_degree=None,
    )
    alpha = render_alphas[0, ..., 0]
    saturated_frac = float((alpha > 0.99).float().mean().item())
    if saturated_frac > 0.05:
        import logging

        logging.getLogger(__name__).warning(
            "rasterize_hit_count: %.1f%% of pixels saturated at eps=%g; hit-count estimate is unreliable there "
            "(consider a smaller eps, but keep it above gsplat's ~1/255 per-Gaussian alpha cutoff)",
            saturated_frac * 100.0,
            eps,
        )
    return hits_from_alpha(alpha, eps)


def visible_hit_count_from_pixel_ids(
    pixel_ids: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Count accepted gsplat contributors for every image pixel.

    ``pixel_ids`` is the flattened pixel-index output of
    ``gsplat.rasterize_to_indices_in_range``. Each entry corresponds to one
    Gaussian that survives gsplat's per-pixel alpha cutoff, depth ordering,
    and transmittance early termination. The returned integer map therefore
    represents actual visible contributors rather than Gaussian kernel mass.
    """
    if pixel_ids.ndim != 1:
        raise ValueError(f"pixel_ids must be one-dimensional, got shape {tuple(pixel_ids.shape)}")
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {(height, width)}")

    num_pixels = height * width
    if pixel_ids.numel() == 0:
        return torch.zeros((height, width), dtype=torch.int32, device=pixel_ids.device)
    if bool((pixel_ids < 0).any()) or bool((pixel_ids >= num_pixels).any()):
        raise ValueError(f"pixel_ids contains values outside [0, {num_pixels})")

    counts = torch.bincount(pixel_ids.to(torch.int64), minlength=num_pixels)
    return counts.reshape(height, width).to(torch.int32)
def effective_hit_count_from_weights(
    pixel_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Compute participation-ratio effective count from per-contributor weights."""
    if pixel_ids.ndim != 1 or weights.ndim != 1:
        raise ValueError("pixel_ids and weights must be one-dimensional")
    if pixel_ids.numel() != weights.numel():
        raise ValueError("pixel_ids and weights must have the same number of elements")
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {(height, width)}")
    num_pixels = height * width
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("weights must be finite and non-negative")

    if pixel_ids.numel() == 0:
        return torch.zeros((height, width), dtype=weights.dtype, device=weights.device)
    if bool((pixel_ids < 0).any()) or bool((pixel_ids >= num_pixels).any()):
        raise ValueError(f"pixel_ids contains values outside [0, {num_pixels})")
    ray_ids = pixel_ids.to(torch.int64)
    sum_w = torch.bincount(ray_ids, weights=weights, minlength=num_pixels)
    sum_w2 = torch.bincount(ray_ids, weights=weights.square(), minlength=num_pixels)
    effective = sum_w.square() / sum_w2.clamp_min(1e-8)
    return effective.reshape(height, width).clamp_min(0.0)




def rasterize_visible_hit_count(scene: GaussianScene, camera: CameraPose) -> torch.Tensor:
    """Return the number of Gaussians actually composited into each pixel.

    Unlike :func:`rasterize_hit_count`, this uses the scene's real opacities
    and asks gsplat for the exact contributor indices selected by its normal
    front-to-back rasterizer. Consequently it follows the same ``1 / 255``
    per-contribution alpha cutoff and ``T <= 1e-4`` early termination as the
    RGB/alpha render.
    """
    _ensure_ninja_on_path()
    try:
        from gsplat import rasterization, rasterize_to_indices_in_range
    except ImportError as exc:
        raise ImportError("gsplat is required for rendering in scene decomposition") from exc

    means, quats, scales, opacities, _, _ = _prepared_params(scene)
    device = means.device
    colors_ones = torch.ones(means.shape[0], 3, device=device, dtype=means.dtype)
    _, _, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors_ones,
        viewmats=camera.extrinsic.to(device).unsqueeze(0),
        Ks=camera.intrinsic.to(device).unsqueeze(0),
        width=camera.width,
        height=camera.height,
        backgrounds=torch.zeros(1, 3, device=device, dtype=means.dtype),
        render_mode="RGB",
        packed=False,
        sh_degree=None,
    )

    flatten_ids = meta["flatten_ids"]
    if flatten_ids.numel() == 0:
        return torch.zeros((camera.height, camera.width), dtype=torch.int32, device=device)

    # range_start/end are tile-local batches of tile_size**2 intersections.
    # The total intersection count is a safe upper bound for every tile, so a
    # single call covers the complete front-to-back contributor sequence.
    block_size = int(meta["tile_size"]) ** 2
    range_end = (int(flatten_ids.numel()) + block_size - 1) // block_size
    transmittances = torch.ones(
        (1, camera.height, camera.width),
        device=device,
        dtype=means.dtype,
    )
    _, pixel_ids, camera_ids = rasterize_to_indices_in_range(
        0,
        range_end,
        transmittances,
        meta["means2d"],
        meta["conics"],
        meta["opacities"],
        camera.width,
        camera.height,
        int(meta["tile_size"]),
        meta["isect_offsets"],
        flatten_ids,
    )
    if camera_ids.numel() and bool((camera_ids != 0).any()):
        raise RuntimeError("single-camera visible-hit rasterization returned non-zero camera ids")
    return visible_hit_count_from_pixel_ids(pixel_ids, height=camera.height, width=camera.width)
def _rasterize_visible_contributors(
    scene: GaussianScene,
    camera: CameraPose,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize once and return the contributors selected by gsplat."""
    _ensure_ninja_on_path()
    try:
        from gsplat import rasterization, rasterize_to_indices_in_range
    except ImportError as exc:
        raise ImportError("gsplat is required for rendering in scene decomposition") from exc

    means, quats, scales, opacities, _, _ = _prepared_params(scene)
    device = means.device
    colors_ones = torch.ones(means.shape[0], 3, device=device, dtype=means.dtype)
    _, _, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors_ones,
        viewmats=camera.extrinsic.to(device).unsqueeze(0),
        Ks=camera.intrinsic.to(device).unsqueeze(0),
        width=camera.width,
        height=camera.height,
        backgrounds=torch.zeros(1, 3, device=device, dtype=means.dtype),
        render_mode="RGB",
        packed=False,
        sh_degree=None,
    )

    flatten_ids = meta["flatten_ids"]
    if flatten_ids.numel() == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return means, meta, empty, empty, empty

    block_size = int(meta["tile_size"]) ** 2
    range_end = (int(flatten_ids.numel()) + block_size - 1) // block_size
    transmittances = torch.ones(
        (1, camera.height, camera.width),
        device=device,
        dtype=means.dtype,
    )
    gaussian_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range(
        0,
        range_end,
        transmittances,
        meta["means2d"],
        meta["conics"],
        meta["opacities"],
        camera.width,
        camera.height,
        int(meta["tile_size"]),
        meta["isect_offsets"],
        flatten_ids,
    )
    if camera_ids.numel() and bool((camera_ids != 0).any()):
        raise RuntimeError("single-camera visible-hit rasterization returned non-zero camera ids")
    return means, meta, gaussian_ids, pixel_ids, camera_ids


def front_to_back_compositing_weights(
    ray_ids: torch.Tensor,
    alpha: torch.Tensor,
    *,
    num_rays: int,
) -> torch.Tensor:
    """Return exact T_i * alpha_i weights for ray-sorted contributors.

    gsplat returns contributors grouped by ray and ordered front-to-back.
    Keeping this calculation public lets local refinement use the same
    visibility semantics as rendering instead of projected hit counts.
    """
    if ray_ids.ndim != 1 or alpha.ndim != 1 or ray_ids.shape != alpha.shape:
        raise ValueError(
            "ray_ids and alpha must be matching one-dimensional tensors, got "
            f"{tuple(ray_ids.shape)} and {tuple(alpha.shape)}"
        )
    if num_rays < 0:
        raise ValueError(f"num_rays must be non-negative, got {num_rays}")
    if ray_ids.numel() == 0:
        return alpha.clone()
    if bool((ray_ids[1:] < ray_ids[:-1]).any()):
        raise ValueError("ray_ids must be grouped in non-decreasing order")

    counts = torch.bincount(ray_ids.to(torch.int64), minlength=num_rays)
    active = counts > 0
    starts = torch.cumsum(counts, dim=0) - counts
    safe_alpha = alpha.clamp(0.0, 1.0 - 1e-12)
    log_survival = torch.log1p(-safe_alpha.double())
    exclusive_prefix = torch.cumsum(log_survival, dim=0) - log_survival
    group_base = exclusive_prefix[starts[active]]
    base_per_contributor = torch.repeat_interleave(group_base, counts[active])
    transmittance = torch.exp(exclusive_prefix - base_per_contributor).to(alpha.dtype)
    return transmittance * alpha


def rasterize_gaussian_contributors(
    scene: GaussianScene,
    camera: CameraPose,
) -> dict[str, torch.Tensor]:
    """Return visible Gaussian-to-pixel mappings and compositing weights."""
    means, meta, gaussian_ids, pixel_ids, camera_ids = _rasterize_visible_contributors(scene, camera)
    if pixel_ids.numel() == 0:
        empty_float = means.new_empty((0,))
        return {
            "gaussian_ids": gaussian_ids.to(torch.int64),
            "pixel_ids": pixel_ids.to(torch.int64),
            "camera_ids": camera_ids.to(torch.int64),
            "alpha": empty_float,
            "weights": empty_float.clone(),
            "radii": empty_float.clone(),
        }

    height, width = camera.height, camera.width
    num_pixels = height * width
    gaussian_ids = gaussian_ids.to(torch.int64)
    pixel_ids = pixel_ids.to(torch.int64)
    camera_ids = camera_ids.to(torch.int64)
    ray_ids = camera_ids * num_pixels + pixel_ids
    means2d = meta["means2d"][camera_ids, gaussian_ids]
    conics = meta["conics"][camera_ids, gaussian_ids]
    opacities = meta["opacities"][camera_ids, gaussian_ids]
    pixel_x = pixel_ids.remainder(width).to(means2d.dtype) + 0.5
    pixel_y = pixel_ids.div(width, rounding_mode="floor").to(means2d.dtype) + 0.5
    delta_x = pixel_x - means2d[:, 0]
    delta_y = pixel_y - means2d[:, 1]
    sigma = (
        0.5 * (conics[:, 0] * delta_x.square() + conics[:, 2] * delta_y.square())
        + conics[:, 1] * delta_x * delta_y
    )
    alpha = (opacities * torch.exp(-sigma)).clamp_max(0.999)
    weights = front_to_back_compositing_weights(ray_ids, alpha, num_rays=num_pixels)
    radii = meta["radii"][camera_ids, gaussian_ids].to(alpha.dtype).clamp_min(1.0)
    return {
        "gaussian_ids": gaussian_ids,
        "pixel_ids": pixel_ids,
        "camera_ids": camera_ids,
        "alpha": alpha,
        "weights": weights,
        "radii": radii,
    }


def _empty_hit_statistics(camera: CameraPose, *, device: torch.device) -> dict[str, torch.Tensor]:
    shape = (camera.height, camera.width)
    zeros = torch.zeros(shape, device=device, dtype=torch.float32)
    return {
        "visible_count": zeros.to(torch.int32),
        "weight_sum": zeros.clone(),
        "weight_squared_sum": zeros.clone(),
        "effective_count": zeros.clone(),
        "projected_resolution": zeros.clone(),
    }


def rasterize_gaussian_hit_statistics(
    scene: GaussianScene,
    camera: CameraPose,
) -> dict[str, torch.Tensor]:
    """Return visible count, effective count, and projected resolution maps."""
    means, meta, gaussian_ids, pixel_ids, camera_ids = _rasterize_visible_contributors(scene, camera)
    stats = _empty_hit_statistics(camera, device=means.device)
    if pixel_ids.numel() == 0:
        return stats

    height, width = camera.height, camera.width
    num_pixels = height * width
    ray_ids = camera_ids.to(torch.int64) * num_pixels + pixel_ids.to(torch.int64)

    means2d = meta["means2d"][camera_ids, gaussian_ids]
    conics = meta["conics"][camera_ids, gaussian_ids]
    opacities = meta["opacities"][camera_ids, gaussian_ids]
    pixel_x = pixel_ids.remainder(width).to(means2d.dtype) + 0.5
    pixel_y = pixel_ids.div(width, rounding_mode="floor").to(means2d.dtype) + 0.5
    delta_x = pixel_x - means2d[:, 0]
    delta_y = pixel_y - means2d[:, 1]
    sigma = (
        0.5 * (conics[:, 0] * delta_x.square() + conics[:, 2] * delta_y.square())
        + conics[:, 1] * delta_x * delta_y
    )
    alpha = (opacities * torch.exp(-sigma)).clamp_max(0.999)

    counts = torch.bincount(ray_ids, minlength=num_pixels)
    active = counts > 0
    starts = torch.cumsum(counts, dim=0) - counts
    log_survival = torch.log1p(-alpha.double())
    exclusive_prefix = torch.cumsum(log_survival, dim=0) - log_survival
    group_base = exclusive_prefix[starts[active]]
    base_per_contributor = torch.repeat_interleave(group_base, counts[active])
    transmittance = torch.exp(exclusive_prefix - base_per_contributor).to(alpha.dtype)
    weights = transmittance * alpha

    sum_w = torch.bincount(ray_ids, weights=weights, minlength=num_pixels)
    sum_w2 = torch.bincount(ray_ids, weights=weights.square(), minlength=num_pixels)
    effective = sum_w.square() / sum_w2.clamp_min(1e-8)

    radii = meta["radii"][camera_ids, gaussian_ids].to(alpha.dtype).clamp_min(1.0)
    inverse_area_sum = torch.bincount(
        ray_ids,
        weights=weights / (radii.square() + 1.0),
        minlength=num_pixels,
    )
    projected_resolution = inverse_area_sum / sum_w.clamp_min(1e-8)

    stats["visible_count"] = visible_hit_count_from_pixel_ids(
        pixel_ids,
        height=height,
        width=width,
    )
    stats["weight_sum"] = sum_w.reshape(height, width)
    stats["weight_squared_sum"] = sum_w2.reshape(height, width)
    stats["effective_count"] = effective.reshape(height, width).clamp_min(0.0)
    stats["projected_resolution"] = projected_resolution.reshape(height, width).clamp_min(0.0)
    return stats


def rasterize_effective_hit_count(scene: GaussianScene, camera: CameraPose) -> torch.Tensor:
    """Return the participation-ratio effective contributor count per pixel."""
    return rasterize_gaussian_hit_statistics(scene, camera)["effective_count"]


def rasterize_projected_resolution(scene: GaussianScene, camera: CameraPose) -> torch.Tensor:
    """Return weighted inverse screen-space Gaussian footprint per pixel."""
    return rasterize_gaussian_hit_statistics(scene, camera)["projected_resolution"]




def rasterize(scene: GaussianScene, camera: CameraPose) -> RenderedView:
    _ensure_ninja_on_path()
    try:
        from gsplat import rasterization
    except ImportError as exc:
        raise ImportError("gsplat is required for rendering in scene decomposition") from exc

    means, quats, scales, opacities, colors, sh_degree = _prepared_params(scene)
    device = means.device
    render_colors, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=camera.extrinsic.to(device).unsqueeze(0),
        Ks=camera.intrinsic.to(device).unsqueeze(0),
        width=camera.width,
        height=camera.height,
        backgrounds=torch.ones(1, 3, device=device, dtype=means.dtype),
        render_mode="RGB+D",
        packed=False,
        sh_degree=sh_degree,
    )
    return RenderedView(
        rgb=render_colors[0, ..., :3].clamp(0, 1),
        depth=render_colors[0, ..., 3],
        alpha=render_alphas[0, ..., 0],
    )


def render_features(
    scene: GaussianScene,
    camera: CameraPose,
    features: torch.Tensor,
    chunk: int = 8,
    *,
    return_alpha: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    from gsplat import rasterization

    p = scene.params
    device = p.means.device
    features = features.to(device)
    outputs = []
    for start in range(0, features.shape[1], chunk):
        part = features[:, start : start + chunk]
        channels = part.shape[1]
        render_channels = max(channels, 3)
        if channels < 3:
            part = torch.nn.functional.pad(part, (0, 3 - channels))
        render_colors, render_alphas, _ = rasterization(
            means=p.means,
            quats=torch.nn.functional.normalize(p.rotations, dim=-1),
            scales=torch.exp(p.scales),
            opacities=torch.sigmoid(p.opacities.squeeze(-1)),
            colors=part,
            viewmats=camera.extrinsic.to(device).unsqueeze(0),
            Ks=camera.intrinsic.to(device).unsqueeze(0),
            width=camera.width,
            height=camera.height,
            backgrounds=torch.zeros(1, render_channels, device=device, dtype=features.dtype),
            render_mode="RGB",
            packed=False,
        )
        outputs.append(render_colors[0, ..., :channels])
    rendered = torch.cat(outputs, dim=-1)
    if return_alpha:
        return rendered, render_alphas[0, ..., 0]
    return rendered


def render_scores(scene: GaussianScene, camera: CameraPose, scores: torch.Tensor, chunk: int = 3) -> torch.Tensor:
    return render_features(scene, camera, scores, chunk=chunk)
