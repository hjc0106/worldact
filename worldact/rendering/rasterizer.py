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
