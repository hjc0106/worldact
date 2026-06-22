from __future__ import annotations

import torch

from worldact.data.gaussians import GaussianParams, GaussianScene


def project_points(scene: GaussianScene, view_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cam = scene.camera_trajectory[view_idx]
    means = scene.params.means
    device = means.device
    extrinsic = cam.extrinsic.to(device)
    intrinsic = cam.intrinsic.to(device)
    cam_pts = (extrinsic[:3, :3] @ means.T + extrinsic[:3, 3:4]).T
    z = cam_pts[:, 2]
    valid = z > 1e-6
    x = intrinsic[0, 0] * cam_pts[:, 0] / z.clamp_min(1e-6) + intrinsic[0, 2]
    y = intrinsic[1, 1] * cam_pts[:, 1] / z.clamp_min(1e-6) + intrinsic[1, 2]
    px = torch.round(x).long()
    py = torch.round(y).long()
    valid = valid & (px >= 0) & (py >= 0) & (px < cam.width) & (py < cam.height)
    return valid, px, py, z


def depth_gate(
    valid: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    z: torch.Tensor,
    depth: torch.Tensor,
    alpha: torch.Tensor,
    max_depth_error: float,
    min_alpha: float,
) -> torch.Tensor:
    if max_depth_error <= 0.0 and min_alpha <= 0.0:
        return valid
    rendered = depth[py.clamp(0, depth.shape[0] - 1), px.clamp(0, depth.shape[1] - 1)]
    rendered_alpha = alpha[py.clamp(0, alpha.shape[0] - 1), px.clamp(0, alpha.shape[1] - 1)]
    depth_ok = rendered > 1e-6
    if max_depth_error > 0.0:
        tolerance = float(max_depth_error) * rendered.abs().clamp(min=0.1)
        depth_ok = depth_ok & ((z - rendered).abs() <= tolerance)
    return valid & depth_ok & (rendered_alpha >= float(min_alpha))


def radius_components(points: torch.Tensor, radius: float) -> list[torch.Tensor]:
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


def subset_scene(scene: GaussianScene, keep: torch.Tensor) -> GaussianScene:
    if keep.dtype == torch.bool:
        indices = keep.nonzero(as_tuple=True)[0]
    else:
        indices = keep.to(scene.device)
    return scene.subset(indices)


def concat_gaussian_params(first: GaussianParams, second: GaussianParams) -> GaussianParams:
    color_dim = max(first.colors.shape[1], second.colors.shape[1])

    def pad_colors(colors: torch.Tensor) -> torch.Tensor:
        if colors.shape[1] == color_dim:
            return colors
        return torch.nn.functional.pad(colors, (0, color_dim - colors.shape[1]))

    return GaussianParams(
        means=torch.cat([first.means, second.means.to(first.device)], dim=0),
        scales=torch.cat([first.scales, second.scales.to(first.device)], dim=0),
        rotations=torch.cat([first.rotations, second.rotations.to(first.device)], dim=0),
        opacities=torch.cat([first.opacities, second.opacities.to(first.device)], dim=0),
        colors=torch.cat([pad_colors(first.colors), pad_colors(second.colors.to(first.device))], dim=0),
    )
