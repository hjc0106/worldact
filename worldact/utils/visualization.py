from __future__ import annotations

from pathlib import Path

import torch

from worldact.data.gaussians import GaussianScene
from worldact.data.scene import DecomposedScene
from worldact.rendering.camera_planning import select_keyframes, trajectory_center_for_keyframes

COLORS = [
    (1.0, 0.0, 0.0),
    (0.0, 0.8, 0.0),
    (0.0, 0.3, 1.0),
    (1.0, 0.8, 0.0),
    (0.8, 0.0, 0.8),
    (0.0, 0.8, 0.8),
]


def save_image(tensor: torch.Tensor, path: Path) -> None:
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (tensor.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
    iio.imwrite(path, arr)


def visualize_decomposition(
    scene: GaussianScene,
    result: DecomposedScene,
    output_dir: Path,
    num_views: int = 4,
    camera_cfg=None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if camera_cfg is None:
        from worldact.rendering.camera import sample_keyframes

        view_indices = sample_keyframes(scene.camera_trajectory, num_views)
    else:
        center = trajectory_center_for_keyframes(scene.params.means, camera_cfg)
        view_indices = select_keyframes(scene.camera_trajectory, num_views, camera_cfg, center=center)
    for out_idx, view_idx in enumerate(view_indices):
        view = scene.render(scene.camera_trajectory[view_idx])
        save_image(view.rgb, output_dir / f"view_{out_idx:02d}_rgb.png")
        overlay = view.rgb.clone()
        if result.masks is not None:
            for obj_idx, obj_mask in enumerate(result.masks.object_masks):
                mask = obj_mask.get_mask(view_idx)
                if mask is None:
                    continue
                color = torch.tensor(COLORS[obj_idx % len(COLORS)], dtype=overlay.dtype, device=overlay.device)
                mask = mask.to(overlay.device).unsqueeze(-1)
                overlay = overlay * (1.0 - 0.45 * mask) + color.view(1, 1, 3) * 0.45 * mask
        save_image(overlay, output_dir / f"view_{out_idx:02d}_masks.png")
