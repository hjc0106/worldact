#!/usr/bin/env python3
"""Restrict full-frame ArtiFixer predictions to an object opacity mask.

Object-only 3DGS renders use a black background.  ArtiFixer still predicts a
complete image, so its unconstrained background should not be used for visual
inspection or ArtiFixer3D distillation.  This utility preserves generated RGB
inside a cleaned object mask and copies the original render outside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def object_mask(alpha: np.ndarray, threshold: float, dilate_pixels: int, feather_pixels: float) -> np.ndarray:
    """Return a soft, largest-component foreground mask from a uint8 alpha image."""
    if alpha.ndim != 2:
        raise ValueError(f"Expected a single-channel alpha image, got shape {alpha.shape}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if dilate_pixels < 0:
        raise ValueError(f"dilate_pixels must be non-negative, got {dilate_pixels}")
    if feather_pixels < 0:
        raise ValueError(f"feather_pixels must be non-negative, got {feather_pixels}")

    foreground = alpha.astype(np.float32) / 255.0 >= threshold
    labels, component_count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count == 0:
        raise ValueError("Opacity mask has no foreground at the selected threshold")

    areas = np.bincount(labels.ravel())
    areas[0] = 0
    foreground = labels == int(areas.argmax())
    if dilate_pixels:
        foreground = ndimage.binary_dilation(foreground, iterations=dilate_pixels)

    mask = foreground.astype(np.float32)
    if feather_pixels:
        mask = ndimage.gaussian_filter(mask, sigma=feather_pixels)
    return np.clip(mask, 0.0, 1.0)


def composite_prediction(prediction: np.ndarray, rendered: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend generated object RGB over the source render using a soft object mask."""
    if prediction.shape != rendered.shape:
        raise ValueError(f"Prediction/render size mismatch: {prediction.shape} vs {rendered.shape}")
    if prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError(f"Expected an RGB prediction, got shape {prediction.shape}")
    if mask.shape != prediction.shape[:2]:
        raise ValueError(f"Mask/image size mismatch: {mask.shape} vs {prediction.shape[:2]}")
    mixed = prediction.astype(np.float32) * mask[..., None] + rendered.astype(np.float32) * (1.0 - mask[..., None])
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def png_names(directory: Path) -> list[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    names = sorted(path.name for path in directory.glob("*.png"))
    if not names:
        raise FileNotFoundError(f"No PNG files found in {directory}")
    return names


def run(args: argparse.Namespace) -> dict[str, object]:
    pred_dir = args.pred_dir.resolve()
    render_dir = args.render_dir.resolve()
    opacity_dir = args.opacity_dir.resolve()
    output_dir = args.output_dir.resolve()

    names = png_names(pred_dir)
    missing = [
        name
        for name in names
        if not (render_dir / name).is_file() or not (opacity_dir / name).is_file()
    ]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise FileNotFoundError(f"Missing matching render/opacity PNGs for: {preview}{suffix}")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite to replace PNGs")
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_coverages: list[float] = []
    for name in names:
        with Image.open(pred_dir / name) as image:
            prediction = np.asarray(image.convert("RGB"))
        with Image.open(render_dir / name) as image:
            rendered = np.asarray(image.convert("RGB"))
        with Image.open(opacity_dir / name) as image:
            alpha = np.asarray(image.convert("L"))

        mask = object_mask(alpha, args.alpha_threshold, args.dilate_pixels, args.feather_pixels)
        result = composite_prediction(prediction, rendered, mask)
        Image.fromarray(result, mode="RGB").save(output_dir / name)
        mask_coverages.append(float((mask > 0.5).mean()))

    summary = {
        "pred_dir": str(pred_dir),
        "render_dir": str(render_dir),
        "opacity_dir": str(opacity_dir),
        "output_dir": str(output_dir),
        "frame_count": len(names),
        "alpha_threshold": args.alpha_threshold,
        "dilate_pixels": args.dilate_pixels,
        "feather_pixels": args.feather_pixels,
        "mask_coverage_min": min(mask_coverages),
        "mask_coverage_max": max(mask_coverages),
    }
    (output_dir / "composite_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--opacity-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.10,
        help="Threshold for defining the core object alpha region (default: 0.10).",
    )
    parser.add_argument(
        "--dilate-pixels",
        type=int,
        default=4,
        help="Expand the cleaned foreground by this many pixels before blending (default: 4).",
    )
    parser.add_argument(
        "--feather-pixels",
        type=float,
        default=1.5,
        help="Gaussian blur radius for a soft object boundary (default: 1.5).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
