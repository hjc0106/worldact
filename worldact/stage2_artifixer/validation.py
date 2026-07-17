"""Validate object packages and refined PLY geometry guards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from worldact.data.gaussians import GaussianParams, GaussianScene
from worldact.stage2_artifixer.contracts import (
    SUPPORTED_SCHEMA_VERSIONS,
    AnchorsDocument,
    NormalizationTransform,
    PackageManifest,
    TrajectoryDocument,
)
from worldact.stage2_artifixer.normalization import robust_bbox_center_radius


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.errors))


def validate_package_dir(package_dir: Path | str, *, max_bytes: int = 2_000_000_000) -> ValidationResult:
    package_dir = Path(package_dir)
    errors: list[str] = []
    warnings: list[str] = []

    required = [
        "manifest.json",
        "normalization.json",
        "object_local.ply",
        "anchors.json",
        "trajectory.json",
    ]
    for name in required:
        if not (package_dir / name).exists():
            errors.append(f"missing {name}")

    if errors:
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    try:
        manifest = PackageManifest.model_validate_json((package_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return ValidationResult(ok=False, errors=[f"manifest invalid: {exc}"], warnings=warnings)

    if manifest.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(f"unsupported schema_version={manifest.schema_version}")

    try:
        NormalizationTransform.model_validate_json((package_dir / "normalization.json").read_text(encoding="utf-8"))
        anchors = AnchorsDocument.model_validate_json((package_dir / "anchors.json").read_text(encoding="utf-8"))
        trajectory = TrajectoryDocument.model_validate_json((package_dir / "trajectory.json").read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"document invalid: {exc}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    if len(anchors.anchors) != manifest.anchor_count:
        errors.append(f"anchor_count mismatch manifest={manifest.anchor_count} actual={len(anchors.anchors)}")
    if len(trajectory.frames) != manifest.target_count:
        errors.append(f"target_count mismatch manifest={manifest.target_count} actual={len(trajectory.frames)}")

    from worldact.stage2_artifixer.package import sha256_file

    for item in manifest.files:
        path = package_dir / item.path
        if not path.exists():
            errors.append(f"listed file missing: {item.path}")
            continue
        digest = sha256_file(path)
        if digest != item.sha256:
            errors.append(f"checksum mismatch for {item.path}")

    # Path traversal / size guards for anchors.
    total = 0
    for path in package_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(package_dir)
        if ".." in rel.parts:
            errors.append(f"illegal path {rel}")
        total += path.stat().st_size
    if total > max_bytes:
        errors.append(f"package too large: {total} bytes")

    for anchor in anchors.anchors:
        if not _anchor_quality_ok(anchor.mask_area_ratio):
            warnings.append(f"anchor {anchor.index} area_ratio={anchor.mask_area_ratio:.3f} outside 0.2-0.8")
        rgb = package_dir / anchor.rgb_path
        mask = package_dir / anchor.mask_path
        if not rgb.exists() or not mask.exists():
            errors.append(f"anchor files missing for index={anchor.index}")

    return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def _anchor_quality_ok(area_ratio: float) -> bool:
    return 0.20 <= area_ratio <= 0.80


@dataclass
class GeometryGuardResult:
    ok: bool
    reasons: list[str]
    metrics: dict


def validate_refined_geometry(
    input_local: GaussianParams,
    refined_local: GaussianParams,
    *,
    max_center_drift_frac: float = 0.05,
    scale_ratio_range: tuple[float, float] = (0.5, 2.0),
    max_gaussian_multiplier: float = 10.0,
) -> GeometryGuardResult:
    reasons: list[str] = []
    if refined_local.num_gaussians < 1:
        reasons.append("empty refined PLY")
    if not torch.isfinite(refined_local.means).all():
        reasons.append("non-finite means")
    if not torch.isfinite(refined_local.scales).all():
        reasons.append("non-finite scales")

    in_center, in_radius = robust_bbox_center_radius(input_local.means)
    out_center, out_radius = robust_bbox_center_radius(refined_local.means)
    center_drift = float((out_center - in_center).norm().item())
    if center_drift > max_center_drift_frac * float(in_radius):
        reasons.append(
            f"bbox center drift {center_drift:.4f} > {max_center_drift_frac * in_radius:.4f}"
        )
    scale_ratio = out_radius / max(in_radius, 1e-6)
    if not (scale_ratio_range[0] <= scale_ratio <= scale_ratio_range[1]):
        reasons.append(f"bbox scale ratio {scale_ratio:.3f} outside {scale_ratio_range}")
    if refined_local.num_gaussians > int(input_local.num_gaussians * max_gaussian_multiplier):
        reasons.append(
            f"gaussian count {refined_local.num_gaussians} > {max_gaussian_multiplier}x input"
        )

    metrics = {
        "input_gaussians": input_local.num_gaussians,
        "refined_gaussians": refined_local.num_gaussians,
        "center_drift": center_drift,
        "scale_ratio": scale_ratio,
        "input_radius": float(in_radius),
        "refined_radius": float(out_radius),
    }
    return GeometryGuardResult(ok=len(reasons) == 0, reasons=reasons, metrics=metrics)


def load_local_ply(path: Path | str) -> GaussianScene:
    return GaussianScene.from_ply(path)
