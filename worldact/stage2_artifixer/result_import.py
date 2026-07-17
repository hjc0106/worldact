"""Download remote artifacts, inverse-normalize PLY, and apply geometry guards."""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from worldact.data.gaussians import GaussianScene
from worldact.stage2_artifixer.contracts import ArtifactManifest, NormalizationTransform
from worldact.stage2_artifixer.normalization import local_to_world_params
from worldact.stage2_artifixer.package import sha256_file
from worldact.stage2_artifixer.validation import GeometryGuardResult, validate_refined_geometry


@dataclass
class ImportResult:
    status: str
    refined_local_ply: Path | None
    refined_world_ply: Path | None
    geometry: GeometryGuardResult | None
    artifact_dir: Path
    messages: list[str]


def extract_artifacts(archive_path: Path | str, output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    artifact_dir = output_dir / "artifacts"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        _safe_extract(zf, artifact_dir)
    # Support archives that nest an artifacts/ folder.
    nested = artifact_dir / "artifacts"
    if nested.is_dir() and (nested / "refined_local.ply").exists():
        return nested
    return artifact_dir


def import_refined_result(
    artifact_dir: Path | str,
    normalization: NormalizationTransform | Path | str,
    output_dir: Path | str,
    *,
    input_local_ply: Path | str | None = None,
    publish_world: bool = True,
) -> ImportResult:
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []

    if isinstance(normalization, (str, Path)):
        norm = NormalizationTransform.model_validate_json(Path(normalization).read_text(encoding="utf-8"))
    else:
        norm = normalization

    manifest_path = artifact_dir / "artifact_manifest.json"
    if manifest_path.exists():
        manifest = ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.files:
            path = artifact_dir / item.path
            if not path.exists():
                messages.append(f"missing artifact {item.path}")
                continue
            digest = sha256_file(path)
            if digest != item.sha256:
                messages.append(f"artifact checksum mismatch: {item.path}")

    local_ply = artifact_dir / "refined_local.ply"
    if not local_ply.exists():
        return ImportResult(
            status="failed",
            refined_local_ply=None,
            refined_world_ply=None,
            geometry=None,
            artifact_dir=artifact_dir,
            messages=messages + ["refined_local.ply missing"],
        )

    refined_local = GaussianScene.from_ply(local_ply)
    if refined_local.num_gaussians < 1:
        return ImportResult(
            status="failed",
            refined_local_ply=local_ply,
            refined_world_ply=None,
            geometry=None,
            artifact_dir=artifact_dir,
            messages=messages + ["empty refined_local.ply"],
        )

    geometry: GeometryGuardResult | None = None
    if input_local_ply is not None and Path(input_local_ply).exists():
        input_local = GaussianScene.from_ply(input_local_ply)
        geometry = validate_refined_geometry(input_local.params, refined_local.params)
        if not geometry.ok:
            messages.extend(geometry.reasons)
            (output_dir / "import_validation.json").write_text(
                json.dumps({"ok": False, "reasons": geometry.reasons, "metrics": geometry.metrics}, indent=2),
                encoding="utf-8",
            )
            # Copy local PLY for debugging but do not publish world PLY.
            shutil.copy2(local_ply, output_dir / "refined_local.ply")
            return ImportResult(
                status="completed_with_validation_failure",
                refined_local_ply=output_dir / "refined_local.ply",
                refined_world_ply=None,
                geometry=geometry,
                artifact_dir=artifact_dir,
                messages=messages,
            )

    shutil.copy2(local_ply, output_dir / "refined_local.ply")
    world_ply = None
    if publish_world:
        world_params = local_to_world_params(refined_local.params, norm)
        world_scene = GaussianScene(params=world_params, metadata={"frame": "world"})
        world_ply = output_dir / "refined_world.ply"
        world_scene.to_ply(world_ply)

    if geometry is not None:
        (output_dir / "import_validation.json").write_text(
            json.dumps({"ok": True, "reasons": [], "metrics": geometry.metrics}, indent=2),
            encoding="utf-8",
        )

    return ImportResult(
        status="completed",
        refined_local_ply=output_dir / "refined_local.ply",
        refined_world_ply=world_ply,
        geometry=geometry,
        artifact_dir=artifact_dir,
        messages=messages,
    )


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not str(target).startswith(str(dest)):
            raise ValueError(f"illegal archive path: {member.filename}")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
