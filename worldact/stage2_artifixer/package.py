"""Build and archive object_package payloads for the remote ArtiFixer service."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

from worldact.data.gaussians import GaussianScene
from worldact.stage2_artifixer.anchors import select_and_export_anchors
from worldact.stage2_artifixer.config import PrepareConfig
from worldact.stage2_artifixer.contracts import (
    SCHEMA_VERSION,
    AnchorsDocument,
    FileChecksum,
    NormalizationTransform,
    PackageManifest,
    TrajectoryDocument,
    utc_now_iso,
)
from worldact.stage2_artifixer.inputs import (
    StageArtifixerInput,
    find_object_mask,
    find_object_record,
    load_object_scene,
)
from worldact.stage2_artifixer.normalization import build_normalization, world_to_local_params
from worldact.stage2_artifixer.trajectory import build_target_trajectory
from worldact.stage2_artifixer.validation import validate_package_dir


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_object_package(
    stage_input: StageArtifixerInput,
    object_id: str,
    output_dir: Path | str,
    cfg: PrepareConfig | None = None,
    *,
    prompt: str | None = None,
) -> tuple[Path, PackageManifest]:
    """Write object_package/ under output_dir and return (package_dir, manifest)."""
    cfg = cfg or PrepareConfig()
    output_dir = Path(output_dir)
    package_dir = output_dir / "object_package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    record = find_object_record(stage_input.objects, object_id)
    object_scene = load_object_scene(stage_input, object_id)
    if object_scene.num_gaussians < 1:
        raise ValueError(f"object {object_id} has no gaussians")

    norm = build_normalization(object_scene.params, cfg.quantile)
    local_params = world_to_local_params(object_scene.params, norm)
    local_scene = GaussianScene(params=local_params, camera_trajectory=[], metadata={"frame": "local"})
    local_ply = package_dir / "object_local.ply"
    local_scene.to_ply(local_ply)

    (package_dir / "normalization.json").write_text(norm.model_dump_json(indent=2), encoding="utf-8")

    obj_mask = find_object_mask(stage_input.masks, object_id)
    anchors = select_and_export_anchors(
        stage_input.scene,
        object_scene,
        obj_mask,
        norm,
        package_dir,
        cfg,
        render_device=stage_input.scene.device,
    )
    trajectory = build_target_trajectory(anchors, cfg, local_radius=1.0)
    (package_dir / "trajectory.json").write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")

    text_prompt = prompt or record.text_prompt or f"a complete {record.name}"
    # Hash every file except manifest.json (avoids self-referential checksum cycles).
    file_checksums = _collect_file_checksums(package_dir, exclude_names={"manifest.json"})
    content_hash = _content_hash(file_checksums)
    manifest = PackageManifest(
        schema_version=SCHEMA_VERSION,
        object_id=str(record.object_id),
        object_name=record.name,
        prompt=text_prompt,
        pipeline="artifixer3d",
        model=cfg.model,
        anchor_count=len(anchors.anchors),
        target_count=len(trajectory.frames),
        base_reconstruction_steps=cfg.base_reconstruction_steps,
        distillation_steps=cfg.distillation_steps,
        run_artifixer3d_plus=cfg.run_artifixer3d_plus,
        image_size=cfg.image_size,
        files=file_checksums,
        input_sha256=content_hash,
        created_at=utc_now_iso(),
    )
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    validate_package_dir(package_dir)
    return package_dir, PackageManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def archive_package(package_dir: Path | str, archive_path: Path | str | None = None) -> Path:
    package_dir = Path(package_dir)
    archive_path = Path(archive_path) if archive_path else package_dir.parent / f"{package_dir.name}.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(package_dir).as_posix())
    return archive_path


def _collect_file_checksums(package_dir: Path, exclude_names: set[str] | None = None) -> list[FileChecksum]:
    exclude_names = exclude_names or set()
    files: list[FileChecksum] = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(package_dir).as_posix()
        if path.name in exclude_names or rel in exclude_names:
            continue
        files.append(FileChecksum(path=rel, sha256=sha256_file(path)))
    return files


def _content_hash(files: Iterable[FileChecksum]) -> str:
    h = hashlib.sha256()
    for item in sorted(files, key=lambda f: f.path):
        h.update(item.path.encode())
        h.update(b":")
        h.update(item.sha256.encode())
        h.update(b"\n")
    return h.hexdigest()


def load_package_documents(package_dir: Path | str) -> tuple[PackageManifest, NormalizationTransform, AnchorsDocument, TrajectoryDocument]:
    package_dir = Path(package_dir)
    manifest = PackageManifest.model_validate_json((package_dir / "manifest.json").read_text(encoding="utf-8"))
    norm = NormalizationTransform.model_validate_json((package_dir / "normalization.json").read_text(encoding="utf-8"))
    anchors = AnchorsDocument.model_validate_json((package_dir / "anchors.json").read_text(encoding="utf-8"))
    trajectory = TrajectoryDocument.model_validate_json((package_dir / "trajectory.json").read_text(encoding="utf-8"))
    return manifest, norm, anchors, trajectory
