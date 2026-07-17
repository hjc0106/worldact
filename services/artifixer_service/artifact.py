"""Collect and checksum job artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from services.artifixer_service.models import ArtifactFile, ArtifactManifest, JobStatus, utc_now_iso


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_artifact_manifest(job_id: str, artifacts_dir: Path, status: JobStatus) -> ArtifactManifest:
    files: list[ArtifactFile] = []
    for path in sorted(Path(artifacts_dir).rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        rel = path.relative_to(artifacts_dir).as_posix()
        files.append(ArtifactFile(path=rel, sha256=sha256_file(path), size_bytes=path.stat().st_size))
    manifest = ArtifactManifest(
        job_id=job_id,
        status=status,
        files=files,
        created_at=utc_now_iso(),
    )
    (Path(artifacts_dir) / "artifact_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    # Re-add manifest itself.
    manifest.files.append(
        ArtifactFile(
            path="artifact_manifest.json",
            sha256=sha256_file(Path(artifacts_dir) / "artifact_manifest.json"),
            size_bytes=(Path(artifacts_dir) / "artifact_manifest.json").stat().st_size,
        )
    )
    (Path(artifacts_dir) / "artifact_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return ArtifactManifest.model_validate_json((Path(artifacts_dir) / "artifact_manifest.json").read_text(encoding="utf-8"))


def write_metrics(artifacts_dir: Path, metrics: dict) -> Path:
    path = Path(artifacts_dir) / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
