"""Orchestrate prepare / submit / wait / download / import phases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from worldact.stage2_artifixer.config import ArtiFixerRefinementConfig
from worldact.stage2_artifixer.contracts import PackageManifest, RemoteJobRecord, TERMINAL_STATUSES
from worldact.stage2_artifixer.inputs import StageArtifixerInput, load_stage_artifixer_input
from worldact.stage2_artifixer.package import archive_package, prepare_object_package
from worldact.stage2_artifixer.remote_client import (
    ArtiFixerRemoteClient,
    load_remote_job_record,
    make_idempotency_key,
    upsert_job_record,
)
from worldact.stage2_artifixer.result_import import extract_artifacts, import_refined_result

Phase = Literal["prepare", "submit", "wait", "download", "import", "validate", "all"]


@dataclass
class RunnerResult:
    phase: str
    object_id: str
    output_dir: Path
    package_dir: Path | None = None
    archive_path: Path | None = None
    manifest: PackageManifest | None = None
    job: RemoteJobRecord | None = None
    import_status: str | None = None
    details: dict[str, Any] | None = None


class ArtiFixerRefinementRunner:
    def __init__(
        self,
        *,
        ply_path: Path | str,
        stage_dir: Path | str,
        object_id: str,
        output_dir: Path | str,
        config: ArtiFixerRefinementConfig | None = None,
        cameras_path: Path | str | None = None,
        object_ply_dir: Path | str | None = None,
        scene_id: str | None = None,
        client_factory: Callable[[], ArtiFixerRemoteClient] | None = None,
    ) -> None:
        self.ply_path = Path(ply_path)
        self.stage_dir = Path(stage_dir)
        self.object_id = str(object_id)
        self.output_dir = Path(output_dir)
        self.config = config or ArtiFixerRefinementConfig()
        self.cameras_path = cameras_path
        self.object_ply_dir = object_ply_dir
        self.scene_id = scene_id or self.stage_dir.name
        self.client_factory = client_factory or (lambda: ArtiFixerRemoteClient(self.config.service))
        self.object_dir = self.output_dir / f"object_{self.object_id}"
        self.job_record_path = self.object_dir / "remote_job.json"

    def run(self, phase: Phase = "all") -> RunnerResult:
        self.object_dir.mkdir(parents=True, exist_ok=True)
        if phase == "prepare":
            return self.prepare()
        if phase == "submit":
            return self.submit()
        if phase == "wait":
            return self.wait()
        if phase == "download":
            return self.download()
        if phase == "import":
            return self.import_result()
        if phase == "validate":
            return self.validate_only()
        if phase == "all":
            self.prepare()
            self.submit()
            self.wait()
            self.download()
            return self.import_result()
        raise ValueError(f"unknown phase {phase!r}")

    def prepare(self) -> RunnerResult:
        stage_input = self._load_input()
        package_dir, manifest = prepare_object_package(
            stage_input,
            self.object_id,
            self.object_dir,
            self.config.prepare,
        )
        archive_path = archive_package(package_dir, self.object_dir / "object_package.zip")
        (self.object_dir / "prepare_summary.json").write_text(
            json.dumps(
                {
                    "object_id": self.object_id,
                    "package_dir": str(package_dir),
                    "archive_path": str(archive_path),
                    "input_sha256": manifest.input_sha256,
                    "anchor_count": manifest.anchor_count,
                    "target_count": manifest.target_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return RunnerResult(
            phase="prepare",
            object_id=self.object_id,
            output_dir=self.object_dir,
            package_dir=package_dir,
            archive_path=archive_path,
            manifest=manifest,
        )

    def submit(self) -> RunnerResult:
        package_dir = self.object_dir / "object_package"
        archive_path = self.object_dir / "object_package.zip"
        manifest = PackageManifest.model_validate_json((package_dir / "manifest.json").read_text(encoding="utf-8"))
        idem = make_idempotency_key(self.object_id, manifest.input_sha256, self.scene_id)

        existing = load_remote_job_record(self.job_record_path)
        if existing is not None and existing.input_sha256 == manifest.input_sha256 and existing.job_id:
            # Resume: do not recreate GPU work for the same package digest.
            if existing.status not in {"failed", "cancelled"}:
                return RunnerResult(
                    phase="submit",
                    object_id=self.object_id,
                    output_dir=self.object_dir,
                    package_dir=package_dir,
                    archive_path=archive_path,
                    manifest=manifest,
                    job=existing,
                    details={"resumed": True},
                )

        if not archive_path.exists():
            archive_path = archive_package(package_dir, archive_path)

        with self.client_factory() as client:
            created = client.submit(
                archive_path,
                idem,
                manifest_json=(package_dir / "manifest.json").read_text(encoding="utf-8"),
            )
            status = client.get_status(created.job_id)
        record = upsert_job_record(
            self.job_record_path,
            job_id=created.job_id,
            object_id=self.object_id,
            input_sha256=manifest.input_sha256,
            idempotency_key=idem,
            status=status.status,
            package_path=str(archive_path),
            last_status=status,
        )
        return RunnerResult(
            phase="submit",
            object_id=self.object_id,
            output_dir=self.object_dir,
            package_dir=package_dir,
            archive_path=archive_path,
            manifest=manifest,
            job=record,
            details={"resumed": False},
        )

    def wait(self) -> RunnerResult:
        record = self._require_job_record()
        with self.client_factory() as client:
            status = client.wait(record.job_id)
        record = upsert_job_record(
            self.job_record_path,
            job_id=record.job_id,
            object_id=record.object_id,
            input_sha256=record.input_sha256,
            idempotency_key=record.idempotency_key,
            status=status.status,
            package_path=record.package_path,
            last_status=status,
        )
        if status.status not in TERMINAL_STATUSES:
            raise RuntimeError(f"job {record.job_id} ended in unexpected status {status.status}")
        if status.status == "failed":
            raise RuntimeError(f"remote job failed: {status.error}")
        return RunnerResult(phase="wait", object_id=self.object_id, output_dir=self.object_dir, job=record)

    def download(self) -> RunnerResult:
        record = self._require_job_record()
        with self.client_factory() as client:
            archive = client.download_artifacts(record.job_id, self.object_dir / "downloads")
        artifact_dir = extract_artifacts(archive, self.object_dir)
        return RunnerResult(
            phase="download",
            object_id=self.object_id,
            output_dir=self.object_dir,
            job=record,
            details={"artifact_dir": str(artifact_dir), "archive": str(archive)},
        )

    def import_result(self) -> RunnerResult:
        package_dir = self.object_dir / "object_package"
        artifact_dir = self.object_dir / "artifacts"
        if not artifact_dir.exists():
            # tolerate nested path from extract
            nested = self.object_dir / "artifacts"
            artifact_dir = nested
        result = import_refined_result(
            artifact_dir,
            package_dir / "normalization.json",
            self.object_dir,
            input_local_ply=package_dir / "object_local.ply",
        )
        (self.object_dir / "import_summary.json").write_text(
            json.dumps(
                {
                    "status": result.status,
                    "refined_local_ply": str(result.refined_local_ply) if result.refined_local_ply else None,
                    "refined_world_ply": str(result.refined_world_ply) if result.refined_world_ply else None,
                    "messages": result.messages,
                    "geometry": None if result.geometry is None else result.geometry.metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return RunnerResult(
            phase="import",
            object_id=self.object_id,
            output_dir=self.object_dir,
            import_status=result.status,
            details={"messages": result.messages},
        )

    def validate_only(self) -> RunnerResult:
        from worldact.stage2_artifixer.validation import validate_package_dir

        package_dir = self.object_dir / "object_package"
        result = validate_package_dir(package_dir)
        result.raise_if_failed()
        return RunnerResult(
            phase="validate",
            object_id=self.object_id,
            output_dir=self.object_dir,
            package_dir=package_dir,
            details={"warnings": result.warnings},
        )

    def _load_input(self) -> StageArtifixerInput:
        return load_stage_artifixer_input(
            self.ply_path,
            self.stage_dir,
            cameras_path=self.cameras_path,
            object_ply_dir=self.object_ply_dir,
            device=self.config.device,
        )

    def _require_job_record(self) -> RemoteJobRecord:
        record = load_remote_job_record(self.job_record_path)
        if record is None:
            raise FileNotFoundError(f"missing {self.job_record_path}; run submit first")
        return record
