"""HTTP client for the asynchronous ArtiFixer refinement Job API."""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from worldact.stage2_artifixer.config import ArtiFixerServiceConfig
from worldact.stage2_artifixer.contracts import (
    TERMINAL_STATUSES,
    ArtifactsResponse,
    JobCreateResponse,
    JobStatusResponse,
    RemoteJobRecord,
    utc_now_iso,
)


class ArtiFixerRemoteClient:
    def __init__(self, config: ArtiFixerServiceConfig | None = None) -> None:
        self.config = config or ArtiFixerServiceConfig()
        self._client = httpx.Client(
            base_url=self.config.base_url.rstrip("/"),
            timeout=self.config.request_timeout_seconds,
            verify=self.config.verify_tls,
            headers=self._headers(),
            # Host http_proxy must not intercept 127.0.0.1 service calls.
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ArtiFixerRemoteClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def health(self) -> dict:
        """Probe service liveness (/healthz, fallback /health)."""
        for path in ("/healthz", "/health"):
            response = self._client.get(path)
            if response.status_code == 200:
                return response.json()
        response.raise_for_status()
        return {}

    def submit(self, package_archive: Path | str, idempotency_key: str, *, manifest_json: str | None = None) -> JobCreateResponse:
        package_archive = Path(package_archive)
        files = {
            "archive": (package_archive.name, package_archive.read_bytes(), "application/zip"),
        }
        data = {}
        if manifest_json is not None:
            data["manifest"] = manifest_json
        response = self._client.post(
            "/v1/refinement-jobs",
            files=files,
            data=data,
            headers={"Idempotency-Key": idempotency_key},
        )
        response.raise_for_status()
        return JobCreateResponse.model_validate(response.json())

    def get_status(self, job_id: str) -> JobStatusResponse:
        response = self._client.get(f"/v1/refinement-jobs/{job_id}")
        response.raise_for_status()
        return JobStatusResponse.model_validate(response.json())

    def download_artifacts(self, job_id: str, output_dir: Path | str) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = self._client.get(f"/v1/refinement-jobs/{job_id}/artifacts")
        meta.raise_for_status()
        artifacts = ArtifactsResponse.model_validate(meta.json())
        # Prefer streaming the archive endpoint content.
        response = self._client.get(f"/v1/refinement-jobs/{job_id}/artifacts/download")
        if response.status_code == 404 and artifacts.download_url:
            response = self._client.get(artifacts.download_url)
        response.raise_for_status()
        archive_path = output_dir / f"{job_id}_artifacts.zip"
        archive_path.write_bytes(response.content)
        return archive_path

    def cancel(self, job_id: str) -> JobStatusResponse:
        response = self._client.delete(f"/v1/refinement-jobs/{job_id}")
        response.raise_for_status()
        return JobStatusResponse.model_validate(response.json())

    def wait(
        self,
        job_id: str,
        *,
        poll_interval: float | None = None,
        timeout: float | None = None,
        on_status=None,
    ) -> JobStatusResponse:
        poll_interval = float(poll_interval if poll_interval is not None else self.config.poll_interval_seconds)
        timeout = float(timeout if timeout is not None else self.config.job_timeout_seconds)
        deadline = time.time() + timeout
        last: JobStatusResponse | None = None
        while time.time() < deadline:
            last = self.get_status(job_id)
            if on_status is not None:
                on_status(last)
            if last.status in TERMINAL_STATUSES:
                return last
            time.sleep(poll_interval)
        raise TimeoutError(f"job {job_id} did not finish within {timeout}s; last={last}")

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        token = self.config.auth_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


def load_remote_job_record(path: Path | str) -> RemoteJobRecord | None:
    path = Path(path)
    if not path.exists():
        return None
    return RemoteJobRecord.model_validate_json(path.read_text(encoding="utf-8"))


def save_remote_job_record(path: Path | str, record: RemoteJobRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def make_idempotency_key(object_id: str, input_sha256: str, scene_id: str | None = None) -> str:
    prefix = scene_id or "scene"
    return f"{prefix}:{object_id}:{input_sha256}"


def upsert_job_record(
    path: Path | str,
    *,
    job_id: str,
    object_id: str,
    input_sha256: str,
    idempotency_key: str,
    status: str,
    package_path: str,
    last_status: JobStatusResponse | None = None,
) -> RemoteJobRecord:
    existing = load_remote_job_record(path)
    now = utc_now_iso()
    record = RemoteJobRecord(
        job_id=job_id,
        object_id=object_id,
        input_sha256=input_sha256,
        idempotency_key=idempotency_key,
        status=status,  # type: ignore[arg-type]
        package_path=package_path,
        created_at=existing.created_at if existing else now,
        updated_at=now,
        last_status=last_status,
    )
    save_remote_job_record(path, record)
    return record
