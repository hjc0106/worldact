"""FastAPI HTTP surface for asynchronous refinement jobs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.artifixer_service.models import (
    ArtifactsResponse,
    JobCreateResponse,
    JobError,
    JobStatusResponse,
    PackageManifest,
    utc_now_iso,
)
from services.artifixer_service.queue import JobQueue, build_queue
from services.artifixer_service.storage import JobStorage, new_job_id
from services.artifixer_service.worker import run_job

app = FastAPI(title="ArtiFixer Refinement Service", version="0.1.0")
security = HTTPBearer(auto_error=False)

def _init_storage() -> JobStorage:
    root = os.environ.get("ARTIFIXER_DATA_ROOT")
    return JobStorage(Path(root) if root else None)


_storage = _init_storage()
_queue: JobQueue = build_queue()


def get_storage() -> JobStorage:
    return _storage


def get_queue() -> JobQueue:
    return _queue


def reset_runtime(storage: JobStorage | None = None, queue: JobQueue | None = None) -> None:
    """Test helper to rebind storage/queue after env changes."""
    global _storage, _queue
    _storage = storage or _init_storage()
    _queue = queue or build_queue()


def require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> None:
    expected = os.environ.get("ARTIFIXER_SERVICE_TOKEN")
    if not expected:
        return
    if credentials is None or credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/v1/refinement-jobs", response_model=JobCreateResponse, dependencies=[Depends(require_auth)])
async def create_job(
    archive: UploadFile = File(...),
    manifest: str | None = Form(default=None),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    storage: JobStorage = Depends(get_storage),
    queue: JobQueue = Depends(get_queue),
) -> JobCreateResponse:
    if idempotency_key:
        existing_id = storage.find_by_idempotency(idempotency_key)
        if existing_id:
            existing = storage.read_status(existing_id)
            if existing is not None:
                return JobCreateResponse(job_id=existing.job_id, status=existing.status, created_at=existing.created_at)

    raw = await archive.read()
    max_bytes = int(os.environ.get("ARTIFIXER_MAX_ARCHIVE_BYTES", str(2_000_000_000)))
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="archive too large")

    input_sha = None
    if manifest:
        try:
            parsed = PackageManifest.model_validate_json(manifest)
            input_sha = parsed.input_sha256
            if parsed.model not in {"artifixer-14b"}:
                raise HTTPException(status_code=400, detail="unknown model alias")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid manifest: {exc}") from exc

    job_id = new_job_id()
    storage.save_upload(job_id, raw, filename="input.zip")
    created_at = utc_now_iso()
    status = JobStatusResponse(
        job_id=job_id,
        status="queued",
        stage="queued",
        progress=0.0,
        created_at=created_at,
        updated_at=created_at,
        artifacts_ready=False,
        input_sha256=input_sha,
        idempotency_key=idempotency_key,
        message="queued",
    )
    storage.write_status(job_id, status)
    queue.enqueue(run_job, job_id)
    return JobCreateResponse(job_id=job_id, status="queued", created_at=created_at)


@app.get("/v1/refinement-jobs/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(require_auth)])
def get_job(job_id: str, storage: JobStorage = Depends(get_storage)) -> JobStatusResponse:
    status = storage.read_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    return status


@app.get("/v1/refinement-jobs/{job_id}/artifacts", response_model=ArtifactsResponse, dependencies=[Depends(require_auth)])
def get_artifacts(job_id: str, storage: JobStorage = Depends(get_storage)) -> ArtifactsResponse:
    status = storage.read_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    if status.status not in {"completed", "completed_with_validation_failure"}:
        raise HTTPException(status_code=409, detail=f"artifacts not ready; status={status.status}")
    zip_path = storage.job_dir(job_id) / "artifacts.zip"
    if not zip_path.exists():
        zip_path = storage.package_artifacts_zip(job_id)
    from services.artifixer_service.artifact import sha256_file

    return ArtifactsResponse(
        job_id=job_id,
        status=status.status,
        download_url=f"/v1/refinement-jobs/{job_id}/artifacts/download",
        checksum=sha256_file(zip_path),
    )


@app.get("/v1/refinement-jobs/{job_id}/artifacts/download", dependencies=[Depends(require_auth)])
def download_artifacts(job_id: str, storage: JobStorage = Depends(get_storage)):
    status = storage.read_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    zip_path = storage.job_dir(job_id) / "artifacts.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="artifacts zip missing")
    return FileResponse(zip_path, filename=f"{job_id}_artifacts.zip", media_type="application/zip")


@app.delete("/v1/refinement-jobs/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(require_auth)])
def cancel_job(job_id: str, storage: JobStorage = Depends(get_storage)) -> JobStatusResponse:
    status = storage.read_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    if status.status in {"completed", "failed", "cancelled"}:
        return status
    cancelled = JobStatusResponse(
        job_id=status.job_id,
        status="cancelled",
        stage="cancelled",
        progress=status.progress,
        created_at=status.created_at,
        updated_at=utc_now_iso(),
        artifacts_ready=False,
        error=JobError(code="CANCELLED", message="cancelled by client", retryable=False),
        input_sha256=status.input_sha256,
        idempotency_key=status.idempotency_key,
        message="cancelled",
    )
    storage.write_status(job_id, cancelled)
    return cancelled
