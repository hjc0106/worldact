from __future__ import annotations

import pytest
from pydantic import ValidationError

from worldact.stage2_artifixer.contracts import (
    SCHEMA_VERSION,
    FileChecksum,
    JobStatusResponse,
    PackageManifest,
)


def test_manifest_rejects_unknown_schema() -> None:
    with pytest.raises(ValidationError):
        PackageManifest(
            schema_version="9.9",
            object_id="1",
            object_name="truck",
            prompt="a truck",
            anchor_count=3,
            target_count=4,
            files=[FileChecksum(path="object_local.ply", sha256="a" * 64)],
            input_sha256="b" * 64,
        )


def test_manifest_rejects_unknown_model() -> None:
    with pytest.raises(ValidationError):
        PackageManifest(
            schema_version=SCHEMA_VERSION,
            object_id="1",
            object_name="truck",
            prompt="a truck",
            model="/etc/passwd",
            anchor_count=3,
            target_count=4,
            files=[FileChecksum(path="object_local.ply", sha256="a" * 64)],
            input_sha256="b" * 64,
        )


def test_job_status_requires_structured_error_fields() -> None:
    status = JobStatusResponse(
        job_id="arf_x",
        status="failed",
        stage="validating",
        progress=0.1,
        created_at="2026-07-17T10:00:00Z",
        updated_at="2026-07-17T10:00:01Z",
        error={"code": "SCHEMA_INVALID", "message": "bad", "retryable": False},
    )
    assert status.error is not None
    assert status.error.code == "SCHEMA_INVALID"
    assert status.error.retryable is False
