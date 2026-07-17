"""Service-side re-exports of shared contracts (same schemas as the local client)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `services/artifixer_service` to import worldact when run from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worldact.stage2_artifixer.contracts import (  # noqa: E402
    SCHEMA_VERSION,
    TERMINAL_STATUSES,
    ArtifactFile,
    ArtifactManifest,
    ArtifactsResponse,
    FileChecksum,
    JobCreateResponse,
    JobError,
    JobStatus,
    JobStatusResponse,
    PackageManifest,
    utc_now_iso,
)

__all__ = [
    "SCHEMA_VERSION",
    "TERMINAL_STATUSES",
    "ArtifactFile",
    "ArtifactManifest",
    "ArtifactsResponse",
    "FileChecksum",
    "JobCreateResponse",
    "JobError",
    "JobStatus",
    "JobStatusResponse",
    "PackageManifest",
    "utc_now_iso",
]
