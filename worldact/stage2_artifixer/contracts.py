"""Versioned schemas shared by the local client and the remote ArtiFixer service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

MODEL_ALIASES = frozenset({"artifixer-14b"})

JobStatus = Literal[
    "queued",
    "validating",
    "preparing",
    "running_artifixer",
    "distilling",
    "exporting",
    "completed",
    "failed",
    "cancelled",
    "completed_with_validation_failure",
]

TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "completed_with_validation_failure"}
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class FileChecksum(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str = Field(min_length=64, max_length=64)

    @field_validator("sha256")
    @classmethod
    def _lower_hex(cls, value: str) -> str:
        value = value.lower().strip()
        if any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("sha256 must be lowercase hex")
        return value


class PackageManifest(BaseModel):
    """object_package/manifest.json"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    object_id: str
    object_name: str
    prompt: str
    pipeline: Literal["artifixer3d"] = "artifixer3d"
    model: str = "artifixer-14b"
    anchor_count: int = Field(ge=3)
    target_count: int = Field(ge=1)
    base_reconstruction_steps: int = Field(default=1000, ge=1)
    distillation_steps: int = Field(default=5000, ge=1)
    run_artifixer3d_plus: bool = False
    image_size: int = Field(default=512, ge=64)
    files: list[FileChecksum]
    input_sha256: str = Field(min_length=64, max_length=64)
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, value: str) -> str:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema_version={value!r}; supported={sorted(SUPPORTED_SCHEMA_VERSIONS)}")
        return value

    @field_validator("model")
    @classmethod
    def _known_model(cls, value: str) -> str:
        if value not in MODEL_ALIASES:
            raise ValueError(f"unknown model alias {value!r}; allowed={sorted(MODEL_ALIASES)}")
        return value

    @field_validator("input_sha256")
    @classmethod
    def _lower_input_hash(cls, value: str) -> str:
        return FileChecksum(path="_", sha256=value).sha256


class NormalizationTransform(BaseModel):
    """object_package/normalization.json — local = (world - center) / radius."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    center: list[float] = Field(min_length=3, max_length=3)
    radius: float = Field(gt=0.0)
    local_to_world: list[list[float]] = Field(min_length=4, max_length=4)
    world_to_local: list[list[float]] = Field(min_length=4, max_length=4)
    quantile: tuple[float, float] = (0.02, 0.98)
    num_gaussians: int = Field(ge=1)

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, value: str) -> str:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema_version={value!r}")
        return value


class AnchorCamera(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    source_view_index: int
    rgb_path: str
    mask_path: str
    transform_matrix: list[list[float]]  # c2w 4x4 in local coords
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    mask_area_ratio: float
    quality_score: float


class AnchorsDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    image_size: int = 512
    anchors: list[AnchorCamera]

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, value: str) -> str:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema_version={value!r}")
        return value


class TrajectoryFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    transform_matrix: list[list[float]]  # c2w 4x4 in local coords
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    kind: Literal["horizontal", "high", "low"] = "horizontal"


class TrajectoryDocument(BaseModel):
    """transforms.json-style target trajectory in local object coordinates."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    camera_model: Literal["OPENCV"] = "OPENCV"
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    frames: list[TrajectoryFrame]

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, value: str) -> str:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema_version={value!r}")
        return value


class JobError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class JobCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus
    created_at: str


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus
    stage: str
    progress: float = Field(ge=0.0, le=1.0)
    created_at: str
    updated_at: str
    artifacts_ready: bool = False
    error: JobError | None = None
    input_sha256: str | None = None
    idempotency_key: str | None = None
    message: str | None = None


class ArtifactFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    size_bytes: int = Field(ge=0)


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    job_id: str
    status: JobStatus
    files: list[ArtifactFile]
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, value: str) -> str:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema_version={value!r}")
        return value


class ArtifactsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus
    download_url: str | None = None
    checksum: str | None = None
    manifest: ArtifactManifest | None = None


class RemoteJobRecord(BaseModel):
    """Persisted at output/object_<id>/remote_job.json for resume."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    object_id: str
    input_sha256: str
    idempotency_key: str
    status: JobStatus
    package_path: str
    created_at: str
    updated_at: str
    last_status: JobStatusResponse | None = None
