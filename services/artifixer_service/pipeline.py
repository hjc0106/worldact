"""Remote ArtiFixer pipeline.

Default mode is a deterministic stub that copies object_local.ply to refined_local.ply
so API/client tests work without CUDA. Set ARTIFIXER_PIPELINE=official to invoke the
real subprocess chain when ArtiFixer + 3DGRUT are mounted in the container.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from services.artifixer_service.artifact import build_artifact_manifest, write_metrics
from services.artifixer_service.colmap_adapter import build_colmap_scene
from services.artifixer_service.models import JobError, JobStatusResponse, PackageManifest, utc_now_iso
from services.artifixer_service.prompt_encoder import write_caption_embedding
from services.artifixer_service.storage import JobStorage


MAX_ARCHIVE_BYTES = int(os.environ.get("ARTIFIXER_MAX_ARCHIVE_BYTES", str(2_000_000_000)))
MAX_FILES = int(os.environ.get("ARTIFIXER_MAX_FILES", "5000"))

MODEL_ALIAS_TO_PATH = {
    "artifixer-14b": os.environ.get("ARTIFIXER_CHECKPOINT", "/models/artifixer/artifixer-14b.pt"),
}


class PipelineError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


def run_pipeline(job_id: str, storage: JobStorage | None = None) -> JobStatusResponse:
    storage = storage or JobStorage()
    status = storage.read_status(job_id)
    if status is None:
        raise PipelineError("JOB_NOT_FOUND", f"unknown job_id={job_id}")

    def update(stage: str, progress: float, job_status: str, message: str | None = None, error: JobError | None = None):
        nonlocal status
        assert status is not None
        status = JobStatusResponse(
            job_id=status.job_id,
            status=job_status,  # type: ignore[arg-type]
            stage=stage,
            progress=progress,
            created_at=status.created_at,
            updated_at=utc_now_iso(),
            artifacts_ready=job_status == "completed",
            error=error,
            input_sha256=status.input_sha256,
            idempotency_key=status.idempotency_key,
            message=message,
        )
        storage.write_status(job_id, status)
        return status

    try:
        update("validating", 0.05, "validating")
        package_dir = _extract_and_validate(job_id, storage)
        manifest = PackageManifest.model_validate_json((package_dir / "manifest.json").read_text(encoding="utf-8"))

        update("preparing", 0.15, "preparing", message="building colmap scene")
        workspace = storage.workspace_dir(job_id)
        scene_root = build_colmap_scene(package_dir, workspace)
        caption_path = write_caption_embedding(manifest.prompt, workspace / "captions" / "caption.h5")

        mode = os.environ.get("ARTIFIXER_PIPELINE", "stub").lower()
        if mode == "official":
            update("running_artifixer", 0.35, "running_artifixer")
            _run_official(manifest, scene_root, workspace, caption_path)
            update("distilling", 0.65, "distilling")
            update("exporting", 0.85, "exporting")
        else:
            update("running_artifixer", 0.40, "running_artifixer", message="stub inference")
            update("distilling", 0.70, "distilling", message="stub distillation")
            update("exporting", 0.85, "exporting")
            _run_stub(package_dir, storage.artifacts_dir(job_id), scene_root, caption_path)

        artifacts = storage.artifacts_dir(job_id)
        write_metrics(
            artifacts,
            {
                "mode": mode,
                "object_id": manifest.object_id,
                "anchor_count": manifest.anchor_count,
                "target_count": manifest.target_count,
                "distillation_steps": manifest.distillation_steps,
            },
        )
        (artifacts / "run.log").write_text(f"job={job_id} mode={mode} completed\n", encoding="utf-8")
        build_artifact_manifest(job_id, artifacts, "completed")
        storage.package_artifacts_zip(job_id)
        return update("completed", 1.0, "completed", message="ok")
    except PipelineError as exc:
        err = JobError(code=exc.code, message=exc.message, retryable=exc.retryable)
        return update("failed", status.progress if status else 0.0, "failed", error=err, message=exc.message)
    except Exception as exc:
        err = JobError(code="INTERNAL_ERROR", message=str(exc), retryable=False)
        return update("failed", status.progress if status else 0.0, "failed", error=err, message=str(exc))


def _extract_and_validate(job_id: str, storage: JobStorage) -> Path:
    job_dir = storage.job_dir(job_id)
    archive = job_dir / "input.zip"
    if not archive.exists():
        raise PipelineError("INPUT_MISSING", "input.zip not found", retryable=False)
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise PipelineError("INPUT_TOO_LARGE", f"archive exceeds {MAX_ARCHIVE_BYTES} bytes", retryable=False)

    package_dir = storage.package_extract_dir(job_id)
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        if len(zf.infolist()) > MAX_FILES:
            raise PipelineError("TOO_MANY_FILES", f"archive has >{MAX_FILES} files", retryable=False)
        for info in zf.infolist():
            rel = Path(info.filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise PipelineError("UNSAFE_PATH", f"illegal path {info.filename}", retryable=False)
            target = (package_dir / info.filename).resolve()
            if not str(target).startswith(str(package_dir.resolve())):
                raise PipelineError("UNSAFE_PATH", f"illegal path {info.filename}", retryable=False)
        zf.extractall(package_dir)

    # Validate via worldact validators when available.
    try:
        from worldact.stage2_artifixer.validation import validate_package_dir

        result = validate_package_dir(package_dir)
        if not result.ok:
            raise PipelineError("SCHEMA_INVALID", "; ".join(result.errors), retryable=False)
    except ImportError:
        if not (package_dir / "manifest.json").exists():
            raise PipelineError("SCHEMA_INVALID", "manifest.json missing", retryable=False)
        PackageManifest.model_validate_json((package_dir / "manifest.json").read_text(encoding="utf-8"))
    return package_dir


def _run_stub(package_dir: Path, artifacts_dir: Path, scene_root: Path, caption_path: Path) -> None:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(package_dir / "object_local.ply", artifacts_dir / "refined_local.ply")
    # Minimal predicted frames / renders for traceability.
    pred = artifacts_dir / "predicted_frames"
    before = artifacts_dir / "renders_before"
    after = artifacts_dir / "renders_after"
    for d in (pred, before, after):
        d.mkdir(parents=True, exist_ok=True)
    anchors_dir = package_dir / "anchors"
    for src in sorted(anchors_dir.glob("frame_*.png")):
        if "_mask" in src.name:
            continue
        shutil.copy2(src, pred / src.name)
        shutil.copy2(src, before / src.name)
        shutil.copy2(src, after / src.name)
    ckpt = artifacts_dir / "artifixer3d_checkpoint.pt"
    ckpt.write_bytes(b"STUB_CHECKPOINT")
    shutil.copy2(caption_path, artifacts_dir / caption_path.name) if caption_path.exists() else None
    (artifacts_dir / "scene_root.txt").write_text(str(scene_root), encoding="utf-8")


def _run_official(manifest: PackageManifest, scene_root: Path, workspace: Path, caption_path: Path) -> None:
    """Invoke official ArtiFixer commands via argv arrays (never shell=True)."""
    artifixer_root = Path(os.environ.get("ARTIFIXER_ROOT", "/opt/ArtiFixer"))
    python_bin = os.environ.get("ARTIFIXER_PYTHON", "python")
    ckpt = MODEL_ALIAS_TO_PATH.get(manifest.model)
    if ckpt is None or not Path(ckpt).exists():
        raise PipelineError("CHECKPOINT_MISSING", f"checkpoint for model {manifest.model} not mounted", retryable=False)
    if not artifixer_root.exists():
        raise PipelineError("ARTIFIXER_MISSING", f"ARTIFIXER_ROOT not found: {artifixer_root}", retryable=False)

    split = {
        "test": {
            f"object_{manifest.object_id}": {
                "transforms_path": str(scene_root / "transforms.json"),
                "image_root": str(scene_root / "images"),
                "render_dir": str(scene_root / "renders"),
                "opacity_dir": str(scene_root / "opacity"),
                "selected_indices_path": str(scene_root / "selected_indices.json"),
                "target_indices_path": str(scene_root / "target_indices.json"),
                "prompt_path": str(caption_path),
                "camera_scale": 1.0,
                "has_gt": False,
            }
        }
    }
    split_path = workspace / "split.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    save_dir = workspace / "artifixer_out"
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_bin,
        "-m",
        "model_eval.run_inference",
        "--ckpt",
        str(ckpt),
        "--evalset",
        "reconstructed_colmap",
        "--split_path",
        str(split_path),
        "--save_dir",
        str(save_dir),
    ]
    _run_argv(cmd, cwd=artifixer_root, timeout=int(os.environ.get("ARTIFIXER_INFER_TIMEOUT", "7200")))

    # Distillation entrypoint is environment-specific; document expected hook.
    distill_script = artifixer_root / "scripts" / "run_artifixer3d.py"
    if distill_script.exists():
        distill_cmd = [
            python_bin,
            str(distill_script),
            "--scene",
            str(scene_root),
            "--predictions",
            str(save_dir),
            "--steps",
            str(manifest.distillation_steps),
            "--out",
            str(workspace / "artifacts_staging"),
        ]
        _run_argv(distill_cmd, cwd=artifixer_root, timeout=int(os.environ.get("ARTIFIXER_DISTILL_TIMEOUT", "21600")))


def _run_argv(cmd: list[str], *, cwd: Path, timeout: int) -> None:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("TIMEOUT", f"command timed out: {cmd[0:3]}...", retryable=True) from exc
    if completed.returncode != 0:
        # Do not echo secrets; truncate stderr for diagnostics.
        stderr = (completed.stderr or "")[-2000:]
        if "out of memory" in stderr.lower() or "cuda oom" in stderr.lower():
            raise PipelineError("GPU_OOM", stderr or "GPU OOM", retryable=True)
        raise PipelineError("SUBPROCESS_FAILED", stderr or f"exit {completed.returncode}", retryable=False)
