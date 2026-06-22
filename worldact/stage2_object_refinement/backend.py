from __future__ import annotations

import subprocess
from pathlib import Path

from worldact.stage2_object_refinement.common import BackendRunResult


class FeedForwardBackend:
    def run(self, request_path: Path, work_dir: Path) -> BackendRunResult:
        raise NotImplementedError


class NoneBackend(FeedForwardBackend):
    def run(self, request_path: Path, work_dir: Path) -> BackendRunResult:
        return BackendRunResult(
            generated_ply=None,
            metadata_path=None,
            status="skipped",
            message=f"No feed-forward backend configured for {request_path}",
        )


class ExternalCommandBackend(FeedForwardBackend):
    def __init__(self, command: str, timeout: float = 600.0) -> None:
        self.command = command
        self.timeout = timeout

    def run(self, request_path: Path, work_dir: Path) -> BackendRunResult:
        command = self._format_command(request_path, work_dir)
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(work_dir),
                check=False,
                capture_output=True,
                text=True,
                timeout=float(self.timeout),
            )
        except subprocess.TimeoutExpired as exc:
            return BackendRunResult(
                generated_ply=None,
                metadata_path=None,
                status="timeout",
                message=f"Backend timed out after {self.timeout}s: {exc}",
            )

        log_path = work_dir / "backend.log"
        log_path.write_text(
            f"$ {command}\n\n[stdout]\n{completed.stdout}\n\n[stderr]\n{completed.stderr}\n",
            encoding="utf-8",
        )
        generated = work_dir / "generated.ply"
        metadata = work_dir / "metadata.json"
        if completed.returncode != 0:
            return BackendRunResult(
                generated_ply=str(generated) if generated.exists() else None,
                metadata_path=str(metadata) if metadata.exists() else None,
                status="failed",
                message=f"Backend exited with code {completed.returncode}; see {log_path}",
            )
        if not generated.exists():
            return BackendRunResult(
                generated_ply=None,
                metadata_path=str(metadata) if metadata.exists() else None,
                status="missing_output",
                message=f"Backend succeeded but did not write {generated}",
            )
        return BackendRunResult(
            generated_ply=str(generated),
            metadata_path=str(metadata) if metadata.exists() else None,
            status="ok",
            message=f"Generated {generated}",
        )

    def _format_command(self, request_path: Path, work_dir: Path) -> str:
        request = str(request_path)
        work = str(work_dir)
        if "{request}" in self.command or "{workdir}" in self.command:
            return self.command.format(request=request, workdir=work)
        return f"{self.command} {request}"


def make_backend(kind: str, command: str | None = None, timeout: float = 600.0, **kwargs) -> FeedForwardBackend:
    if kind == "none":
        return NoneBackend()
    if kind == "external":
        if not command:
            raise ValueError("--backend-command is required when --backend external")
        return ExternalCommandBackend(command, timeout=timeout)
    if kind == "sv3d":
        from worldact.stage2_object_refinement.sv3d_backend import SV3DBackend
        return SV3DBackend(
            generation_model=kwargs.get("generation_model", "sv3d"),
            reconstruction_model=kwargs.get("reconstruction_model", "gs_lrm"),
            inpaint_model=kwargs.get("inpaint_model", "lama"),
            num_generation_views=kwargs.get("num_generation_views", 6),
            generation_resolution=kwargs.get("generation_resolution", 512),
            icp_iterations=kwargs.get("icp_iterations", 50),
            device=kwargs.get("device", "cuda"),
        )
    raise ValueError(f"Unknown backend: {kind}")
