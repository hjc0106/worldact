"""Job workspace storage under /data/jobs/<job_id> (or local override)."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

from services.artifixer_service.models import JobStatusResponse, utc_now_iso


def data_root() -> Path:
    return Path(os.environ.get("ARTIFIXER_DATA_ROOT", "/data/jobs"))


class JobStorage:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else data_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "index.json"
        if not self.meta_path.exists():
            self._write_index({})

    def job_dir(self, job_id: str) -> Path:
        path = self.root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_upload(self, job_id: str, archive_bytes: bytes, filename: str = "input.zip") -> Path:
        job_dir = self.job_dir(job_id)
        archive_path = job_dir / filename
        archive_path.write_bytes(archive_bytes)
        return archive_path

    def write_status(self, job_id: str, status: JobStatusResponse) -> None:
        job_dir = self.job_dir(job_id)
        (job_dir / "status.json").write_text(status.model_dump_json(indent=2), encoding="utf-8")
        index = self._read_index()
        index[job_id] = status.model_dump()
        # Idempotency map
        if status.idempotency_key:
            idem = index.setdefault("_idempotency", {})
            idem[status.idempotency_key] = job_id
        self._write_index(index)

    def read_status(self, job_id: str) -> JobStatusResponse | None:
        path = self.job_dir(job_id) / "status.json"
        if not path.exists():
            return None
        return JobStatusResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def find_by_idempotency(self, key: str) -> str | None:
        index = self._read_index()
        return (index.get("_idempotency") or {}).get(key)

    def artifacts_dir(self, job_id: str) -> Path:
        path = self.job_dir(job_id) / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def workspace_dir(self, job_id: str) -> Path:
        path = self.job_dir(job_id) / "workspace"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def package_extract_dir(self, job_id: str) -> Path:
        path = self.job_dir(job_id) / "package"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, job_id: str, name: str, payload: dict[str, Any]) -> Path:
        path = self.job_dir(job_id) / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def package_artifacts_zip(self, job_id: str) -> Path:
        artifacts = self.artifacts_dir(job_id)
        out = self.job_dir(job_id) / "artifacts.zip"
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(artifacts.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(artifacts).as_posix())
        return out

    def delete_job(self, job_id: str) -> None:
        path = self.root / job_id
        if path.exists():
            shutil.rmtree(path)
        index = self._read_index()
        index.pop(job_id, None)
        idem = index.get("_idempotency") or {}
        drop = [k for k, v in idem.items() if v == job_id]
        for k in drop:
            idem.pop(k, None)
        index["_idempotency"] = idem
        self._write_index(index)

    def _read_index(self) -> dict[str, Any]:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _write_index(self, data: dict[str, Any]) -> None:
        self.meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def new_job_id() -> str:
    stamp = utc_now_iso().replace("-", "").replace(":", "")[:15]
    import secrets

    return f"arf_{stamp}_{secrets.token_hex(2)}"
