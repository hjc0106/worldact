"""Client integration against the root ArtiFixer refinement Job API (stub).

The service implementation lives in the parent ArtiFixer repo:
`../artifixer_service` (started via ArtiFixer/run_svc.sh).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_artifixer_package import make_synthetic_stage
from worldact.stage2_artifixer.config import ArtiFixerRefinementConfig, ArtiFixerServiceConfig, PrepareConfig
from worldact.stage2_artifixer.inputs import load_stage_artifixer_input
from worldact.stage2_artifixer.package import archive_package, prepare_object_package
from worldact.stage2_artifixer.remote_client import ArtiFixerRemoteClient, make_idempotency_key
from worldact.stage2_artifixer.result_import import extract_artifacts, import_refined_result
from worldact.stage2_artifixer.runner import ArtiFixerRefinementRunner

# Parent ArtiFixer repo (…/ArtiFixer). Insert after worldact imports so `tests.*`
# still resolves to worldact/tests, not ArtiFixer/tests.
_ARTIFIXER_ROOT = Path(__file__).resolve().parents[2]


def _ensure_artifixer_on_path() -> None:
    root = str(_ARTIFIXER_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@pytest.fixture()
def service_env(tmp_path, monkeypatch):
    _ensure_artifixer_on_path()
    pytest.importorskip("artifixer_service")
    monkeypatch.setenv("ARTIFIXER_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("ARTIFIXER_PIPELINE", "stub")
    monkeypatch.setenv("ARTIFIXER_SERVICE_TOKEN", "test-token")

    from artifixer_service.refinement.queue import InMemoryQueue
    from artifixer_service.refinement.routes import reset_runtime, router
    from artifixer_service.refinement.storage import JobStorage

    reset_runtime(JobStorage(tmp_path / "jobs"), InMemoryQueue(max_workers=1))
    app = FastAPI()
    app.include_router(router)
    return app


def _wait_completed(client: TestClient, job_id: str, token: str, timeout: float = 10.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/v1/refinement-jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in {"completed", "failed", "cancelled"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {last}")


def test_service_submit_idempotency_and_download(tmp_path, service_env):
    chunk, stage = make_synthetic_stage(tmp_path / "scene")
    stage_input = load_stage_artifixer_input(chunk, stage, device="cpu")
    cfg = PrepareConfig(
        anchor_count=3,
        min_anchor_count=3,
        target_count=4,
        horizontal_targets=4,
        high_elevation_targets=0,
        low_elevation_targets=0,
        image_size=64,
    )
    package_dir, manifest = prepare_object_package(stage_input, "1", tmp_path / "prep", cfg)
    archive = archive_package(package_dir)

    client = TestClient(service_env)
    headers = {"Authorization": "Bearer test-token", "Idempotency-Key": make_idempotency_key("1", manifest.input_sha256)}
    files = {"archive": ("object_package.zip", archive.read_bytes(), "application/zip")}
    data = {"manifest": (package_dir / "manifest.json").read_text(encoding="utf-8")}

    r1 = client.post("/v1/refinement-jobs", headers=headers, files=files, data=data)
    assert r1.status_code == 200, r1.text
    job_id = r1.json()["job_id"]

    r2 = client.post("/v1/refinement-jobs", headers=headers, files=files, data=data)
    assert r2.status_code == 200
    assert r2.json()["job_id"] == job_id

    status = _wait_completed(client, job_id, "test-token")
    assert status["status"] == "completed"
    assert status["error"] is None
    assert status.get("message") == "stub_ok"

    arts = client.get(f"/v1/refinement-jobs/{job_id}/artifacts", headers={"Authorization": "Bearer test-token"})
    assert arts.status_code == 200
    dl = client.get(f"/v1/refinement-jobs/{job_id}/artifacts/download", headers={"Authorization": "Bearer test-token"})
    assert dl.status_code == 200
    out_zip = tmp_path / "artifacts.zip"
    out_zip.write_bytes(dl.content)
    artifact_dir = extract_artifacts(out_zip, tmp_path / "imported")
    summary = __import__("json").loads((artifact_dir / "refinement_summary.json").read_text(encoding="utf-8"))
    assert summary["pipeline"] == "stub"
    assert summary["refinement_applied"] is False
    result = import_refined_result(
        artifact_dir,
        package_dir / "normalization.json",
        tmp_path / "imported_out",
        input_local_ply=package_dir / "object_local.ply",
    )
    assert result.status == "completed"
    assert result.messages == []
    assert result.refined_world_ply is not None
    assert result.refined_world_ply.exists()


def test_runner_resume_does_not_resubmit(tmp_path, service_env, monkeypatch):
    chunk, stage = make_synthetic_stage(tmp_path / "scene")
    cfg = ArtiFixerRefinementConfig(
        prepare=PrepareConfig(
            anchor_count=3,
            min_anchor_count=3,
            target_count=4,
            horizontal_targets=4,
            high_elevation_targets=0,
            low_elevation_targets=0,
            image_size=64,
        ),
        service=ArtiFixerServiceConfig(
            base_url="http://testserver",
            token_env="ARTIFIXER_SERVICE_TOKEN",
            poll_interval_seconds=0.05,
        ),
        device="cpu",
    )
    monkeypatch.setenv("ARTIFIXER_SERVICE_TOKEN", "test-token")

    asgi_client = TestClient(service_env)

    class _TestClientAdapter(ArtiFixerRemoteClient):
        def __init__(self):
            self.config = cfg.service
            self._client = asgi_client

        def close(self) -> None:
            return None

        def submit(self, package_archive, idempotency_key, *, manifest_json=None):
            package_archive = Path(package_archive)
            files = {"archive": (package_archive.name, package_archive.read_bytes(), "application/zip")}
            data = {"manifest": manifest_json} if manifest_json else {}
            response = self._client.post(
                "/v1/refinement-jobs",
                files=files,
                data=data,
                headers={"Idempotency-Key": idempotency_key, "Authorization": "Bearer test-token"},
            )
            response.raise_for_status()
            from worldact.stage2_artifixer.contracts import JobCreateResponse

            return JobCreateResponse.model_validate(response.json())

        def get_status(self, job_id: str):
            response = self._client.get(
                f"/v1/refinement-jobs/{job_id}",
                headers={"Authorization": "Bearer test-token"},
            )
            response.raise_for_status()
            from worldact.stage2_artifixer.contracts import JobStatusResponse

            return JobStatusResponse.model_validate(response.json())

    runner = ArtiFixerRefinementRunner(
        ply_path=chunk,
        stage_dir=stage,
        object_id="1",
        output_dir=tmp_path / "run",
        config=cfg,
        client_factory=_TestClientAdapter,
    )
    runner.prepare()
    first = runner.submit()
    assert first.details and first.details.get("resumed") is False
    second = runner.submit()
    assert second.details and second.details.get("resumed") is True
    assert second.job is not None and first.job is not None
    assert second.job.job_id == first.job.job_id
