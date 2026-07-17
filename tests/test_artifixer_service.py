import os
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from tests.test_artifixer_package import make_synthetic_stage
from worldact.stage2_artifixer.config import ArtiFixerRefinementConfig, ArtiFixerServiceConfig, PrepareConfig
from worldact.stage2_artifixer.inputs import load_stage_artifixer_input
from worldact.stage2_artifixer.package import archive_package, prepare_object_package
from worldact.stage2_artifixer.remote_client import ArtiFixerRemoteClient, make_idempotency_key
from worldact.stage2_artifixer.result_import import extract_artifacts, import_refined_result
from worldact.stage2_artifixer.runner import ArtiFixerRefinementRunner


@pytest.fixture()
def service_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFIXER_DATA_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("ARTIFIXER_PIPELINE", "stub")
    monkeypatch.setenv("ARTIFIXER_SERVICE_TOKEN", "test-token")
    monkeypatch.setenv("ARTIFIXER_USE_REDIS", "0")
    from services.artifixer_service.app import reset_runtime
    from services.artifixer_service.queue import InMemoryQueue
    from services.artifixer_service.storage import JobStorage
    import services.artifixer_service.app as app_mod

    reset_runtime(JobStorage(tmp_path / "jobs"), InMemoryQueue(max_workers=1))
    return app_mod


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

    app = service_env.app
    client = TestClient(app)
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
    assert "progress" in status

    arts = client.get(f"/v1/refinement-jobs/{job_id}/artifacts", headers={"Authorization": "Bearer test-token"})
    assert arts.status_code == 200
    dl = client.get(f"/v1/refinement-jobs/{job_id}/artifacts/download", headers={"Authorization": "Bearer test-token"})
    assert dl.status_code == 200
    out_zip = tmp_path / "artifacts.zip"
    out_zip.write_bytes(dl.content)
    artifact_dir = extract_artifacts(out_zip, tmp_path / "imported")
    result = import_refined_result(
        artifact_dir,
        package_dir / "normalization.json",
        tmp_path / "imported_out",
        input_local_ply=package_dir / "object_local.ply",
    )
    assert result.status == "completed"
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

    app = service_env.app
    asgi_client = TestClient(app)

    class _TestClientAdapter(ArtiFixerRemoteClient):
        def __init__(self):
            # Skip real httpx client init networking; reuse TestClient.
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
