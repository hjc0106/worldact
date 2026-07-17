from __future__ import annotations

from worldact.stage2_artifixer.config import ArtiFixerServiceConfig
from worldact.stage2_artifixer.remote_client import make_idempotency_key


def test_idempotency_key_stable() -> None:
    a = make_idempotency_key("1", "abc", "sceneA")
    b = make_idempotency_key("1", "abc", "sceneA")
    assert a == b
    assert a == "sceneA:1:abc"


def test_service_config_reads_token_env(monkeypatch) -> None:
    monkeypatch.setenv("ARTIFIXER_SERVICE_TOKEN", "secret")
    cfg = ArtiFixerServiceConfig()
    assert cfg.auth_token() == "secret"
