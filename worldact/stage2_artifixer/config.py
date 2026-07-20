"""Configuration for local preparation and the remote ArtiFixer service client."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PrepareConfig:
    anchor_count: int = 6
    min_anchor_count: int = 3
    target_count: int = 32
    horizontal_targets: int = 24
    high_elevation_targets: int = 4
    low_elevation_targets: int = 4
    image_size: int = 512
    min_mask_area_ratio: float = 0.20
    max_mask_area_ratio: float = 0.80
    max_boundary_edges: int = 2
    angular_dedup_deg: float = 25.0
    target_anchor_min_angle_deg: float = 5.0
    fov_deg: float = 60.0
    high_elevation_deg: float = 45.0
    low_elevation_deg: float = -20.0
    camera_distance_scale: float = 2.2
    quantile: tuple[float, float] = (0.02, 0.98)
    base_reconstruction_steps: int = 1000
    distillation_steps: int = 5000
    run_artifixer3d_plus: bool = False
    model: str = "artifixer-14b"
    pipeline: str = "artifixer3d"


@dataclass
class ArtiFixerServiceConfig:
    # Default points at the ArtiFixer Docker service (host port 8000).
    base_url: str = field(
        default_factory=lambda: os.environ.get("ARTIFIXER_SERVICE_URL", "http://127.0.0.1:8000")
    )
    token_env: str = "ARTIFIXER_SERVICE_TOKEN"
    request_timeout_seconds: float = 60.0
    job_timeout_seconds: float = 21600.0
    poll_interval_seconds: float = 10.0
    verify_tls: bool = True
    model: str = "artifixer-14b"
    distillation_steps: int = 5000
    run_artifixer3d_plus: bool = False

    def auth_token(self) -> str | None:
        return os.environ.get(self.token_env) or None

    def health_url(self) -> str:
        """Prefer /healthz (WorldAct contract); /health remains for legacy probes."""
        return f"{self.base_url.rstrip('/')}/healthz"


@dataclass
class ArtiFixerRefinementConfig:
    prepare: PrepareConfig = field(default_factory=PrepareConfig)
    service: ArtiFixerServiceConfig = field(default_factory=ArtiFixerServiceConfig)
    device: str = "cpu"


def load_artifixer_config(path: Path | str | None = None, overrides: dict[str, Any] | None = None) -> ArtiFixerRefinementConfig:
    cfg = ArtiFixerRefinementConfig()
    if path is not None:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg = _from_dict(raw)
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def _from_dict(raw: dict[str, Any]) -> ArtiFixerRefinementConfig:
    prepare_raw = dict(raw.get("prepare") or raw.get("artifixer_prepare") or {})
    service_raw = dict(raw.get("artifixer_service") or raw.get("service") or {})
    prepare = PrepareConfig(**{k: v for k, v in prepare_raw.items() if k in PrepareConfig.__dataclass_fields__})
    service = ArtiFixerServiceConfig(
        **{k: v for k, v in service_raw.items() if k in ArtiFixerServiceConfig.__dataclass_fields__}
    )
    # Env wins over YAML so smoke can retarget port without editing configs.
    env_url = os.environ.get("ARTIFIXER_SERVICE_URL")
    if env_url:
        service.base_url = env_url
    return ArtiFixerRefinementConfig(
        prepare=prepare,
        service=service,
        device=str(raw.get("device", "cpu")),
    )


def _merge(cfg: ArtiFixerRefinementConfig, overrides: dict[str, Any]) -> ArtiFixerRefinementConfig:
    prepare = PrepareConfig(**{**cfg.prepare.__dict__, **dict(overrides.get("prepare") or {})})
    service = ArtiFixerServiceConfig(**{**cfg.service.__dict__, **dict(overrides.get("service") or {})})
    return ArtiFixerRefinementConfig(
        prepare=prepare,
        service=service,
        device=str(overrides.get("device", cfg.device)),
    )
