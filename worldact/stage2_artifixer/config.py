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
    # Optional elevation bands (deg). When set, override the fixed *_elevation_deg
    # values and sample evenly across [lo, hi] for that ring.
    horizontal_elevation_range_deg: tuple[float, float] | None = None
    high_elevation_range_deg: tuple[float, float] | None = None
    # Hard floor for all generated target elevations (deg). Use e.g. 8.0 to
    # forbid underside / negative-pitch cameras.
    min_elevation_deg: float = -90.0
    camera_distance_scale: float = 2.2
    # Optional per-camera distance jitter around the fitted base distance.
    camera_distance_scale_range: tuple[float, float] | None = None
    # Keep this fraction of the image as empty border when fitting distance.
    edge_margin: float = 0.12
    # Aim cameras at the robust (quantile) object center instead of the origin.
    look_at_robust_center: bool = False
    # Manual vehicle front direction in local XY. Prefer front_yaw_deg; if both
    # are unset, cameras keep the legacy world-Z up convention.
    front_yaw_deg: float | None = None
    front_axis: tuple[float, float, float] | None = None
    # Vehicle upright axis. None → PCA (non-length axis most aligned with +Z).
    up_axis: tuple[float, float, float] | None = None
    # Sphere-ring orbit: cameras on a sphere concentric with the object
    # circumsphere, at fixed elevation above the vehicle equator.
    sphere_ring: bool = False
    ring_elevation_deg: float = 12.0
    # Shift look-at toward +up by this fraction of circumradius (avoids underbody).
    look_at_up_bias: float = 0.08
    # 1.0 places cameras on the circumsphere; >1 moves them outward.
    circumradius_scale: float = 1.0
    # When set, restrict target azimuth to front_yaw ± this half-span (deg).
    front_azimuth_half_span_deg: float | None = None
    # Optional front-hemisphere rings (used when > 0). Counts are in addition to
    # the legacy horizontal/high/low rings unless those are set to 0.
    front_targets: int = 0
    front_elevation_range_deg: tuple[float, float] = (25.0, 45.0)
    side_targets: int = 0
    side_elevation_range_deg: tuple[float, float] = (25.0, 35.0)
    overhead_targets: int = 0
    overhead_elevation_range_deg: tuple[float, float] = (50.0, 60.0)
    # Sort generated frames into a smooth azimuth/elevation path.
    sort_trajectory_smooth: bool = False
    # object_centered keeps Stage-1 view directions for selection, but renders
    # from fresh cameras aimed at the normalized object. source_crop retains
    # the legacy Stage-1-camera crop path.
    anchor_camera_mode: str = "object_centered"
    anchor_alpha_threshold: float = 0.01
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
    prepare_kwargs = {k: v for k, v in prepare_raw.items() if k in PrepareConfig.__dataclass_fields__}
    for key in (
        "horizontal_elevation_range_deg",
        "high_elevation_range_deg",
        "camera_distance_scale_range",
        "quantile",
        "front_axis",
        "up_axis",
        "front_elevation_range_deg",
        "side_elevation_range_deg",
        "overhead_elevation_range_deg",
    ):
        if key in prepare_kwargs and prepare_kwargs[key] is not None:
            prepare_kwargs[key] = tuple(prepare_kwargs[key])
    prepare = PrepareConfig(**prepare_kwargs)
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
