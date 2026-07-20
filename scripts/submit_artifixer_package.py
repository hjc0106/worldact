#!/usr/bin/env python3
"""Submit an existing object_package.zip to the ArtiFixer refinement API.

Does not require Stage-1 PLY/masks — use this to smoke-test the remote Job
contract against ARTIFIXER_PIPELINE=stub.

Example:
  export ARTIFIXER_SERVICE_URL=http://127.0.0.1:8000
  python scripts/submit_artifixer_package.py \\
    --archive /path/to/object_package.zip \\
    --output-dir /tmp/artifixer-smoke \\
    --wait
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldact.stage2_artifixer.config import ArtiFixerServiceConfig
from worldact.stage2_artifixer.contracts import PackageManifest
from worldact.stage2_artifixer.remote_client import (
    ArtiFixerRemoteClient,
    make_idempotency_key,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Submit object_package.zip to ArtiFixer refinement Jobs")
    p.add_argument("--archive", required=True, help="Path to object_package.zip")
    p.add_argument("--output-dir", required=True, help="Directory for downloaded artifacts")
    p.add_argument("--service-url", default=None, help="Override ARTIFIXER_SERVICE_URL")
    p.add_argument("--scene-id", default="smoke")
    p.add_argument("--idempotency-key", default=None)
    p.add_argument("--wait", action="store_true", help="Poll until terminal status")
    p.add_argument("--download", action="store_true", help="Download artifacts after wait")
    p.add_argument("--poll-interval", type=float, default=2.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive = Path(args.archive)
    if not archive.is_file():
        raise SystemExit(f"archive not found: {archive}")

    import zipfile

    with zipfile.ZipFile(archive) as zf:
        manifest = PackageManifest.model_validate_json(zf.read("manifest.json"))

    cfg = ArtiFixerServiceConfig()
    if args.service_url:
        cfg.base_url = args.service_url
    cfg.poll_interval_seconds = args.poll_interval

    key = args.idempotency_key or make_idempotency_key(manifest.object_id, manifest.input_sha256, args.scene_id)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with ArtiFixerRemoteClient(cfg) as client:
        health = client.health()
        print(json.dumps({"health": health}, indent=2))
        created = client.submit(archive, key, manifest_json=manifest.model_dump_json())
        print(json.dumps(created.model_dump(), indent=2))
        if not args.wait:
            return 0
        status = client.wait(created.job_id, on_status=lambda s: print(f"{s.status} {s.stage} {s.progress:.2f}"))
        print(json.dumps(status.model_dump(), indent=2))
        if args.download and status.status in {"completed", "completed_with_validation_failure"}:
            path = client.download_artifacts(created.job_id, out)
            print(json.dumps({"downloaded": str(path)}, indent=2))
            if status.status != "completed":
                return 2
        elif status.status != "completed":
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
