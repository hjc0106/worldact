"""GPU worker entrypoint. One concurrent job per process/GPU."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.artifixer_service.pipeline import run_pipeline
from services.artifixer_service.storage import JobStorage


def run_job(job_id: str) -> dict:
    storage = JobStorage()
    status = run_pipeline(job_id, storage)
    return status.model_dump()


def main() -> None:
    """RQ worker process: rq worker artifixer."""
    redis_url = os.environ.get("ARTIFIXER_REDIS_URL") or os.environ.get("REDIS_URL") or "redis://redis:6379/0"
    from redis import Redis
    from rq import Worker, Queue

    queues = [Queue("artifixer", connection=Redis.from_url(redis_url))]
    Worker(queues).work(with_scheduler=False)


if __name__ == "__main__":
    main()
