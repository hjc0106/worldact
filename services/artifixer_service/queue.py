"""Task queue: Redis/RQ when available, otherwise in-process thread pool for tests/MVP."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class JobQueue:
    def enqueue(self, func: Callable[..., Any], *args, **kwargs) -> None:
        raise NotImplementedError


class InMemoryQueue(JobQueue):
    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def enqueue(self, func: Callable[..., Any], *args, **kwargs) -> None:
        self._executor.submit(func, *args, **kwargs)


class RedisRQQueue(JobQueue):
    def __init__(self, redis_url: str, queue_name: str = "artifixer") -> None:
        from redis import Redis
        from rq import Queue

        self._queue = Queue(queue_name, connection=Redis.from_url(redis_url))

    def enqueue(self, func: Callable[..., Any], *args, **kwargs) -> None:
        # RQ requires importable callables; worker.py:run_job is the entrypoint.
        self._queue.enqueue(func, *args, **kwargs, job_timeout=os.environ.get("ARTIFIXER_JOB_TIMEOUT", "21600"))


def build_queue() -> JobQueue:
    redis_url = os.environ.get("ARTIFIXER_REDIS_URL") or os.environ.get("REDIS_URL")
    if redis_url and os.environ.get("ARTIFIXER_USE_REDIS", "0") == "1":
        return RedisRQQueue(redis_url)
    workers = int(os.environ.get("ARTIFIXER_QUEUE_WORKERS", "1"))
    return InMemoryQueue(max_workers=workers)
