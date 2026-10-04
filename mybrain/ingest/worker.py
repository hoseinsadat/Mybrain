"""Background worker: claims ingestion jobs one at a time (parsing is the memory hog on a small VPS)."""

from __future__ import annotations

import logging
import signal
import time
from typing import Any

from ..config import get_settings
from ..db import connection, run_migrations
from .pipeline import ingest_document
from .queue import claim_next, complete, requeue_stale

log = logging.getLogger("mybrain.worker")
_stop = False


def _handle_stop(*_: Any) -> None:
    global _stop
    _stop = True


def process(job: dict[str, Any]) -> None:
    payload = job.get("payload") or {}
    with connection() as conn:
        try:
            if job["kind"] in ("ingest", "reindex"):
                ingest_document(
                    conn,
                    str(job["document_id"]),
                    reparse=bool(payload.get("reparse")),
                    index_version_id=payload.get("index_version_id"),
                )
            else:
                raise ValueError(f"unknown job kind {job['kind']}")
            complete(conn, job["id"])
        except Exception as e:  # noqa: BLE001
            log.exception("job %s failed", job["id"])
            complete(conn, job["id"], error=f"{type(e).__name__}: {e}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    run_migrations()
    settings = get_settings()
    with connection() as conn:
        n = requeue_stale(conn)
        if n:
            log.info("requeued %d stale jobs", n)
    log.info("worker started")
    while not _stop:
        with connection() as conn:
            job = claim_next(conn)
        if job is None:
            time.sleep(settings.worker_poll_seconds)
            continue
        log.info("job %s (%s) for document %s", job["id"], job["kind"], job["document_id"])
        process(job)
    log.info("worker stopped")


if __name__ == "__main__":
    main()
