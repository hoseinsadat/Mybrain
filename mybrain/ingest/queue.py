"""Tiny Postgres-backed job queue (FOR UPDATE SKIP LOCKED). One table, no extra service."""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb


def enqueue(
    conn: psycopg.Connection[Any],
    document_id: str | None,
    kind: str = "ingest",
    payload: dict[str, Any] | None = None,
) -> int:
    row = conn.execute(
        "INSERT INTO ingestion_jobs (document_id, kind, payload) VALUES (%s, %s, %s) RETURNING id",
        (document_id, kind, Jsonb(payload or {})),
    ).fetchone()
    assert row is not None
    conn.commit()
    return int(row["id"])


def claim_next(conn: psycopg.Connection[Any]) -> dict[str, Any] | None:
    row = conn.execute(
        """
        UPDATE ingestion_jobs SET status = 'running', locked_at = now(), started_at = now(), attempts = attempts + 1
        WHERE id = (
            SELECT id FROM ingestion_jobs WHERE status = 'queued' ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1
        )
        RETURNING *
        """
    ).fetchone()
    conn.commit()
    return row


def complete(conn: psycopg.Connection[Any], job_id: int, error: str | None = None) -> None:
    conn.execute(
        "UPDATE ingestion_jobs SET status = %s, error = %s, finished_at = now(), locked_at = NULL WHERE id = %s",
        ("failed" if error else "done", (error or None) and error[:4000], job_id),
    )
    conn.commit()


def requeue_stale(conn: psycopg.Connection[Any], older_than_minutes: int = 90) -> int:
    cur = conn.execute(
        """
        UPDATE ingestion_jobs SET status = 'queued', locked_at = NULL
        WHERE status = 'running' AND locked_at < now() - make_interval(mins => %s) AND attempts < 3
        """,
        (older_than_minutes,),
    )
    conn.commit()
    return cur.rowcount


def retry(conn: psycopg.Connection[Any], job_id: int) -> bool:
    cur = conn.execute(
        "UPDATE ingestion_jobs SET status = 'queued', error = NULL WHERE id = %s AND status = 'failed'",
        (job_id,),
    )
    conn.commit()
    return cur.rowcount == 1
