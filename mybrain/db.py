"""Postgres access: connection pool, migrations, lexical-backend detection, vector helpers."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import get_settings

log = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
_lexical_backend: str | None = None

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            get_settings().database_url, min_size=1, max_size=8, kwargs={"row_factory": dict_row}, open=True
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection[dict[str, Any]]]:
    with get_pool().connection() as conn:
        yield conn


def vec(values: list[float]) -> str:
    """Serialize a float list for pgvector ('[0.1,0.2,...]')."""
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


def parse_vec(text: str | None) -> list[float] | None:
    if text is None:
        return None
    return [float(x) for x in text.strip("[]").split(",") if x]


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def _extension_available(conn: psycopg.Connection[Any], name: str) -> bool:
    row = conn.execute("SELECT 1 FROM pg_available_extensions WHERE name = %s", (name,)).fetchone()
    return row is not None


def run_migrations(conn: psycopg.Connection[Any] | None = None) -> list[str]:
    """Apply SQL files in migrations/ in order. Files named *.pg_search.sql only run when pg_search is available."""
    own = conn is None
    if own:
        conn = psycopg.connect(get_settings().database_url, row_factory=dict_row)
    assert conn is not None
    applied: list[str] = []
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY, applied_at timestamptz DEFAULT now())"
        )
        done = {r["name"] for r in conn.execute("SELECT name FROM schema_migrations").fetchall()}
        has_pg_search = _extension_available(conn, "pg_search")
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue
            if path.name.endswith(".pg_search.sql") and not has_pg_search:
                log.info("skipping %s (pg_search not available, tsvector fallback will be used)", path.name)
                continue
            log.info("applying migration %s", path.name)
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
            conn.commit()
            applied.append(path.name)
    finally:
        if own:
            conn.close()
    return applied


def detect_lexical_backend() -> str:
    """pg_search (ParadeDB BM25) when installed, else Postgres tsvector."""
    global _lexical_backend
    if _lexical_backend:
        return _lexical_backend
    forced = get_settings().lexical_backend
    if forced != "auto":
        _lexical_backend = forced
        return forced
    with connection() as conn:
        row = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_search'").fetchone()
    _lexical_backend = "pg_search" if row else "tsvector"
    log.info("lexical backend: %s", _lexical_backend)
    return _lexical_backend


def reset_lexical_backend_cache() -> None:
    global _lexical_backend
    _lexical_backend = None


# ---------------------------------------------------------------------------
# Index versions (embedding model + chunking config are versioned together)
# ---------------------------------------------------------------------------


def ensure_index_version(conn: psycopg.Connection[Any]) -> dict[str, Any]:
    """Return the active index version, creating one from current settings if none exists."""
    s = get_settings()
    row = conn.execute("SELECT * FROM index_versions WHERE is_active ORDER BY id DESC LIMIT 1").fetchone()
    if row:
        return row
    row = create_index_version(
        conn,
        name="v1",
        embedding_provider=s.embed_provider,
        embedding_model=s.embed_model if s.embed_provider != "openai_compat" else s.openai_compat_embed_model,
        dims=s.embed_dims,
        chunk_tokens=s.chunk_tokens,
        chunk_overlap=s.chunk_overlap_tokens,
        contextualize=s.contextualize,
        activate=True,
    )
    return row


def create_index_version(
    conn: psycopg.Connection[Any],
    *,
    name: str,
    embedding_provider: str,
    embedding_model: str,
    dims: int,
    chunk_tokens: int,
    chunk_overlap: int,
    contextualize: bool,
    activate: bool = False,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", name):
        raise ValueError("index version name must be alphanumeric")
    row = conn.execute(
        """
        INSERT INTO index_versions (name, embedding_provider, embedding_model, dims, chunk_tokens, chunk_overlap, contextualize, is_active)
        VALUES (%s, %s, %s, %s, %s, %s, %s, false) RETURNING *
        """,
        (name, embedding_provider, embedding_model, dims, chunk_tokens, chunk_overlap, contextualize),
    ).fetchone()
    assert row is not None
    # Expression + partial HNSW index: one vector column serves every dimensionality/version.
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS chunks_hnsw_iv{row['id']} ON chunks USING hnsw ((embedding::vector({int(dims)})) vector_cosine_ops) "
        f"WHERE index_version_id = {int(row['id'])}"
    )
    if activate:
        activate_index_version(conn, row["id"])
        row["is_active"] = True
    conn.commit()
    return row


def activate_index_version(conn: psycopg.Connection[Any], version_id: int) -> None:
    conn.execute("UPDATE index_versions SET is_active = (id = %s)", (version_id,))
    conn.commit()


def main() -> None:  # console script: mybrain-migrate
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    applied = run_migrations()
    print("applied:", applied or "nothing (up to date)")
    with connection() as conn:
        iv = ensure_index_version(conn)
    print(
        f"active index version: {iv['name']} ({iv['embedding_provider']}/{iv['embedding_model']}, {iv['dims']} dims)"
    )
    print("lexical backend:", detect_lexical_backend())
    sys.exit(0)


if __name__ == "__main__":
    main()
