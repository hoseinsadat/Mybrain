"""Document ingestion: parse -> normalize -> chunk -> contextualize -> embed -> upsert -> entities.

Parsed Markdown is stored on the document, so re-indexing with a new embedding model or chunking
configuration (a new index_version) never re-parses files.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from ..config import Settings, get_settings
from ..db import ensure_index_version, vec
from ..providers import get_extract_llm, make_embedder
from .chunker import chunk_markdown
from .contextualize import contextualize
from .entities import extract, store_entities
from .normalize_fa import lexical_form
from .parsers import parse_file

log = logging.getLogger(__name__)


def chunk_id_for(document_id: str, index_version_id: int, ord_: int, text: str) -> str:
    return hashlib.sha256(f"{document_id}:{index_version_id}:{ord_}:{text}".encode()).hexdigest()[:32]


def _set_status(
    conn: psycopg.Connection[Any], document_id: str, status: str, error: str | None = None
) -> None:
    conn.execute(
        "UPDATE documents SET status = %s, error = %s, updated_at = now() WHERE id = %s",
        (status, error[:4000] if error else None, document_id),
    )
    conn.commit()


def parse_stage(conn: psycopg.Connection[Any], doc: dict[str, Any], settings: Settings) -> dict[str, Any]:
    path = Path(doc["storage_path"])
    parsed = parse_file(path, settings, filename=doc["filename"])
    meta = dict(doc.get("metadata") or {})
    meta.update(parsed.metadata)
    if parsed.warnings:
        meta["warnings"] = parsed.warnings
    title = doc.get("title") or parsed.title or Path(doc["filename"]).stem
    row = conn.execute(
        """
        UPDATE documents SET raw_text = %s, title = %s, lang = %s, doc_date = COALESCE(doc_date, %s),
          page_count = %s, parser = %s, source_type = CASE WHEN source_type = 'upload' THEN %s ELSE source_type END,
          metadata = %s, updated_at = now()
        WHERE id = %s RETURNING *
        """,
        (
            parsed.markdown,
            title,
            parsed.lang,
            parsed.doc_date,
            parsed.page_count,
            parsed.parser,
            parsed.source_type,
            Jsonb(meta),
            doc["id"],
        ),
    ).fetchone()
    conn.commit()
    assert row is not None
    return row


def index_stage(
    conn: psycopg.Connection[Any], doc: dict[str, Any], settings: Settings, index_version: dict[str, Any]
) -> dict[str, Any]:
    iv = index_version
    document_id = str(doc["id"])
    title = doc.get("title")
    markdown = doc.get("raw_text") or ""
    chunks = chunk_markdown(
        markdown, title=title, max_tokens=iv["chunk_tokens"], overlap_tokens=iv["chunk_overlap"]
    )
    stats: dict[str, Any] = {"chunks": len(chunks)}
    if not chunks:
        conn.execute(
            "DELETE FROM chunks WHERE document_id = %s AND index_version_id = %s", (document_id, iv["id"])
        )
        conn.commit()
        return stats

    t0 = time.time()
    contexts = [""] * len(chunks)
    if settings.contextualize and iv.get("contextualize", True):
        contexts = contextualize(
            get_extract_llm(), title, markdown, chunks, settings.contextualize_max_doc_chars
        )
    stats["contextualize_s"] = round(time.time() - t0, 1)

    embedder = make_embedder(iv["embedding_provider"], iv["embedding_model"], iv["dims"])
    t0 = time.time()
    embed_texts = [(f"{ctx}\n\n{c.text}" if ctx else c.text) for c, ctx in zip(chunks, contexts, strict=True)]
    vectors = embedder.embed_documents(embed_texts, titles=[c.breadcrumb or title for c in chunks])
    stats["embed_s"] = round(time.time() - t0, 1)

    rows: list[dict[str, Any]] = []
    for c, ctx, v in zip(chunks, contexts, vectors, strict=True):
        cid = chunk_id_for(document_id, iv["id"], c.ord, c.text)
        rows.append(
            {
                "id": cid,
                "document_id": document_id,
                "index_version_id": iv["id"],
                "ord": c.ord,
                "text": c.text,
                "context": ctx or None,
                "breadcrumb": c.breadcrumb or None,
                "lexical_text": lexical_form(" ".join(x for x in (c.breadcrumb, ctx, c.text) if x)),
                "page": c.page,
                "token_count": c.token_count,
                "metadata": Jsonb({"kinds": c.kinds}),
                "embedding": vec(v),
            }
        )
    conn.execute(
        "DELETE FROM chunks WHERE document_id = %s AND index_version_id = %s", (document_id, iv["id"])
    )
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (id, document_id, index_version_id, ord, text, context, breadcrumb, lexical_text, page, token_count, metadata, embedding)
            VALUES (%(id)s, %(document_id)s, %(index_version_id)s, %(ord)s, %(text)s, %(context)s, %(breadcrumb)s, %(lexical_text)s,
                    %(page)s, %(token_count)s, %(metadata)s, %(embedding)s::vector)
            """,
            rows,
        )
    conn.commit()

    if settings.extract_entities and iv.get("is_active", True):
        t0 = time.time()
        entities, relationships = extract(get_extract_llm(), title, markdown)
        conn.execute("DELETE FROM relationships WHERE document_id = %s", (document_id,))
        stats.update(store_entities(conn, document_id, rows, entities, relationships))
        conn.commit()
        stats["entities_s"] = round(time.time() - t0, 1)
    return stats


def ingest_document(
    conn: psycopg.Connection[Any],
    document_id: str,
    *,
    reparse: bool = False,
    index_version_id: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    doc = conn.execute("SELECT * FROM documents WHERE id = %s", (document_id,)).fetchone()
    if doc is None:
        raise ValueError(f"document {document_id} not found")
    try:
        if reparse or not doc.get("raw_text"):
            _set_status(conn, document_id, "parsing")
            doc = parse_stage(conn, doc, settings)
        _set_status(conn, document_id, "indexing")
        if index_version_id is not None:
            iv = conn.execute("SELECT * FROM index_versions WHERE id = %s", (index_version_id,)).fetchone()
            if iv is None:
                raise ValueError(f"index version {index_version_id} not found")
        else:
            iv = ensure_index_version(conn)
        stats = index_stage(conn, doc, settings, iv)
        conn.execute(
            "UPDATE documents SET status = 'ready', error = NULL, chunk_count = %s, updated_at = now() WHERE id = %s",
            (stats.get("chunks", 0), document_id),
        )
        conn.commit()
        log.info("ingested %s (%s): %s", doc.get("title"), document_id, stats)
        return stats
    except Exception as e:
        conn.rollback()
        _set_status(conn, document_id, "failed", f"{type(e).__name__}: {e}")
        raise


def enqueue_reindex(conn: psycopg.Connection[Any], index_version_id: int) -> int:
    """Queue a re-index of every parsed document into another index version (blue/green migration)."""
    from .queue import enqueue

    ids = [r["id"] for r in conn.execute("SELECT id FROM documents WHERE raw_text IS NOT NULL").fetchall()]
    for did in ids:
        enqueue(conn, str(did), kind="reindex", payload={"index_version_id": index_version_id})
    return len(ids)
