"""Hybrid retrieval in one database: vector (pgvector HNSW) + lexical (pg_search BM25 or tsvector),
fused with reciprocal rank fusion, plus an entity boost from the entity layer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import psycopg

from ..db import vec
from ..ingest.normalize_fa import lexical_form

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    chunk_id: str
    score: float = 0.0
    vector_rank: int | None = None
    lexical_rank: int | None = None
    entity_hit: bool = False
    sources: list[str] = field(default_factory=list)


def vector_search(
    conn: psycopg.Connection[Any], qvec: list[float], index_version: dict[str, Any], limit: int
) -> list[str]:
    dims = int(index_version["dims"])
    ivid = int(index_version["id"])
    conn.execute("SET LOCAL hnsw.ef_search = 120")
    rows = conn.execute(
        f"""
        SELECT id FROM chunks
        WHERE index_version_id = {ivid}
        ORDER BY embedding::vector({dims}) <=> %s::vector({dims}), id
        LIMIT %s
        """,
        (vec(qvec), limit),
    ).fetchall()
    return [r["id"] for r in rows]


def lexical_search(
    conn: psycopg.Connection[Any], query: str, index_version: dict[str, Any], limit: int, backend: str
) -> list[str]:
    q = lexical_form(query)
    if not q:
        return []
    ivid = int(index_version["id"])
    if backend == "pg_search":
        rows = conn.execute(
            f"""
            SELECT id FROM chunks
            WHERE lexical_text ||| %s AND index_version_id = {ivid}
            ORDER BY pdb.score(id) DESC, id
            LIMIT %s
            """,
            (q, limit),
        ).fetchall()
    else:
        tokens = [t for t in q.split() if len(t) >= 2][:32]
        if not tokens:
            return []
        tsq = " | ".join(tokens)  # OR semantics for recall; the reranker restores precision
        rows = conn.execute(
            f"""
            SELECT id FROM chunks, to_tsquery('simple', %s) AS q
            WHERE lexical_tsv @@ q AND index_version_id = {ivid}
            ORDER BY ts_rank_cd(lexical_tsv, q) DESC, id
            LIMIT %s
            """,
            (tsq, limit),
        ).fetchall()
    return [r["id"] for r in rows]


def rrf_fuse(ranked_lists: list[tuple[str, list[str], float]], k: int = 60) -> dict[str, Candidate]:
    """ranked_lists: (label, [chunk ids best-first], weight)."""
    fused: dict[str, Candidate] = {}
    for label, ids, weight in ranked_lists:
        for rank, cid in enumerate(ids, start=1):
            c = fused.setdefault(cid, Candidate(cid))
            c.score += weight / (k + rank)
            c.sources.append(f"{label}#{rank}")
            if label.startswith("vector") and (c.vector_rank is None or rank < c.vector_rank):
                c.vector_rank = rank
            if label.startswith("lexical") and (c.lexical_rank is None or rank < c.lexical_rank):
                c.lexical_rank = rank
    return fused


def entity_chunks(
    conn: psycopg.Connection[Any], entity_ids: list[str], index_version_id: int, limit: int = 20
) -> list[str]:
    if not entity_ids:
        return []
    rows = conn.execute(
        """
        SELECT em.chunk_id, count(*) AS n, max(em.confidence) AS conf
        FROM entity_mentions em JOIN chunks c ON c.id = em.chunk_id
        WHERE em.entity_id = ANY(%s::uuid[]) AND c.index_version_id = %s
        GROUP BY em.chunk_id ORDER BY n DESC, conf DESC, em.chunk_id LIMIT %s
        """,
        (entity_ids, index_version_id, limit),
    ).fetchall()
    return [r["chunk_id"] for r in rows]


def hybrid_search(
    conn: psycopg.Connection[Any],
    queries: list[str],
    qvecs: list[list[float]],
    index_version: dict[str, Any],
    backend: str,
    *,
    candidates: int = 40,
    per_list: int | None = None,
    entity_ids: list[str] | None = None,
    entity_boost: float = 0.5,
    rrf_k: int = 60,
) -> list[Candidate]:
    per_list = per_list or candidates
    lists: list[tuple[str, list[str], float]] = []
    for i, (q, v) in enumerate(zip(queries, qvecs, strict=True)):
        weight = 1.0 if i == 0 else 0.8  # the original question outranks rewrites
        lists.append((f"vector{i}", vector_search(conn, v, index_version, per_list), weight))
        lists.append((f"lexical{i}", lexical_search(conn, q, index_version, per_list, backend), weight))
    fused = rrf_fuse(lists, k=rrf_k)

    if entity_ids:
        ent = entity_chunks(conn, entity_ids, int(index_version["id"]))
        bonus = entity_boost / rrf_k
        for cid in ent:
            c = fused.setdefault(cid, Candidate(cid))
            c.score += bonus
            c.entity_hit = True
            c.sources.append("entity")
    ranked = sorted(fused.values(), key=lambda c: (-c.score, c.chunk_id))
    return ranked[:candidates]


def fetch_chunks(conn: psycopg.Connection[Any], chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return {}
    rows = conn.execute(
        """
        SELECT c.id, c.document_id, c.ord, c.text, c.context, c.breadcrumb, c.page, c.token_count,
               d.title, d.filename, d.source_type, d.doc_date, d.lang
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.id = ANY(%s)
        """,
        (chunk_ids,),
    ).fetchall()
    return {r["id"]: r for r in rows}
