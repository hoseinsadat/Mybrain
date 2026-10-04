"""Retrieval orchestration: plan -> hybrid search per query -> entity boost -> rerank -> optional second pass."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import psycopg

from ..config import Settings, get_settings
from ..db import detect_lexical_backend, ensure_index_version
from ..providers import get_embedder, get_extract_llm, get_reranker
from ..providers.base import LLM, Embedder, Reranker
from .hybrid import Candidate, fetch_chunks, hybrid_search
from .planner import Plan, plan_query, resolve_entities

log = logging.getLogger(__name__)

FOLLOWUP_SYSTEM = """You check whether retrieved passages are enough to answer a question. Return JSON:
{"sufficient": true|false, "queries": ["up to 2 additional short search queries (in the language of the documents) for the missing facts"]}.
Only ask for more searches when a needed fact is clearly missing."""


@dataclass
class RetrievedChunk:
    id: str
    document_id: str
    title: str | None
    filename: str
    source_type: str
    page: int | None
    breadcrumb: str | None
    text: str
    context: str | None
    doc_date: str | None
    fused_score: float
    rerank_score: float | None
    vector_rank: int | None
    lexical_rank: int | None
    entity_hit: bool
    sources: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    question: str
    plan: dict[str, Any]
    entities: list[dict[str, Any]]
    chunks: list[RetrievedChunk]
    candidates: int
    timings_ms: dict[str, int]
    second_pass_queries: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _to_chunk(c: Candidate, row: dict[str, Any], rerank_score: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        id=c.chunk_id,
        document_id=str(row["document_id"]),
        title=row.get("title"),
        filename=row["filename"],
        source_type=row["source_type"],
        page=row.get("page"),
        breadcrumb=row.get("breadcrumb"),
        text=row["text"],
        context=row.get("context"),
        doc_date=row["doc_date"].isoformat() if row.get("doc_date") else None,
        fused_score=round(c.score, 6),
        rerank_score=rerank_score,
        vector_rank=c.vector_rank,
        lexical_rank=c.lexical_rank,
        entity_hit=c.entity_hit,
        sources=c.sources,
    )


def _rerank(
    reranker: Reranker, question: str, cands: list[Candidate], rows: dict[str, dict[str, Any]], top_k: int
) -> list[tuple[Candidate, float | None]]:
    docs: list[str] = []
    keep: list[Candidate] = []
    for c in cands:
        r = rows.get(c.chunk_id)
        if r is None:
            continue
        docs.append("\n".join(x for x in (r.get("breadcrumb"), r.get("context"), r["text"]) if x))
        keep.append(c)
    if reranker.provider == "none":
        return [(c, None) for c in keep[:top_k]]
    try:
        order = reranker.rerank(question, docs, top_n=min(top_k, len(docs)))
        return [(keep[i], score) for i, score in order]
    except Exception as e:  # noqa: BLE001
        log.warning("reranker failed (%s); using fused order", e)
        return [(c, None) for c in keep[:top_k]]


def retrieve(
    conn: psycopg.Connection[Any],
    question: str,
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    planner_llm: LLM | None = None,
    index_version: dict[str, Any] | None = None,
    use_planner: bool = True,
    use_rerank: bool = True,
    use_second_pass: bool | None = None,
    candidates: int | None = None,
    top_k: int | None = None,
) -> SearchResult:
    settings = settings or get_settings()
    embedder = embedder or get_embedder()
    reranker = reranker or (get_reranker() if use_rerank else None)
    iv = index_version or ensure_index_version(conn)
    backend = detect_lexical_backend()
    candidates = candidates or settings.retrieve_candidates
    top_k = top_k or settings.final_top_k
    use_second_pass = settings.second_pass if use_second_pass is None else use_second_pass
    timings: dict[str, int] = {}

    t = time.time()
    llm = planner_llm if planner_llm is not None else (get_extract_llm() if use_planner else None)
    plan: Plan = plan_query(llm if use_planner else None, question)
    entities = resolve_entities(conn, plan) if plan.entities else []
    timings["plan"] = int((time.time() - t) * 1000)

    t = time.time()
    queries = plan.queries or [question]
    qvecs = embedder.embed_queries(queries)
    timings["embed"] = int((time.time() - t) * 1000)

    t = time.time()
    cands = hybrid_search(
        conn,
        queries,
        qvecs,
        iv,
        backend,
        candidates=candidates,
        entity_ids=[e["id"] for e in entities],
        entity_boost=settings.entity_boost,
        rrf_k=settings.rrf_k,
    )
    rows = fetch_chunks(conn, [c.chunk_id for c in cands])
    timings["search"] = int((time.time() - t) * 1000)

    t = time.time()
    rr = reranker if (use_rerank and reranker is not None) else _NO_RERANK
    ranked = _rerank(rr, question, cands, rows, top_k)
    timings["rerank"] = int((time.time() - t) * 1000)

    followups: list[str] = []
    if use_second_pass and llm is not None and ranked:
        t = time.time()
        top_scores = [s for _, s in ranked if s is not None]
        weak = (max(top_scores) < 0.35) if top_scores else False
        multi = plan.intent in ("relationship", "comparison", "timeline", "aggregate")
        if weak or multi:
            snippets = "\n\n".join(
                f"[{i + 1}] {rows[c.chunk_id]['text'][:600]}" for i, (c, _) in enumerate(ranked[:6])
            )
            try:
                data = llm.complete_json(
                    FOLLOWUP_SYSTEM, f"Question: {question}\n\nPassages:\n{snippets}", max_tokens=600
                )
                followups = (
                    [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()][:2]
                    if not data.get("sufficient", True)
                    else []
                )
            except Exception as e:  # noqa: BLE001
                log.warning("follow-up planning failed: %s", e)
        if followups:
            fv = embedder.embed_queries(followups)
            extra = hybrid_search(
                conn, followups, fv, iv, backend, candidates=candidates // 2, rrf_k=settings.rrf_k
            )
            known = {c.chunk_id for c in cands}
            new = [c for c in extra if c.chunk_id not in known]
            if new:
                rows.update(fetch_chunks(conn, [c.chunk_id for c in new]))
                merged = cands + new
                ranked = _rerank(rr, question, merged, rows, top_k)
                cands = merged
        timings["second_pass"] = int((time.time() - t) * 1000)

    chunks = [_to_chunk(c, rows[c.chunk_id], s) for c, s in ranked if c.chunk_id in rows]
    return SearchResult(
        question=question,
        plan=plan.to_dict(),
        entities=[
            {k: (v if not hasattr(v, "isoformat") else v.isoformat()) for k, v in e.items()} for e in entities
        ],
        chunks=chunks,
        candidates=len(cands),
        timings_ms=timings,
        second_pass_queries=followups,
        config={
            "index_version": iv["name"],
            "embedding": f"{iv['embedding_provider']}/{iv['embedding_model']}",
            "lexical": backend,
            "reranker": f"{rr.provider}/{rr.model}",
            "candidates": candidates,
            "top_k": top_k,
            "planner": use_planner,
            "second_pass": use_second_pass,
        },
    )


class _NoRerank:
    provider = "none"
    model = "none"

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        return [(i, 1.0) for i in range(min(top_n, len(documents)))]


_NO_RERANK = _NoRerank()
