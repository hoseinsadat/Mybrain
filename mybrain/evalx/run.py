"""Eval harness: run a question set through the retrieval stack under a given configuration and
report Recall@k / Hit@k / MRR / nDCG at document and chunk level. Used for the vendor bake-off."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from ..config import get_settings
from ..db import connection
from ..providers import make_reranker
from ..retrieval.search import retrieve
from .metrics import hit_at_k, mean, mrr, ndcg_at_k, recall_at_k

log = logging.getLogger(__name__)
KS = (5, 10, 20)


def _resolve_gold_docs(conn: psycopg.Connection[Any], names: list[str]) -> list[str]:
    ids: list[str] = []
    for n in names:
        row = conn.execute(
            "SELECT id FROM documents WHERE filename = %s OR title = %s OR filename ILIKE %s OR title ILIKE %s ORDER BY created_at LIMIT 1",
            (n, n, f"%{n}%", f"%{n}%"),
        ).fetchone()
        if row:
            ids.append(str(row["id"]))
        else:
            log.warning("gold document not found: %s", n)
    return ids


def upsert_eval_set(conn: psycopg.Connection[Any], name: str, questions: list[dict[str, Any]]) -> int:
    row = conn.execute(
        "INSERT INTO eval_sets (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
        (name,),
    ).fetchone()
    assert row is not None
    set_id = row["id"]
    n = 0
    for q in questions:
        gold_docs = list(q.get("gold_doc_ids") or []) + _resolve_gold_docs(conn, q.get("gold_docs") or [])
        conn.execute(
            """
            INSERT INTO eval_questions (set_id, qid, question, lang, qtype, gold_doc_ids, gold_chunk_ids, gold_answer)
            VALUES (%s, %s, %s, %s, %s, %s::uuid[], %s, %s)
            ON CONFLICT (set_id, qid) DO UPDATE SET question = EXCLUDED.question, lang = EXCLUDED.lang, qtype = EXCLUDED.qtype,
              gold_doc_ids = EXCLUDED.gold_doc_ids, gold_chunk_ids = EXCLUDED.gold_chunk_ids, gold_answer = EXCLUDED.gold_answer
            """,
            (
                set_id,
                q["qid"],
                q["question"],
                q.get("lang"),
                q.get("type"),
                gold_docs,
                list(q.get("gold_chunk_ids") or []),
                q.get("gold_answer"),
            ),
        )
        n += 1
    conn.commit()
    return n


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(json.loads(line))
    return out


def score_question(
    ranked_chunks: list[str], ranked_docs: list[str], gold_chunks: list[str], gold_docs: list[str]
) -> dict[str, float]:
    m: dict[str, float] = {}
    if gold_docs:
        for k in KS:
            m[f"doc_hit@{k}"] = hit_at_k(ranked_docs, gold_docs, k)
            m[f"doc_recall@{k}"] = recall_at_k(ranked_docs, gold_docs, k)
        m["doc_mrr"] = mrr(ranked_docs, gold_docs)
        m["doc_ndcg@10"] = ndcg_at_k(ranked_docs, gold_docs, 10)
    if gold_chunks:
        for k in KS:
            m[f"chunk_recall@{k}"] = recall_at_k(ranked_chunks, gold_chunks, k)
        m["chunk_mrr"] = mrr(ranked_chunks, gold_chunks)
        m["chunk_ndcg@10"] = ndcg_at_k(ranked_chunks, gold_chunks, 10)
    return m


def run_eval_set(
    conn: psycopg.Connection[Any],
    set_name: str,
    *,
    label: str | None = None,
    use_planner: bool = True,
    use_rerank: bool = True,
    second_pass: bool = False,
    rerank_provider: str | None = None,
    rerank_model: str | None = None,
    index_version_name: str | None = None,
    top_k: int = 20,
    candidates: int | None = None,
) -> dict[str, Any]:
    s = get_settings()
    es = conn.execute("SELECT id FROM eval_sets WHERE name = %s", (set_name,)).fetchone()
    if not es:
        raise ValueError(f"eval set {set_name} not found")
    questions = conn.execute(
        "SELECT * FROM eval_questions WHERE set_id = %s ORDER BY id", (es["id"],)
    ).fetchall()
    iv = None
    if index_version_name:
        iv = conn.execute("SELECT * FROM index_versions WHERE name = %s", (index_version_name,)).fetchone()
        if not iv:
            raise ValueError(f"index version {index_version_name} not found")
    reranker = (
        make_reranker(rerank_provider, rerank_model)
        if (use_rerank and (rerank_provider or rerank_model))
        else None
    )
    from ..providers import make_embedder

    embedder = make_embedder(iv["embedding_provider"], iv["embedding_model"], iv["dims"]) if iv else None

    per_q: list[dict[str, Any]] = []
    for q in questions:
        gold_docs = [str(x) for x in (q["gold_doc_ids"] or [])]
        gold_chunks = list(q["gold_chunk_ids"] or [])
        try:
            res = retrieve(
                conn,
                q["question"],
                embedder=embedder,
                reranker=reranker,
                index_version=iv,
                use_planner=use_planner,
                use_rerank=use_rerank,
                use_second_pass=second_pass,
                top_k=top_k,
                candidates=candidates or max(s.retrieve_candidates, top_k * 2),
            )
            ranked_chunks = [c.id for c in res.chunks]
            ranked_docs = list(dict.fromkeys(c.document_id for c in res.chunks))
            metrics = score_question(ranked_chunks, ranked_docs, gold_chunks, gold_docs)
            per_q.append(
                {
                    "qid": q["qid"],
                    "question": q["question"],
                    "type": q["qtype"],
                    "lang": q["lang"],
                    "metrics": metrics,
                    "ranked_docs": ranked_docs[:10],
                    "timings_ms": res.timings_ms,
                    "second_pass_queries": res.second_pass_queries,
                }
            )
        except Exception as e:  # noqa: BLE001
            log.exception("eval question %s failed", q["qid"])
            per_q.append(
                {
                    "qid": q["qid"],
                    "question": q["question"],
                    "type": q["qtype"],
                    "error": str(e),
                    "metrics": {},
                }
            )

    keys = sorted({k for p in per_q for k in p["metrics"]})
    summary = {k: round(mean([p["metrics"][k] for p in per_q if k in p["metrics"]]), 4) for k in keys}
    summary["questions"] = len(per_q)
    summary["scored"] = sum(1 for p in per_q if p["metrics"])
    summary["errors"] = sum(1 for p in per_q if p.get("error"))
    summary["mean_latency_ms"] = round(
        mean([sum(p.get("timings_ms", {}).values()) for p in per_q if p.get("timings_ms")]), 0
    )
    config = {
        "use_planner": use_planner,
        "use_rerank": use_rerank,
        "second_pass": second_pass,
        "reranker": f"{(reranker or (make_reranker() if use_rerank else None)).provider if use_rerank else 'none'}",
        "rerank_model": rerank_model or (s.rerank_model if use_rerank else None),
        "index_version": index_version_name or "active",
        "top_k": top_k,
    }
    row = conn.execute(
        "INSERT INTO eval_runs (set_id, label, config, metrics, per_question) VALUES (%s, %s, %s, %s, %s) RETURNING id, created_at",
        (es["id"], label, Jsonb(config), Jsonb(summary), Jsonb(per_q)),
    ).fetchone()
    conn.commit()
    assert row is not None
    return {
        "run_id": row["id"],
        "set": set_name,
        "label": label,
        "config": config,
        "metrics": summary,
        "per_question": per_q,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the retrieval eval")
    ap.add_argument("--set", required=True, help="eval set name (created from --file if given)")
    ap.add_argument("--file", help="JSONL with questions to (re)load into the set first")
    ap.add_argument("--label")
    ap.add_argument("--no-planner", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--second-pass", action="store_true")
    ap.add_argument("--rerank-provider")
    ap.add_argument("--rerank-model")
    ap.add_argument("--index-version")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--json", action="store_true", help="print full JSON")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with connection() as conn:
        if args.file:
            n = upsert_eval_set(conn, args.set, load_jsonl(Path(args.file)))
            print(f"loaded {n} questions into set {args.set}")
        out = run_eval_set(
            conn,
            args.set,
            label=args.label,
            use_planner=not args.no_planner,
            use_rerank=not args.no_rerank,
            second_pass=args.second_pass,
            rerank_provider=args.rerank_provider,
            rerank_model=args.rerank_model,
            index_version_name=args.index_version,
            top_k=args.top_k,
        )
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    else:
        print(f"run {out['run_id']}  config={out['config']}")
        for k, v in out["metrics"].items():
            print(f"  {k:>16}: {v}")
    sys.exit(0)


if __name__ == "__main__":
    main()
