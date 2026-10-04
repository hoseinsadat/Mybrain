"""Managed baseline: Gemini File Search over the same documents. If the custom stack cannot beat this on
your own questions, you do not need the custom stack.

Usage: python -m mybrain.evalx.filesearch_baseline --set NAME [--store-name mybrain-eval] [--model gemini-3.8-flash]
Document-level metrics only (File Search returns cited files, not ranked chunks).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from ..config import get_settings
from ..db import connection
from .metrics import hit_at_k, mean, mrr, ndcg_at_k, recall_at_k

log = logging.getLogger(__name__)


def _client():
    from google import genai

    return genai.Client(api_key=get_settings().gemini_api_key or None)


def ensure_store(client: Any, display_name: str) -> str:
    for st in client.file_search_stores.list():
        if getattr(st, "display_name", None) == display_name:
            return st.name
    st = client.file_search_stores.create(
        config={"display_name": display_name, "embedding_model": "models/gemini-embedding-2"}
    )
    return st.name


def upload_documents(client: Any, store: str, docs: list[dict[str, Any]]) -> dict[str, str]:
    """Upload original files; display_name carries our document id so citations can be mapped back."""
    mapping: dict[str, str] = {}
    ops = []
    for d in docs:
        path = Path(d["storage_path"])
        if not path.exists():
            continue
        op = client.file_search_stores.upload_to_file_search_store(
            file=str(path),
            file_search_store_name=store,
            config={
                "display_name": str(d["id"]),
                "custom_metadata": [{"key": "doc_id", "string_value": str(d["id"])}],
            },
        )
        ops.append((str(d["id"]), op))
    for doc_id, op in ops:
        while not op.done:
            time.sleep(3)
            op = client.operations.get(op)
        mapping[doc_id] = doc_id
    return mapping


def query_store(client: Any, store: str, model: str, question: str) -> tuple[str, list[str]]:
    from google.genai import types

    resp = client.models.generate_content(
        model=model,
        contents=question,
        config=types.GenerateContentConfig(
            tools=[types.Tool(file_search=types.FileSearch(file_search_store_names=[store]))]
        ),
    )
    cited: list[str] = []
    try:
        gm = resp.candidates[0].grounding_metadata
        for gc in gm.grounding_chunks or []:
            ctx = getattr(gc, "retrieved_context", None)
            title = getattr(ctx, "title", None) or ""
            if title and title not in cited:
                cited.append(title)
    except Exception:  # noqa: BLE001
        pass
    return (resp.text or ""), cited


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--store-name", default="mybrain-eval")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    client = _client()
    store = ensure_store(client, args.store_name)
    with connection() as conn:
        es = conn.execute("SELECT id FROM eval_sets WHERE name = %s", (args.set,)).fetchone()
        if not es:
            raise SystemExit(f"eval set {args.set} not found")
        if not args.skip_upload:
            docs = conn.execute("SELECT id, storage_path FROM documents WHERE status = 'ready'").fetchall()
            upload_documents(client, store, docs)
        questions = conn.execute(
            "SELECT * FROM eval_questions WHERE set_id = %s ORDER BY id", (es["id"],)
        ).fetchall()
        per_q = []
        for q in questions:
            gold = [str(x) for x in (q["gold_doc_ids"] or [])]
            try:
                answer, cited = query_store(client, store, args.model, q["question"])
                m = {}
                if gold:
                    for k in (5, 10, 20):
                        m[f"doc_hit@{k}"] = hit_at_k(cited, gold, k)
                        m[f"doc_recall@{k}"] = recall_at_k(cited, gold, k)
                    m["doc_mrr"] = mrr(cited, gold)
                    m["doc_ndcg@10"] = ndcg_at_k(cited, gold, 10)
                per_q.append(
                    {
                        "qid": q["qid"],
                        "question": q["question"],
                        "metrics": m,
                        "cited": cited,
                        "answer": answer[:500],
                    }
                )
            except Exception as e:  # noqa: BLE001
                per_q.append({"qid": q["qid"], "question": q["question"], "metrics": {}, "error": str(e)})
        keys = sorted({k for p in per_q for k in p["metrics"]})
        summary = {k: round(mean([p["metrics"][k] for p in per_q if k in p["metrics"]]), 4) for k in keys}
        summary["questions"] = len(per_q)
        config = {"baseline": "gemini_file_search", "model": args.model, "store": store}
        row = conn.execute(
            "INSERT INTO eval_runs (set_id, label, config, metrics, per_question) VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (es["id"], "gemini-file-search", Jsonb(config), Jsonb(summary), Jsonb(per_q)),
        ).fetchone()
        conn.commit()
    print(json.dumps({"run_id": row["id"], "metrics": summary}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
