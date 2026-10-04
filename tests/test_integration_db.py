"""End-to-end against a real Postgres (pgvector required; pg_search optional -> tsvector fallback).
Set MYBRAIN_TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:5433/mybrain_test to run."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row

TEST_URL = os.environ.get("MYBRAIN_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="MYBRAIN_TEST_DATABASE_URL not set")

DOC1 = """# Pars Holding — investment memo

## Management
Ali Rezaei (علی رضایی) is the CFO of Pars Holding since 2023. He previously worked at Bank Melli.

## مدیریت
علی رضایی مدیر مالی هلدینگ پارس است و از سال ۱۴۰۲ در این سمت فعالیت می‌کند.

## Financials
| Year | Revenue | EBITDA |
| --- | --- | --- |
| 2023 | 130 | 31 |
"""

DOC2 = """# Tehran Metro expansion

The Tehran Metro budget for 2024 line 7 extension is 2 trillion rials. Contractor selection is pending.

بودجه مترو تهران برای توسعه خط ۷ در سال ۲۰۲۴ دو هزار میلیارد ریال است.
"""


@pytest.fixture(scope="module")
def db_url(tmp_path_factory):
    os.environ["DATABASE_URL"] = TEST_URL
    os.environ["DATA_DIR"] = str(tmp_path_factory.mktemp("data"))
    from mybrain import db as dbmod
    from mybrain.config import get_settings

    get_settings.cache_clear()
    dbmod.close_pool()
    dbmod.reset_lexical_backend_cache()
    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    dbmod.run_migrations()
    return TEST_URL


@pytest.fixture(scope="module")
def ingested(db_url, tmp_path_factory, request):
    """Ingest two documents with the fake embedder / fake LLM; returns ids."""
    from mybrain import db as dbmod
    from mybrain.ingest import pipeline, queue
    from mybrain.ingest.worker import process
    from tests.conftest import FakeEmbedder, FakeLLM

    mp = pytest.MonkeyPatch()
    request.addfinalizer(mp.undo)
    mp.setattr(
        pipeline, "make_embedder", lambda provider=None, model=None, dims=None: FakeEmbedder(dims or 8)
    )
    mp.setattr(pipeline, "get_extract_llm", lambda: FakeLLM())

    folder = tmp_path_factory.mktemp("uploads")
    ids = {}
    with dbmod.connection() as conn:
        dbmod.create_index_version(
            conn,
            name="v1",
            embedding_provider="fake",
            embedding_model="bow",
            dims=8,
            chunk_tokens=120,
            chunk_overlap=10,
            contextualize=True,
            activate=True,
        )
        for name, text in (("pars_memo.md", DOC1), ("metro.md", DOC2)):
            p = folder / name
            p.write_text(text, encoding="utf-8")
            row = conn.execute(
                "INSERT INTO documents (content_hash, filename, storage_path, bytes) VALUES (%s, %s, %s, %s) RETURNING id",
                (name, name, str(p), len(text)),
            ).fetchone()
            ids[name] = str(row["id"])
            conn.commit()
            queue.enqueue(conn, ids[name])
        for _ in range(2):
            job = queue.claim_next(conn)
            assert job is not None
            process(job)
        assert queue.claim_next(conn) is None
    return ids


def test_documents_ready_with_chunks_and_contexts(ingested, db_url):
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        docs = conn.execute(
            "SELECT id, status, chunk_count, title, lang, error FROM documents ORDER BY filename"
        ).fetchall()
        assert all(d["status"] == "ready" for d in docs), docs
        pars = next(d for d in docs if str(d["id"]) == ingested["pars_memo.md"])
        assert pars["title"] == "Pars Holding — investment memo" and pars["chunk_count"] >= 2
        chunks = conn.execute(
            "SELECT * FROM chunks WHERE document_id = %s ORDER BY ord", (ingested["pars_memo.md"],)
        ).fetchall()
        assert all(c["context"].startswith("Context for chunk") for c in chunks)
        assert all(c["embedding"] is not None for c in chunks)
        assert any("| 2023 | 130 | 31 |" in c["text"] for c in chunks)
        assert all("هلدینگ پارس" in c["lexical_text"] or "pars holding" in c["lexical_text"] for c in chunks)
        jobs = conn.execute("SELECT status FROM ingestion_jobs").fetchall()
        assert {j["status"] for j in jobs} == {"done"}


def test_entities_resolved_with_aliases_mentions_relationships(ingested, db_url):
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        ents = {e["canonical_name"]: e for e in conn.execute("SELECT * FROM entities").fetchall()}
        assert "Ali Rezaei" in ents and "Pars Holding" in ents and "Tehran Metro" in ents
        aliases = {
            a["alias_norm"]
            for a in conn.execute(
                "SELECT alias_norm FROM entity_aliases WHERE entity_id = %s", (ents["Ali Rezaei"]["id"],)
            ).fetchall()
        }
        assert "علی رضایی" in aliases and "ali rezaei" in aliases
        assert ents["Ali Rezaei"]["mention_count"] >= 1
        rel = conn.execute("SELECT r.predicate, r.valid_from, r.chunk_id FROM relationships r").fetchone()
        assert (
            rel["predicate"] == "CFO of"
            and str(rel["valid_from"]) == "2023-01-01"
            and rel["chunk_id"] is not None
        )
        m = conn.execute(
            "SELECT count(*) AS n FROM entity_mentions em JOIN entities e ON e.id = em.entity_id WHERE e.canonical_name = 'Ali Rezaei'"
        ).fetchone()
        assert m["n"] >= 2  # English and Persian chunks both mention him


def test_reingest_is_idempotent_for_entities(ingested, db_url):
    from mybrain import db as dbmod
    from mybrain.ingest import pipeline
    from tests.conftest import FakeEmbedder, FakeLLM

    mp = pytest.MonkeyPatch()
    mp.setattr(
        pipeline, "make_embedder", lambda provider=None, model=None, dims=None: FakeEmbedder(dims or 8)
    )
    mp.setattr(pipeline, "get_extract_llm", lambda: FakeLLM())
    try:
        with dbmod.connection() as conn:
            before = conn.execute("SELECT count(*) AS n FROM entities").fetchone()["n"]
            pipeline.ingest_document(conn, ingested["pars_memo.md"])
            after = conn.execute("SELECT count(*) AS n FROM entities").fetchone()["n"]
            assert before == after, "re-ingesting must not duplicate entities"
            assert conn.execute("SELECT count(*) AS n FROM relationships").fetchone()["n"] == 1
    finally:
        mp.undo()


def test_hybrid_retrieval_en_and_fa(ingested, db_url, fake_embedder, fake_llm):
    from mybrain import db as dbmod
    from mybrain.providers.rerank import NoReranker
    from mybrain.retrieval.search import retrieve

    with dbmod.connection() as conn:
        res = retrieve(
            conn,
            "Who is the CFO of Pars Holding?",
            embedder=fake_embedder,
            reranker=NoReranker(),
            planner_llm=fake_llm,
            use_second_pass=False,
            top_k=5,
        )
        assert res.chunks and res.chunks[0].document_id == ingested["pars_memo.md"]
        assert res.config["lexical"] in ("pg_search", "tsvector")
        assert res.plan["intent"] == "lookup" and len(res.plan["queries"]) == 3
        assert res.chunks[0].lexical_rank is not None

        res = retrieve(
            conn,
            "مدیر مالی هلدینگ پارس کیست؟",
            embedder=fake_embedder,
            reranker=NoReranker(),
            planner_llm=None,
            use_planner=False,
            use_second_pass=False,
            top_k=5,
        )
        assert res.chunks[0].document_id == ingested["pars_memo.md"]
        assert "علی رضایی" in res.chunks[0].text

        res = retrieve(
            conn,
            "What do we know about Ali Rezaei?",
            embedder=fake_embedder,
            reranker=NoReranker(),
            planner_llm=fake_llm,
            use_second_pass=False,
            top_k=5,
        )
        assert (
            res.entities
            and res.entities[0]["canonical_name"] == "Ali Rezaei"
            and res.entities[0]["sim"] == 1.0
        )
        assert any(c.entity_hit for c in res.chunks)

        res = retrieve(
            conn,
            "metro budget line 7",
            embedder=fake_embedder,
            reranker=NoReranker(),
            planner_llm=None,
            use_planner=False,
            use_second_pass=False,
            top_k=3,
        )
        assert res.chunks[0].document_id == ingested["metro.md"]


def test_answer_with_citations_and_query_log(ingested, db_url, fake_embedder, fake_llm):
    from mybrain import db as dbmod
    from mybrain.answer.cited import answer_question
    from mybrain.providers.rerank import NoReranker

    with dbmod.connection() as conn:
        payload = answer_question(
            conn,
            "What do we know about Ali Rezaei?",
            llm=fake_llm,
            embedder=fake_embedder,
            reranker=NoReranker(),
            planner_llm=fake_llm,
            use_second_pass=False,
            top_k=4,
        )
        assert payload.answer.endswith("[1]")
        assert payload.citations[0]["marker"] == 1 and payload.citations[0]["kind"] == "chunk"
        assert payload.citations[0]["chunk_id"] == payload.chunks[0]["id"]
        assert payload.entities and payload.entities[0]["relationships"][0]["predicate"] == "CFO of"
        assert payload.query_log_id is not None
        row = conn.execute("SELECT * FROM query_logs WHERE id = %s", (payload.query_log_id,)).fetchone()
        assert row["answer"] == payload.answer and row["citations"][0]["marker"] == 1


def test_api_end_to_end(ingested, db_url, fake_embedder, fake_llm, tmp_path: Path):
    from fastapi.testclient import TestClient

    from mybrain import main as m
    from mybrain.answer import cited
    from mybrain.ingest import pipeline, queue
    from mybrain.ingest.worker import process
    from mybrain.providers.rerank import NoReranker
    from mybrain.retrieval import search as srch
    from tests.conftest import FakeEmbedder, FakeLLM

    mp = pytest.MonkeyPatch()
    mp.setattr(
        pipeline, "make_embedder", lambda provider=None, model=None, dims=None: FakeEmbedder(dims or 8)
    )
    mp.setattr(pipeline, "get_extract_llm", lambda: FakeLLM())
    mp.setattr(srch, "get_embedder", lambda: fake_embedder)
    mp.setattr(srch, "get_reranker", lambda: NoReranker())
    mp.setattr(srch, "get_extract_llm", lambda: fake_llm)
    mp.setattr(cited, "get_answer_llm", lambda: fake_llm)
    try:
        with TestClient(m.app) as client:
            h = client.get("/healthz").json()
            assert h["status"] == "ok" and h["documents"] == 2

            r = client.post(
                "/api/documents",
                files=[
                    (
                        "files",
                        (
                            "note.md",
                            b"# Call with Pars Holding\n\nAli Rezaei confirmed the 40m valuation.",
                            "text/markdown",
                        ),
                    )
                ],
                data={"source_type": "note", "tags": "deal,pars"},
            )
            assert r.status_code == 200, r.text
            doc = r.json()[0]
            assert doc["status"] == "queued" and doc["job_id"]
            dup = client.post(
                "/api/documents",
                files=[
                    (
                        "files",
                        (
                            "note.md",
                            b"# Call with Pars Holding\n\nAli Rezaei confirmed the 40m valuation.",
                            "text/markdown",
                        ),
                    )
                ],
            ).json()[0]
            assert dup["duplicate"] is True
            bad = client.post(
                "/api/documents", files=[("files", ("x.exe", b"nope", "application/octet-stream"))]
            ).json()[0]
            assert bad["status"] == "rejected"

            from mybrain import db as dbmod

            with dbmod.connection() as conn:
                job = queue.claim_next(conn)
                process(job)
            d = client.get(f"/api/documents/{doc['id']}").json()
            assert d["status"] == "ready" and d["metadata"]["tags"] == ["deal", "pars"]
            assert any(e["canonical_name"] == "Ali Rezaei" for e in d["entities"])

            ents = client.get("/api/entities", params={"q": "rezaei"}).json()
            assert ents and ents[0]["canonical_name"] == "Ali Rezaei"
            ents_fa = client.get("/api/entities", params={"q": "رضایی"}).json()
            assert ents_fa and ents_fa[0]["id"] == ents[0]["id"]
            prof = client.get(f"/api/entities/{ents[0]['id']}").json()
            assert prof["relationships"] and prof["mentions"]

            s = client.post(
                "/api/search", json={"question": "Pars Holding valuation", "second_pass": False}
            ).json()
            assert s["chunks"] and s["chunks"][0]["document_id"] in (doc["id"], ingested["pars_memo.md"])

            a = client.post(
                "/api/ask", json={"question": "Who is the CFO of Pars Holding?", "second_pass": False}
            ).json()
            assert a["answer"].endswith("[1]") and a["citations"] and a["query_log_id"]
            assert client.post(
                "/api/feedback", json={"query_log_id": a["query_log_id"], "score": 1}
            ).json() == {"ok": True}

            jobs = client.get("/api/jobs").json()
            assert jobs and all(j["status"] == "done" for j in jobs)
            ivs = client.get("/api/index-versions").json()
            assert ivs[0]["is_active"] and ivs[0]["chunk_count"] > 0

            es = client.post(
                "/api/eval/sets",
                json={
                    "name": "smoke",
                    "questions": [
                        {
                            "qid": "q1",
                            "question": "Who is the CFO of Pars Holding?",
                            "lang": "en",
                            "type": "lookup",
                            "gold_docs": ["pars_memo.md"],
                        },
                        {
                            "qid": "q2",
                            "question": "بودجه مترو تهران",
                            "lang": "fa",
                            "type": "lookup",
                            "gold_docs": ["metro.md"],
                        },
                    ],
                },
            ).json()
            assert es["questions"] == 2
            run = client.post(
                "/api/eval/run", json={"set": "smoke", "label": "fake", "second_pass": False}
            ).json()
            assert run["metrics"]["doc_hit@5"] == 1.0 and run["metrics"]["questions"] == 2
            assert client.get("/api/eval/runs").json()[0]["label"] == "fake"

            assert client.delete(f"/api/documents/{doc['id']}").json()["deleted"] == doc["id"]
            assert client.get("/").status_code == 200
    finally:
        mp.undo()
