"""FastAPI application: upload, jobs, ask, search, entities, index versions, eval, and the static UI."""

from __future__ import annotations

import base64
import hashlib
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from . import __version__
from .answer.cited import answer_question, entity_profiles
from .config import get_settings
from .db import (
    activate_index_version,
    connection,
    create_index_version,
    detect_lexical_backend,
    ensure_index_version,
    run_migrations,
)
from .ingest import queue as jobq
from .ingest.normalize_fa import normalize_name
from .ingest.parsers import TEXT_EXTS
from .ingest.pipeline import enqueue_reindex
from .retrieval.search import retrieve

log = logging.getLogger("mybrain.api")
UI_DIR = Path(__file__).parent / "ui"
SUPPORTED_EXTS = TEXT_EXTS | {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xlsm",
    ".csv",
    ".tsv",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".eml",
    ".mbox",
    ".html",
    ".htm",
    ".json",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_migrations()
    with connection() as conn:
        ensure_index_version(conn)
    detect_lexical_backend()
    yield


app = FastAPI(title="Mybrain", version=__version__, lifespan=lifespan)


# ---------------------------------------------------------------------------
# auth (single user): Bearer <APP_PASSWORD> or Basic any:<APP_PASSWORD>; empty password disables it
# ---------------------------------------------------------------------------


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    pw = get_settings().app_password
    if pw and request.url.path not in ("/healthz",):
        header = request.headers.get("authorization", "")
        ok = False
        if header.startswith("Bearer "):
            ok = header[7:].strip() == pw
        elif header.startswith("Basic "):
            try:
                raw = base64.b64decode(header[6:]).decode()
                ok = raw.split(":", 1)[-1] == pw
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            return JSONResponse(
                {"detail": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Basic realm=mybrain"},
            )
    return await call_next(request)


def db():
    with connection() as conn:
        yield conn


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz(conn=Depends(db)) -> dict[str, Any]:
    s = get_settings()
    iv = ensure_index_version(conn)
    counts = conn.execute(
        "SELECT (SELECT count(*) FROM documents) AS documents, (SELECT count(*) FROM chunks) AS chunks, "
        "(SELECT count(*) FROM entities) AS entities, (SELECT count(*) FROM ingestion_jobs WHERE status IN ('queued','running')) AS pending_jobs"
    ).fetchone()
    return {
        "status": "ok",
        "version": __version__,
        "index_version": {
            "name": iv["name"],
            "embedding": f"{iv['embedding_provider']}/{iv['embedding_model']}",
            "dims": iv["dims"],
        },
        "lexical_backend": detect_lexical_backend(),
        "reranker": f"{s.rerank_provider}/{s.rerank_model}",
        "answer_model": f"{s.answer_provider}/{s.answer_model}",
        "extract_model": f"{s.extract_provider}/{s.extract_model if s.extract_provider == 'gemini' else s.extract_anthropic_model}",
        "pdf_parser": s.pdf_parser,
        **dict(counts or {}),
    }


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------


def _doc_public(row: dict[str, Any]) -> dict[str, Any]:
    d = {k: v for k, v in row.items() if k != "raw_text"}
    d["id"] = str(d["id"])
    for k in ("doc_date",):
        if d.get(k) is not None and hasattr(d[k], "isoformat"):
            d[k] = d[k].isoformat()
    for k in ("created_at", "updated_at"):
        if d.get(k) is not None:
            d[k] = d[k].isoformat()
    return d


@app.post("/api/documents")
def upload_documents(
    files: list[UploadFile] = File(...),
    title: str | None = Form(None),
    source_type: str | None = Form(None),
    doc_date: str | None = Form(None),
    tags: str | None = Form(None),
    conn=Depends(db),
) -> list[dict[str, Any]]:
    s = get_settings()
    out: list[dict[str, Any]] = []
    parsed_date = date.fromisoformat(doc_date) if doc_date else None
    for f in files:
        name = Path(f.filename or "upload").name
        if Path(name).suffix.lower() not in SUPPORTED_EXTS:
            out.append({"filename": name, "status": "rejected", "error": "unsupported file type"})
            continue
        data = f.file.read()
        content_hash = hashlib.sha256(data).hexdigest()
        existing = conn.execute("SELECT * FROM documents WHERE content_hash = %s", (content_hash,)).fetchone()
        if existing:
            out.append({**_doc_public(existing), "duplicate": True})
            continue
        doc_id = str(uuid.uuid4())
        folder = s.uploads_dir / doc_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_bytes(data)
        meta = {"tags": [t.strip() for t in tags.split(",") if t.strip()]} if tags else {}
        row = conn.execute(
            """
            INSERT INTO documents (id, content_hash, filename, title, source_type, mime, bytes, storage_path, doc_date, metadata, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'queued') RETURNING *
            """,
            (
                doc_id,
                content_hash,
                name,
                title if len(files) == 1 else None,
                source_type or "upload",
                f.content_type,
                len(data),
                str(path),
                parsed_date,
                Jsonb(meta),
            ),
        ).fetchone()
        conn.commit()
        job_id = jobq.enqueue(conn, doc_id, "ingest")
        out.append({**_doc_public(row), "job_id": job_id})
    return out


@app.get("/api/documents")
def list_documents(
    status: str | None = None, q: str | None = None, limit: int = 100, offset: int = 0, conn=Depends(db)
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, filename, title, source_type, mime, bytes, lang, doc_date, status, error, page_count, chunk_count, parser, metadata, created_at, updated_at
        FROM documents
        WHERE (%s::text IS NULL OR status = %s) AND (%s::text IS NULL OR title ILIKE %s OR filename ILIKE %s)
        ORDER BY created_at DESC LIMIT %s OFFSET %s
        """,
        (status, status, q, f"%{q}%" if q else None, f"%{q}%" if q else None, min(limit, 500), offset),
    ).fetchall()
    return [_doc_public(r) for r in rows]


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str, conn=Depends(db)) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM documents WHERE id = %s", (doc_id,)).fetchone()
    if not row:
        raise HTTPException(404, "document not found")
    ents = conn.execute(
        """
        SELECT e.id, e.type, e.canonical_name, count(*) AS mentions FROM entity_mentions em JOIN entities e ON e.id = em.entity_id
        WHERE em.document_id = %s GROUP BY e.id ORDER BY mentions DESC LIMIT 30
        """,
        (doc_id,),
    ).fetchall()
    jobs = conn.execute(
        "SELECT id, kind, status, attempts, error, created_at, finished_at FROM ingestion_jobs WHERE document_id = %s ORDER BY id DESC LIMIT 5",
        (doc_id,),
    ).fetchall()
    d = _doc_public(row)
    d["entities"] = [{**e, "id": str(e["id"])} for e in ents]
    d["jobs"] = [
        {
            **j,
            "created_at": j["created_at"].isoformat(),
            "finished_at": j["finished_at"].isoformat() if j["finished_at"] else None,
        }
        for j in jobs
    ]
    d["text_preview"] = (row.get("raw_text") or "")[:3000]
    return d


@app.get("/api/documents/{doc_id}/text", response_class=PlainTextResponse)
def get_document_text(doc_id: str, conn=Depends(db)) -> str:
    row = conn.execute("SELECT raw_text FROM documents WHERE id = %s", (doc_id,)).fetchone()
    if not row:
        raise HTTPException(404, "document not found")
    return row["raw_text"] or ""


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, conn=Depends(db)) -> dict[str, Any]:
    row = conn.execute("DELETE FROM documents WHERE id = %s RETURNING storage_path", (doc_id,)).fetchone()
    if not row:
        raise HTTPException(404, "document not found")
    conn.commit()
    if row["storage_path"]:
        shutil.rmtree(Path(row["storage_path"]).parent, ignore_errors=True)
    conn.execute(
        "DELETE FROM entities WHERE mention_count = 0 AND NOT EXISTS (SELECT 1 FROM entity_mentions WHERE entity_id = entities.id)"
    )
    conn.commit()
    return {"deleted": doc_id}


class ReingestRequest(BaseModel):
    reparse: bool = False


@app.post("/api/documents/{doc_id}/reingest")
def reingest_document(doc_id: str, body: ReingestRequest, conn=Depends(db)) -> dict[str, Any]:
    if not conn.execute("SELECT 1 FROM documents WHERE id = %s", (doc_id,)).fetchone():
        raise HTTPException(404, "document not found")
    conn.execute("UPDATE documents SET status = 'queued', error = NULL WHERE id = %s", (doc_id,))
    conn.commit()
    return {"job_id": jobq.enqueue(conn, doc_id, "ingest", {"reparse": body.reparse})}


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


@app.get("/api/jobs")
def list_jobs(status: str | None = None, limit: int = 50, conn=Depends(db)) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT j.id, j.document_id, d.title, d.filename, j.kind, j.status, j.attempts, j.error, j.created_at, j.started_at, j.finished_at
        FROM ingestion_jobs j LEFT JOIN documents d ON d.id = j.document_id
        WHERE (%s::text IS NULL OR j.status = %s) ORDER BY j.id DESC LIMIT %s
        """,
        (status, status, min(limit, 500)),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["document_id"] = str(d["document_id"]) if d["document_id"] else None
        for k in ("created_at", "started_at", "finished_at"):
            d[k] = d[k].isoformat() if d[k] else None
        out.append(d)
    return out


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: int, conn=Depends(db)) -> dict[str, Any]:
    if not jobq.retry(conn, job_id):
        raise HTTPException(400, "job is not in failed state")
    return {"requeued": job_id}


# ---------------------------------------------------------------------------
# ask / search
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=40)
    candidates: int | None = Field(default=None, ge=5, le=200)
    use_planner: bool = True
    use_rerank: bool = True
    second_pass: bool | None = None


@app.post("/api/ask")
def ask(body: AskRequest, conn=Depends(db)) -> dict[str, Any]:
    payload = answer_question(
        conn,
        body.question,
        top_k=body.top_k,
        candidates=body.candidates,
        use_planner=body.use_planner,
        use_rerank=body.use_rerank,
        use_second_pass=body.second_pass,
    )
    return payload.__dict__


@app.post("/api/search")
def search(body: AskRequest, conn=Depends(db)) -> dict[str, Any]:
    res = retrieve(
        conn,
        body.question,
        top_k=body.top_k,
        candidates=body.candidates,
        use_planner=body.use_planner,
        use_rerank=body.use_rerank,
        use_second_pass=body.second_pass,
    )
    return res.to_dict()


class FeedbackRequest(BaseModel):
    query_log_id: int
    score: int = Field(ge=-1, le=1)


@app.post("/api/feedback")
def feedback(body: FeedbackRequest, conn=Depends(db)) -> dict[str, Any]:
    conn.execute("UPDATE query_logs SET feedback = %s WHERE id = %s", (body.score, body.query_log_id))
    conn.commit()
    return {"ok": True}


@app.get("/api/queries")
def recent_queries(limit: int = 50, conn=Depends(db)) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, question, answer, model, timings_ms, feedback, created_at FROM query_logs ORDER BY id DESC LIMIT %s",
        (min(limit, 500),),
    ).fetchall()
    return [{**r, "created_at": r["created_at"].isoformat()} for r in rows]


# ---------------------------------------------------------------------------
# entities
# ---------------------------------------------------------------------------


@app.get("/api/entities")
def list_entities(
    q: str | None = None, type: str | None = None, limit: int = 50, conn=Depends(db)
) -> list[dict[str, Any]]:
    norm = normalize_name(q) if q else ""
    if norm:
        conn.execute("SELECT set_limit(0.4::real)")
    rows = conn.execute(
        """
        SELECT e.id, e.type, e.canonical_name, e.name_en, e.name_fa, e.description, e.mention_count,
               (SELECT count(*) FROM relationships r WHERE r.subject_id = e.id OR r.object_id = e.id) AS relationship_count
        FROM entities e
        WHERE (%s::text IS NULL OR e.type = %s)
          AND (%s = '' OR EXISTS (SELECT 1 FROM entity_aliases ea WHERE ea.entity_id = e.id AND (ea.alias_norm LIKE %s OR ea.alias_norm %% %s)))
        ORDER BY e.mention_count DESC, e.canonical_name LIMIT %s
        """,
        (type, type, norm, f"%{norm}%", norm, min(limit, 500)),
    ).fetchall()
    return [{**r, "id": str(r["id"])} for r in rows]


@app.get("/api/entities/{entity_id}")
def get_entity(entity_id: str, conn=Depends(db)) -> dict[str, Any]:
    profiles = entity_profiles(conn, [entity_id], limit=1)
    if not profiles:
        raise HTTPException(404, "entity not found")
    mentions = conn.execute(
        """
        SELECT em.chunk_id, em.document_id, d.title, c.page, c.breadcrumb, em.snippet, d.doc_date
        FROM entity_mentions em JOIN chunks c ON c.id = em.chunk_id JOIN documents d ON d.id = em.document_id
        WHERE em.entity_id = %s ORDER BY d.doc_date DESC NULLS LAST, c.ord LIMIT 50
        """,
        (entity_id,),
    ).fetchall()
    p = profiles[0]
    p["mentions"] = [
        {
            **m,
            "document_id": str(m["document_id"]),
            "doc_date": m["doc_date"].isoformat() if m["doc_date"] else None,
        }
        for m in mentions
    ]
    return p


class MergeRequest(BaseModel):
    into: str


@app.post("/api/entities/{entity_id}/merge")
def merge_entity(entity_id: str, body: MergeRequest, conn=Depends(db)) -> dict[str, Any]:
    """Merge a duplicate entity into another (aliases, mentions, relationships move; the duplicate is deleted)."""
    if entity_id == body.into:
        raise HTTPException(400, "cannot merge an entity into itself")
    for table, col in (
        ("entity_aliases", "entity_id"),
        ("entity_mentions", "entity_id"),
        ("relationships", "subject_id"),
        ("relationships", "object_id"),
    ):
        conn.execute(f"UPDATE {table} SET {col} = %s WHERE {col} = %s", (body.into, entity_id))  # noqa: S608 - fixed identifiers
    conn.execute("DELETE FROM entities WHERE id = %s", (entity_id,))
    conn.execute(
        "UPDATE entities SET mention_count = (SELECT count(*) FROM entity_mentions WHERE entity_id = %s) WHERE id = %s",
        (body.into, body.into),
    )
    conn.commit()
    return {"merged": entity_id, "into": body.into}


# ---------------------------------------------------------------------------
# index versions (embedding model + chunking are versioned together; blue/green re-index)
# ---------------------------------------------------------------------------


@app.get("/api/index-versions")
def list_index_versions(conn=Depends(db)) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT iv.*, (SELECT count(*) FROM chunks c WHERE c.index_version_id = iv.id) AS chunk_count FROM index_versions iv ORDER BY id"
    ).fetchall()
    return [{**r, "created_at": r["created_at"].isoformat()} for r in rows]


class IndexVersionRequest(BaseModel):
    name: str
    embedding_provider: str
    embedding_model: str
    dims: int = Field(ge=64, le=2000)
    chunk_tokens: int = 400
    chunk_overlap: int = 40
    contextualize: bool = True
    reindex: bool = True


@app.post("/api/index-versions")
def new_index_version(body: IndexVersionRequest, conn=Depends(db)) -> dict[str, Any]:
    try:
        iv = create_index_version(
            conn,
            name=body.name,
            embedding_provider=body.embedding_provider,
            embedding_model=body.embedding_model,
            dims=body.dims,
            chunk_tokens=body.chunk_tokens,
            chunk_overlap=body.chunk_overlap,
            contextualize=body.contextualize,
        )
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        raise HTTPException(400, str(e)) from e
    queued = enqueue_reindex(conn, iv["id"]) if body.reindex else 0
    return {**iv, "created_at": iv["created_at"].isoformat(), "queued_jobs": queued}


@app.post("/api/index-versions/{version_id}/activate")
def activate_version(version_id: int, conn=Depends(db)) -> dict[str, Any]:
    if not conn.execute("SELECT 1 FROM index_versions WHERE id = %s", (version_id,)).fetchone():
        raise HTTPException(404, "index version not found")
    activate_index_version(conn, version_id)
    return {"active": version_id}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


class EvalQuestion(BaseModel):
    qid: str
    question: str
    lang: str | None = None
    type: str | None = None
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_docs: list[str] = Field(
        default_factory=list, description="filename or title substrings, resolved to ids"
    )
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_answer: str | None = None


class EvalSetRequest(BaseModel):
    name: str
    questions: list[EvalQuestion]


@app.post("/api/eval/sets")
def create_eval_set(body: EvalSetRequest, conn=Depends(db)) -> dict[str, Any]:
    from .evalx.run import upsert_eval_set

    n = upsert_eval_set(conn, body.name, [q.model_dump() for q in body.questions])
    return {"set": body.name, "questions": n}


@app.get("/api/eval/sets")
def list_eval_sets(conn=Depends(db)) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT s.id, s.name, s.created_at, (SELECT count(*) FROM eval_questions q WHERE q.set_id = s.id) AS questions FROM eval_sets s ORDER BY id"
    ).fetchall()
    return [{**r, "created_at": r["created_at"].isoformat()} for r in rows]


class EvalRunRequest(BaseModel):
    set: str
    label: str | None = None
    use_planner: bool = True
    use_rerank: bool = True
    second_pass: bool = False
    rerank_provider: str | None = None
    rerank_model: str | None = None
    index_version: str | None = None
    top_k: int = 20


@app.post("/api/eval/run")
def run_eval(body: EvalRunRequest, conn=Depends(db)) -> dict[str, Any]:
    from .evalx.run import run_eval_set

    return run_eval_set(
        conn,
        body.set,
        label=body.label,
        use_planner=body.use_planner,
        use_rerank=body.use_rerank,
        second_pass=body.second_pass,
        rerank_provider=body.rerank_provider,
        rerank_model=body.rerank_model,
        index_version_name=body.index_version,
        top_k=body.top_k,
    )


@app.get("/api/eval/runs")
def list_eval_runs(limit: int = 50, conn=Depends(db)) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT r.id, s.name AS set, r.label, r.config, r.metrics, r.created_at FROM eval_runs r JOIN eval_sets s ON s.id = r.set_id ORDER BY r.id DESC LIMIT %s",
        (min(limit, 500),),
    ).fetchall()
    return [{**r, "created_at": r["created_at"].isoformat()} for r in rows]


@app.get("/api/eval/runs/{run_id}")
def get_eval_run(run_id: int, conn=Depends(db)) -> dict[str, Any]:
    r = conn.execute(
        "SELECT r.*, s.name AS set FROM eval_runs r JOIN eval_sets s ON s.id = r.set_id WHERE r.id = %s",
        (run_id,),
    ).fetchone()
    if not r:
        raise HTTPException(404, "run not found")
    return {**r, "created_at": r["created_at"].isoformat()}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")
