-- Mybrain core schema. One Postgres for vectors, lexical search, entities, jobs and logs.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE TABLE IF NOT EXISTS index_versions (
    id              serial PRIMARY KEY,
    name            text UNIQUE NOT NULL,
    embedding_provider text NOT NULL,
    embedding_model text NOT NULL,
    dims            int NOT NULL,
    chunk_tokens    int NOT NULL,
    chunk_overlap   int NOT NULL,
    contextualize   boolean NOT NULL DEFAULT true,
    is_active       boolean NOT NULL DEFAULT false,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash    text NOT NULL UNIQUE,
    filename        text NOT NULL,
    title           text,
    source_type     text NOT NULL DEFAULT 'upload',   -- upload | email | chat | spreadsheet | database | note | web
    mime            text,
    bytes           bigint,
    storage_path    text,
    raw_text        text,                              -- parsed markdown (re-embedding never re-parses)
    parser          text,
    lang            text,
    doc_date        date,
    metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    acl_principals  text[] NOT NULL DEFAULT '{owner}',
    status          text NOT NULL DEFAULT 'queued',    -- queued | parsing | indexing | ready | failed
    error           text,
    page_count      int,
    chunk_count     int,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_created_idx ON documents (created_at DESC);
CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source_type);

CREATE TABLE IF NOT EXISTS chunks (
    id              text PRIMARY KEY,                 -- deterministic: sha256(doc, version, ord, text)
    document_id     uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    index_version_id int NOT NULL REFERENCES index_versions(id) ON DELETE CASCADE,
    ord             int NOT NULL,
    text            text NOT NULL,                     -- display text (normalized Persian, original casing)
    context         text,                              -- LLM-written situating context (contextual retrieval)
    breadcrumb      text,                              -- "Title > Section > Subsection"
    lexical_text    text NOT NULL,                     -- breadcrumb + context + text, lexical-normalized (for BM25)
    page            int,
    token_count     int,
    metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector,                            -- untyped; per-version expression HNSW index (see db.py)
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, index_version_id, ord)
);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_version_idx ON chunks (index_version_id);
-- tsvector fallback (used when pg_search is not installed). 'simple' keeps Persian tokens intact.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS lexical_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', lexical_text)) STORED;
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (lexical_tsv);

CREATE TABLE IF NOT EXISTS entities (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    type            text NOT NULL,                     -- person | organization | place | other
    canonical_name  text NOT NULL,
    name_en         text,
    name_fa         text,
    description     text,
    attributes      jsonb NOT NULL DEFAULT '{}'::jsonb,
    mention_count   int NOT NULL DEFAULT 0,
    first_seen      timestamptz,
    last_seen       timestamptz,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS entities_type_idx ON entities (type);

CREATE TABLE IF NOT EXISTS entity_aliases (
    id              bigserial PRIMARY KEY,
    entity_id       uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias           text NOT NULL,
    alias_norm      text NOT NULL,
    lang            text,
    UNIQUE (entity_id, alias_norm)
);
CREATE INDEX IF NOT EXISTS entity_aliases_norm_idx ON entity_aliases (alias_norm);
CREATE INDEX IF NOT EXISTS entity_aliases_trgm_idx ON entity_aliases USING gin (alias_norm gin_trgm_ops);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id              bigserial PRIMARY KEY,
    entity_id       uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id        text NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    document_id     uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    role            text,
    snippet         text,
    confidence      real,
    UNIQUE (entity_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS entity_mentions_chunk_idx ON entity_mentions (chunk_id);
CREATE INDEX IF NOT EXISTS entity_mentions_entity_idx ON entity_mentions (entity_id);

CREATE TABLE IF NOT EXISTS relationships (
    id              bigserial PRIMARY KEY,
    subject_id      uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    object_id       uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate       text NOT NULL,                     -- e.g. "CFO of", "invested in", "met with"
    description     text,
    valid_from      date,
    valid_to        date,
    chunk_id        text REFERENCES chunks(id) ON DELETE SET NULL,
    document_id     uuid REFERENCES documents(id) ON DELETE CASCADE,
    confidence      real,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (subject_id, object_id, predicate, document_id)
);
CREATE INDEX IF NOT EXISTS relationships_subject_idx ON relationships (subject_id);
CREATE INDEX IF NOT EXISTS relationships_object_idx ON relationships (object_id);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id              bigserial PRIMARY KEY,
    document_id     uuid REFERENCES documents(id) ON DELETE CASCADE,
    kind            text NOT NULL DEFAULT 'ingest',    -- ingest | reindex
    status          text NOT NULL DEFAULT 'queued',    -- queued | running | done | failed
    attempts        int NOT NULL DEFAULT 0,
    payload         jsonb NOT NULL DEFAULT '{}'::jsonb,
    error           text,
    locked_at       timestamptz,
    created_at      timestamptz NOT NULL DEFAULT now(),
    started_at      timestamptz,
    finished_at     timestamptz
);
CREATE INDEX IF NOT EXISTS ingestion_jobs_status_idx ON ingestion_jobs (status, id);

CREATE TABLE IF NOT EXISTS query_logs (
    id              bigserial PRIMARY KEY,
    question        text NOT NULL,
    plan            jsonb,
    retrieved       jsonb,
    answer          text,
    citations       jsonb,
    entities        jsonb,
    model           text,
    timings_ms      jsonb,
    usage           jsonb,
    feedback        int,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_sets (
    id              serial PRIMARY KEY,
    name            text UNIQUE NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_questions (
    id              serial PRIMARY KEY,
    set_id          int NOT NULL REFERENCES eval_sets(id) ON DELETE CASCADE,
    qid             text NOT NULL,
    question        text NOT NULL,
    lang            text,
    qtype           text,                               -- lookup | multi_hop | entity | unanswerable
    gold_doc_ids    uuid[] NOT NULL DEFAULT '{}',
    gold_chunk_ids  text[] NOT NULL DEFAULT '{}',
    gold_answer     text,
    UNIQUE (set_id, qid)
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id              serial PRIMARY KEY,
    set_id          int NOT NULL REFERENCES eval_sets(id) ON DELETE CASCADE,
    label           text,
    config          jsonb NOT NULL,
    metrics         jsonb NOT NULL,
    per_question    jsonb NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);
