# Mybrain architecture and vendor choices

A personal "second brain" for a private-equity / business / political analyst: documents in, cited answers
about companies and people out. English + Persian. Runs on a small CPU-only VPS; every heavy model is an API.

## Pipeline

```
 upload / bulk ingest ──▶ worker (one document at a time)
   1 parse        LlamaParse (agentic tier) for complex or scanned PDFs when a key is set; PyMuPDF with
                  font-size headings and table extraction otherwise; python-docx / python-pptx / openpyxl /
                  csv / sqlite / email+mailbox / WhatsApp + Telegram exports / HTML / Markdown locally
   2 normalize    Persian canonical form (yeh/kaf/alef variants, digits, tatweel, diacritics; ZWNJ kept),
                  Markdown with headings and <!-- page: N --> markers
   3 chunk        structure-aware: headings -> breadcrumbs, tables and code kept whole (long tables split by
                  rows with the header repeated), 400-token budget, sentence-level overlap
   4 contextualize cheap LLM (Gemini Flash by default) writes 1-2 situating sentences per chunk, batched 10
                  per call (Anthropic's contextual retrieval, without one call per chunk)
   5 entities     cheap LLM extracts people / organizations / places with aliases in both scripts and typed,
                  dated relationships; resolved against the registry by normalized alias (then trigram for
                  multi-word names); chunk mentions linked by alias string match
   6 embed        pluggable: Gemini Embedding 2 (default), Voyage 4, any OpenAI-compatible endpoint
                  (Together/Fireworks for Qwen3-Embedding-8B, Jina); stored per index_version
   7 upsert       Postgres: chunks(text, context, lexical_text, embedding) + entity tables

 question ──▶ planner (cheap LLM): language, intent, entities, 2-4 search queries incl. the other language,
              sub-questions for multi-hop
          ──▶ per query: pgvector HNSW cosine + BM25 (ParadeDB pg_search, ICU tokenizer) -> reciprocal rank
              fusion (k=60, original question weighted 1.0, rewrites 0.8) -> entity boost for chunks that
              mention a resolved query entity -> top 40 candidates
          ──▶ cross-encoder rerank (ZeroEntropy zerank-2 default; Cohere Rerank 4 / Voyage rerank-2.5
              pluggable) -> top 10
          ──▶ optional second pass: if the intent is relational/comparative/timeline or the best rerank
              score is weak, the cheap LLM proposes up to 2 follow-up queries, merged and reranked once
          ──▶ answer model (Claude Opus 5.5 default, Claude Sonnet 5.5 for lower cost) with the Citations
              API: each claim carries the exact supporting passage; entity profiles (aliases, relationships,
              documents) are passed as citable documents too
          ──▶ query_logs: plan, scores per stage, citations, timings, token usage, thumbs feedback
```

Why this beats a NotebookLM-style tool for this use case: an **entity layer** (people/companies with
Persian+English aliases, dated relationships, chunk mentions) that profile and relationship questions hit
first; **hybrid + multilingual + reranked** retrieval so exact names, amounts and Persian terms are found by
BM25 while meaning is found by dense vectors; and **versioned, pluggable models with a built-in eval** so the
vendor choice is made on your own questions.

## Storage: one Postgres

- Image `paradedb/paradedb:latest`: Postgres + pg_search (true BM25, Tantivy, ICU tokenizer) + pgvector.
- `chunks.embedding` is an untyped `vector`; each index version gets a partial expression HNSW index
  (`(embedding::vector(DIMS)) WHERE index_version_id = N`), so different models/dimensionalities coexist and a
  blue/green re-index is: create version -> worker re-embeds from stored Markdown -> eval -> activate.
- BM25 index: `CREATE INDEX ... USING bm25 (id, (lexical_text::pdb.icu), index_version_id, document_id) WITH (key_field='id')`,
  queried with `lexical_text ||| 'terms' ORDER BY pdb.score(id) DESC`. Without pg_search (plain Postgres) the
  app falls back automatically to a `tsvector('simple')` GIN index with OR semantics.
- pgvector HNSW indexes support up to 2000 dimensions: Gemini Embedding 2 is used at 1536 (Matryoshka),
  Voyage 4 at 1024, Qwen3-Embedding-8B truncated to <= 2000 (it is Matryoshka-trained; the client truncates
  and re-normalizes when a host ignores `dimensions`).
- Job queue: `ingestion_jobs` with `FOR UPDATE SKIP LOCKED`; no Redis.

## Vendor shortlist (prices checked October 2026; verify before budgeting)

### Embeddings (default: Gemini Embedding 2)

| Model | Why | Price / 1M tokens |
|---|---|---|
| **Gemini Embedding 2** (`gemini-embedding-2`) | top proprietary multilingual MTEB, 100+ languages, multimodal (deck pages as images later), 8k input, 128–3072 dims | $0.20 ($0.10 batch) |
| **Voyage voyage-4-large** | top retrieval quality, 32k context, 200M free tokens; `voyage-context-4` for contextualized chunk embeddings without an LLM pass | $0.12 |
| **Qwen3-Embedding-8B** via Together/Fireworks | #1 open model on MTEB multilingual (119 languages incl. Persian); US-hosted | ~$0.05–0.10 |
| jina-embeddings-v5 / v3 | jina-v3 scored best among models tested on FaMTEB (Persian MTEB) | ~$0.02–0.05 |

### Rerankers (default: zerank-2)

| Model | Agentset ELO (Feb 2026) | Latency | Price |
|---|---|---|---|
| **ZeroEntropy zerank-2** | 1638 (#1) | 265 ms | $0.025 / 1M tokens, multilingual |
| Cohere Rerank 4 Pro (`rerank-v4.0-pro`) | 1629 | 614 ms | ~$2 / 1k searches, 100+ languages |
| Voyage rerank-2.5 / lite | 1544 / 1520 | ~615 ms | $0.05 / $0.02 per 1M tokens |
| Qwen3-Reranker-8B (self-host) | 1473 | 4.7 s CPU | Apache 2.0, only if going self-hosted |

### Parsing (default: LlamaParse agentic for complex PDFs, PyMuPDF otherwise)

| Service | Accuracy (openbenchmarks legal set, Sep 2026) | Price / 1k pages |
|---|---|---|
| **LlamaParse agentic** (`tier=agentic`, `agentic_plus` for dense financials with charts) | 80.0% (plus: 83.4%) | $12.50 ($56) |
| Reducto | 78.7%, fastest | $10 |
| Gemini 3.x Flash as a vision parser | n/a | ~$24 |
| PyMuPDF / python-docx / python-pptx / openpyxl | born-digital files | free |

### LLMs

| Role | Model | Price in/out per 1M |
|---|---|---|
| Final cited answer | **Claude Opus 5.5** (`claude-opus-5-5`, default) | $4 / $20 |
| Lower-cost answer | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | $2 / $10 |
| Whole-data-room reading (no retrieval) | Gemini 3.1 Pro (1M context) | $2 / $12 (≤200k), $4 / $18 above |
| Chunk context, entity extraction, query planning | Gemini 3.8 Flash (default) or Claude Haiku 4.5 | ~$0.75 / $3.75 ; $1 / $5 |
| Chinese open models (Qwen 3.x, DeepSeek V4, Kimi K2.x, GLM-5) | only via US/EU hosts (Together, Fireworks, OpenRouter); official endpoints are China-hosted | $0.3–1.3 / $1.2–4 |

Data policy: all defaults are US/EU-hosted APIs on paid, no-training terms. Anthropic Citations and Claude
Opus 5.5 are eligible for zero-data-retention organizations; check each vendor's DPA before loading sensitive
material. No China-hosted endpoint is used.

### Managed baseline

Gemini File Search ($0.15 / 1M tokens indexing; storage and query embeddings free; metadata filters; page-level
citations). `python -m mybrain.evalx.filesearch_baseline` scores it on the same question set.

## Expected cost for a pilot (5,000 pages, ~2.5M tokens of text)

| Item | Estimate |
|---|---|
| LlamaParse agentic on 2,000 complex pages | ~$25 |
| Contextual chunk text (Gemini Flash, ~6k chunks, document re-sent per 10 chunks) | ~$10–20 |
| Entity extraction (Gemini Flash) | ~$3 |
| Embeddings (Gemini Embedding 2, 2.5M tokens + contexts) | ~$1 |
| Per question (plan + rerank + Opus 5.5 answer over ~10 chunks) | ~$0.05–0.15 |

## Security notes

- Bind the API to localhost and reach it over Tailscale, or enable the `public` compose profile (Caddy with
  automatic HTTPS and basic auth). `APP_PASSWORD` adds an application-level password on top.
- Retrieved text is treated as data: the answer prompt instructs the model to ignore instructions inside
  documents, and the answer agent has no tools.
- Back up `pgdata` and the `data` volume nightly (`pg_dump` + uploads). Vectors are derivable; the parsed
  Markdown and entity tables are the crown jewels.

## Later (not in the pilot)

Visual page retrieval for decks (Gemini Embedding 2 image embeddings or ColQwen on a GPU host), IMAP / Telegram
live connectors, per-entity news monitoring, text-to-SQL over spreadsheets, Graphiti/FalkorDB when graph
traversal questions dominate, multi-user ACLs (the schema already carries `acl_principals`).
