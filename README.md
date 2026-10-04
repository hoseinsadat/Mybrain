# Mybrain

A personal knowledge brain for private-equity, business and political work: drop in memos, decks, financial
statements, spreadsheets, e-mails, chat exports and meeting notes (English or Persian), then ask questions
about the companies and people in them and get cited answers.

What makes it more than "chat with PDFs":

- **Entity layer**: people and organizations are extracted with aliases in both scripts and dated
  relationships, so "what do I know about X" and "who at Y have I met" hit entity profiles first.
- **Hybrid multilingual retrieval**: pgvector + true BM25 (ParadeDB, ICU tokenizer) fused with RRF, an
  LLM query planner that searches in both languages, and a cross-encoder reranker.
- **Cited answers**: Claude's Citations API returns the exact passage behind every claim.
- **Pluggable, versioned vendors + built-in eval**: swap embedding models, rerankers, parsers and LLMs
  through `.env`, re-index without re-parsing, and measure each change on your own questions.

See `docs/ARCHITECTURE.md` for the design and vendor comparison, `docs/EVAL.md` for the bake-off.

## Quick start (VPS, 4 GB RAM / 2 vCPU)

```bash
git clone <this repo> && cd Mybrain
cp .env.example .env            # fill in API keys: GEMINI, ZEROENTROPY, ANTHROPIC (+ LLAMAPARSE for PDFs)
docker compose up -d --build    # ParadeDB + API + worker
open http://localhost:8000      # or reach it over Tailscale
```

Then upload files in the **Documents** tab (or `docker compose exec api python -m scripts.ingest_dir /data/inbox`
after copying a folder into the `data` volume), watch the **Jobs** tab, and ask in **Ask**.

Public exposure with HTTPS and basic auth:

```bash
export DOMAIN=brain.example.com BASIC_USER=me BASIC_HASH="$(docker run --rm caddy:2 caddy hash-password --plaintext 'secret')"
docker compose --profile public up -d
```

## Configuration

Everything is an environment variable (see `.env.example`). The important ones:

| Variable | Default | Options |
|---|---|---|
| `EMBED_PROVIDER` / `EMBED_MODEL` / `EMBED_DIMS` | `gemini` / `gemini-embedding-2` / 1536 | `voyage` (`voyage-4-large`, 1024), `openai_compat` (Together `Qwen/Qwen3-Embedding-8B`, Jina) |
| `RERANK_PROVIDER` / `RERANK_MODEL` | `zeroentropy` / `zerank-2` | `cohere` (`rerank-v4.0-pro`), `voyage` (`rerank-2.5`), `none` |
| `ANSWER_PROVIDER` / `ANSWER_MODEL` / `ANSWER_EFFORT` | `anthropic` / `claude-opus-5-5` / `medium` | `claude-sonnet-5-5`; `gemini` + `gemini-3.1-pro-preview` |
| `EXTRACT_PROVIDER` / `EXTRACT_MODEL` | `gemini` / `gemini-3.8-flash` | `anthropic` (uses `EXTRACT_ANTHROPIC_MODEL`, `claude-haiku-4-5`) |
| `PDF_PARSER` | `auto` | `llamaparse`, `local`; `LLAMAPARSE_TIER=agentic|agentic_plus|cost_effective|fast` |
| `CONTEXTUALIZE`, `EXTRACT_ENTITIES`, `SECOND_PASS` | true | toggles for the LLM-powered stages |
| `LEXICAL_BACKEND` | `auto` | `pg_search` or `tsvector` (plain Postgres) |
| `APP_PASSWORD` | empty | single-user password (Bearer or Basic) |

Changing the embedding model or chunking is a new **index version**: `POST /api/index-versions`
(re-embeds from stored text), run the eval, then `POST /api/index-versions/{id}/activate`.

## API

`POST /api/documents` (multipart upload) · `GET /api/documents` · `GET /api/documents/{id}` ·
`DELETE /api/documents/{id}` · `POST /api/documents/{id}/reingest` · `GET /api/jobs` ·
`POST /api/ask {question}` · `POST /api/search {question}` · `POST /api/feedback` · `GET /api/entities?q=` ·
`GET /api/entities/{id}` · `POST /api/entities/{id}/merge` · `GET|POST /api/index-versions` ·
`POST /api/eval/sets` · `POST /api/eval/run` · `GET /api/eval/runs` · `GET /healthz`

## Development

```bash
uv venv && uv pip install -e '.[dev]'
ruff check mybrain scripts tests && pytest                       # unit tests
MYBRAIN_TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:5433/mybrain_test pytest   # + end-to-end (needs pgvector)
uvicorn mybrain.main:app --reload & mybrain-worker               # run locally against a Postgres with pgvector
```

The end-to-end test uses fake embedding/LLM providers, so it runs without API keys; with plain Postgres it
exercises the tsvector fallback, with ParadeDB the BM25 path.

## Layout

```
mybrain/
  config.py            settings (pydantic-settings)
  db.py                pool, migrations, index versions, lexical backend detection
  migrations/          001 core schema, 002 pg_search BM25 (applied only when available)
  providers/           embeddings (gemini, voyage, openai_compat), rerank (zeroentropy, cohere, voyage), llm (anthropic w/ citations, gemini), parse (LlamaParse v2)
  ingest/              parsers, Persian normalization, chunker, contextualize, entities, pipeline, queue, worker
  retrieval/           hybrid SQL + RRF + entity boost, planner, search orchestration (rerank, second pass)
  answer/cited.py      entity profiles + cited answer + query log
  evalx/               metrics, eval runner, Gemini File Search baseline
  ui/index.html        single-page UI
scripts/ingest_dir.py  bulk loader
eval/                  question sets
```
