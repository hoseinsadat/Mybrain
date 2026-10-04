# Evaluation: measure before you trust

Every vendor choice in Mybrain (embedding model, reranker, contextual chunks, planner, second pass) is a
configuration. The eval harness tells you which configuration wins **on your own documents and questions**,
instead of trusting public leaderboards that contain no Persian private-equity memos.

## 1. Build a question set (60–100 questions)

Write `eval/questions.jsonl`, one JSON object per line (see `eval/questions.example.jsonl`):

| field | meaning |
|---|---|
| `qid` | stable id |
| `question` | as you would really ask it (mix English and Persian; ask in one language about a document in the other) |
| `type` | `lookup`, `entity`, `multi_hop`, `timeline`, `aggregate`, `unanswerable` |
| `gold_docs` | filename or title substrings of the documents that contain the answer (resolved to ids at load time) |
| `gold_chunk_ids` | optional exact chunk ids (copy from the UI's retrieved-chunks table) for chunk-level metrics |
| `gold_answer` | optional reference answer, for manual review |

Good sets include: exact-identifier lookups (amounts, dates, tickers), people/company profile questions,
relationship questions ("who introduced X to Y"), cross-language questions, and a few unanswerable ones.

## 2. Load and run

```bash
# load (or reload) the set and run with the current configuration
mybrain-eval --set pilot --file eval/questions.jsonl --label "gemini2+zerank2+ctx"

# ablations
mybrain-eval --set pilot --no-rerank --label "no reranker"
mybrain-eval --set pilot --rerank-provider cohere --rerank-model rerank-v4.0-pro --label "cohere"
mybrain-eval --set pilot --no-planner --label "no planner"
mybrain-eval --set pilot --second-pass --label "agentic second pass"

# a different embedding model = a new index version (re-embeds from stored text, no re-parsing)
curl -X POST localhost:8000/api/index-versions -H 'content-type: application/json' -d '{"name":"voyage4","embedding_provider":"voyage","embedding_model":"voyage-4-large","dims":1024,"reindex":true}'
# wait for the worker, then:
mybrain-eval --set pilot --index-version voyage4 --label "voyage-4-large"
```

Runs are stored in `eval_runs` and listed in the UI's **Eval** tab and `GET /api/eval/runs`.

## 3. Metrics

- `doc_hit@k` / `doc_recall@k` (k = 5, 10, 20): is a gold document in the top k? Anthropic's "top-20 failure rate" is `1 - doc_hit@20`.
- `doc_mrr`, `doc_ndcg@10`: ranking quality.
- `chunk_*`: same at chunk level when `gold_chunk_ids` are given.
- `mean_latency_ms`: plan + embed + search + rerank (+ second pass).

Retrieval metrics are deterministic and cheap: run them on every configuration change. Judge answer quality
separately by reading answers for 20 questions with the citations open.

## 4. Managed baseline: Gemini File Search

```bash
python -m mybrain.evalx.filesearch_baseline --set pilot
```

Uploads the same source files to a Gemini File Search store (indexing costs $0.15 per 1M tokens, storage and
query-time embeddings are free) and scores document-level hits from its citations. If the custom stack does not
beat this baseline on your questions, keep the baseline and drop the infrastructure.

## 5. Locking a configuration

When a configuration wins, make it the default: `.env` for reranker/planner/second pass, and
`POST /api/index-versions/{id}/activate` for the embedding model. Keep the losing index version until you have
used the new one for a week, then delete its chunks (`DELETE FROM chunks WHERE index_version_id = ...`).
