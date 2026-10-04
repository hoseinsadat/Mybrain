"""Embedding providers. Document/query asymmetry is handled per vendor."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

PGVECTOR_MAX_INDEXED_DIMS = 2000


def _retry(fn, attempts: int = 4, base: float = 1.5):
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - vendor SDKs raise many types
            last = e
            if i == attempts - 1:
                break
            delay = base * (2**i)
            log.warning("embedding call failed (%s); retry %d in %.1fs", e, i + 1, delay)
            time.sleep(delay)
    assert last is not None
    raise last


def _fit_dims(vectors: list[list[float]], dims: int) -> list[list[float]]:
    """Truncate (Matryoshka) and L2-normalize when a vendor ignores the requested size."""
    out = []
    for v in vectors:
        if len(v) > dims:
            v = v[:dims]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


class GeminiEmbedder:
    provider = "gemini"
    batch_size = 32

    def __init__(self, model: str, dims: int, api_key: str):
        from google import genai

        self.model = model
        self.dims = dims
        self._client = genai.Client(api_key=api_key or None)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # One Content per text => one vector per text (a flat list would be aggregated into one vector).
            contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]

            def call(contents=contents):
                return self._client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=types.EmbedContentConfig(output_dimensionality=self.dims),
                )

            res = _retry(call)
            out.extend([list(e.values) for e in res.embeddings])
        return _fit_dims(out, self.dims)

    def embed_documents(self, texts: list[str], titles: list[str | None] | None = None) -> list[list[float]]:
        titles = titles or [None] * len(texts)
        return self._embed([f"title: {t or 'none'} | text: {x}" for x, t in zip(texts, titles, strict=True)])

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed([f"task: search result | query: {q}" for q in texts])


class VoyageEmbedder:
    provider = "voyage"
    batch_size = 64

    def __init__(self, model: str, dims: int, api_key: str):
        self.model = model
        self.dims = dims
        self._client = httpx.Client(
            base_url="https://api.voyageai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            body: dict[str, Any] = {
                "input": batch,
                "model": self.model,
                "input_type": input_type,
                "output_dimension": self.dims,
            }

            def call(body=body):
                r = self._client.post("/embeddings", json=body)
                r.raise_for_status()
                return r.json()

            data = _retry(call)
            rows = sorted(data["data"], key=lambda d: d["index"])
            out.extend([row["embedding"] for row in rows])
        return _fit_dims(out, self.dims)

    def embed_documents(self, texts: list[str], titles: list[str | None] | None = None) -> list[list[float]]:
        titles = titles or [None] * len(texts)
        return self._embed([f"{t}\n{x}" if t else x for x, t in zip(texts, titles, strict=True)], "document")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "query")


class OpenAICompatEmbedder:
    """Works for Together/Fireworks (Qwen3-Embedding), Jina, OpenAI. Sends `dimensions` for Matryoshka models."""

    provider = "openai_compat"
    batch_size = 32

    def __init__(self, model: str, dims: int, api_key: str, base_url: str):
        self.model = model
        self.dims = dims
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {api_key}"}, timeout=120
        )
        self._send_dims = True

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            def call(batch=batch):
                body: dict[str, Any] = {"model": self.model, "input": batch}
                if self._send_dims:
                    body["dimensions"] = self.dims
                r = self._client.post("/embeddings", json=body)
                if r.status_code == 400 and self._send_dims and "dimension" in r.text.lower():
                    self._send_dims = False  # host does not support MRL truncation; we truncate client-side
                    return call(batch)
                r.raise_for_status()
                return r.json()

            data = _retry(call)
            rows = sorted(data["data"], key=lambda d: d["index"])
            out.extend([row["embedding"] for row in rows])
        return _fit_dims(out, self.dims)

    def embed_documents(self, texts: list[str], titles: list[str | None] | None = None) -> list[list[float]]:
        titles = titles or [None] * len(texts)
        return self._embed([f"{t}\n{x}" if t else x for x, t in zip(texts, titles, strict=True)])

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        # Qwen3-Embedding recommends an instruction prefix on the query side only.
        return self._embed(
            [f"Instruct: Given a search query, retrieve relevant passages\nQuery: {q}" for q in texts]
        )
