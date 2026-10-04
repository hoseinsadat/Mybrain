"""Cross-encoder rerankers behind one interface. All are multilingual."""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

MAX_DOC_CHARS = 4000


def _clip(docs: list[str]) -> list[str]:
    return [d[:MAX_DOC_CHARS] for d in docs]


class NoReranker:
    provider = "none"
    model = "none"

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        return [(i, 1.0 - i / max(len(documents), 1)) for i in range(min(top_n, len(documents)))]


class ZeroEntropyReranker:
    provider = "zeroentropy"

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.zeroentropy.dev/v1"):
        self.model = model
        self._client = httpx.Client(
            base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60
        )

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        r = self._client.post(
            "/models/rerank",
            json={"model": self.model, "query": query, "documents": _clip(documents), "top_n": top_n},
        )
        r.raise_for_status()
        return [(x["index"], float(x["relevance_score"])) for x in r.json()["results"]]


class CohereReranker:
    provider = "cohere"

    def __init__(self, model: str, api_key: str):
        self.model = model
        self._client = httpx.Client(
            base_url="https://api.cohere.com/v2", headers={"Authorization": f"Bearer {api_key}"}, timeout=60
        )

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        r = self._client.post(
            "/rerank",
            json={"model": self.model, "query": query, "documents": _clip(documents), "top_n": top_n},
        )
        r.raise_for_status()
        return [(x["index"], float(x["relevance_score"])) for x in r.json()["results"]]


class VoyageReranker:
    provider = "voyage"

    def __init__(self, model: str, api_key: str):
        self.model = model
        self._client = httpx.Client(
            base_url="https://api.voyageai.com/v1", headers={"Authorization": f"Bearer {api_key}"}, timeout=60
        )

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        r = self._client.post(
            "/rerank",
            json={"model": self.model, "query": query, "documents": _clip(documents), "top_k": top_n},
        )
        r.raise_for_status()
        rows = sorted(r.json()["data"], key=lambda x: -float(x["relevance_score"]))
        return [(x["index"], float(x["relevance_score"])) for x in rows]
