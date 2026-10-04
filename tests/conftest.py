from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any

import pytest

os.environ.setdefault("DATABASE_URL", os.environ.get("MYBRAIN_TEST_DATABASE_URL", ""))
os.environ.setdefault("DATA_DIR", "/tmp/mybrain-test-data")
os.environ.setdefault("CONTEXTUALIZE", "true")
os.environ.setdefault("EXTRACT_ENTITIES", "true")
os.environ.setdefault("SECOND_PASS", "false")

from mybrain.providers.base import AnswerResult, Citation, CitedDocument  # noqa: E402


class FakeEmbedder:
    """Bag-of-words hashing embedder: deterministic, overlapping tokens => cosine similarity."""

    provider = "fake"
    model = "bow"

    def __init__(self, dims: int = 8):
        self.dims = dims

    def _vec(self, text: str) -> list[float]:
        from mybrain.ingest.normalize_fa import lexical_form

        v = [0.0] * self.dims
        for tok in lexical_form(text).split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dims] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str], titles: list[str | None] | None = None) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class FakeLLM:
    provider = "fake"
    model = "fake"
    calls: list[tuple[str, str]] = []

    def complete_text(self, system: str, user: str, max_tokens: int = 2048) -> str:
        return "ok"

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> dict[str, Any]:
        self.calls.append((system[:40], user[:80]))
        if "Chunks to situate" in user:
            n = len(re.findall(r'<chunk id="\d+">', user))
            return {
                "contexts": [
                    {"id": i + 1, "context": f"Context for chunk {i + 1} of the test document."}
                    for i in range(n)
                ]
            }
        if system.startswith("You extract entities"):
            if "Pars Holding" in user or "هلدینگ پارس" in user:
                return {
                    "entities": [
                        {
                            "name": "Ali Rezaei",
                            "type": "person",
                            "aliases": ["علی رضایی", "Rezaei"],
                            "name_en": "Ali Rezaei",
                            "name_fa": "علی رضایی",
                            "description": "CFO of Pars Holding",
                            "attributes": {"role": "CFO"},
                        },
                        {
                            "name": "Pars Holding",
                            "type": "organization",
                            "aliases": ["هلدینگ پارس"],
                            "name_en": "Pars Holding",
                            "name_fa": "هلدینگ پارس",
                            "description": "Tehran-based industrial holding",
                            "attributes": {"country": "Iran"},
                        },
                    ],
                    "relationships": [
                        {
                            "subject": "Ali Rezaei",
                            "predicate": "CFO of",
                            "object": "Pars Holding",
                            "description": "Ali Rezaei is the CFO of Pars Holding since 2023",
                            "valid_from": "2023-01-01",
                            "valid_to": None,
                            "confidence": 0.9,
                        }
                    ],
                }
            return {
                "entities": [
                    {
                        "name": "Tehran Metro",
                        "type": "organization",
                        "aliases": ["مترو تهران"],
                        "description": "Public transport operator",
                    }
                ],
                "relationships": [],
            }
        if system.startswith("You plan retrieval"):
            fa = any("؀" <= ch <= "ۿ" for ch in user)
            return {
                "language": "fa" if fa else "en",
                "intent": "entity_profile" if "Rezaei" in user or "رضایی" in user else "lookup",
                "entities": [{"name": "Ali Rezaei", "type": "person", "aliases": ["علی رضایی"]}]
                if ("Rezaei" in user or "رضایی" in user)
                else [],
                "queries": [
                    {"text": "Pars Holding CFO", "lang": "en"},
                    {"text": "مدیر مالی هلدینگ پارس", "lang": "fa"},
                ],
                "keywords": [],
                "time": {"from": None, "to": None},
            }
        if system.startswith("You check whether"):
            return {"sufficient": True, "queries": []}
        return {}

    def answer_with_citations(
        self, system: str, question: str, documents: list[CitedDocument], max_tokens: int = 4096
    ) -> AnswerResult:
        return AnswerResult(
            text="Ali Rezaei is the CFO of Pars Holding.[1]",
            citations=[Citation(marker=1, document_index=0, cited_text=documents[0].text[:40])]
            if documents
            else [],
            model="fake",
            usage={"input_tokens": 10, "output_tokens": 5},
        )


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
