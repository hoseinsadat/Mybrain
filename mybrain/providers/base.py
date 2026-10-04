from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class Embedder(Protocol):
    provider: str
    model: str
    dims: int

    def embed_documents(
        self, texts: list[str], titles: list[str | None] | None = None
    ) -> list[list[float]]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    provider: str
    model: str

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """Return (original_index, relevance_score) sorted best-first."""
        ...


@dataclass
class CitedDocument:
    """A retrieved chunk handed to the answer model."""

    id: str
    title: str
    text: str
    context: str | None = None


@dataclass
class Citation:
    marker: int  # 1-based index into the documents list
    document_index: int  # 0-based
    cited_text: str
    start: int | None = None
    end: int | None = None


@dataclass
class AnswerResult:
    text: str  # markdown with [n] markers
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    refused: bool = False
    raw_blocks: list[dict[str, Any]] = field(default_factory=list)


class LLM(Protocol):
    provider: str
    model: str

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> dict[str, Any]: ...

    def complete_text(self, system: str, user: str, max_tokens: int = 2048) -> str: ...

    def answer_with_citations(
        self, system: str, question: str, documents: list[CitedDocument], max_tokens: int = 4096
    ) -> AnswerResult: ...
