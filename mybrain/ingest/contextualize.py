"""Contextual retrieval: a cheap LLM writes a 1-2 sentence situating context per chunk (batched)."""

from __future__ import annotations

import logging
from typing import Any

from ..providers.base import LLM
from .chunker import Chunk

log = logging.getLogger(__name__)

SYSTEM = """You prepare chunks of a document for a search index used by a private-equity / business / political analyst.
For each chunk, write 1-2 sentences of context that situates it within the whole document so that it can be
understood and found on its own: name the document, the companies/people/places involved (in the same script as
the chunk, add the Latin spelling of names if the document gives it), the date/period, and what the chunk is about.
Write in the same language as the chunk. Do not summarize the chunk itself; add the missing context around it."""

BATCH = 10


def contextualize(
    llm: LLM, title: str | None, markdown: str, chunks: list[Chunk], max_doc_chars: int
) -> list[str]:
    doc_excerpt = markdown[:max_doc_chars]
    contexts: list[str] = [""] * len(chunks)
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start : start + BATCH]
        numbered = "\n\n".join(
            f'<chunk id="{i + 1}">\n{c.text[:3000]}\n</chunk>' for i, c in enumerate(batch)
        )
        user = (
            f'<document title="{title or "untitled"}">\n{doc_excerpt}\n</document>\n\n'
            f"Chunks to situate:\n{numbered}\n\n"
            f'Return JSON: {{"contexts": [{{"id": 1, "context": "..."}}, ...]}} with exactly {len(batch)} items.'
        )
        try:
            data: dict[str, Any] = llm.complete_json(SYSTEM, user, max_tokens=4096)
            for item in data.get("contexts", []):
                idx = int(item.get("id", 0)) - 1
                if 0 <= idx < len(batch):
                    contexts[start + idx] = str(item.get("context", "")).strip()[:1000]
        except Exception as e:  # noqa: BLE001
            log.warning(
                "contextualize batch failed (%s); chunks %d-%d keep breadcrumb only",
                e,
                start,
                start + len(batch),
            )
    return contexts
