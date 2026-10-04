"""Query planning: detect entities, rewrite the question for search in both languages, decompose multi-hop
questions. One cheap LLM call; falls back to the raw question when the model is unavailable."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import psycopg

from ..ingest.normalize_fa import normalize_name
from ..providers.base import LLM

log = logging.getLogger(__name__)

SYSTEM = """You plan retrieval over a private knowledge base of a private-equity / business / political analyst.
Documents are in Persian and English. Given a question, return JSON:
{"language": "fa|en|mixed",
 "intent": "lookup|entity_profile|relationship|comparison|timeline|aggregate|other",
 "entities": [{"name": "as written", "type": "person|organization|place|other", "aliases": ["same name in the other script or common variants"]}],
 "queries": [{"text": "...", "lang": "fa|en"}],
 "keywords": ["exact identifiers, amounts, tickers, code names that must match literally"],
 "time": {"from": "YYYY-MM-DD or null", "to": "YYYY-MM-DD or null"}}
queries: 2 to 4 search queries: (1) the question rephrased as a concise search query, (2) the same in the other
language (Persian<->English, transliterating names), (3+) sub-questions when the question needs several facts
(multi-hop). Keep queries short and keyword-rich."""

MAX_QUERIES = 4


@dataclass
class PlanEntity:
    name: str
    type: str = "other"
    aliases: list[str] = field(default_factory=list)


@dataclass
class Plan:
    language: str = "unknown"
    intent: str = "lookup"
    entities: list[PlanEntity] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    time_from: str | None = None
    time_to: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "intent": self.intent,
            "entities": [{"name": e.name, "type": e.type, "aliases": e.aliases} for e in self.entities],
            "queries": self.queries,
            "keywords": self.keywords,
            "time": {"from": self.time_from, "to": self.time_to},
            "error": self.error,
        }


def plan_query(llm: LLM | None, question: str) -> Plan:
    plan = Plan(queries=[question])
    if llm is None:
        return plan
    try:
        data = llm.complete_json(SYSTEM, f"Question: {question}", max_tokens=1500)
    except Exception as e:  # noqa: BLE001
        log.warning("planner failed: %s", e)
        plan.error = str(e)
        return plan
    plan.raw = data
    plan.language = str(data.get("language") or "unknown")
    plan.intent = str(data.get("intent") or "lookup")
    for e in data.get("entities") or []:
        name = str(e.get("name") or "").strip()
        if name:
            plan.entities.append(
                PlanEntity(
                    name, str(e.get("type") or "other"), [str(a) for a in (e.get("aliases") or []) if a]
                )
            )
    seen = {question.strip().casefold()}
    for q in data.get("queries") or []:
        text = str(q.get("text") if isinstance(q, dict) else q).strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            plan.queries.append(text)
    plan.queries = plan.queries[: MAX_QUERIES + 1]
    plan.keywords = [str(k) for k in (data.get("keywords") or []) if k][:10]
    t = data.get("time") or {}
    plan.time_from = t.get("from") or None
    plan.time_to = t.get("to") or None
    return plan


def resolve_entities(
    conn: psycopg.Connection[Any], plan: Plan, limit_per_name: int = 3
) -> list[dict[str, Any]]:
    """Map planner entities to registry entities (exact alias, then trigram)."""
    found: dict[str, dict[str, Any]] = {}
    for pe in plan.entities:
        norms = [normalize_name(n) for n in [pe.name, *pe.aliases]]
        norms = [n for n in norms if len(n) >= 2]
        if not norms:
            continue
        rows = conn.execute(
            """
            SELECT DISTINCT ON (en.id) en.id, en.type, en.canonical_name, en.name_en, en.name_fa, en.description, en.mention_count, 1.0 AS sim
            FROM entity_aliases ea JOIN entities en ON en.id = ea.entity_id
            WHERE ea.alias_norm = ANY(%s)
            ORDER BY en.id, en.mention_count DESC
            LIMIT %s
            """,
            (norms, limit_per_name),
        ).fetchall()
        if not rows:
            conn.execute("SELECT set_limit(0.55::real)")
            rows = conn.execute(
                """
                SELECT en.id, en.type, en.canonical_name, en.name_en, en.name_fa, en.description, en.mention_count,
                       max(similarity(ea.alias_norm, cand)) AS sim
                FROM entity_aliases ea JOIN entities en ON en.id = ea.entity_id, unnest(%s::text[]) AS cand
                WHERE ea.alias_norm %% cand
                GROUP BY en.id ORDER BY sim DESC, en.mention_count DESC LIMIT %s
                """,
                (norms, limit_per_name),
            ).fetchall()
        for r in rows:
            found[str(r["id"])] = {**r, "id": str(r["id"]), "sim": float(r["sim"]), "query_name": pe.name}
    return sorted(found.values(), key=lambda r: (-r["sim"], -int(r["mention_count"] or 0)))
