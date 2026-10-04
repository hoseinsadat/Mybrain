"""Entity layer: LLM extraction of people/organizations/relationships, resolution against the registry,
and chunk-level mention linking. GraphRAG-lite, all in Postgres."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from ..providers.base import LLM
from .normalize_fa import lexical_form, normalize_name, normalize_persian

log = logging.getLogger(__name__)

ENTITY_TYPES = {"person", "organization", "place", "other"}
WINDOW_CHARS = 30_000
TRGM_LINK = 0.88

SYSTEM = """You extract entities and relationships from documents for a private knowledge base about companies,
people, deals and politics (private equity and business context; Persian and English text).
Return JSON:
{"entities": [{"name": "canonical name as most often written", "type": "person|organization|place|other",
   "aliases": ["other spellings, abbreviations, transliterations, the other script"],
   "name_en": "Latin-script name or null", "name_fa": "Persian-script name or null",
   "description": "one sentence: who/what, role, sector, nationality",
   "attributes": {"role": "...", "organization": "...", "country": "...", "sector": "..."}}],
 "relationships": [{"subject": "entity name", "predicate": "short phrase: CEO of | board member of | invested in |
   owns stake in | advisor to | met with | subsidiary of | partner of | relative of | negotiating with",
   "object": "entity name", "description": "the evidence sentence", "valid_from": "YYYY-MM-DD or null",
   "valid_to": "YYYY-MM-DD or null", "confidence": 0.0}]}
Rules: only entities that actually appear; merge duplicates; keep names in their original script and give the
other script when the document shows it; skip generic roles without a name; include family names as aliases for
people; at most 60 entities and 80 relationships per call."""


@dataclass
class ExtractedEntity:
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)
    name_en: str | None = None
    name_fa: str | None = None
    description: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRelationship:
    subject: str
    predicate: str
    object: str
    description: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float | None = None


def _clean(s: Any) -> str | None:
    if s is None:
        return None
    s = normalize_persian(str(s)).strip()
    return s or None


def _date(s: Any) -> str | None:
    if not s or not isinstance(s, str):
        return None
    m = re.match(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$", s.strip())
    if not m:
        return None
    y, mo, d = m.group(1), m.group(2) or "01", m.group(3) or "01"
    try:
        datetime(int(y), int(mo), int(d))
    except ValueError:
        return None
    return f"{y}-{mo}-{d}"


def extract(
    llm: LLM, title: str | None, markdown: str
) -> tuple[list[ExtractedEntity], list[ExtractedRelationship]]:
    ents: dict[str, ExtractedEntity] = {}
    rels: list[ExtractedRelationship] = []
    for start in range(0, max(len(markdown), 1), WINDOW_CHARS):
        window = markdown[start : start + WINDOW_CHARS]
        if not window.strip():
            continue
        user = f'<document title="{title or "untitled"}" part="{start // WINDOW_CHARS + 1}">\n{window}\n</document>'
        try:
            data = llm.complete_json(SYSTEM, user, max_tokens=8192)
        except Exception as e:  # noqa: BLE001
            log.warning("entity extraction failed for window at %d: %s", start, e)
            continue
        for e in data.get("entities", []) or []:
            name = _clean(e.get("name"))
            if not name:
                continue
            etype = str(e.get("type", "other")).lower()
            etype = etype if etype in ENTITY_TYPES else "other"
            key = f"{etype}:{normalize_name(name)}"
            aliases = [a for a in (_clean(x) for x in (e.get("aliases") or [])) if a]
            cur = ents.get(key)
            if cur is None:
                ents[key] = ExtractedEntity(
                    name=name,
                    type=etype,
                    aliases=aliases,
                    name_en=_clean(e.get("name_en")),
                    name_fa=_clean(e.get("name_fa")),
                    description=_clean(e.get("description")),
                    attributes={k: v for k, v in (e.get("attributes") or {}).items() if v},
                )
            else:
                cur.aliases = list(dict.fromkeys(cur.aliases + aliases))
                cur.name_en = cur.name_en or _clean(e.get("name_en"))
                cur.name_fa = cur.name_fa or _clean(e.get("name_fa"))
                cur.description = cur.description or _clean(e.get("description"))
                cur.attributes = {**(e.get("attributes") or {}), **cur.attributes}
        for r in data.get("relationships", []) or []:
            s, p, o = _clean(r.get("subject")), _clean(r.get("predicate")), _clean(r.get("object"))
            if not (s and p and o):
                continue
            try:
                conf = float(r.get("confidence")) if r.get("confidence") is not None else None
            except (TypeError, ValueError):
                conf = None
            rels.append(
                ExtractedRelationship(
                    subject=s,
                    predicate=p[:120],
                    object=o,
                    description=_clean(r.get("description")),
                    valid_from=_date(r.get("valid_from")),
                    valid_to=_date(r.get("valid_to")),
                    confidence=conf,
                )
            )
    return list(ents.values()), rels


# ---------------------------------------------------------------------------
# resolution + storage
# ---------------------------------------------------------------------------


def _alias_set(e: ExtractedEntity) -> list[tuple[str, str]]:
    names = [e.name, *e.aliases, e.name_en, e.name_fa]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for n in names:
        if not n:
            continue
        norm = normalize_name(n)
        if len(norm) < 2 or norm in seen:
            continue
        seen.add(norm)
        out.append((n, norm))
    return out


def resolve_entity(conn: psycopg.Connection[Any], e: ExtractedEntity) -> str:
    """Return the entity id for an extracted entity, creating it when no alias matches."""
    aliases = _alias_set(e)
    norms = [n for _, n in aliases]
    row = conn.execute(
        """
        SELECT ea.entity_id, en.mention_count
        FROM entity_aliases ea JOIN entities en ON en.id = ea.entity_id
        WHERE ea.alias_norm = ANY(%s) AND en.type = %s
        ORDER BY en.mention_count DESC LIMIT 1
        """,
        (norms, e.type),
    ).fetchone()
    if row is None:
        # near-exact spelling variants of multi-word names (trigram); single tokens are too ambiguous
        multi = [n for n in norms if " " in n and len(n) >= 8]
        if multi:
            conn.execute("SELECT set_limit(%s::real)", (TRGM_LINK,))
            row = conn.execute(
                """
                SELECT ea.entity_id, max(similarity(ea.alias_norm, cand)) AS s
                FROM entity_aliases ea JOIN entities en ON en.id = ea.entity_id, unnest(%s::text[]) AS cand
                WHERE en.type = %s AND ea.alias_norm %% cand
                GROUP BY ea.entity_id ORDER BY s DESC LIMIT 1
                """,
                (multi, e.type),
            ).fetchone()
    if row is not None:
        entity_id = str(row["entity_id"])
        conn.execute(
            """
            UPDATE entities SET
              description = COALESCE(description, %s),
              name_en = COALESCE(name_en, %s), name_fa = COALESCE(name_fa, %s),
              attributes = attributes || %s::jsonb, updated_at = now()
            WHERE id = %s
            """,
            (e.description, e.name_en, e.name_fa, Jsonb(e.attributes), entity_id),
        )
    else:
        row = conn.execute(
            """
            INSERT INTO entities (type, canonical_name, name_en, name_fa, description, attributes, first_seen, last_seen)
            VALUES (%s, %s, %s, %s, %s, %s, now(), now()) RETURNING id
            """,
            (e.type, e.name, e.name_en, e.name_fa, e.description, Jsonb(e.attributes)),
        ).fetchone()
        assert row is not None
        entity_id = str(row["id"])
    for alias, norm in aliases:
        conn.execute(
            "INSERT INTO entity_aliases (entity_id, alias, alias_norm, lang) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (entity_id, alias, norm, "fa" if any("\u0600" <= ch <= "\u06ff" for ch in alias) else "en"),
        )
    return entity_id


def _mentions_in(lexical_text: str, norms: list[str]) -> bool:
    padded = f" {lexical_text} "
    return any(len(n) >= 3 and f" {n} " in padded for n in norms)


def store_entities(
    conn: psycopg.Connection[Any],
    document_id: str,
    chunks: list[dict[str, Any]],
    entities: list[ExtractedEntity],
    relationships: list[ExtractedRelationship],
) -> dict[str, int]:
    """Resolve entities, link chunk mentions and store relationships. Returns counts."""
    name_to_id: dict[str, str] = {}
    id_norms: dict[str, list[str]] = {}
    for e in entities:
        eid = resolve_entity(conn, e)
        norms = [n for _, n in _alias_set(e)]
        id_norms[eid] = list(dict.fromkeys(id_norms.get(eid, []) + norms))
        for n in [e.name, *e.aliases, e.name_en, e.name_fa]:
            if n:
                name_to_id[normalize_name(n)] = eid

    mentions = 0
    chunks_by_entity: dict[str, list[str]] = {}
    for ch in chunks:
        lex = ch["lexical_text"]
        for eid, norms in id_norms.items():
            if _mentions_in(lex, norms):
                conn.execute(
                    """
                    INSERT INTO entity_mentions (entity_id, chunk_id, document_id, snippet, confidence)
                    VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
                    """,
                    (eid, ch["id"], document_id, ch["text"][:300], 0.9),
                )
                mentions += 1
                chunks_by_entity.setdefault(eid, []).append(ch["id"])
    # entities the LLM saw but whose aliases never string-match a chunk (e.g. inflected forms): link to the first chunk
    for eid in id_norms:
        if eid not in chunks_by_entity and chunks:
            conn.execute(
                """
                INSERT INTO entity_mentions (entity_id, chunk_id, document_id, snippet, confidence)
                VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
                """,
                (eid, chunks[0]["id"], document_id, chunks[0]["text"][:300], 0.5),
            )
            chunks_by_entity[eid] = [chunks[0]["id"]]

    rels = 0
    for r in relationships:
        s_id = name_to_id.get(normalize_name(r.subject))
        o_id = name_to_id.get(normalize_name(r.object))
        if not s_id or not o_id or s_id == o_id:
            continue
        evidence = None
        s_chunks, o_chunks = set(chunks_by_entity.get(s_id, [])), set(chunks_by_entity.get(o_id, []))
        both = [c["id"] for c in chunks if c["id"] in s_chunks and c["id"] in o_chunks]
        if both:
            evidence = both[0]
        elif r.description:
            needle = lexical_form(r.description)[:60]
            evidence = next((c["id"] for c in chunks if needle and needle in c["lexical_text"]), None)
        conn.execute(
            """
            INSERT INTO relationships (subject_id, object_id, predicate, description, valid_from, valid_to, chunk_id, document_id, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (subject_id, object_id, predicate, document_id) DO UPDATE SET
              description = COALESCE(EXCLUDED.description, relationships.description),
              chunk_id = COALESCE(EXCLUDED.chunk_id, relationships.chunk_id)
            """,
            (
                s_id,
                o_id,
                r.predicate,
                r.description,
                r.valid_from,
                r.valid_to,
                evidence,
                document_id,
                r.confidence,
            ),
        )
        rels += 1

    for eid in id_norms:
        conn.execute(
            """
            UPDATE entities SET mention_count = (SELECT count(*) FROM entity_mentions WHERE entity_id = %s),
              last_seen = now(), first_seen = COALESCE(first_seen, now()) WHERE id = %s
            """,
            (eid, eid),
        )
    return {"entities": len(id_norms), "mentions": mentions, "relationships": rels}
