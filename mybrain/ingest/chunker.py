"""Structure-aware chunking of parsed Markdown.

Headings become breadcrumbs, tables and code blocks stay whole (tables split by rows with the header
repeated when too long), paragraphs are packed to a token budget with a small sentence-level overlap.
Page markers (`<!-- page: N -->`) emitted by the parsers are tracked, not embedded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PAGE_RE = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENT_RE = re.compile(r"(?<=[.!?؟۔])\s+")


def estimate_tokens(text: str) -> int:
    """Cheap tokenizer-free estimate (Persian words tokenize into more pieces than English)."""
    n = 0
    for w in text.split():
        if w.isascii():
            n += 1 + len(w) // 5
        else:
            n += 1 + len(w) // 3
    return n


@dataclass
class Block:
    kind: str  # heading | paragraph | table | code
    text: str
    page: int | None
    breadcrumb: str
    level: int = 0
    tokens: int = 0


@dataclass
class Chunk:
    ord: int
    text: str
    breadcrumb: str
    page: int | None
    token_count: int
    kinds: list[str] = field(default_factory=list)


def _split_table(lines: list[str], max_tokens: int) -> list[str]:
    header = (
        lines[:2] if len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= set("-: ") else lines[:1]
    )
    body = lines[len(header) :]
    parts: list[str] = []
    cur: list[str] = list(header)
    for row in body:
        if estimate_tokens("\n".join(cur + [row])) > max_tokens and len(cur) > len(header):
            parts.append("\n".join(cur))
            cur = list(header)
        cur.append(row)
    parts.append("\n".join(cur))
    return parts


def _split_paragraph(text: str, max_tokens: int) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]
    out: list[str] = []
    cur: list[str] = []
    for sent in _SENT_RE.split(text):
        if cur and estimate_tokens(" ".join(cur + [sent])) > max_tokens:
            out.append(" ".join(cur))
            cur = []
        cur.append(sent)
    if cur:
        out.append(" ".join(cur))
    # a single giant "sentence" (e.g. a dump without punctuation): hard split by words
    final: list[str] = []
    for piece in out:
        if estimate_tokens(piece) <= max_tokens * 1.5:
            final.append(piece)
            continue
        words = piece.split()
        step = max(20, int(len(words) * max_tokens / max(estimate_tokens(piece), 1)))
        for i in range(0, len(words), step):
            final.append(" ".join(words[i : i + step]))
    return final


def parse_blocks(markdown: str, title: str | None, max_tokens: int) -> list[Block]:
    lines = markdown.split("\n")
    blocks: list[Block] = []
    page: int | None = None
    crumbs: list[tuple[int, str]] = []
    para: list[str] = []
    i = 0

    def breadcrumb() -> str:
        parts = [title] if title else []
        parts += [c[1] for c in crumbs]
        return " > ".join(p for p in parts if p)

    def flush_para() -> None:
        nonlocal para
        if para:
            text = "\n".join(para).strip()
            for piece in _split_paragraph(text, max_tokens):
                blocks.append(Block("paragraph", piece, page, breadcrumb(), tokens=estimate_tokens(piece)))
            para = []

    while i < len(lines):
        line = lines[i]
        m = _PAGE_RE.match(line.strip())
        if m:
            flush_para()
            page = int(m.group(1))
            i += 1
            continue
        if line.strip().startswith("```"):
            flush_para()
            code = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            if i < len(lines):
                code.append(lines[i])
                i += 1
            text = "\n".join(code)
            blocks.append(Block("code", text, page, breadcrumb(), tokens=estimate_tokens(text)))
            continue
        hm = _HEADING_RE.match(line)
        if hm:
            flush_para()
            level = len(hm.group(1))
            name = hm.group(2).strip()
            crumbs = [c for c in crumbs if c[0] < level] + [(level, name)]
            blocks.append(
                Block("heading", line.strip(), page, breadcrumb(), level=level, tokens=estimate_tokens(line))
            )
            i += 1
            continue
        if line.lstrip().startswith("|"):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            for part in _split_table(rows, max_tokens):
                blocks.append(Block("table", part, page, breadcrumb(), tokens=estimate_tokens(part)))
            continue
        if not line.strip():
            flush_para()
            i += 1
            continue
        para.append(line.rstrip())
        i += 1
    flush_para()
    return blocks


def chunk_markdown(
    markdown: str, title: str | None = None, max_tokens: int = 400, overlap_tokens: int = 40
) -> list[Chunk]:
    # paragraphs are split to (budget - overlap) so a piece plus its carried overlap still fits the budget
    blocks = parse_blocks(markdown, title, max(max_tokens - overlap_tokens, max_tokens // 2))
    chunks: list[Chunk] = []
    cur: list[Block] = []
    cur_tokens = 0
    min_flush = max(max_tokens // 3, 1)

    def flush() -> None:
        nonlocal cur, cur_tokens
        if not cur:
            return
        text = "\n\n".join(b.text for b in cur).strip()
        if text:
            first = next((b for b in cur if b.kind != "heading"), cur[0])
            chunks.append(
                Chunk(
                    ord=len(chunks),
                    text=text,
                    breadcrumb=first.breadcrumb,
                    page=cur[0].page,
                    token_count=estimate_tokens(text),
                    kinds=sorted({b.kind for b in cur}),
                )
            )
        # sentence-level overlap: carry the tail of the last paragraph into the next chunk
        carry: list[Block] = []
        if overlap_tokens > 0 and cur and cur[-1].kind == "paragraph":
            sents = _SENT_RE.split(cur[-1].text)
            tail: list[str] = []
            for s in reversed(sents):
                if estimate_tokens(" ".join([s] + tail)) > overlap_tokens:
                    break
                tail.insert(0, s)
            if tail and len(tail) < len(sents):
                t = " ".join(tail)
                carry = [Block("paragraph", t, cur[-1].page, cur[-1].breadcrumb, tokens=estimate_tokens(t))]
        cur = carry
        cur_tokens = sum(b.tokens for b in cur)

    for b in blocks:
        if b.kind == "heading" and b.level <= 3 and cur_tokens >= min_flush:
            flush()
            cur, cur_tokens = [], 0  # new section: no overlap carried across sections
        if cur and cur_tokens + b.tokens > max_tokens:
            flush()
            if b.kind in ("table", "code") or cur_tokens + b.tokens > max_tokens:
                cur, cur_tokens = [], 0  # never glue an overlap tail onto a table, or overflow the budget
        cur.append(b)
        cur_tokens += b.tokens
    flush()

    # merge a tiny trailing chunk into its predecessor
    if (
        len(chunks) >= 2
        and chunks[-1].token_count < max_tokens // 5
        and chunks[-2].token_count + chunks[-1].token_count <= int(max_tokens * 1.3)
    ):
        last = chunks.pop()
        prev = chunks[-1]
        prev.text = prev.text + "\n\n" + last.text
        prev.token_count = estimate_tokens(prev.text)
        prev.kinds = sorted(set(prev.kinds) | set(last.kinds))
    for n, c in enumerate(chunks):
        c.ord = n
    return chunks
