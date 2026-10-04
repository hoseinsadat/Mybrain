"""LlamaParse (v2 REST) client. Used for complex or scanned PDFs; local parsers handle the rest."""

from __future__ import annotations

import logging
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

TERMINAL_OK = {"SUCCESS", "COMPLETED", "DONE"}
TERMINAL_FAIL = {"ERROR", "FAILED", "CANCELLED", "CANCELED"}


@dataclass
class ParsedPages:
    markdown: str
    pages: list[tuple[int, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _find_key(obj: Any, key: str) -> Any:
    """Depth-first search for a key in a nested JSON structure."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def pages_to_markdown(pages: list[tuple[int, str]]) -> str:
    return "\n\n".join(f"<!-- page: {n} -->\n{t.strip()}" for n, t in pages if t and t.strip())


class LlamaParseClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        tier: str = "agentic",
        version: str = "latest",
        language: str | None = None,
    ):
        self.tier = tier
        self.version = version
        self.language = language
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {api_key}"}, timeout=120
        )

    def parse(self, path: Path, mime: str | None = None, timeout_s: int = 1800) -> ParsedPages:
        mime = mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data: dict[str, str] = {"tier": self.tier, "version": self.version}
        if self.language:
            data["language"] = self.language
        with path.open("rb") as f:
            r = self._client.post("/api/v2/parse/upload", files={"file": (path.name, f, mime)}, data=data)
        r.raise_for_status()
        job = r.json()
        job_id = job.get("id") or job.get("job_id")
        if not job_id:
            raise RuntimeError(f"LlamaParse did not return a job id: {job}")

        deadline = time.time() + timeout_s
        delay = 3.0
        status = (job.get("status") or "").upper()
        while status not in TERMINAL_OK:
            if status in TERMINAL_FAIL:
                raise RuntimeError(f"LlamaParse job {job_id} failed: {job}")
            if time.time() > deadline:
                raise TimeoutError(f"LlamaParse job {job_id} timed out")
            time.sleep(delay)
            delay = min(delay * 1.5, 15)
            j = self._client.get(f"/api/v2/parse/{job_id}")
            j.raise_for_status()
            job = j.json()
            status = (job.get("status") or "").upper()

        res = self._client.get(f"/api/v2/parse/{job_id}", params={"expand": "markdown"})
        res.raise_for_status()
        body = res.json()
        md = _find_key(body, "markdown")
        pages: list[tuple[int, str]] = []
        if isinstance(md, str):
            return ParsedPages(markdown=md, metadata={"job_id": job_id})
        items = md.get("pages") if isinstance(md, dict) else md
        if isinstance(items, list):
            for i, p in enumerate(items):
                if isinstance(p, str):
                    pages.append((i + 1, p))
                elif isinstance(p, dict):
                    n = p.get("page_number") or p.get("page") or (i + 1)
                    txt = p.get("markdown") or p.get("md") or p.get("text") or ""
                    if isinstance(txt, dict):
                        txt = txt.get("content") or txt.get("text") or ""
                    pages.append((int(n), str(txt)))
        if not pages:
            raise RuntimeError(f"LlamaParse result had no markdown: keys={list(body)[:10]}")
        return ParsedPages(markdown=pages_to_markdown(pages), pages=pages, metadata={"job_id": job_id})
