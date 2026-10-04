"""LLM providers: Anthropic (final answers with native Citations) and Gemini (cheap extraction, optional answers)."""

from __future__ import annotations

import logging
import re
from typing import Any

from .base import AnswerResult, Citation, CitedDocument
from .util import parse_json_text, retry

log = logging.getLogger(__name__)

JSON_SUFFIX = "\n\nRespond with a single JSON object only. No prose, no code fences."
MAX_DOC_CHARS = 12_000


def _append_markers(text: str, markers: list[int]) -> str:
    if not markers:
        return text
    stripped = text.rstrip()
    tail = text[len(stripped) :]
    return stripped + "".join(f"[{m}]" for m in markers) + tail


class AnthropicLLM:
    provider = "anthropic"

    def __init__(self, model: str, api_key: str, effort: str | None = None):
        import anthropic

        self.model = model
        self.effort = effort
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key or None, max_retries=3)

    def _supports_effort(self) -> bool:
        return not self.model.startswith("claude-haiku")

    def _kwargs(self, **kw: Any) -> dict[str, Any]:
        if self.effort and self._supports_effort():
            kw.setdefault("output_config", {"effort": self.effort})
        return kw

    def complete_text(self, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self._client.messages.create(
            **self._kwargs(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("model refused the request")
        return "".join(b.text for b in resp.content if b.type == "text")

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> dict[str, Any]:
        return parse_json_text(self.complete_text(system + JSON_SUFFIX, user, max_tokens))

    def answer_with_citations(
        self, system: str, question: str, documents: list[CitedDocument], max_tokens: int = 4096
    ) -> AnswerResult:
        content: list[dict[str, Any]] = []
        for d in documents:
            block: dict[str, Any] = {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": d.text[:MAX_DOC_CHARS]},
                "title": (d.title or "untitled")[:200],
                "citations": {"enabled": True},
            }
            if d.context:
                block["context"] = d.context[:1000]
            content.append(block)
        content.append({"type": "text", "text": question})
        kw = self._kwargs(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        try:
            # Server-side refusal fallbacks: a policy decline is re-run on a fallback model inside the same call.
            resp = self._client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw
            )
        except (self._anthropic.BadRequestError, TypeError) as e:
            if "fallback" not in str(e).lower():
                raise
            log.info("fallbacks unsupported here (%s); retrying without", e)
            resp = self._client.messages.create(**kw)

        if resp.stop_reason == "refusal":
            return AnswerResult(
                text="The answer model declined this request.", refused=True, model=resp.model
            )

        parts: list[str] = []
        citations: list[Citation] = []
        raw_blocks: list[dict[str, Any]] = []
        for block in resp.content:
            if block.type != "text":
                continue
            markers: list[int] = []
            for c in getattr(block, "citations", None) or []:
                di = int(c.document_index)
                marker = di + 1
                citations.append(
                    Citation(
                        marker=marker,
                        document_index=di,
                        cited_text=c.cited_text,
                        start=getattr(c, "start_char_index", None),
                        end=getattr(c, "end_char_index", None),
                    )
                )
                if marker not in markers:
                    markers.append(marker)
            parts.append(_append_markers(block.text, markers))
            raw_blocks.append({"text": block.text, "markers": markers})
        usage = resp.usage
        return AnswerResult(
            text="".join(parts).strip(),
            citations=citations,
            model=resp.model,
            usage={
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            },
            raw_blocks=raw_blocks,
        )


class GeminiLLM:
    provider = "gemini"

    def __init__(self, model: str, api_key: str):
        from google import genai

        self.model = model
        self._client = genai.Client(api_key=api_key or None)

    def _gen(self, system: str, user: str, max_tokens: int, json_mode: bool) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.1,
            response_mime_type="application/json" if json_mode else None,
        )
        resp = retry(
            lambda: self._client.models.generate_content(model=self.model, contents=user, config=cfg),
            what="gemini",
        )
        return resp.text or ""

    def complete_text(self, system: str, user: str, max_tokens: int = 2048) -> str:
        return self._gen(system, user, max_tokens, json_mode=False)

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> dict[str, Any]:
        return parse_json_text(self._gen(system, user, max_tokens, json_mode=True))

    def answer_with_citations(
        self, system: str, question: str, documents: list[CitedDocument], max_tokens: int = 4096
    ) -> AnswerResult:
        # No native citation spans: the model places [n] markers itself.
        src = "\n\n".join(
            f"[{i + 1}] {d.title}\n{(d.context or '').strip()}\n{d.text[:MAX_DOC_CHARS]}"
            for i, d in enumerate(documents)
        )
        sys_prompt = (
            system
            + "\nCite the numbered sources you rely on by appending [n] markers to the relevant sentences."
        )
        text = self._gen(sys_prompt, f"SOURCES:\n{src}\n\nQUESTION: {question}", max_tokens, json_mode=False)
        markers = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 1 <= int(m) <= len(documents)})
        return AnswerResult(
            text=text.strip(),
            citations=[Citation(marker=m, document_index=m - 1, cited_text="") for m in markers],
            model=self.model,
        )
