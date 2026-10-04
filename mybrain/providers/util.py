from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")


def retry(fn: Callable[[], T], attempts: int = 4, base: float = 1.5, what: str = "call") -> T:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - vendor SDKs raise many types
            last = e
            if i == attempts - 1:
                break
            delay = base * (2**i)
            log.warning("%s failed (%s); retry %d in %.1fs", what, e, i + 1, delay)
            time.sleep(delay)
    assert last is not None
    raise last


def parse_json_text(text: str) -> dict[str, Any]:
    """Tolerant JSON parse: strips code fences and finds the outermost object."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj
