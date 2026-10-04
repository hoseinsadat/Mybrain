"""Retrieval metrics with binary relevance."""

from __future__ import annotations

import math
from collections.abc import Iterable


def hit_at_k(ranked: list[str], gold: Iterable[str], k: int) -> float:
    g = set(gold)
    return 1.0 if g and any(x in g for x in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], gold: Iterable[str], k: int) -> float:
    g = set(gold)
    if not g:
        return 0.0
    return len(g & set(ranked[:k])) / len(g)


def mrr(ranked: list[str], gold: Iterable[str]) -> float:
    g = set(gold)
    for i, x in enumerate(ranked, start=1):
        if x in g:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: Iterable[str], k: int) -> float:
    g = set(gold)
    if not g:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, x in enumerate(ranked[:k], start=1) if x in g)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(g), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
