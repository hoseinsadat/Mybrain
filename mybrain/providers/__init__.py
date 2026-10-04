"""Pluggable vendors. Every stage (embed, rerank, LLM, parse) is an interface with several implementations."""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from .base import Embedder, Reranker


def make_embedder(provider: str | None = None, model: str | None = None, dims: int | None = None) -> Embedder:
    from .embeddings import GeminiEmbedder, OpenAICompatEmbedder, VoyageEmbedder

    s = get_settings()
    provider = provider or s.embed_provider
    dims = dims or s.embed_dims
    if provider == "gemini":
        return GeminiEmbedder(model=model or s.embed_model, dims=dims, api_key=s.gemini_api_key)
    if provider == "voyage":
        return VoyageEmbedder(model=model or s.embed_model, dims=dims, api_key=s.voyage_api_key)
    if provider == "openai_compat":
        return OpenAICompatEmbedder(
            model=model or s.openai_compat_embed_model,
            dims=dims,
            api_key=s.openai_compat_api_key,
            base_url=s.openai_compat_base_url,
        )
    raise ValueError(f"unknown embedding provider {provider}")


@lru_cache
def get_embedder() -> Embedder:
    return make_embedder()


def make_reranker(provider: str | None = None, model: str | None = None) -> Reranker:
    from .rerank import CohereReranker, NoReranker, VoyageReranker, ZeroEntropyReranker

    s = get_settings()
    provider = provider or s.rerank_provider
    if provider == "none":
        return NoReranker()
    if provider == "zeroentropy":
        return ZeroEntropyReranker(model=model or s.rerank_model, api_key=s.zeroentropy_api_key)
    if provider == "cohere":
        return CohereReranker(model=model or "rerank-v4.0-pro", api_key=s.cohere_api_key)
    if provider == "voyage":
        return VoyageReranker(model=model or "rerank-2.5", api_key=s.voyage_api_key)
    raise ValueError(f"unknown rerank provider {provider}")


@lru_cache
def get_reranker() -> Reranker:
    return make_reranker()


@lru_cache
def get_extract_llm():
    """Cheap, high-volume model: chunk context, entity extraction, query planning."""
    from .llm import AnthropicLLM, GeminiLLM

    s = get_settings()
    if s.extract_provider == "gemini":
        return GeminiLLM(model=s.extract_model, api_key=s.gemini_api_key)
    return AnthropicLLM(model=s.extract_anthropic_model, api_key=s.anthropic_api_key)


@lru_cache
def get_answer_llm():
    """Final cited answer."""
    from .llm import AnthropicLLM, GeminiLLM

    s = get_settings()
    if s.answer_provider == "anthropic":
        return AnthropicLLM(model=s.answer_model, api_key=s.anthropic_api_key, effort=s.answer_effort)
    return GeminiLLM(model=s.answer_model, api_key=s.gemini_api_key)
