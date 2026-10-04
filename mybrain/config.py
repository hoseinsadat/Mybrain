"""Runtime settings. Every model and vendor is swappable through environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # core
    database_url: str = "postgresql://mybrain:mybrain@localhost:5432/mybrain"
    data_dir: Path = Path("./data")
    app_password: str = ""
    lexical_backend: Literal["auto", "pg_search", "tsvector"] = "auto"

    # embeddings
    embed_provider: Literal["gemini", "voyage", "openai_compat"] = "gemini"
    embed_model: str = "gemini-embedding-2"
    embed_dims: int = 1536
    gemini_api_key: str = ""
    voyage_api_key: str = ""
    openai_compat_base_url: str = "https://api.together.xyz/v1"
    openai_compat_api_key: str = ""
    openai_compat_embed_model: str = "Qwen/Qwen3-Embedding-8B"

    # reranker
    rerank_provider: Literal["zeroentropy", "cohere", "voyage", "none"] = "zeroentropy"
    rerank_model: str = "zerank-2"
    zeroentropy_api_key: str = ""
    cohere_api_key: str = ""
    retrieve_candidates: int = 40
    final_top_k: int = 10

    # llms
    anthropic_api_key: str = ""
    answer_provider: Literal["anthropic", "gemini"] = "anthropic"
    answer_model: str = "claude-opus-5-5"
    answer_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    extract_provider: Literal["gemini", "anthropic"] = "gemini"
    extract_model: str = "gemini-3.8-flash"
    extract_anthropic_model: str = "claude-haiku-4-5"

    # parsing
    pdf_parser: Literal["auto", "llamaparse", "local"] = "auto"
    llamaparse_api_key: str = ""
    llamaparse_tier: str = "agentic"
    llamaparse_version: str = "latest"
    llamaparse_language: str = "en"
    llamaparse_base_url: str = "https://api.cloud.llamaindex.ai"

    # ingestion
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 40
    contextualize: bool = True
    contextualize_max_doc_chars: int = 120_000
    extract_entities: bool = True
    worker_poll_seconds: float = 2.0

    # retrieval
    rrf_k: int = 60
    entity_boost: float = Field(
        default=0.5, description="RRF-scale bonus for chunks mentioning a query entity"
    )
    second_pass: bool = True

    @property
    def uploads_dir(self) -> Path:
        p = self.data_dir / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
