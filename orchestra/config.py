"""Configuration via pydantic-settings.

Every value can be overridden by an environment variable prefixed ``OARAG_``,
or by a ``.env`` file. Secrets (API keys) are read from their conventional env
vars (``ANTHROPIC_API_KEY``, ``HF_TOKEN``) and never hardcoded.
"""

from __future__ import annotations

from typing import Optional

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_PYDANTIC_SETTINGS = True
except ImportError:  # pragma: no cover - fallback for minimal installs
    _HAS_PYDANTIC_SETTINGS = False


if _HAS_PYDANTIC_SETTINGS:

    class Settings(BaseSettings):
        """Runtime configuration for orchestra-rag."""

        model_config = SettingsConfigDict(
            env_prefix="OARAG_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # LLM backend
        backend: str = Field(default="mock", description="mock | anthropic | huggingface")

        # Per-role model ids (used by anthropic/hf backends).
        planner_model: str = Field(default="claude-haiku-4-5")
        researcher_model: str = Field(default="claude-sonnet-4-6")
        coder_model: str = Field(default="claude-opus-4-8")
        critic_model: str = Field(default="claude-sonnet-4-6")
        synthesizer_model: str = Field(default="claude-opus-4-8")

        # Retrieval
        k: int = Field(default=4, description="Number of passages to retrieve")
        embedder: str = Field(
            default="auto", description="auto | hashing | ml | sentence-transformers"
        )
        embedder_dimension: int = Field(default=512)
        store: str = Field(default="numpy", description="numpy | hnsw | chroma | pgvector | qdrant")
        chunk_size: int = Field(default=180)
        chunk_overlap: int = Field(default=40)
        hybrid: bool = Field(default=True, description="Fuse dense + BM25 sparse retrieval")
        rerank: bool = Field(
            default=False, description="Cross-encoder reranking (from-scratch ML or sentence-transformers)"
        )

        # Orchestration
        strategy: str = Field(default="linear", description="linear | blackboard")
        max_rounds: int = Field(default=3, description="Max critic/revision rounds")
        max_cost_usd: float = Field(default=1.0, description="Per-query cost budget (USD)")

        # Security / API
        require_auth: bool = Field(default=False, description="Require X-API-Key on the API")
        cors_origins: str = Field(default="*", description="Comma-separated CORS allowlist")
        rate_limit_per_minute: int = Field(default=60, description="Requests/min per API key")
        max_upload_bytes: int = Field(default=10 * 1024 * 1024, description="Max upload size per file")
        max_upload_files: int = Field(default=20, description="Max files per upload")

        # Persistence
        storage_dir: str = Field(default="storage")
        database_url: str = Field(default="sqlite:///./orchestra.sqlite")

        # Secrets (read from conventional env vars, not OARAG_-prefixed).
        anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
        hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")

else:  # pragma: no cover - exercised only without pydantic-settings

    import os
    from dataclasses import dataclass

    @dataclass
    class Settings:  # type: ignore[no-redef]
        """Minimal env-driven settings fallback when pydantic-settings is absent."""

        backend: str = os.environ.get("OARAG_BACKEND", "mock")
        planner_model: str = os.environ.get("OARAG_PLANNER_MODEL", "claude-haiku-4-5")
        researcher_model: str = os.environ.get("OARAG_RESEARCHER_MODEL", "claude-sonnet-4-6")
        coder_model: str = os.environ.get("OARAG_CODER_MODEL", "claude-opus-4-8")
        critic_model: str = os.environ.get("OARAG_CRITIC_MODEL", "claude-sonnet-4-6")
        synthesizer_model: str = os.environ.get("OARAG_SYNTHESIZER_MODEL", "claude-opus-4-8")
        k: int = int(os.environ.get("OARAG_K", "4"))
        embedder: str = os.environ.get("OARAG_EMBEDDER", "auto")
        embedder_dimension: int = int(os.environ.get("OARAG_EMBEDDER_DIMENSION", "512"))
        store: str = os.environ.get("OARAG_STORE", "numpy")
        chunk_size: int = int(os.environ.get("OARAG_CHUNK_SIZE", "180"))
        chunk_overlap: int = int(os.environ.get("OARAG_CHUNK_OVERLAP", "40"))
        hybrid: bool = os.environ.get("OARAG_HYBRID", "true").lower() in ("1", "true", "yes")
        rerank: bool = os.environ.get("OARAG_RERANK", "false").lower() in ("1", "true", "yes")
        strategy: str = os.environ.get("OARAG_STRATEGY", "linear")
        max_rounds: int = int(os.environ.get("OARAG_MAX_ROUNDS", "3"))
        max_cost_usd: float = float(os.environ.get("OARAG_MAX_COST_USD", "1.0"))
        require_auth: bool = os.environ.get("OARAG_REQUIRE_AUTH", "false").lower() in ("1", "true", "yes")
        cors_origins: str = os.environ.get("OARAG_CORS_ORIGINS", "*")
        rate_limit_per_minute: int = int(os.environ.get("OARAG_RATE_LIMIT_PER_MINUTE", "60"))
        max_upload_bytes: int = int(os.environ.get("OARAG_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
        max_upload_files: int = int(os.environ.get("OARAG_MAX_UPLOAD_FILES", "20"))
        storage_dir: str = os.environ.get("OARAG_STORAGE_DIR", "storage")
        database_url: str = os.environ.get("OARAG_DATABASE_URL", "sqlite:///./orchestra.sqlite")
        anthropic_api_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
        hf_token: Optional[str] = os.environ.get("HF_TOKEN")


def load_settings() -> "Settings":
    """Load settings from the environment / .env file."""
    return Settings()


__all__ = ["Settings", "load_settings"]
