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
        embedder: str = Field(default="auto", description="auto | hashing | sentence-transformers")
        embedder_dimension: int = Field(default=512)
        store: str = Field(default="numpy", description="numpy | chroma")
        chunk_size: int = Field(default=180)
        chunk_overlap: int = Field(default=40)

        # Orchestration
        strategy: str = Field(default="linear", description="linear | blackboard")
        max_rounds: int = Field(default=3, description="Max critic/revision rounds")

        # Persistence
        storage_dir: str = Field(default="storage")

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
        strategy: str = os.environ.get("OARAG_STRATEGY", "linear")
        max_rounds: int = int(os.environ.get("OARAG_MAX_ROUNDS", "3"))
        storage_dir: str = os.environ.get("OARAG_STORAGE_DIR", "storage")
        anthropic_api_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
        hf_token: Optional[str] = os.environ.get("HF_TOKEN")


def load_settings() -> "Settings":
    """Load settings from the environment / .env file."""
    return Settings()


__all__ = ["Settings", "load_settings"]
