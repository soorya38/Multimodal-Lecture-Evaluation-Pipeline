"""
Central application configuration.

All runtime configuration is declared here as a single typed ``Settings`` object
loaded from environment variables (and an optional ``.env`` file). Modules should
call :func:`get_settings` at *runtime* rather than reading ``os.getenv`` at import
time — this keeps configuration in one place, validated once, and avoids the
import-order pitfalls of module-level ``os.getenv`` globals.

Backward compatibility: the historical environment variable names (e.g.
``OLLAMA_EVAL_MODEL``, ``WHISPER_MODEL_SIZE``) are all still accepted via
``AliasChoices`` so existing deployments keep working unchanged.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, populated from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- General ---------------------------------------------------------
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    port: int = Field(default=8000, validation_alias="PORT")

    # --- MinIO object storage -------------------------------------------
    minio_endpoint: str = Field(default="localhost:9000", validation_alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", validation_alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", validation_alias="MINIO_SECRET_KEY")
    minio_secure: bool = Field(default=False, validation_alias="MINIO_SECURE")
    minio_default_bucket: str = Field(default="lectures", validation_alias="MINIO_DEFAULT_BUCKET")

    # --- Whisper transcription ------------------------------------------
    whisper_device: str = Field(default="cuda", validation_alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="float16", validation_alias="WHISPER_COMPUTE_TYPE")
    whisper_model_size: str = Field(default="large-v3", validation_alias="WHISPER_MODEL_SIZE")
    whisper_cpu_threads: int = Field(default=0, validation_alias="WHISPER_CPU_THREADS")

    # --- LLM provider (OCR + evaluation) --------------------------------
    # "ollama"  -> local Ollama server (default, fully self-hosted)
    # "openai"  -> any OpenAI-compatible chat/completions endpoint
    #              (OpenAI, vLLM, Together, LM Studio, Ollama's /v1, etc.)
    llm_provider: Literal["ollama", "openai"] = Field(
        default="ollama", validation_alias="LLM_PROVIDER"
    )

    # Ollama transport
    ollama_host: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_HOST")

    # OpenAI-compatible transport
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", validation_alias="OPENAI_BASE_URL"
    )
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    # Models. New canonical names are LLM_EVAL_MODEL / LLM_OCR_MODEL; the old
    # OLLAMA_* names are still accepted for backward compatibility.
    eval_model: str = Field(
        default="llama3.2",
        validation_alias=AliasChoices("LLM_EVAL_MODEL", "OLLAMA_EVAL_MODEL"),
    )
    ocr_model: str = Field(
        default="llava-phi3",
        validation_alias=AliasChoices("LLM_OCR_MODEL", "OLLAMA_OCR_MODEL"),
    )

    # Sampling / determinism
    eval_seed: int = Field(default=42, validation_alias=AliasChoices("LLM_EVAL_SEED", "OLLAMA_EVAL_SEED"))
    eval_temperature: float = Field(default=0.1, validation_alias="LLM_EVAL_TEMPERATURE")
    ocr_temperature: float = Field(default=0.0, validation_alias="LLM_OCR_TEMPERATURE")

    # Concurrency + resilience
    ocr_max_workers: int = Field(default=0, validation_alias="OCR_MAX_WORKERS")
    llm_timeout_seconds: float = Field(default=120.0, validation_alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=2, validation_alias="LLM_MAX_RETRIES")

    # --- Derived helpers -------------------------------------------------
    def resolved_cpu_threads(self) -> int:
        """CPU thread count for Whisper — 0 means 'use all available cores'."""
        import os

        return self.whisper_cpu_threads or (os.cpu_count() or 4)

    def resolved_ocr_workers(self) -> int:
        """OCR concurrency — 0 means 'scale with CPUs, capped at 8'."""
        import os

        return self.ocr_max_workers or min(os.cpu_count() or 4, 8)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide, cached ``Settings`` instance."""
    return Settings()
