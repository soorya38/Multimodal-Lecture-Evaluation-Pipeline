"""
Pluggable LLM client abstraction.

Both the OCR stage (vision) and the evaluation stage (text scoring) talk to an
LLM through the same small interface, so the concrete provider can be swapped
via configuration (``LLM_PROVIDER=ollama`` or ``LLM_PROVIDER=openai``) without
touching call sites.

Two backends are supported:

* ``ollama``  — the local Ollama server (default, fully self-hosted). Images are
  passed as file paths, which the Ollama SDK base64-encodes for us.
* ``openai``  — any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI,
  vLLM, Together, LM Studio, Ollama's own ``/v1``, …). Images are inlined as
  base64 ``data:`` URLs.

Every provider returns JSON-mode output and raises on failure after exhausting
retries — callers get either a parsed ``dict`` or an exception, never a silent
empty result.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Protocol

import structlog

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class LLMClient(Protocol):
    """Minimal chat interface shared by all providers."""

    def chat_json(
        self,
        prompt: str,
        *,
        model: str,
        images: list[str] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Send a single-user-turn chat request and return the parsed JSON reply."""
        ...


def _coerce_json(content: Any) -> dict[str, Any]:
    """Normalise a model reply (str or dict) into a parsed JSON object."""
    if isinstance(content, dict):
        return content
    text = str(content or "{}").strip()
    if not text:
        return {}
    return json.loads(text)


class _OllamaClient:
    """LLM client backed by a local Ollama server."""

    def __init__(self, settings: Settings) -> None:
        import ollama

        self._settings = settings
        self._client = ollama.Client(host=settings.ollama_host)

    def chat_json(
        self,
        prompt: str,
        *,
        model: str,
        images: list[str] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            message["images"] = images  # Ollama accepts file paths directly

        options: dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed

        response = self._client.chat(
            model=model,
            messages=[message],
            format="json",
            options=options,
        )
        return _coerce_json(response["message"]["content"])


class _OpenAIClient:
    """LLM client backed by any OpenAI-compatible chat/completions endpoint."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._settings = settings
        self._client = OpenAI(
            base_url=settings.openai_base_url,
            # Some self-hosted gateways don't require a key; send a placeholder
            # so the SDK doesn't refuse to construct.
            api_key=settings.openai_api_key or "not-needed",
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # retries handled by the shell in get_llm_client()
        )

    @staticmethod
    def _image_to_data_url(path: str) -> str:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        # JPEG is what the frame extractor writes; data URLs are content-sniffed
        # by most providers regardless, so the mime hint here is a safe default.
        return f"data:image/jpeg;base64,{encoded}"

    def chat_json(
        self,
        prompt: str,
        *,
        model: str,
        images: list[str] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for path in images:
                content.append(
                    {"type": "image_url", "image_url": {"url": self._image_to_data_url(path)}}
                )
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            kwargs["seed"] = seed

        response = self._client.chat.completions.create(**kwargs)
        return _coerce_json(response.choices[0].message.content)


class _RetryingClient:
    """Wrap any provider with bounded exponential-backoff retries + logging."""

    def __init__(self, inner: LLMClient, max_retries: int) -> None:
        self._inner = inner
        self._max_retries = max_retries

    def chat_json(
        self,
        prompt: str,
        *,
        model: str,
        images: list[str] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                return self._inner.chat_json(
                    prompt,
                    model=model,
                    images=images,
                    temperature=temperature,
                    seed=seed,
                )
            except Exception as e:  # noqa: BLE001 — retry all transient failures
                attempt += 1
                if attempt > self._max_retries:
                    logger.error(
                        "LLM call failed after retries",
                        model=model,
                        attempts=attempt,
                        error=str(e),
                    )
                    raise
                backoff = min(2.0 ** (attempt - 1), 8.0)
                logger.warning(
                    "LLM call failed, retrying",
                    model=model,
                    attempt=attempt,
                    max_retries=self._max_retries,
                    backoff_seconds=backoff,
                    error=str(e),
                )
                time.sleep(backoff)


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """
    Build the configured LLM client, wrapped with retry handling.

    A fresh client is returned per call so it is safe to use from worker threads
    (each thread gets its own underlying HTTP client, avoiding shared state).
    """
    settings = settings or get_settings()

    inner: LLMClient
    if settings.llm_provider == "openai":
        inner = _OpenAIClient(settings)
    else:
        inner = _OllamaClient(settings)

    return _RetryingClient(inner, max_retries=settings.llm_max_retries)
