"""Thin wrapper around supported LLM providers that enforces JSON outputs."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

try:  # pragma: no cover - optional dependencies
    import orjson
except Exception:  # pragma: no cover
    orjson = None  # type: ignore

try:  # pragma: no cover
    import openai
except Exception:  # pragma: no cover
    openai = None  # type: ignore

try:  # pragma: no cover
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

logger = logging.getLogger(__name__)


class LLMClient:
    """Simple JSON-enforcing client for OpenAI or Anthropic models."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self._api_key = api_key

        if self.provider == "openai":
            if openai is None:
                raise RuntimeError("openai package not installed")
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OpenAI API key not configured")
            openai.api_key = key
            self._anthropic_client = None
        elif self.provider == "anthropic":
            if anthropic is None:
                raise RuntimeError("anthropic package not installed")
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("Anthropic API key not configured")
            self._anthropic_client = anthropic.Anthropic(api_key=key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def complete_json(self, prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload_json = (
            orjson.dumps(payload).decode("utf-8") if orjson else json.dumps(payload, ensure_ascii=False)
        )
        formatted_prompt = prompt.format(payload=payload_json)

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_response = self._invoke_model(formatted_prompt)
                if not raw_response:
                    raise ValueError("Empty response from LLM")
                if not isinstance(raw_response, str):
                    raw_response = str(raw_response)
                raw_response = raw_response.strip()
                return json.loads(raw_response)
            except json.JSONDecodeError as exc:
                logger.error("LLM returned invalid JSON (attempt %s/%s): %s", attempt, self.max_retries, exc)
                time.sleep(min(2 ** attempt, 10))
            except Exception as exc:  # pragma: no cover - depends on provider
                logger.error("LLM call failed (attempt %s/%s): %s", attempt, self.max_retries, exc)
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError("Unable to obtain valid JSON from LLM after retries")

    # Internal helpers -------------------------------------------------

    def _invoke_model(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._invoke_openai(prompt)
        if self.provider == "anthropic":
            return self._invoke_anthropic(prompt)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _invoke_openai(self, prompt: str) -> str:
        if openai is None:
            raise RuntimeError("openai package not installed")
        completion = openai.ChatCompletion.create(  # type: ignore[call-arg]
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Respond strictly in JSON with double quotes and UTF-8 encoding.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            timeout=self.timeout,
        )
        return completion.choices[0].message["content"]  # type: ignore[index]

    def _invoke_anthropic(self, prompt: str) -> str:
        if anthropic is None or not getattr(self, "_anthropic_client", None):
            raise RuntimeError("anthropic package not installed or client not configured")
        client = self._anthropic_client
        response = client.messages.create(  # type: ignore[call-arg]
            model=self.model,
            max_tokens=1500,
            temperature=self.temperature,
            system="Respond strictly in JSON with double quotes and UTF-8 encoding.",
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns a list of content blocks
        content_block = response.content[0]  # type: ignore[index]
        return content_block.text  # type: ignore[attr-defined]


def _read_streamlit_secrets() -> Dict[str, Any]:
    try:  # pragma: no cover - Streamlit not always available in tests
        import streamlit as st  # type: ignore

        secrets = getattr(st, "secrets", {})
        if not isinstance(secrets, dict):
            return {}
        return {key: secrets[key] for key in secrets}
    except Exception:
        return {}


def _resolve_llm_settings() -> tuple[str | None, str | None, str | None]:
    secrets = _read_streamlit_secrets()
    llm_section = secrets.get("llm", {}) if isinstance(secrets.get("llm"), dict) else {}

    provider = (llm_section.get("provider") or secrets.get("llm_provider") or os.getenv("LLM_PROVIDER"))
    model = llm_section.get("model") or secrets.get("llm_model") or os.getenv("LLM_MODEL")

    provider_lower = (provider or "").lower()
    api_key = (
        llm_section.get("api_key")
        or (
            llm_section.get("openai_api_key")
            if provider_lower == "openai"
            else llm_section.get("anthropic_api_key")
        )
    )

    if not api_key and provider_lower == "openai":
        api_key = secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    elif not api_key and provider_lower == "anthropic":
        api_key = secrets.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        api_key = os.getenv("LLM_API_KEY")

    return provider, model, api_key


DEFAULT_MODELS = {
    "anthropic": "claude-3-haiku-20240307",
    "openai": "gpt-4o-mini",
}


def load_llm_client() -> LLMClient | None:
    """Instantiate an :class:`LLMClient` from secrets or environment variables."""

    provider, model, api_key = _resolve_llm_settings()
    provider = (provider or "anthropic").lower()
    model = model or DEFAULT_MODELS.get(provider)

    if not api_key:
        logger.info("No API key configured for %s provider; falling back to template responses.", provider)
        return None

    if not model:
        logger.info("No model configured for %s provider; falling back to template responses.", provider)
        return None

    try:
        return LLMClient(provider=provider, model=model, api_key=api_key)
    except Exception as exc:  # pragma: no cover - configuration errors
        logger.error("Failed to initialise LLM client: %s", exc)
        return None


__all__ = ["LLMClient", "load_llm_client"]
