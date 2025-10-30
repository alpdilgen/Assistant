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
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

        if self.provider == "openai" and openai is not None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
        if self.provider == "anthropic" and anthropic is not None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self._anthropic_client = anthropic.Anthropic(api_key=api_key)
            else:  # pragma: no cover - configuration issue
                self._anthropic_client = None
        else:
            self._anthropic_client = None

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


__all__ = ["LLMClient"]
