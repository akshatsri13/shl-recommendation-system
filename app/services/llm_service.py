"""
app/services/llm_service.py

Groq LLM wrapper service.

Provides a clean interface around the Groq SDK with:
- Retry logic with exponential backoff (via tenacity).
- Timeout enforcement.
- JSON mode parsing.
- Structured logging of token usage.

All other services depend on this — it is the single integration point
with the external LLM provider.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from groq import Groq, APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.config import Settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Groq LLM client wrapper.

    Encapsulates all communication with the Groq API so that the rest
    of the application never imports groq directly.

    Attributes:
        model: The Groq model name (e.g. 'llama-3.3-70b-versatile').
        temperature: Sampling temperature (low = more deterministic).
        max_tokens: Maximum tokens in the LLM response.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        logger.info("LLMService initialised with model: %s", self.model)

    # ── Public API ────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        """
        Send a prompt to the Groq LLM and return the text response.

        Args:
            system_prompt: Instructions / persona for the LLM.
            user_prompt: The current user-facing input (may include context).
            json_mode: If True, instructs Groq to return valid JSON only.

        Returns:
            The LLM's text response as a string.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        return self._call_with_retry(system_prompt, user_prompt, json_mode)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Send a prompt and parse the response as JSON.

        Args:
            system_prompt: Instructions / persona.
            user_prompt: User-facing input with context.

        Returns:
            Parsed JSON dict.

        Raises:
            ValueError: If the response cannot be parsed as JSON.
            RuntimeError: If all retries are exhausted.
        """
        raw = self.generate(system_prompt, user_prompt, json_mode=True)
        return self._parse_json(raw)

    # ── Internal ──────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool,
    ) -> str:
        """Execute the Groq API call with retry on transient errors."""
        t_start = time.monotonic()

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**kwargs)
        except APIError as exc:
            logger.error("Groq API error: %s", exc)
            raise

        elapsed = time.monotonic() - t_start
        content = response.choices[0].message.content or ""

        # Log token usage for observability
        usage = response.usage
        if usage:
            logger.info(
                "LLM call completed in %.2fs | prompt=%d tokens | "
                "completion=%d tokens | total=%d tokens",
                elapsed,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        else:
            logger.info("LLM call completed in %.2fs", elapsed)

        return content.strip()

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """
        Parse a JSON string, stripping markdown fences if present.

        Args:
            raw: Raw LLM output string.

        Returns:
            Parsed dict.

        Raises:
            ValueError: On JSON parse failure.
        """
        # Strip common markdown code fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove first and last fence lines
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse LLM JSON response. Raw output:\n%s\nError: %s",
                raw,
                exc,
            )
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc
