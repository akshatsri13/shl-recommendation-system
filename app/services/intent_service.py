"""
app/services/intent_service.py

Intent Classifier Service.

Classifies the user's latest message into one of the supported intents
using the LLM with a structured JSON response.

Supported intents:
  CLARIFY | RECOMMEND | REFINE | COMPARE | OFF_TOPIC | PROMPT_INJECTION | UNKNOWN
"""

import logging
from dataclasses import dataclass
from typing import List

from app.models.request_models import Message
from app.prompts.classifier_prompt import (
    CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFIER_USER_TEMPLATE,
)
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

VALID_INTENTS = {
    "CLARIFY",
    "RECOMMEND",
    "REFINE",
    "COMPARE",
    "OFF_TOPIC",
    "PROMPT_INJECTION",
    "UNKNOWN",
}


@dataclass
class ClassificationResult:
    """Result of an intent classification call."""

    intent: str
    confidence: float
    assessment_names: List[str]  # populated for COMPARE intent


class IntentService:
    """
    Classifies user intent from the full conversation history.

    Uses the LLM with a strict JSON-only system prompt to ensure
    structured, reliable classification results.
    """

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    def classify(self, messages: List[Message]) -> ClassificationResult:
        """
        Classify the intent of the last user message in the conversation.

        Args:
            messages: Full conversation history.

        Returns:
            ClassificationResult with intent, confidence, and any named assessments.
        """
        conversation_text = self._format_messages(messages)

        user_prompt = CLASSIFIER_USER_TEMPLATE.format(
            conversation_history=conversation_text
        )

        try:
            result = self._llm.generate_json(
                system_prompt=CLASSIFIER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except (ValueError, Exception) as exc:
            logger.warning(
                "Intent classification failed: %s. Defaulting to CLARIFY.", exc
            )
            return ClassificationResult(
                intent="CLARIFY", confidence=0.5, assessment_names=[]
            )

        intent = str(result.get("intent", "UNKNOWN")).upper().strip()
        if intent not in VALID_INTENTS:
            logger.warning(
                "LLM returned invalid intent '%s'. Defaulting to UNKNOWN.", intent
            )
            intent = "UNKNOWN"

        confidence = float(result.get("confidence", 0.5))
        assessment_names = list(result.get("assessment_names", []))

        logger.info(
            "Intent classified: %s (confidence=%.2f, assessments=%s)",
            intent,
            confidence,
            assessment_names,
        )

        return ClassificationResult(
            intent=intent,
            confidence=confidence,
            assessment_names=assessment_names,
        )

    @staticmethod
    def _format_messages(messages: List[Message]) -> str:
        """Format messages as a readable dialogue string."""
        lines = []
        for msg in messages:
            label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{label}: {msg.content}")
        return "\n".join(lines)
