"""
app/services/clarification_service.py

Clarification Engine.

When the classifier determines there is not enough context to make
a recommendation (CLARIFY or UNKNOWN intent), this service generates
ONE targeted clarification question using the LLM.

The question is informed by what information is already known
(from ConversationState) and what is still missing.
"""

import logging

from app.models.response_models import ChatResponse
from app.prompts.clarification_prompt import (
    CLARIFICATION_SYSTEM_PROMPT,
    CLARIFICATION_USER_TEMPLATE,
)
from app.services.conversation_service import ConversationState
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

# Fallback clarification when LLM fails
_FALLBACK_QUESTION = (
    "To help me find the right SHL assessments for you, could you tell me "
    "what role you're hiring for?"
)


class ClarificationService:
    """
    Generates a single targeted clarification question.

    Uses the LLM to produce a context-aware question based on what
    information is already known vs. still missing.
    """

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    def clarify(
        self,
        state: ConversationState,
        conversation_history: str,
    ) -> ChatResponse:
        """
        Generate one clarification question given the current state.

        Args:
            state: Reconstructed conversation state.
            conversation_history: Formatted conversation string.

        Returns:
            ChatResponse with the clarification question as reply.
            recommendations is always empty for CLARIFY responses.
        """
        user_prompt = CLARIFICATION_USER_TEMPLATE.format(
            conversation_history=conversation_history,
            role=state.role or "Unknown",
            seniority=state.seniority or "Unknown",
            skills=", ".join(state.skills) or "None specified",
            test_types=", ".join(state.test_types_needed) or "None specified",
            industry=state.industry or "Unknown",
            constraints=", ".join(state.constraints) or "None",
        )

        try:
            question = self._llm.generate(
                system_prompt=CLARIFICATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_mode=False,
            )
            # Clean up any accidental JSON wrapping
            question = question.strip().strip('"').strip()
        except Exception as exc:
            logger.warning("LLM clarification failed: %s. Using fallback.", exc)
            question = _FALLBACK_QUESTION

        logger.info("Clarification question: %s", question[:80])

        return ChatResponse(
            reply=question,
            recommendations=[],
            end_of_conversation=False,
        )
