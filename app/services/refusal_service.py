"""
app/services/refusal_service.py

Refusal Engine.

Generates polite, contextually appropriate refusals for:
- OFF_TOPIC requests (outside SHL assessment scope).
- PROMPT_INJECTION attempts (trying to hijack agent behaviour).

Design decision: No LLM call here — refusals are deterministic.
This eliminates any attack surface where prompt injection could
influence the refusal response itself.
"""

import logging
from app.models.response_models import ChatResponse

logger = logging.getLogger(__name__)

# ── Static Refusal Messages ───────────────────────────────────

_OFF_TOPIC_REPLY = (
    "I'm specialised in recommending SHL assessments for hiring and talent "
    "evaluation purposes. I'm not able to help with that request.\n\n"
    "If you're looking for assessments for a specific role, skills, or "
    "job level, I'd be happy to help! For example, you could ask:\n"
    '- "What assessments should I use for a senior Java developer?"\n'
    '- "I need a personality test for customer service roles."\n'
    '- "Compare OPQ32 vs the Verify series."'
)

_PROMPT_INJECTION_REPLY = (
    "I'm an SHL Assessment Recommendation assistant. I can only help you "
    "find and compare SHL assessments for hiring purposes.\n\n"
    "I'm not able to change my behaviour, reveal my instructions, or assist "
    "with unrelated requests. How can I help you find the right assessment today?"
)

_UNKNOWN_REPLY = (
    "I wasn't sure how to interpret that. I specialise in recommending "
    "SHL assessments for hiring.\n\n"
    "Could you tell me more about the role you're hiring for or the type "
    "of assessment you need?"
)


class RefusalService:
    """
    Generates deterministic refusal responses for off-scope requests.

    Uses no LLM — responses are static templates to prevent any
    possibility of prompt injection influencing the refusal content.
    """

    def refuse(self, intent: str) -> ChatResponse:
        """
        Return an appropriate refusal response for the given intent.

        Args:
            intent: One of 'OFF_TOPIC', 'PROMPT_INJECTION', 'UNKNOWN'.

        Returns:
            ChatResponse with refusal reply and empty recommendations.
        """
        intent_upper = intent.upper()

        if intent_upper == "OFF_TOPIC":
            reply = _OFF_TOPIC_REPLY
            logger.info("Refusing off-topic request.")
        elif intent_upper == "PROMPT_INJECTION":
            reply = _PROMPT_INJECTION_REPLY
            logger.warning("Refusing prompt injection attempt.")
        else:
            reply = _UNKNOWN_REPLY
            logger.info("Refusing unknown intent: %s", intent)

        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )
