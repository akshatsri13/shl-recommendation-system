"""
app/services/comparison_service.py

Comparison Engine.

When the user asks to compare two named SHL assessments, this service:
1. Looks up both assessments in ChromaDB by name.
2. Passes the retrieved documents to the LLM for a grounded comparison.
3. Returns a structured ChatResponse with the comparison reply
   and both assessments listed in recommendations.

Anti-hallucination: comparison is generated only from retrieved docs.
"""

import logging
from typing import List, Optional

from app.models.response_models import ChatResponse, Recommendation
from app.prompts.comparison_prompt import (
    COMPARISON_SYSTEM_PROMPT,
    COMPARISON_USER_TEMPLATE,
    COMPARISON_NOT_FOUND_TEMPLATE,
)
from app.services.conversation_service import ConversationState
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class ComparisonService:
    """
    Generates a data-grounded comparison between two SHL assessments.

    Uses the RetrievalService to fetch both assessments from ChromaDB
    and the LLMService to generate the comparison narrative.
    """

    def __init__(
        self,
        llm_service: LLMService,
        retrieval_service: RetrievalService,
    ) -> None:
        self._llm = llm_service
        self._retrieval = retrieval_service

    def compare(
        self,
        assessment_name_a: str,
        assessment_name_b: str,
        state: ConversationState,
    ) -> ChatResponse:
        """
        Compare two named SHL assessments.

        Args:
            assessment_name_a: First assessment name (from user message).
            assessment_name_b: Second assessment name (from user message).
            state: Current conversation state (for hiring context).

        Returns:
            ChatResponse with comparison reply and both assessments.
        """
        doc_a = self._retrieval.retrieve_by_name(assessment_name_a)
        doc_b = self._retrieval.retrieve_by_name(assessment_name_b)

        # Handle not found cases
        not_found = []
        if not doc_a:
            not_found.append(f"'{assessment_name_a}'")
        if not doc_b:
            not_found.append(f"'{assessment_name_b}'")

        if not_found:
            reply = COMPARISON_NOT_FOUND_TEMPLATE.format(
                not_found=" and ".join(not_found)
            )
            logger.warning("Assessment(s) not found for comparison: %s", not_found)
            return ChatResponse(
                reply=reply,
                recommendations=[],
                end_of_conversation=False,
            )

        # Format both documents for the prompt
        formatted_a = self._format_assessment(doc_a)
        formatted_b = self._format_assessment(doc_b)

        user_prompt = COMPARISON_USER_TEMPLATE.format(
            role=state.role or "Not specified",
            seniority=state.seniority or "Not specified",
            skills=", ".join(state.skills) or "Not specified",
            assessment_a=formatted_a,
            assessment_b=formatted_b,
        )

        try:
            result = self._llm.generate_json(
                system_prompt=COMPARISON_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except ValueError as exc:
            logger.error("LLM JSON parse error in comparison: %s", exc)
            return self._fallback_comparison(doc_a, doc_b)

        reply = str(
            result.get(
                "reply",
                f"Here is a comparison of {doc_a['name']} and {doc_b['name']}.",
            )
        )

        # Always include both assessments in recommendations for reference
        recommendations = [
            Recommendation(
                name=doc_a["name"],
                url=doc_a["url"],
                test_type=doc_a["test_type"],
            ),
            Recommendation(
                name=doc_b["name"],
                url=doc_b["url"],
                test_type=doc_b["test_type"],
            ),
        ]

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=False,
        )

    # ── Private ───────────────────────────────────────────────

    @staticmethod
    def _format_assessment(doc: dict) -> str:
        """Format an assessment dict as structured text for the prompt."""
        return (
            f"Name: {doc.get('name', 'Unknown')}\n"
            f"URL: {doc.get('url', 'N/A')}\n"
            f"Test Type: {doc.get('test_type', 'Unknown')}\n"
            f"Description: {doc.get('description', 'Not specified')}\n"
            f"Job Levels: {', '.join(doc.get('job_levels', [])) or 'Not specified'}\n"
            f"Duration: {doc.get('duration', 'Not specified')}\n"
            f"Languages: {', '.join(doc.get('languages', [])) or 'Not specified'}\n"
            f"Remote Testing: {'Yes' if doc.get('remote_testing') else 'No'}\n"
            f"Adaptive Testing: {'Yes' if doc.get('adaptive') else 'No'}\n"
            f"Categories: {', '.join(doc.get('categories', [])) or 'Not specified'}"
        )

    @staticmethod
    def _fallback_comparison(doc_a: dict, doc_b: dict) -> ChatResponse:
        """Return a basic comparison when LLM fails."""
        reply = (
            f"I found both assessments. "
            f"**{doc_a['name']}** is a {doc_a['test_type']} assessment "
            f"({doc_a.get('duration', 'duration unknown')}). "
            f"**{doc_b['name']}** is a {doc_b['test_type']} assessment "
            f"({doc_b.get('duration', 'duration unknown')}). "
            f"Both support remote testing. Please visit the links for full details."
        )
        return ChatResponse(
            reply=reply,
            recommendations=[
                Recommendation(
                    name=doc_a["name"],
                    url=doc_a["url"],
                    test_type=doc_a["test_type"],
                ),
                Recommendation(
                    name=doc_b["name"],
                    url=doc_b["url"],
                    test_type=doc_b["test_type"],
                ),
            ],
            end_of_conversation=False,
        )
