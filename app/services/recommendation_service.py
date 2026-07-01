"""
app/services/recommendation_service.py

Recommendation Engine.

Combines retrieved assessments with hiring context and calls the LLM
to produce a ranked, grounded list of SHL assessment recommendations.

Anti-hallucination guarantee: all assessment data passed to the LLM
comes from ChromaDB-retrieved documents, never from model memory.
"""

import json
import logging
from typing import Any, Dict, List

from app.models.response_models import ChatResponse, Recommendation
from app.prompts.recommendation_prompt import (
    RECOMMENDATION_SYSTEM_PROMPT,
    RECOMMENDATION_USER_TEMPLATE,
)
from app.services.conversation_service import ConversationState
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

# URL whitelist for post-generation validation
_VALID_URL_PREFIXES = ("https://www.shl.com/", "https://shl.com/")


class RecommendationService:
    """
    Generates ranked SHL assessment recommendations using LLM + RAG.

    The service sends retrieved assessment context to the LLM and asks
    it to select and rank the most relevant ones. It then validates
    that all returned URLs exist in the retrieved set.
    """

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    def recommend(
        self,
        state: ConversationState,
        retrieved_docs: List[Dict[str, Any]],
        conversation_history: str,
    ) -> ChatResponse:
        """
        Produce a ranked list of recommendations from retrieved documents.

        Args:
            state: Inferred hiring context.
            retrieved_docs: Top-K candidates from ChromaDB.
            conversation_history: Formatted conversation string for context.

        Returns:
            ChatResponse with reply text and ordered recommendations.
        """
        if not retrieved_docs:
            logger.warning("No retrieved documents — returning empty recommendation.")
            return ChatResponse(
                reply=(
                    "I wasn't able to find any assessments matching your requirements "
                    "in the SHL catalog. Could you provide more details about the role "
                    "or skills you're assessing for?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # Build a reference set of valid URLs for post-validation
        valid_urls: Dict[str, Dict[str, Any]] = {
            doc["url"]: doc for doc in retrieved_docs
        }

        # Format retrieved docs for the prompt
        formatted_docs = self._format_docs(retrieved_docs)

        user_prompt = RECOMMENDATION_USER_TEMPLATE.format(
            role=state.role or "Not specified",
            seniority=state.seniority or "Not specified",
            skills=", ".join(state.skills) or "Not specified",
            test_types=", ".join(state.test_types_needed) or "Not specified",
            industry=state.industry or "Not specified",
            constraints=", ".join(state.constraints) or "None",
            conversation_history=conversation_history,
            retrieved_assessments=formatted_docs,
        )

        try:
            result = self._llm.generate_json(
                system_prompt=RECOMMENDATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except ValueError as exc:
            logger.error("LLM JSON parse error in recommendation: %s", exc)
            return self._fallback_response(retrieved_docs)

        reply = str(result.get("reply", "Here are my recommendations based on your requirements."))
        raw_recs = result.get("recommendations", [])

        # Validate and build Recommendation objects
        recommendations = self._validate_recommendations(raw_recs, valid_urls, retrieved_docs)

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=bool(result.get("end_of_conversation", False)),
        )

    # ── Private Helpers ───────────────────────────────────────

    @staticmethod
    def _format_docs(docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved docs as a numbered list for prompt injection.

        Args:
            docs: Retrieved assessment dicts from ChromaDB.

        Returns:
            Formatted string with all assessment details.
        """
        lines = []
        for i, doc in enumerate(docs, start=1):
            lines.append(
                f"{i}. Name: {doc['name']}\n"
                f"   URL: {doc['url']}\n"
                f"   Test Type: {doc['test_type']}\n"
                f"   Description: {doc['description']}\n"
                f"   Job Levels: {', '.join(doc['job_levels'])}\n"
                f"   Duration: {doc['duration']}\n"
                f"   Remote: {'Yes' if doc['remote_testing'] else 'No'}\n"
                f"   Adaptive: {'Yes' if doc['adaptive'] else 'No'}"
            )
        return "\n\n".join(lines)

    def _validate_recommendations(
        self,
        raw_recs: list,
        valid_urls: Dict[str, Dict[str, Any]],
        retrieved_docs: List[Dict[str, Any]],
    ) -> List[Recommendation]:
        """
        Validate LLM-returned recommendations against the retrieved document set.

        - Rejects any URLs not in the retrieved set (anti-hallucination).
        - Falls back to top retrieved docs if LLM returns nothing valid.
        """
        validated: List[Recommendation] = []

        for rec in raw_recs:
            if not isinstance(rec, dict):
                continue

            url = rec.get("url", "").strip()
            name = rec.get("name", "").strip()
            test_type = rec.get("test_type", "").strip()

            if url in valid_urls:
                # URL is from our retrieved set — safe to use
                validated.append(
                    Recommendation(name=name, url=url, test_type=test_type)
                )
            else:
                logger.warning(
                    "LLM returned URL not in retrieved set (possible hallucination): %s",
                    url,
                )

        if not validated:
            logger.warning(
                "No valid recommendations after validation — using top retrieved docs."
            )
            return self._fallback_recommendations(retrieved_docs)

        return validated

    @staticmethod
    def _fallback_recommendations(docs: List[Dict[str, Any]]) -> List[Recommendation]:
        """Return top-3 retrieved docs as fallback recommendations."""
        return [
            Recommendation(
                name=doc["name"],
                url=doc["url"],
                test_type=doc["test_type"],
            )
            for doc in docs[:3]
        ]

    @staticmethod
    def _fallback_response(docs: List[Dict[str, Any]]) -> ChatResponse:
        """Return a safe fallback response when LLM fails."""
        recs = [
            Recommendation(name=d["name"], url=d["url"], test_type=d["test_type"])
            for d in docs[:3]
        ]
        return ChatResponse(
            reply=(
                "Based on your requirements, here are the most relevant "
                "SHL assessments I found:"
            ),
            recommendations=recs,
            end_of_conversation=False,
        )
