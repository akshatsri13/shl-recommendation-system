"""
app/services/conversation_service.py

Stateless conversation state reconstruction.

Every API request includes the full message history. This service
analyses that history using the LLM to extract a structured
ConversationState object — the inferred hiring context.

Design principles:
- Pure function: same messages → same state (no side effects).
- LLM-backed: robust to varied phrasing, implicit context, multi-turn.
- Fail-safe: returns empty/default state on any error.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from app.models.request_models import Message
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Domain Model
# ──────────────────────────────────────────────────────────────


@dataclass
class ConversationState:
    """
    Inferred hiring context extracted from conversation history.

    All fields default to empty/None — populated as the conversation
    provides more information. The recommendation engine uses these
    fields to build a targeted search query.
    """

    role: Optional[str] = None
    seniority: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    test_types_needed: List[str] = field(default_factory=list)
    industry: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    named_assessments: List[str] = field(default_factory=list)  # for COMPARE
    has_prior_recommendations: bool = False

    def has_enough_context(self) -> bool:
        """
        Return True if enough context exists to attempt a recommendation.

        Minimum requirement: a role OR at least one skill/test type.
        """
        return bool(self.role or self.skills or self.test_types_needed)

    def to_search_query(self) -> str:
        """
        Build a rich natural-language query string for semantic search.

        Combines all available context into a single descriptive sentence
        that captures the hiring need for embedding-based retrieval.
        """
        parts: List[str] = []

        if self.role:
            parts.append(f"assessment for {self.role}")
        if self.seniority:
            parts.append(f"{self.seniority} level")
        if self.skills:
            parts.append(f"skills: {', '.join(self.skills)}")
        if self.test_types_needed:
            parts.append(f"test types: {', '.join(self.test_types_needed)}")
        if self.industry:
            parts.append(f"industry: {self.industry}")
        if self.constraints:
            parts.append(f"constraints: {', '.join(self.constraints)}")

        return " | ".join(parts) if parts else "SHL individual test assessment"

    def describe(self) -> str:
        """Human-readable summary of current state (for prompt injection)."""
        lines = [
            f"Role: {self.role or 'Unknown'}",
            f"Seniority: {self.seniority or 'Unknown'}",
            f"Skills: {', '.join(self.skills) or 'None specified'}",
            f"Test types: {', '.join(self.test_types_needed) or 'None specified'}",
            f"Industry: {self.industry or 'Unknown'}",
            f"Constraints: {', '.join(self.constraints) or 'None'}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# System Prompt for State Extraction
# ──────────────────────────────────────────────────────────────

_STATE_EXTRACTION_PROMPT = """You are a conversation analyser for an SHL Assessment Recommendation system.

Analyse the provided conversation history and extract the hiring context.

Return ONLY valid JSON. No markdown, no explanation.

## Fields to Extract

- role: The job role being hired for (string or null)
- seniority: Job level — one of: Entry-Level, Graduate, Mid-Professional, Professional Individual Contributor, Manager, Front Line Manager, Director, Executive, General Population (string or null)
- skills: List of technical/domain skills mentioned (list of strings)
- test_types_needed: Types of assessments needed — from: Knowledge & Skills, Personality & Behavior, Ability & Aptitude, Simulations, Competencies, Development & 360, Biodata & Situational Judgment (list of strings)
- industry: Industry or domain context if mentioned (string or null)
- constraints: Any constraints mentioned — language, duration, remote testing, adaptive (list of strings)
- named_assessments: Specific SHL assessment names mentioned by the user (list of strings)
- has_prior_recommendations: True if the assistant has already provided recommendations in this conversation (boolean)

## Example Output

{
  "role": "Software Engineer",
  "seniority": "Mid-Professional",
  "skills": ["Java", "Python", "REST APIs"],
  "test_types_needed": ["Knowledge & Skills", "Personality & Behavior"],
  "industry": "Technology",
  "constraints": ["English only", "under 30 minutes"],
  "named_assessments": [],
  "has_prior_recommendations": false
}
"""


# ──────────────────────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────────────────────


class ConversationService:
    """
    Reconstructs ConversationState from a list of messages.

    Stateless — called fresh on every API request.
    Uses the LLM to extract structured context from free-form conversation.
    """

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    def reconstruct_state(self, messages: List[Message]) -> ConversationState:
        """
        Analyse the full conversation history and return a ConversationState.

        Args:
            messages: Ordered list of user/assistant messages.

        Returns:
            A populated ConversationState (fields default to None/empty
            if not found in conversation).
        """
        if not messages:
            return ConversationState()

        conversation_text = self._format_conversation(messages)

        user_prompt = f"""## Conversation History

{conversation_text}

Extract the hiring context. Return only JSON."""

        try:
            result = self._llm.generate_json(
                system_prompt=_STATE_EXTRACTION_PROMPT,
                user_prompt=user_prompt,
            )
            return self._parse_state(result)
        except (ValueError, Exception) as exc:
            logger.warning(
                "Failed to extract conversation state: %s. Returning default state.",
                exc,
            )
            return ConversationState()

    def format_for_prompt(self, messages: List[Message]) -> str:
        """
        Format messages as a human-readable dialogue for inclusion in prompts.

        Args:
            messages: Full conversation history.

        Returns:
            Formatted string with role labels and message content.
        """
        return self._format_conversation(messages)

    # ── Private ───────────────────────────────────────────────

    @staticmethod
    def _format_conversation(messages: List[Message]) -> str:
        """Format messages list as a readable dialogue string."""
        lines = []
        for msg in messages:
            label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{label}: {msg.content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_state(data: dict) -> ConversationState:
        """
        Convert parsed JSON dict into a ConversationState dataclass.

        Validates types and applies sensible defaults.
        """
        def _list(val) -> list:
            if isinstance(val, list):
                return [str(v) for v in val if v]
            return []

        def _str_or_none(val) -> Optional[str]:
            if isinstance(val, str) and val.strip():
                return val.strip()
            return None

        return ConversationState(
            role=_str_or_none(data.get("role")),
            seniority=_str_or_none(data.get("seniority")),
            skills=_list(data.get("skills")),
            test_types_needed=_list(data.get("test_types_needed")),
            industry=_str_or_none(data.get("industry")),
            constraints=_list(data.get("constraints")),
            named_assessments=_list(data.get("named_assessments")),
            has_prior_recommendations=bool(data.get("has_prior_recommendations", False)),
        )
