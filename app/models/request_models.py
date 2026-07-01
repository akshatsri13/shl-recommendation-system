"""
app/models/request_models.py

Pydantic request models for the /chat endpoint.
Schema is contractual — must not change without versioning.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single turn in the conversation."""

    role: Literal["user", "assistant"] = Field(
        ...,
        description="Who sent this message — 'user' or 'assistant'.",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The text content of the message.",
    )


class ChatRequest(BaseModel):
    """
    Payload for POST /chat.

    The caller sends their latest message and an optional session_id.
    """

    session_id: Optional[str] = Field(
        None,
        description="Optional session ID. If not provided, a new session is created.",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The latest user message.",
    )
