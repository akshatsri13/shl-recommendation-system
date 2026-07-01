"""
app/api/routes.py
FastAPI route definitions for the SHL Assessment Recommender API.
Endpoints:
  GET  /health  — Health check (no auth required).
  POST /chat    — Main conversational endpoint.
All dependencies are injected via FastAPI's Depends() system,
which resolves from the application state set in main.py.
"""
import logging
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Request, status
from app.models.request_models import ChatRequest
from app.models.response_models import ChatResponse
from app.services.agent import AgentOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Dependency: Agent from app state ────────────────────────
def get_agent(request: Request) -> AgentOrchestrator:
    """
    Dependency: retrieve the AgentOrchestrator from app state.
    The agent is initialised once in main.py lifespan and stored
    in app.state.agent. This avoids recreating expensive objects
    (embedding model, ChromaDB client) on every request.
    """
    agent: AgentOrchestrator | None = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialised. Service is starting up.",
        )
    return agent


AgentDep = Annotated[AgentOrchestrator, Depends(get_agent)]


# ── Routes ────────────────────────────────────────────────────
@router.get(
    "/health",
    summary="Health check",
    tags=["Health"],
    response_description="Service health status",
)
async def health_check() -> dict:
    """
    Returns service health status.
    Used by load balancers and monitoring tools to verify the
    service is running and responsive.
    """
    return {"status": "ok"}


@router.post(
    "/chat",
    summary="Conversational SHL assessment recommendation",
    tags=["Chat"],
    response_model=ChatResponse,
    response_description="Agent reply with optional assessment recommendations",
    status_code=status.HTTP_200_OK,
)
async def chat(
    request: ChatRequest,
    agent: AgentDep,
) -> ChatResponse:
    """
    Process a conversational message and return SHL assessment recommendations.

    The API is stateless. The caller sends the FULL conversation history
    on every call; the service stores no per-conversation state.

    **Request body:**
    - `messages`: Full conversation history, oldest message first.

    **Response:**
    - `reply`: The agent's natural-language response.
    - `recommendations`: List of recommended assessments (may be empty).
    - `end_of_conversation`: Whether the conversation is complete.

    **Schema is fixed** — do not expect field additions without versioning.
    """
    last_user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), ""
    )
    logger.info(
        "POST /chat | turns=%d | latest_user_message=%s",
        len(request.messages),
        last_user_msg[:60],
    )
    try:
        response = agent.process(request)
    except Exception as exc:
        logger.exception("Unhandled error in agent.process(): %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request. Please try again.",
        ) from exc

    logger.info(
        "Response | recommendations=%d | end=%s",
        len(response.recommendations),
        response.end_of_conversation,
    )
    return response
