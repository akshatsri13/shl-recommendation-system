"""
tests/conftest.py

Shared pytest fixtures for the SHL Assessment Recommender test suite.

Design:
- Uses a mock LLM service to avoid real Groq API calls in tests.
- Uses a temporary ChromaDB populated with a small test catalog.
- All tests are deterministic and do not require API keys.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models.response_models import ChatResponse, Recommendation
from app.services.agent import AgentOrchestrator
from app.services.clarification_service import ClarificationService
from app.services.comparison_service import ComparisonService
from app.services.conversation_service import ConversationService
from app.services.intent_service import IntentService, ClassificationResult
from app.services.llm_service import LLMService
from app.services.recommendation_service import RecommendationService
from app.services.refusal_service import RefusalService
from app.services.retrieval_service import RetrievalService
from app.services.scraper import Assessment, CatalogLoader
from app.services.session_service import SessionService
from app.vectorstore.chroma import ChromaStore
from app.vectorstore.embedding import EmbeddingService


# ── Test Assessment Catalog ───────────────────────────────────

SAMPLE_ASSESSMENTS = [
    Assessment(
        entity_id="1001",
        name="Java (Advanced Level)",
        url="https://www.shl.com/products/product-catalog/view/java-advanced/",
        description="Advanced Java test covering OOP, concurrency, and JVM.",
        test_type="Knowledge & Skills",
        categories=["Knowledge & Skills"],
        job_levels=["Mid-Professional", "Professional Individual Contributor"],
        languages=["English (USA)"],
        duration="30 minutes",
        remote_testing=True,
        adaptive=True,
    ),
    Assessment(
        entity_id="1002",
        name="OPQ32",
        url="https://www.shl.com/products/product-catalog/view/opq32/",
        description="Personality questionnaire measuring 32 competency-related characteristics.",
        test_type="Personality & Behavior",
        categories=["Personality & Behavior", "Competencies"],
        job_levels=["Manager", "Director", "Mid-Professional"],
        languages=["English (USA)", "Spanish"],
        duration="25 minutes",
        remote_testing=True,
        adaptive=False,
    ),
    Assessment(
        entity_id="1003",
        name="Verify Numerical Reasoning",
        url="https://www.shl.com/products/product-catalog/view/verify-numerical/",
        description="Numerical reasoning ability test for graduate and professional roles.",
        test_type="Ability & Aptitude",
        categories=["Ability & Aptitude"],
        job_levels=["Graduate", "Mid-Professional"],
        languages=["English (USA)"],
        duration="17 minutes",
        remote_testing=True,
        adaptive=True,
    ),
    Assessment(
        entity_id="1004",
        name="Python (Advanced Level)",
        url="https://www.shl.com/products/product-catalog/view/python-advanced/",
        description="Advanced Python programming test covering data structures, algorithms, and frameworks.",
        test_type="Knowledge & Skills",
        categories=["Knowledge & Skills"],
        job_levels=["Mid-Professional", "Professional Individual Contributor"],
        languages=["English (USA)"],
        duration="35 minutes",
        remote_testing=True,
        adaptive=False,
    ),
]


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_settings(tmp_path_factory) -> Settings:
    """Settings pointing to temp directories for isolation."""
    tmp = tmp_path_factory.mktemp("shl_test")
    return Settings(
        groq_api_key="test-key",
        groq_model="llama-3.3-70b-versatile",
        embedding_model="all-MiniLM-L6-v2",
        chroma_db_path=str(tmp / "chroma_db"),
        catalog_path=str(tmp / "catalog.json"),
        port=8002,
    )


@pytest.fixture(scope="session")
def embedding_service() -> EmbeddingService:
    """Real embedding service (uses cached model for speed)."""
    return EmbeddingService(model_name="all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def chroma_store(test_settings, embedding_service) -> ChromaStore:
    """Test ChromaDB populated with sample assessments."""
    store = ChromaStore(
        embedding_service=embedding_service,
        db_path=test_settings.chroma_db_path,
        collection_name="shl_test",
    )
    store.ingest(SAMPLE_ASSESSMENTS)
    return store


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """
    Mock LLMService that returns deterministic responses.

    Each test can override .generate_json.return_value to control
    the LLM output for that specific scenario.
    """
    mock = MagicMock(spec=LLMService)
    mock.generate.return_value = "What role are you hiring for?"
    mock.generate_json.return_value = {
        "intent": "CLARIFY",
        "confidence": 0.9,
        "assessment_names": [],
    }
    return mock


@pytest.fixture
def retrieval_service(chroma_store) -> RetrievalService:
    """Real RetrievalService using the test ChromaDB."""
    return RetrievalService(chroma_store=chroma_store, top_k=5)


@pytest.fixture
def refusal_service() -> RefusalService:
    """Real RefusalService (deterministic, no mocking needed)."""
    return RefusalService()


@pytest.fixture
def session_service(tmp_path) -> SessionService:
    """Temporary SQLite session database for testing."""
    db_file = tmp_path / "test_sessions.db"
    service = SessionService(db_path=str(db_file))
    service.init_db()
    return service


@pytest.fixture
def agent(mock_llm_service, retrieval_service, refusal_service, session_service) -> AgentOrchestrator:
    """Agent wired with mock LLM and real retrieval."""
    intent_service = IntentService(llm_service=mock_llm_service)
    conversation_service = ConversationService(llm_service=mock_llm_service)
    recommendation_service = RecommendationService(llm_service=mock_llm_service)
    comparison_service = ComparisonService(
        llm_service=mock_llm_service,
        retrieval_service=retrieval_service,
    )
    clarification_service = ClarificationService(llm_service=mock_llm_service)

    return AgentOrchestrator(
        intent_service=intent_service,
        conversation_service=conversation_service,
        retrieval_service=retrieval_service,
        recommendation_service=recommendation_service,
        comparison_service=comparison_service,
        clarification_service=clarification_service,
        refusal_service=refusal_service,
        session_service=session_service,
    )


@pytest.fixture
def test_client(agent):
    """
    FastAPI TestClient with pre-wired agent injected into app.state.
    No real server, no network calls. Runs within lifespan context.
    """
    app = create_app()

    # Override lifespan — inject test agent directly
    async def mock_lifespan(app):
        app.state.agent = agent
        yield

    app.router.lifespan_context = mock_lifespan
    with TestClient(app) as client:
        yield client
