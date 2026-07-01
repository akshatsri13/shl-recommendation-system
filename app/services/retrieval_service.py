"""
app/services/retrieval_service.py

Semantic retrieval from ChromaDB.

Builds a rich search query from ConversationState and retrieves
the top-K most relevant assessments. Applies optional metadata
filtering for practical constraints (language, remote, adaptive).
"""

import logging
from typing import Any, Dict, List, Optional

from app.services.conversation_service import ConversationState
from app.vectorstore.chroma import ChromaStore

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Retrieves relevant SHL assessments from ChromaDB based on hiring context.

    Responsibilities:
    - Build a rich semantic query from ConversationState.
    - Apply metadata pre-filters where appropriate.
    - Return top-K candidates for the recommendation engine to rank.
    """

    def __init__(self, chroma_store: ChromaStore, top_k: int = 10) -> None:
        self._store = chroma_store
        self._top_k = top_k

    def retrieve(
        self,
        state: ConversationState,
        query_override: Optional[str] = None,
        n_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K assessments relevant to the hiring context.

        Args:
            state: Reconstructed conversation state with hiring requirements.
            query_override: Optional manual query (skips state.to_search_query()).
            n_results: Override the default top-K count.

        Returns:
            List of assessment dicts from ChromaDB (sorted by relevance).
        """
        query = query_override or state.to_search_query()
        k = n_results or self._top_k

        logger.info("Retrieving top-%d assessments for query: '%s'", k, query)

        # Build metadata filters from hard constraints
        where_filter = self._build_filter(state)

        try:
            results = self._store.query(
                query_text=query,
                n_results=k,
                where=where_filter if where_filter else None,
            )
        except Exception as exc:
            logger.error(
                "ChromaDB query failed (filter=%s): %s. Retrying without filter.",
                where_filter,
                exc,
            )
            # Retry without metadata filter (graceful degradation)
            results = self._store.query(query_text=query, n_results=k)

        logger.info("Retrieved %d candidates.", len(results))
        return results

    def retrieve_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a specific assessment by name for comparison purposes.

        Args:
            name: Assessment name (exact or partial).

        Returns:
            Assessment dict or None if not found.
        """
        logger.info("Looking up assessment by name: '%s'", name)
        return self._store.get_by_name(name)

    # ── Private Helpers ───────────────────────────────────────

    @staticmethod
    def _build_filter(state: ConversationState) -> Optional[Dict[str, Any]]:
        """
        Build a ChromaDB `where` filter from hard constraints in state.

        Only applies constraints that can be reliably matched on metadata.
        Avoids over-filtering which could return zero results.

        Returns:
            A ChromaDB-compatible filter dict, or None if no constraints.
        """
        conditions = []

        # Remote testing constraint
        remote_keywords = {"remote", "online", "virtual"}
        if any(k in " ".join(state.constraints).lower() for k in remote_keywords):
            conditions.append({"remote_testing": {"$eq": True}})

        # Adaptive testing constraint
        adaptive_keywords = {"adaptive", "cat", "computer adaptive"}
        if any(k in " ".join(state.constraints).lower() for k in adaptive_keywords):
            conditions.append({"adaptive": {"$eq": True}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
