"""
app/vectorstore/chroma.py

ChromaDB vector store wrapper for SHL assessment retrieval.

Responsibilities:
- Initialise and persist the ChromaDB collection.
- Ingest Assessment objects (text + metadata + embeddings).
- Query by semantic similarity with optional metadata filters.
- Look up individual assessments by name (for comparison).

Design decisions:
- Uses ChromaDB's persistent client so the index survives restarts.
- Collection is seeded once on startup; subsequent runs skip ingest.
- Metadata filters are applied *after* semantic search (ChromaDB where-clause).
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.services.scraper import Assessment
from app.vectorstore.embedding import EmbeddingService

logger = logging.getLogger(__name__)

_COLLECTION_VERSION = "v1"   # bump to force re-ingest after schema changes


class ChromaStore:
    """
    Persistent ChromaDB vector store for SHL assessments.

    Attributes:
        collection_name: Name of the ChromaDB collection.
        db_path: Filesystem path for persistent storage.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        db_path: str = "app/data/chroma_db",
        collection_name: str = "shl_assessments",
    ) -> None:
        self._embedding = embedding_service
        self.db_path = db_path
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        logger.info(
            "ChromaDB collection '%s' ready at '%s'. Documents: %d",
            collection_name,
            db_path,
            self._collection.count(),
        )

    # ── Ingest ────────────────────────────────────────────────

    def is_populated(self) -> bool:
        """Return True if the collection already contains documents."""
        return self._collection.count() > 0

    def ingest(self, assessments: List[Assessment]) -> None:
        """
        Embed and store all assessments in ChromaDB.

        Skips ingestion if the collection already contains the same number
        of documents (idempotent on re-runs).

        Args:
            assessments: Processed Assessment objects from CatalogLoader.
        """
        if self.is_populated():
            logger.info(
                "Collection '%s' already contains %d documents — skipping ingest.",
                self.collection_name,
                self._collection.count(),
            )
            return

        logger.info("Ingesting %d assessments into ChromaDB...", len(assessments))

        texts = [a.to_embedding_text() for a in assessments]
        embeddings = self._embedding.embed_batch(texts)
        metadatas = [self._to_metadata(a) for a in assessments]
        ids = [f"{a.entity_id}" for a in assessments]

        # Batch upsert (ChromaDB handles chunking internally)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(
            "Ingest complete. Collection now contains %d documents.",
            self._collection.count(),
        )

    # ── Query ─────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search against the ChromaDB collection.

        Args:
            query_text: Natural language query to embed and search.
            n_results: Maximum number of results to return.
            where: Optional ChromaDB metadata filter dict.
                   Example: {"remote_testing": True}

        Returns:
            List of dicts with keys: name, url, test_type, description,
            job_levels, languages, duration, remote_testing, adaptive,
            categories, distance.
        """
        query_embedding = self._embedding.embed(query_text)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        return self._parse_results(results)

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific assessment by its exact name.

        Used by the Comparison Engine to fetch both named assessments.

        Args:
            name: The assessment name to look up (case-insensitive partial match).

        Returns:
            Assessment metadata dict, or None if not found.
        """
        # Semantic search with the name as query (most reliable approach)
        results = self.query(query_text=name, n_results=5)
        if not results:
            return None

        # Try exact match first, then closest
        name_lower = name.lower().strip()
        for result in results:
            if result["name"].lower().strip() == name_lower:
                return result

        # Partial match fallback
        for result in results:
            if name_lower in result["name"].lower():
                return result

        # Return best semantic match
        return results[0] if results else None

    def count(self) -> int:
        """Return the total number of documents in the collection."""
        return self._collection.count()

    # ── Private Helpers ───────────────────────────────────────

    @staticmethod
    def _to_metadata(assessment: Assessment) -> Dict[str, Any]:
        """
        Convert an Assessment into a ChromaDB-compatible metadata dict.

        ChromaDB metadata values must be str, int, float, or bool.
        Lists are serialised as pipe-delimited strings.
        """
        return {
            "name": assessment.name,
            "url": assessment.url,
            "test_type": assessment.test_type,
            "description": assessment.description[:1000],  # ChromaDB metadata limit
            "job_levels": " | ".join(assessment.job_levels),
            "languages": " | ".join(assessment.languages),
            "duration": assessment.duration,
            "remote_testing": assessment.remote_testing,
            "adaptive": assessment.adaptive,
            "categories": " | ".join(assessment.categories),
            "entity_id": assessment.entity_id,
        }

    @staticmethod
    def _parse_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten ChromaDB query results into a list of readable dicts.

        Args:
            results: Raw ChromaDB query response.

        Returns:
            List of assessment dicts with distance scores.
        """
        parsed = []
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for meta, dist in zip(metadatas, distances):
            parsed.append(
                {
                    "name": meta.get("name", ""),
                    "url": meta.get("url", ""),
                    "test_type": meta.get("test_type", ""),
                    "description": meta.get("description", ""),
                    "job_levels": meta.get("job_levels", "").split(" | "),
                    "languages": meta.get("languages", "").split(" | "),
                    "duration": meta.get("duration", ""),
                    "remote_testing": meta.get("remote_testing", True),
                    "adaptive": meta.get("adaptive", False),
                    "categories": meta.get("categories", "").split(" | "),
                    "entity_id": meta.get("entity_id", ""),
                    "distance": round(float(dist), 4),
                }
            )

        return parsed
