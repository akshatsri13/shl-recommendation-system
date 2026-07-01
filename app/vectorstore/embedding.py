"""
app/vectorstore/embedding.py

Embedding service using HuggingFace sentence-transformers.

Wraps the SentenceTransformer model behind a clean interface so the
rest of the codebase never imports sentence_transformers directly.
This makes it easy to swap embedding models without touching callers.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates dense vector embeddings for text using sentence-transformers.

    The model is loaded once and reused for all subsequent calls (singleton
    lifecycle managed externally via dependency injection).

    Attributes:
        model_name: HuggingFace model identifier, e.g. 'all-MiniLM-L6-v2'.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")

    def embed(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the dense embedding vector.
        """
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts efficiently.

        Uses sentence-transformers' batch encoding for speed.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        if not texts:
            return []

        logger.debug("Embedding batch of %d texts.", len(texts))
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=len(texts) > 100,
        )
        return [v.tolist() for v in vectors]

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension for the loaded model."""
        return self._model.get_sentence_embedding_dimension()
