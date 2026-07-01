"""
scripts/build_vectorstore.py

One-time ChromaDB population script.

Run this after build_catalog.py to:
1. Load catalog.json.
2. Generate embeddings for all assessments.
3. Ingest into ChromaDB.

This pre-warms the vector store so server startup is fast.

Usage:
    python scripts/build_vectorstore.py [--reset]

Options:
    --reset    Delete existing ChromaDB collection and re-ingest from scratch.
"""

import logging
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.services.scraper import CatalogLoader
from app.vectorstore.chroma import ChromaStore
from app.vectorstore.embedding import EmbeddingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Build and populate the ChromaDB vector store."""
    settings = get_settings()
    reset = "--reset" in sys.argv

    # ── Optional reset ────────────────────────────────────────
    chroma_path = Path(settings.chroma_db_abs_path)
    if reset and chroma_path.exists():
        logger.info("--reset flag detected. Deleting existing ChromaDB at %s", chroma_path)
        shutil.rmtree(chroma_path)
        logger.info("ChromaDB deleted.")

    # ── Load catalog ──────────────────────────────────────────
    if not Path(settings.catalog_abs_path).exists():
        logger.error(
            "catalog.json not found at %s. Run build_catalog.py first.",
            settings.catalog_abs_path,
        )
        sys.exit(1)

    logger.info("Loading catalog from %s", settings.catalog_abs_path)
    loader = CatalogLoader(catalog_path=settings.catalog_abs_path)
    assessments = loader.load()
    logger.info("Loaded %d assessments.", len(assessments))

    # ── Load embedding model ──────────────────────────────────
    logger.info("Loading embedding model: %s", settings.embedding_model)
    embedding_service = EmbeddingService(model_name=settings.embedding_model)
    logger.info("Embedding dimension: %d", embedding_service.dimension)

    # ── Ingest into ChromaDB ──────────────────────────────────
    logger.info("Initialising ChromaDB at: %s", settings.chroma_db_abs_path)
    store = ChromaStore(
        embedding_service=embedding_service,
        db_path=settings.chroma_db_abs_path,
        collection_name=settings.chroma_collection_name,
    )

    if store.is_populated() and not reset:
        logger.info(
            "ChromaDB already contains %d documents. Use --reset to re-ingest.",
            store.count(),
        )
    else:
        logger.info("Ingesting %d assessments into ChromaDB...", len(assessments))
        # Force re-ingest by temporarily deleting collection if reset
        store.ingest(assessments)

    logger.info("=" * 50)
    logger.info("Vector store build complete!")
    logger.info("  Documents in collection: %d", store.count())
    logger.info("  ChromaDB path          : %s", settings.chroma_db_abs_path)
    logger.info("=" * 50)
    logger.info("Ready to start the server: uvicorn app.main:app --port 8002")


if __name__ == "__main__":
    main()
