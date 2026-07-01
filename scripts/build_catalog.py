"""
scripts/build_catalog.py

One-time catalog preparation script.

Run this before starting the server to:
1. Download the SHL product catalog JSON from the remote endpoint.
2. Normalise it into the catalog.json format.
3. Save it to app/data/catalog.json.

This ensures fast startup (local file load instead of network fetch).

Usage:
    python scripts/build_catalog.py

The script is idempotent — safe to re-run. It will overwrite catalog.json.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.services.scraper import CatalogLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

settings = get_settings()
CATALOG_PATH = Path(settings.catalog_abs_path)


def main() -> None:
    """Download, normalise, and save the SHL product catalog."""
    logger.info("Building SHL catalog...")

    # Force a fresh download by temporarily removing existing file
    if CATALOG_PATH.exists():
        logger.info("Existing catalog found — will overwrite with fresh data.")
        CATALOG_PATH.unlink()

    loader = CatalogLoader(catalog_path=str(CATALOG_PATH))
    assessments = loader.load()

    logger.info("=" * 50)
    logger.info("Catalog built successfully!")
    logger.info("  Total assessments : %d", len(assessments))
    logger.info("  Saved to          : %s", CATALOG_PATH.resolve())

    # Print test type distribution
    from collections import Counter
    type_counts = Counter(a.test_type for a in assessments)
    logger.info("  Assessment types:")
    for test_type, count in type_counts.most_common():
        logger.info("    %-35s : %d", test_type, count)

    logger.info("=" * 50)
    logger.info("Next step: python scripts/build_vectorstore.py")


if __name__ == "__main__":
    main()
