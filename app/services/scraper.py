"""
app/services/scraper.py

Catalog loader / scraper service.

Primary path: Load the pre-fetched SHL product catalog JSON.
Secondary path: Stub for future live scraping of shl.com product catalog.

The CatalogLoader is the single source of truth for raw assessment data.
All other services consume Assessment objects produced here.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Domain Model
# ──────────────────────────────────────────────────────────────

REMOTE_CATALOG_URL = (
    "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
)


@dataclass
class Assessment:
    """
    Normalised representation of one SHL assessment.

    All fields map directly to data available in the SHL catalog JSON.
    The test_type field is derived from the `keys` list.
    """

    entity_id: str
    name: str
    url: str
    description: str
    test_type: str                         # primary category label
    categories: List[str] = field(default_factory=list)   # all keys
    job_levels: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    duration: str = ""
    remote_testing: bool = True
    adaptive: bool = False

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for catalog.json persistence)."""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "test_type": self.test_type,
            "categories": self.categories,
            "job_levels": self.job_levels,
            "languages": self.languages,
            "duration": self.duration,
            "remote_testing": self.remote_testing,
            "adaptive": self.adaptive,
        }

    def to_embedding_text(self) -> str:
        """
        Build a rich text string for embedding.

        Combines all searchable fields so the embedding captures
        semantic meaning across name, description, levels, and skills.
        """
        parts = [
            f"Assessment: {self.name}",
            f"Description: {self.description}",
            f"Test Type: {self.test_type}",
            f"Categories: {', '.join(self.categories)}",
            f"Job Levels: {', '.join(self.job_levels)}",
            f"Languages: {', '.join(self.languages)}",
            f"Duration: {self.duration}",
            f"Remote Testing: {'Yes' if self.remote_testing else 'No'}",
            f"Adaptive: {'Yes' if self.adaptive else 'No'}",
        ]
        return "\n".join(parts)


# ──────────────────────────────────────────────────────────────
# Key → Test Type Mapping
# ──────────────────────────────────────────────────────────────

_KEY_TO_TEST_TYPE: dict[str, str] = {
    "Knowledge & Skills": "Knowledge & Skills",
    "Personality & Behavior": "Personality & Behavior",
    "Simulations": "Simulations",
    "Ability & Aptitude": "Ability & Aptitude",
    "Competencies": "Competencies",
    "Development & 360": "Development & 360",
    "Biodata & Situational Judgment": "Biodata & Situational Judgment",
    "Assessment Exercises": "Assessment Exercises",
}


def _resolve_test_type(keys: List[str]) -> str:
    """
    Derive the primary test_type from the `keys` list.

    Uses priority order so a well-known type is chosen first.
    Falls back to the first key, or 'Unknown' if empty.
    """
    priority = [
        "Personality & Behavior",
        "Ability & Aptitude",
        "Knowledge & Skills",
        "Simulations",
        "Competencies",
        "Development & 360",
        "Biodata & Situational Judgment",
        "Assessment Exercises",
    ]
    for p in priority:
        if p in keys:
            return _KEY_TO_TEST_TYPE.get(p, p)
    return keys[0] if keys else "Unknown"


# ──────────────────────────────────────────────────────────────
# Catalog Loader
# ──────────────────────────────────────────────────────────────


class CatalogLoader:
    """
    Loads SHL assessments from the pre-fetched catalog JSON.

    Design:
    - Primary: read from local catalog.json (fast, reliable).
    - Fallback: download fresh copy from the SHL remote endpoint.
    - Scrape: future stub for live shl.com HTML scraping.
    """

    def __init__(self, catalog_path: str) -> None:
        self._catalog_path = Path(catalog_path)

    # ── Public API ────────────────────────────────────────────

    def load(self) -> List[Assessment]:
        """
        Load assessments from disk. If catalog.json does not exist,
        download and process from the remote URL first.

        Returns:
            List of Assessment objects ready for embedding.
        """
        if not self._catalog_path.exists():
            logger.warning(
                "catalog.json not found at %s — downloading from remote.",
                self._catalog_path,
            )
            assessments = self._fetch_from_remote()
            self._save_catalog(assessments)
            return assessments

        logger.info("Loading catalog from %s", self._catalog_path)
        return self._load_from_disk()

    # ── Private Helpers ───────────────────────────────────────

    def _load_from_disk(self) -> List[Assessment]:
        """Deserialise catalog.json into Assessment objects."""
        with self._catalog_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        assessments = [self._from_dict(item) for item in raw]
        logger.info("Loaded %d assessments from disk.", len(assessments))
        return assessments

    def _fetch_from_remote(self) -> List[Assessment]:
        """
        Download the SHL product catalog JSON from the remote endpoint
        and normalise it into Assessment objects.
        """
        logger.info("Fetching catalog from %s", REMOTE_CATALOG_URL)
        resp = requests.get(REMOTE_CATALOG_URL, timeout=30)
        resp.raise_for_status()
        # Use strict=False to tolerate unescaped control characters in the remote JSON response
        raw_items: List[dict] = json.loads(resp.text, strict=False)
        assessments = [self._normalise_raw_item(item) for item in raw_items]
        logger.info("Fetched %d assessments from remote.", len(assessments))
        return assessments

    def _normalise_raw_item(self, item: dict) -> Assessment:
        """
        Convert a raw catalog JSON entry into an Assessment object.

        Handles field aliasing and type coercion from the source JSON schema.
        """
        keys: List[str] = item.get("keys", [])
        return Assessment(
            entity_id=str(item.get("entity_id", "")),
            name=item.get("name", "").strip(),
            url=item.get("link", "").strip(),
            description=item.get("description", "").strip(),
            test_type=_resolve_test_type(keys),
            categories=keys,
            job_levels=item.get("job_levels", []),
            languages=item.get("languages", []),
            duration=item.get("duration", "").strip(),
            remote_testing=item.get("remote", "no").lower() == "yes",
            adaptive=item.get("adaptive", "no").lower() == "yes",
        )

    def _from_dict(self, item: dict) -> Assessment:
        """Deserialise a normalised catalog dict (from catalog.json)."""
        return Assessment(
            entity_id=item.get("entity_id", ""),
            name=item.get("name", ""),
            url=item.get("url", ""),
            description=item.get("description", ""),
            test_type=item.get("test_type", "Unknown"),
            categories=item.get("categories", []),
            job_levels=item.get("job_levels", []),
            languages=item.get("languages", []),
            duration=item.get("duration", ""),
            remote_testing=item.get("remote_testing", True),
            adaptive=item.get("adaptive", False),
        )

    def _save_catalog(self, assessments: List[Assessment]) -> None:
        """Persist processed assessments to catalog.json for future fast loads."""
        self._catalog_path.parent.mkdir(parents=True, exist_ok=True)
        data = [a.to_dict() for a in assessments]
        with self._catalog_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d assessments to %s.", len(data), self._catalog_path)

    # ── Scraper Stub ─────────────────────────────────────────

    @staticmethod
    def scrape_live(base_url: str = "https://www.shl.com/solutions/products/productcatalog/") -> List[dict]:
        """
        STUB: Future live HTML scraper for the SHL product catalog page.

        Currently returns an empty list — the page may be behind JavaScript
        rendering or require authentication. Use _fetch_from_remote() instead.

        Args:
            base_url: SHL product catalog URL.

        Returns:
            List of raw dicts (empty until implemented).
        """
        logger.warning(
            "scrape_live() is not yet implemented. "
            "Using JSON catalog endpoint instead."
        )
        return []
