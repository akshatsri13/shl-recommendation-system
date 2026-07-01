"""
app/config.py

Centralised settings loaded from environment variables via Pydantic Settings.
All modules import from here — never read os.environ directly elsewhere.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Application-wide configuration.

    Values are resolved in this priority order:
    1. OS environment variables
    2. .env file
    3. Field defaults below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Groq LLM ──────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1          # low = more deterministic
    llm_max_tokens: int = 2048
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 30

    # ── Embeddings ────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── ChromaDB ──────────────────────────────────────────────
    chroma_db_path: str = "app/data/chroma_db"
    chroma_collection_name: str = "shl_assessments"

    # ── API ───────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8002
    log_level: str = "INFO"

    # ── Catalog ───────────────────────────────────────────────
    catalog_path: str = "app/data/catalog.json"

    # ── SQLite Database ───────────────────────────────────────
    sqlite_db_path: str = "app/data/sessions.db"

    # ── Agent Behaviour ───────────────────────────────────────
    max_recommendations: int = 10
    min_recommendations: int = 1
    retrieval_top_k: int = 10

    @property
    def chroma_db_abs_path(self) -> str:
        """Resolve ChromaDB path relative to project root."""
        return str((PROJECT_ROOT / self.chroma_db_path).resolve())

    @property
    def catalog_abs_path(self) -> str:
        """Resolve catalog path relative to project root."""
        return str((PROJECT_ROOT / self.catalog_path).resolve())

    @property
    def sqlite_db_abs_path(self) -> str:
        """Resolve SQLite path relative to project root."""
        return str((PROJECT_ROOT / self.sqlite_db_path).resolve())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Uses lru_cache so the .env file is read only once per process.
    """
    return Settings()
