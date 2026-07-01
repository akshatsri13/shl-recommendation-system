"""
app/services/session_service.py

SQLite-backed session storage service.

Saves conversation messages on the server to make the /chat endpoint stateful.
Uses standard Python sqlite3 with thread-safe on-demand connection contexts.
"""

import logging
import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional

from app.models.request_models import Message

logger = logging.getLogger(__name__)


class SessionService:
    """
    Manages session lifecycle and persists message history in SQLite.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def init_db(self) -> None:
        """
        Create the schema and indexes if they do not exist.
        Runs once on FastAPI startup lifespan.
        """
        # Ensure parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_id ON chat_messages(session_id);
                """
            )
            conn.commit()
        logger.info("SQLite session database ready at: %s", self._db_path)

    def get_or_create_session(self, session_id: Optional[str]) -> str:
        """
        Validate the given session ID or generate a new UUIDv4 string.
        """
        if session_id and session_id.strip():
            return session_id.strip()
        new_id = str(uuid.uuid4())
        logger.info("Generated new session ID: %s", new_id)
        return new_id

    def save_message(self, session_id: str, role: str, content: str) -> None:
        """
        Append a new message (user or assistant) to the chat history.
        """
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_messages (session_id, role, content)
                VALUES (?, ?, ?);
                """,
                (session_id, role, content),
            )
            conn.commit()
        logger.debug("Saved message to session '%s': role=%s", session_id, role)

    def get_history(self, session_id: str) -> List[Message]:
        """
        Retrieve the full ordered conversation history for the session.
        """
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC;
                """,
                (session_id,),
            )
            rows = cursor.fetchall()

        messages = [Message(role=row[0], content=row[1]) for row in rows]
        logger.debug("Retrieved %d history messages for session '%s'", len(messages), session_id)
        return messages

    def clear_history(self, session_id: str) -> None:
        """
        Clear all messages for the session (useful for resetting/tests).
        """
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM chat_messages WHERE session_id = ?;",
                (session_id,),
            )
            conn.commit()
        logger.info("Cleared history for session: %s", session_id)
