"""
database.py - SQLite database schema and CRUD helpers for the Face Tracker system.

Tables:
  faces  - Registered unique faces (one row per unique person)
  events - Entry / exit events (one row per enter or exit)
"""

import sqlite3
import logging
import os
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_FACES = """
CREATE TABLE IF NOT EXISTS faces (
    face_id       TEXT PRIMARY KEY,       -- UUID string
    first_seen    TEXT NOT NULL,          -- ISO-8601 timestamp
    last_seen     TEXT NOT NULL,          -- ISO-8601 timestamp
    embedding     BLOB NOT NULL,          -- NumPy float32 array serialised to bytes
    thumbnail_path TEXT                   -- path to the first cropped image saved
);
"""

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id       TEXT NOT NULL,          -- FK → faces.face_id
    event_type    TEXT NOT NULL,          -- 'entry' or 'exit'
    timestamp     TEXT NOT NULL,          -- ISO-8601 timestamp
    image_path    TEXT,                   -- path to the saved face crop
    FOREIGN KEY(face_id) REFERENCES faces(face_id)
);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    """Return a connection with row_factory set to Row."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    """
    Initialise the SQLite database, creating tables if they don't exist.

    Args:
        db_path: Filesystem path to the .sqlite file.
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = _connect(db_path)
    with conn:
        conn.execute(_CREATE_FACES)
        conn.execute(_CREATE_EVENTS)
    conn.close()
    logger.info("Database initialised at '%s'", db_path)


# ---------------------------------------------------------------------------
# Face registration
# ---------------------------------------------------------------------------

def register_face(
    db_path: str,
    face_id: str,
    embedding: np.ndarray,
    thumbnail_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> None:
    """
    Insert a new face record into the faces table.

    Args:
        db_path: Path to the SQLite database.
        face_id: Unique identifier string (UUID).
        embedding: 512-d float32 ArcFace embedding.
        thumbnail_path: Optional path to the saved face crop image.
        timestamp: Optional ISO-8601 timestamp; defaults to now.
    """
    ts = timestamp or datetime.now().isoformat()
    emb_blob = embedding.astype(np.float32).tobytes()
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO faces (face_id, first_seen, last_seen, embedding, thumbnail_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (face_id, ts, ts, emb_blob, thumbnail_path),
            )
        logger.info("Registered new face: %s", face_id)
    except sqlite3.IntegrityError:
        logger.warning("Face %s already exists in DB — skipping registration.", face_id)
    finally:
        conn.close()


def update_face_last_seen(db_path: str, face_id: str, timestamp: Optional[str] = None) -> None:
    """Update the last_seen timestamp for an existing face."""
    ts = timestamp or datetime.now().isoformat()
    conn = _connect(db_path)
    with conn:
        conn.execute(
            "UPDATE faces SET last_seen = ? WHERE face_id = ?",
            (ts, face_id),
        )
    conn.close()


# ---------------------------------------------------------------------------
# Embedding retrieval
# ---------------------------------------------------------------------------

def get_all_embeddings(db_path: str) -> List[Tuple[str, np.ndarray]]:
    """
    Load all stored face embeddings from the database.

    Returns:
        List of (face_id, embedding_array) tuples.
    """
    conn = _connect(db_path)
    rows = conn.execute("SELECT face_id, embedding FROM faces").fetchall()
    conn.close()
    result = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        result.append((row["face_id"], emb))
    return result


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

def log_event(
    db_path: str,
    face_id: str,
    event_type: str,
    image_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> None:
    """
    Record a face entry or exit event.

    Args:
        db_path: Path to the SQLite database.
        face_id: The face identifier.
        event_type: 'entry' or 'exit'.
        image_path: Optional path to the saved crop image.
        timestamp: Optional ISO-8601 string; defaults to now.
    """
    ts = timestamp or datetime.now().isoformat()
    conn = _connect(db_path)
    with conn:
        conn.execute(
            "INSERT INTO events (face_id, event_type, timestamp, image_path) "
            "VALUES (?, ?, ?, ?)",
            (face_id, event_type, ts, image_path),
        )
    conn.close()
    logger.debug("Event logged: face=%s type=%s ts=%s", face_id, event_type, ts)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def get_visitor_count(db_path: str) -> int:
    """
    Return the total number of unique registered faces (unique visitors).

    Args:
        db_path: Path to the SQLite database.

    Returns:
        Integer count of unique faces.
    """
    conn = _connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    conn.close()
    return count


def get_recent_events(db_path: str, limit: int = 50) -> List[dict]:
    """
    Return the most recent events as a list of dicts (for the dashboard).

    Args:
        db_path: Path to the SQLite database.
        limit: Maximum number of events to return.

    Returns:
        List of event dicts ordered by timestamp descending.
    """
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT event_id, face_id, event_type, timestamp, image_path "
        "FROM events ORDER BY event_id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
