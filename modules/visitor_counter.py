"""
visitor_counter.py - Maintains an accurate count of unique visitors.

A face is counted as a new unique visitor only the first time it is
registered (i.e. its face_id appears for the first time). Re-identification
of the same face in later frames does NOT increment the count.
"""

import logging
from threading import Lock
from typing import Set

logger = logging.getLogger(__name__)


class VisitorCounter:
    """
    Thread-safe unique visitor counter.

    Attributes:
        _seen_ids: Set of face_id strings that have already been counted.
        _lock: Thread lock for concurrent access.
    """

    def __init__(self) -> None:
        self._seen_ids: Set[str] = set()
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, face_id: str) -> bool:
        """
        Register a face_id as a unique visitor.

        Args:
            face_id: Unique face identifier string.

        Returns:
            True if this is a *new* visitor (first time seen),
            False if the face was already counted.
        """
        with self._lock:
            if face_id in self._seen_ids:
                return False
            self._seen_ids.add(face_id)
            logger.info("New unique visitor registered: %s  (total=%d)", face_id, len(self._seen_ids))
            return True

    def get_count(self) -> int:
        """Return current unique visitor count."""
        with self._lock:
            return len(self._seen_ids)

    def get_ids(self) -> Set[str]:
        """Return a snapshot copy of all seen face IDs."""
        with self._lock:
            return set(self._seen_ids)

    def reset(self) -> None:
        """Reset the counter (useful for testing)."""
        with self._lock:
            self._seen_ids.clear()
            logger.info("VisitorCounter reset.")
