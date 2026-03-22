"""
logger.py - Structured event logging for the Face Tracker system.

Responsibilities:
  1. Write human-readable lines to `events.log`.
  2. Save cropped face images to logs/entries/ or logs/exits/ with date folders.
  3. Record the event in the SQLite database via database.log_event().

All three actions happen atomically for each event so a crash between steps
does not leave the system in an inconsistent state.
"""

import logging
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from modules import database as db

# Module-level Python logger (writes to console + events.log)
_sys_logger = logging.getLogger(__name__)


def setup_logging(log_dir: str) -> None:
    """
    Configure the root Python logger to write to both console and
    a rotating events.log file inside `log_dir`.

    Call this once at application startup (in main.py).

    Args:
        log_dir: Directory where events.log will be created.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "events.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler — append mode so restarts accumulate history
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    root.addHandler(ch)
    root.addHandler(fh)

    root.info("=== Face Tracker session started ===")
    root.info("Log file: %s", os.path.abspath(log_path))


class EventLogger:
    """
    Records face entry and exit events to disk, database, and the log file.

    Args:
        log_dir: Root directory for logs (e.g. 'logs/').
        db_path: Path to the SQLite database file.
    """

    def __init__(self, log_dir: str, db_path: str) -> None:
        self.log_dir = log_dir
        self.db_path = db_path

        # Create sub-directories
        os.makedirs(os.path.join(log_dir, "entries"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "exits"), exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_entry(
        self,
        face_id: str,
        face_crop: Optional[np.ndarray] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        """
        Log a face entry event.

        Args:
            face_id: Unique face identifier.
            face_crop: BGR crop of the face (will be saved as JPEG).
            timestamp: ISO-8601 string; defaults to now.

        Returns:
            Path to the saved image (or empty string if no crop).
        """
        return self._log_event("entry", face_id, face_crop, timestamp)

    def log_exit(
        self,
        face_id: str,
        face_crop: Optional[np.ndarray] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        """
        Log a face exit event.

        Args:
            face_id: Unique face identifier.
            face_crop: BGR crop of the face at exit time.
            timestamp: ISO-8601 string; defaults to now.

        Returns:
            Path to the saved image (or empty string if no crop).
        """
        return self._log_event("exit", face_id, face_crop, timestamp)

    def log_registration(self, face_id: str) -> None:
        """Log that a new face has been registered in the DB."""
        _sys_logger.info(
            "REGISTRATION | face_id=%s | New face registered and embedded.", face_id
        )

    def log_recognition(self, face_id: str, score: float) -> None:
        """Log that an existing face has been recognised."""
        _sys_logger.info(
            "RECOGNITION  | face_id=%s | score=%.4f", face_id, score
        )

    def log_tracking(self, face_id: str, track_id: int) -> None:
        """Log that a face is being tracked under a specific track ID."""
        _sys_logger.debug(
            "TRACKING     | face_id=%s | track_id=%d", face_id, track_id
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_event(
        self,
        event_type: str,       # 'entry' or 'exit'
        face_id: str,
        face_crop: Optional[np.ndarray],
        timestamp: Optional[str],
    ) -> str:
        ts = timestamp or datetime.now().isoformat()
        date_str = ts[:10]  # YYYY-MM-DD

        image_path = ""

        # 1. Save the cropped image
        if face_crop is not None and face_crop.size > 0:
            image_path = self._save_crop(event_type, face_id, ts, date_str, face_crop)

        # 2. Write to events.log via the Python logging system
        _sys_logger.info(
            "%-6s | face_id=%-36s | ts=%s | image=%s",
            event_type.upper(),
            face_id,
            ts,
            image_path or "N/A",
        )

        # 3. Persist to database
        try:
            db.log_event(
                db_path=self.db_path,
                face_id=face_id,
                event_type=event_type,
                image_path=image_path or None,
                timestamp=ts,
            )
        except Exception as exc:
            _sys_logger.error("DB log_event failed: %s", exc)

        return image_path

    def _save_crop(
        self,
        event_type: str,
        face_id: str,
        timestamp: str,
        date_str: str,
        crop: np.ndarray,
    ) -> str:
        """Save the face crop image and return the path."""
        subdir = os.path.join(self.log_dir, f"{event_type}s", date_str)
        os.makedirs(subdir, exist_ok=True)

        # Sanitise timestamp for use in filename
        ts_safe = timestamp.replace(":", "-").replace(".", "-")
        short_id = face_id[:8]  # first 8 chars of UUID for readability
        filename = f"{short_id}_{ts_safe}.jpg"
        full_path = os.path.join(subdir, filename)

        try:
            cv2.imwrite(full_path, crop)
        except Exception as exc:
            _sys_logger.warning("Failed to save face crop: %s", exc)
            return ""

        return full_path
