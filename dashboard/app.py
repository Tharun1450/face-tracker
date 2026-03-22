"""
dashboard/app.py - Optional Flask web dashboard for the Face Tracker.

Provides:
  GET /               → Live dashboard HTML page
  GET /api/count      → JSON: { "unique_visitors": N }
  GET /api/events     → JSON: list of recent events
  GET /api/stream     → Server-Sent Events stream for live visitor count updates

The dashboard is started as a daemon thread from main.py and shares the
same SQLite database as the tracker pipeline.
"""

import logging
import os
from typing import Generator

from flask import Flask, Response, jsonify, render_template

from modules.database import get_recent_events, get_visitor_count

logger = logging.getLogger(__name__)


def create_app(cfg: dict) -> Flask:
    """
    Create and configure the Flask app.

    Args:
        cfg: Loaded configuration dictionary (must contain 'db_path' and 'log_dir').

    Returns:
        Configured Flask application instance.
    """
    # Template folder is relative to this file's directory
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    app = Flask(__name__, template_folder=template_dir)
    app.config["DB_PATH"] = cfg["db_path"]
    app.config["LOG_DIR"] = cfg["log_dir"]

    # ----------------------------------------------------------------
    # Routes
    # ----------------------------------------------------------------

    @app.route("/")
    def index():
        """Render the main dashboard."""
        return render_template("index.html")

    @app.route("/api/count")
    def api_count():
        """Return the current unique visitor count as JSON."""
        count = get_visitor_count(app.config["DB_PATH"])
        return jsonify({"unique_visitors": count})

    @app.route("/api/events")
    def api_events():
        """Return the 50 most recent events as JSON."""
        events = get_recent_events(app.config["DB_PATH"], limit=50)
        return jsonify(events)

    @app.route("/api/stream")
    def api_stream():
        """
        Server-Sent Events endpoint that pushes the visitor count
        every second. Compatible with EventSource in the browser.
        """
        def event_generator() -> Generator[str, None, None]:
            import time
            while True:
                try:
                    count = get_visitor_count(app.config["DB_PATH"])
                    yield f"data: {count}\n\n"
                    time.sleep(1)
                except GeneratorExit:
                    break
                except Exception as exc:
                    logger.warning("SSE error: %s", exc)
                    break

        return Response(event_generator(), mimetype="text/event-stream")

    return app
