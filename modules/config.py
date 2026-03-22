"""
config.py - Configuration loader for the Face Tracker system.
Reads config.json and provides a validated config dictionary to all modules.
"""

import json
import os
import logging

logger = logging.getLogger(__name__)

# Default values used when a key is missing from config.json
DEFAULTS = {
    "video_source": "data/sample_video.mp4",
    "detection_skip_frames": 5,
    "similarity_threshold": 0.45,
    "face_min_confidence": 0.55,
    "tracker_max_disappeared": 30,
    "bytetrack_track_thresh": 0.45,
    "bytetrack_track_buffer": 30,
    "bytetrack_match_thresh": 0.8,
    "log_dir": "logs",
    "db_path": "data/face_db.sqlite",
    "dashboard_port": 5000,
    "enable_dashboard": True,
    "show_preview": True,
    "save_output_video": False,
    "output_video_path": "data/output.avi",
    "insightface_model": "buffalo_sc",
    "insightface_provider": "CPUExecutionProvider",
}


def load_config(config_path: str = "config.json") -> dict:
    """
    Load configuration from a JSON file, falling back to defaults
    for any missing keys.

    Args:
        config_path: Path to the config.json file (relative or absolute).

    Returns:
        Dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the config file is malformed JSON.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Please create config.json (see README.md for sample)."
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Merge loaded values over defaults so missing keys fall back gracefully
    config = {**DEFAULTS, **raw}

    # Ensure required directories exist
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(config["db_path"]) or ".", exist_ok=True)

    logger.info("Config loaded from '%s': %s", config_path, config)
    return config
