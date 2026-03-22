"""
select_video.py - Pick which video file to use for the face tracker.

Lists all video files in the data/ folder, lets you choose one,
and writes the choice into config.json as the video_source.

Usage:
    python select_video.py
"""

import json
import os
import sys

CONFIG_PATH = "config.json"
DATA_DIR = "data"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def get_current_source() -> str:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh).get("video_source", "")
    except Exception:
        return ""


def main():
    # ── Find all video files ─────────────────────────────────────────────────
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Folder '{DATA_DIR}' not found.")
        print("Download a video first using:  python download_video.py")
        sys.exit(1)

    videos = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(VIDEO_EXTS)
    )

    if not videos:
        print(f"No video files found in '{DATA_DIR}/'.")
        sys.exit(1)

    current = get_current_source()

    # ── Display list ─────────────────────────────────────────────────────────
    print("\nAvailable videos in data/:\n")
    for i, name in enumerate(videos, 1):
        path = os.path.join(DATA_DIR, name)
        size_mb = os.path.getsize(path) / 1_048_576
        is_current = os.path.basename(current) == name
        marker = "  ◄ currently selected" if is_current else ""
        print(f"  [{i}] {name}  ({size_mb:.1f} MB){marker}")

    # ── Prompt ───────────────────────────────────────────────────────────────
    print()
    while True:
        try:
            raw = input(f"Select video number (1–{len(videos)}): ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(videos):
                chosen = videos[idx]
                break
            print("  ✗ Out of range, try again.")
        except (ValueError, EOFError):
            print("  ✗ Invalid input.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    # ── Update config.json ───────────────────────────────────────────────────
    chosen_path = os.path.join(DATA_DIR, chosen).replace("\\", "/")

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except FileNotFoundError:
        print(f"ERROR: {CONFIG_PATH} not found.")
        sys.exit(1)

    old = cfg.get("video_source", "")
    cfg["video_source"] = chosen_path

    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"\n✅  config.json updated:")
    print(f"    Before : {old}")
    print(f"    After  : {chosen_path}")
    print("\nNow run:  python main.py")


if __name__ == "__main__":
    main()
