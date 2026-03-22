"""
reset_db.py - Clear the face tracker database and logs for a fresh session.

Usage:
    python reset_db.py          # prompts for confirmation
    python reset_db.py --yes    # skip confirmation (for scripting)

What it clears:
  - data/face_db.sqlite  (all registered faces and events)
  - logs/events.log      (event log file)
  - logs/entries/        (entry face images)
  - logs/exits/          (exit face images)
"""

import json
import os
import shutil
import sys

CONFIG_PATH = "config.json"


def load_paths():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cfg.get("db_path", "data/face_db.sqlite"), cfg.get("log_dir", "logs")
    except Exception:
        return "data/face_db.sqlite", "logs"


def main():
    auto_yes = "--yes" in sys.argv
    db_path, log_dir = load_paths()

    print("=" * 50)
    print("  Face Tracker — Reset Session")
    print("=" * 50)
    print(f"\nThis will DELETE:")
    print(f"  • {db_path}  (all registered faces & events)")
    print(f"  • {log_dir}/events.log")
    print(f"  • {log_dir}/entries/  (entry images)")
    print(f"  • {log_dir}/exits/    (exit images)")
    print()

    if not auto_yes:
        try:
            ans = input("Are you sure? (yes/no): ").strip().lower()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
        if ans not in ("yes", "y"):
            print("Cancelled.")
            sys.exit(0)

    deleted = []
    errors = []

    # 1. Delete the SQLite database
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            deleted.append(db_path)
        except Exception as e:
            errors.append(f"{db_path}: {e}")

    # 2. Delete events.log
    events_log = os.path.join(log_dir, "events.log")
    if os.path.exists(events_log):
        try:
            os.remove(events_log)
            deleted.append(events_log)
        except Exception as e:
            errors.append(f"{events_log}: {e}")

    # 3. Delete entries/ and exits/ image folders
    for sub in ("entries", "exits"):
        folder = os.path.join(log_dir, sub)
        if os.path.isdir(folder):
            try:
                shutil.rmtree(folder)
                deleted.append(folder + "/")
            except Exception as e:
                errors.append(f"{folder}: {e}")

    # Summary
    print()
    if deleted:
        print(f"✅ Deleted {len(deleted)} item(s):")
        for d in deleted:
            print(f"   • {d}")
    if errors:
        print(f"\n⚠️  {len(errors)} error(s):")
        for e in errors:
            print(f"   • {e}")

    if not deleted and not errors:
        print("Nothing to delete — already clean.")

    print("\nReady for a fresh run:  python main.py")


if __name__ == "__main__":
    main()
