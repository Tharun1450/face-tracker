"""
main.py - Orchestrator for the Intelligent Face Tracker pipeline.

Entry point for processing a video file (or RTSP stream).
Ties together: detection → recognition → ByteTrack → logging → counting.

Usage:
    python main.py                                    # uses config.json
    python main.py --config path/to/config.json
    python main.py --source data/video.mp4 --headless
    python main.py --source rtsp://192.168.1.1/stream
    python main.py --reset                            # clear DB+logs, then run
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config import load_config
from modules.database import (
    get_all_embeddings,
    get_visitor_count,
    init_db,
    register_face,
    update_face_last_seen,
)
from modules.face_detector import FaceDetector
from modules.face_recognizer import FaceRecognizer
from modules.line_counter import LineCrossingCounter
from modules.logger import EventLogger, setup_logging
from modules.tracker import ByteTracker
from modules.visitor_counter import VisitorCounter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker")
    parser.add_argument(
        "--config", default="config.json", help="Path to config.json"
    )
    parser.add_argument(
        "--source", default=None,
        help="Override video source (file path or rtsp:// URL)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Disable preview window (useful for servers / CI)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear the database and logs before starting (fresh session)"
    )
    return parser.parse_args()


def _reset_session(cfg: dict) -> None:
    """Delete DB, events.log, and face-image folders for a clean start."""
    import shutil
    db_path = cfg["db_path"]
    log_dir = cfg["log_dir"]

    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info("Reset: deleted %s", db_path)

    events_log = os.path.join(log_dir, "events.log")
    if os.path.exists(events_log):
        os.remove(events_log)
        logger.info("Reset: deleted %s", events_log)

    for sub in ("entries", "exits"):
        folder = os.path.join(log_dir, sub)
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            logger.info("Reset: deleted %s", folder)

    print("[RESET] Database and logs cleared — starting fresh session.")


# ---------------------------------------------------------------------------
# IoU helper for matching InsightFace detections to ByteTrack bboxes
# ---------------------------------------------------------------------------

def _bbox_iou(a, b) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _center_dist(a, b) -> float:
    """Euclidean distance between bbox centres."""
    cx_a = (a[0]+a[2])/2; cy_a = (a[1]+a[3])/2
    cx_b = (b[0]+b[2])/2; cy_b = (b[1]+b[3])/2
    return ((cx_a-cx_b)**2 + (cy_a-cy_b)**2) ** 0.5


def _match_embedding(track_bbox, frame_embeddings):
    """
    Find the InsightFace embedding closest to a ByteTrack bounding box.
    Uses IoU first; falls back to centre-distance for partial overlaps.
    Returns the best matching embedding or None.
    """
    if not frame_embeddings:
        return None
    best_emb = None
    best_iou = 0.0
    for (fx1, fy1, fx2, fy2), emb in frame_embeddings:
        iou = _bbox_iou(track_bbox[:4], (fx1, fy1, fx2, fy2))
        if iou > best_iou:
            best_iou = iou
            best_emb = emb
    if best_iou >= 0.1:          # at least 10 % overlap
        return best_emb
    # Fallback: closest centre within 120 px
    best_dist = float('inf')
    for (fx1, fy1, fx2, fy2), emb in frame_embeddings:
        d = _center_dist(track_bbox[:4], (fx1, fy1, fx2, fy2))
        if d < best_dist:
            best_dist = d
            best_emb = emb
    if best_dist < 120:
        return best_emb
    return None


# ---------------------------------------------------------------------------
# Drawing overlay
# ---------------------------------------------------------------------------

def _draw_overlay(
    frame: np.ndarray,
    boxes: list,      # (tid, x1,y1,x2,y2, conf, person_label|None)
    visitor_count: int,
    entry_count: int = 0,
    exit_count: int = 0,
) -> np.ndarray:
    """Draw Person-N labels with large readable font + stats panel."""
    overlay = frame.copy()

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85          # large, clearly readable
    thickness  = 2
    box_color  = (0, 60, 255)  # red-orange (like reference repo)
    txt_color  = (255, 255, 255)

    for item in boxes:
        tid, x1, y1, x2, y2, conf, person_label = item
        label    = person_label if person_label else f"Person ?"
        color    = box_color if person_label else (180, 90, 0)

        # Bounding box (thick border)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Label background + text (at top of box, with padding)
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, thickness)
        tag_y1 = max(y1 - th - bl - 6, 0)
        tag_y2 = max(y1, th + bl + 6)
        cv2.rectangle(overlay, (x1, tag_y1), (x1 + tw + 6, tag_y2), color, -1)
        cv2.putText(overlay, label, (x1 + 3, tag_y2 - bl - 2),
                    font, font_scale, txt_color, thickness, cv2.LINE_AA)

    # ── Stats banner (top-centre) ─────────────────────────────────────────
    h, w = overlay.shape[:2]
    banner = f"  IN : {entry_count}     OUT : {exit_count}  "
    (bw, bh), _ = cv2.getTextSize(banner, font, 0.9, 2)
    bx = (w - bw) // 2
    cv2.rectangle(overlay, (bx - 10, 4), (bx + bw + 10, bh + 16), (30, 30, 30), -1)
    cv2.putText(overlay, banner, (bx, bh + 8),
                font, 0.9, (0, 230, 255), 2, cv2.LINE_AA)

    # ── Visitor count (top-left) ──────────────────────────────────────────
    cv2.rectangle(overlay, (0, 0), (260, 36), (15, 15, 15), -1)
    cv2.putText(overlay, f"Visitors : {visitor_count}", (8, 26),
                font, 0.75, (0, 255, 180), 2, cv2.LINE_AA)

    return overlay




# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cfg: dict, headless: bool = False) -> None:
    """
    Face tracking pipeline — inspired by mahdi-marjani/object-tracking-entry-exit.

    Key design (same as reference repos):
      • model.track(persist=True) for single-call detect+track with stable IDs
      • InsightFace only on padded face crops (fast, no full-frame slowness)
      • Entry  = face first becomes tracked
      • Exit   = face gone > 5 s (cooldown absorbs brief occlusions)
      • Line counter draws IN/OUT on preview (visual reference)
    """
    db_path = cfg["db_path"]
    log_dir = cfg["log_dir"]
    DISPLAY_W, DISPLAY_H = 960, 540
    REENTRY_COOLDOWN = 2.0   # seconds — short enough to catch real exits in video

    # ── Subsystems ───────────────────────────────────────────────────────
    init_db(db_path)
    event_logger = EventLogger(log_dir=log_dir, db_path=db_path)
    visitor_counter = VisitorCounter()

    detector = FaceDetector(
        min_confidence=cfg["face_min_confidence"],
        skip_frames=cfg["detection_skip_frames"],
        device="cpu",
    )
    recognizer = FaceRecognizer(
        model_name=cfg["insightface_model"],
        providers=[cfg["insightface_provider"]],
        threshold=cfg["similarity_threshold"],
    )
    existing = get_all_embeddings(db_path)
    recognizer.load_gallery(existing)
    logger.info("Loaded %d known face(s) for re-ID.", len(existing))

    # ── Line counter (visual only) ────────────────────────────────────────
    line_cfg = cfg.get("counting_line", {})
    lx1, ly1 = line_cfg.get("start", [0, 300])
    lx2, ly2 = line_cfg.get("end",   [640, 300])
    line_counter = LineCrossingCounter(
        line_start=(lx1, ly1), line_end=(lx2, ly2),
        entry_side=line_cfg.get("entry_side", 1),
    )

    # ── Per-session state ─────────────────────────────────────────────────
    track_face: Dict[int, str] = {}     # track_id → face_id
    face_last_crop: dict = {}           # face_id → last crop image
    currently_present: set = set()      # face_ids currently tracked
    pending_exits: dict = {}            # face_id → exit deadline
    entry_count: int = 0               # real entries logged this session
    exit_count: int  = 0               # real exits logged this session
    person_num_map: Dict[str, str] = {} # face_id → "Person N" label
    _next_person: list = [0]           # mutable counter (avoids nonlocal)

    # ── Video source ──────────────────────────────────────────────────────
    source = cfg["video_source"]
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", source)
        sys.exit(1)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video: %s | %dx%d | %.1f fps | %s frames",
                source, vid_w, vid_h, fps, total or "?")

    # Auto-fit line to full video width
    if not line_cfg.get("start"):
        line_counter.x1, line_counter.y1 = 0,     vid_h // 2
        line_counter.x2, line_counter.y2 = vid_w, vid_h // 2

    writer = None
    if cfg.get("save_output_video"):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(cfg["output_video_path"], fourcc, fps, (vid_w, vid_h))

    WIN_NAME = "Face Tracker  |  Q = quit"
    show = not headless and cfg.get("show_preview", True)
    if show:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)   # fully resizable by user
        # Start at a sensible size; user can drag borders to any size
        cv2.resizeWindow(WIN_NAME, DISPLAY_W, DISPLAY_H)

    frame_idx = 0
    prev_track_ids: set = set()
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream.")
                break

            frame_idx += 1
            now = time.time()

            # ── 1. Detect + track (single Ultralytics call, persist=True) ──
            tracked = detector.track(frame)   # → [(tid, x1,y1,x2,y2, conf), ...]
            current_ids = {t[0] for t in tracked}

            # ── 2. Flush expired exits ────────────────────────────────────────────────
            for fid, deadline in list(pending_exits.items()):
                if now >= deadline:
                    del pending_exits[fid]
                    ts = datetime.now().isoformat()
                    event_logger.log_exit(fid, face_crop=face_last_crop.get(fid),
                                         timestamp=ts)
                    update_face_last_seen(db_path, fid, timestamp=ts)
                    exit_count += 1
                    logger.info("EXIT  | face=%s  total_out=%d", fid, exit_count)

            # ── 3. Handle disappeared tracks ──────────────────────────────
            for tid in prev_track_ids - current_ids:
                fid = track_face.get(tid)
                if fid and fid in currently_present:
                    currently_present.discard(fid)
                    pending_exits[fid] = now + REENTRY_COOLDOWN
                line_counter.remove_track(tid)

            prev_track_ids = current_ids

            # ── 4. Process active tracks ──────────────────────────────────
            box_list = []   # for drawing

            for tid, x1, y1, x2, y2, conf in tracked:
                state = (x1, y1, x2, y2, conf)
                crop = detector.crop_face(frame, state, padding=0.3)

                # Already identified?
                if tid in track_face:
                    fid = track_face[tid]
                    if crop is not None:
                        face_last_crop[fid] = crop
                    if fid in pending_exits:          # came back in time
                        del pending_exits[fid]
                        currently_present.add(fid)
                    line_counter.update(tid, fid, state)
                    event_logger.log_tracking(fid, tid)
                    box_list.append((tid, x1, y1, x2, y2, conf, person_num_map.get(fid)))
                    continue

                # ── New track: try to identify with InsightFace ───────────
                if crop is None:
                    box_list.append((tid, x1, y1, x2, y2, conf, None))
                    continue

                embedding = recognizer.get_embedding(crop)
                if embedding is None:
                    box_list.append((tid, x1, y1, x2, y2, conf, None))
                    continue

                face_id, score = recognizer.match(embedding)

                if face_id is None:
                    face_id = FaceRecognizer.new_face_id()
                    ts = datetime.now().isoformat()
                    thumb_dir = os.path.join(log_dir, "entries", ts[:10])
                    os.makedirs(thumb_dir, exist_ok=True)
                    thumb_path = os.path.join(thumb_dir, f"{face_id[:8]}.jpg")
                    cv2.imwrite(thumb_path, crop)
                    register_face(db_path, face_id, embedding, thumb_path, ts)
                    recognizer.add_to_gallery(face_id, embedding)
                    event_logger.log_registration(face_id)
                    logger.info("NEW  face=%s", face_id)
                else:
                    ts = datetime.now().isoformat()
                    event_logger.log_recognition(face_id, score)
                    update_face_last_seen(db_path, face_id, ts)

                # Assign sequential Person N label (once per unique face)
                if face_id not in person_num_map:
                    _next_person[0] += 1
                    person_num_map[face_id] = f"Person {_next_person[0]}"

                track_face[tid] = face_id
                face_last_crop[face_id] = crop
                visitor_counter.register(face_id)

                # ENTRY: first time this face becomes active
                if face_id not in currently_present:
                    currently_present.add(face_id)
                    pending_exits.pop(face_id, None)
                    event_logger.log_entry(face_id, face_crop=crop, timestamp=ts)
                    entry_count += 1
                    logger.info("ENTRY face=%s  total_in=%d",
                                face_id, entry_count)

                line_counter.update(tid, face_id, state)
                box_list.append((tid, x1, y1, x2, y2, conf, person_num_map.get(face_id)))

            # ── 5. Display ────────────────────────────────────────────────
            count   = visitor_counter.get_count()
            display = _draw_overlay(frame, box_list, count,
                                    entry_count=entry_count,
                                    exit_count=exit_count)
            if show:
                cv2.imshow(WIN_NAME, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Q pressed — stopping.")
                    break

            if writer is not None:
                writer.write(display)

            if frame_idx % 100 == 0:
                elapsed = time.time() - t_start
                logger.info("Frame %d | tracks=%d | visitors=%d | %.1f fps",
                            frame_idx, len(tracked), count,
                            frame_idx / elapsed if elapsed else 0)

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Flush everyone still present as exits
        all_present = currently_present | set(pending_exits.keys())
        for fid in all_present:
            event_logger.log_exit(fid, face_crop=face_last_crop.get(fid))
            update_face_last_seen(db_path, fid)

        final = visitor_counter.get_count()
        logger.info("=== Done | unique=%d | line_in=%d | line_out=%d ===",
                    final, line_counter.entry_count, line_counter.exit_count)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    # --reset must happen BEFORE setup_logging() opens events.log,
    # otherwise Windows locks the file and os.remove() raises PermissionError.
    if args.reset:
        _reset_session(cfg)

    # Now set up logging (creates a fresh events.log after any reset)
    setup_logging(cfg["log_dir"])

    if args.source:
        cfg["video_source"] = args.source

    if args.headless:
        cfg["show_preview"] = False

    # Optional: start dashboard in background thread
    if cfg.get("enable_dashboard"):
        try:
            from dashboard.app import create_app
            import threading
            app = create_app(cfg)
            t = threading.Thread(
                target=lambda: app.run(
                    host="0.0.0.0",
                    port=cfg["dashboard_port"],
                    debug=False,
                    use_reloader=False,
                ),
                daemon=True,
            )
            t.start()
            logger.info("Dashboard running at http://localhost:%d", cfg["dashboard_port"])
        except Exception as e:
            logger.warning("Dashboard could not start: %s", e)

    run(cfg, headless=args.headless)
