# Intelligent Face Tracker with Auto-Registration and Visitor Counting

> **This project is a part of a hackathon run by [https://katomaran.com](https://katomaran.com)**

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Compute Load Estimate](#compute-load-estimate)
5. [Setup Instructions](#setup-instructions)
6. [Sample `config.json`](#sample-configjson)
7. [Running the System](#running-the-system)
8. [Dashboard](#dashboard)
9. [Output Structure](#output-structure)
10. [Assumptions](#assumptions)
11. [AI Planning Document](#ai-planning-document)

---

## Overview

An end-to-end real-time face detection, recognition, tracking, and visitor counting system built with:

| Component | Technology |
|---|---|
| Face Detection | YOLOv8n-face (ultralytics) |
| Face Recognition | InsightFace ArcFace (`buffalo_sc`) |
| Tracking | ByteTrack (Kalman Filter + Hungarian matching — pure Python) |
| Database | SQLite |
| Logging | Python `logging` → `events.log` + JPEG images |
| Dashboard | Flask + Server-Sent Events |
| Configuration | `config.json` |

---

## Architecture

```
Video File / RTSP Stream
         │
         ▼
┌────────────────────┐   every N frames (detection_skip_frames)
│   Face Detector    │  ◄────────────────────────────────────────┐
│   (YOLOv8n-face)   │                                           │
└────────────────────┘                                           │
         │  [x1,y1,x2,y2, conf] list                            │
         ▼                                                       │
┌────────────────────┐                                           │
│    ByteTracker     │  Kalman predict + Hungarian IoU match     │
│  (tracker.py)      │  → track_id per bounding box             │
└────────────────────┘                                           │
         │  active tracks                                        │
         ▼                                                       │
┌────────────────────┐                                           │
│  Face Recognizer   │  InsightFace ArcFace 512-d embedding     │
│ (face_recognizer)  │  Cosine similarity vs. gallery           │
└────────────────────┘                                           │
         │  face_id (known) or NEW                               │
         ▼                                                       │
┌────────────────────┐   ┌─────────────────────┐                │
│  Visitor Counter   │   │   SQLite Database   │                │
│ (visitor_counter)  │──▶│  faces + events     │                │
└────────────────────┘   └─────────────────────┘                │
         │                                                       │
         ▼                                                       │
┌────────────────────┐   ┌─────────────────────┐                │
│   Event Logger     │──▶│  logs/               │                │
│   (logger.py)      │   │  events.log          │                │
└────────────────────┘   │  entries/YYYY-MM-DD/ │                │
         │               │  exits/YYYY-MM-DD/   │                │
         ▼               └─────────────────────┘                │
┌────────────────────┐                                           │
│  Flask Dashboard   │  http://localhost:5000                   │
│  (dashboard/app)   │  SSE live count + events table           │
└────────────────────┘
```

---

## Features

1. **Real-time face detection** — YOLOv8n-face with configurable frame-skip to reduce CPU load.
2. **Auto-registration** — First-time faces are assigned a UUID, embedded, and stored in both the database and the local filesystem.
3. **Face recognition** — ArcFace (InsightFace) generates 512-d embeddings; cosine similarity matching re-identifies known faces.
4. **ByteTrack tracking** — Multi-face tracking across frames using Kalman-filter prediction + two-round Hungarian matching (high-conf / low-conf detections separately).
5. **Entry / exit logging** — Exactly one entry and one exit event per appearance with a timestamped JPEG crop.
6. **Unique visitor count** — Monotonically increasing; re-identification does not increment the count.
7. **SQLite database** — Persists `faces` and `events` tables; survives restarts.
8. **Mandatory `events.log`** — Human-readable log of every registration, recognition, tracking update, entry, and exit.
9. **Flask dashboard** — Optional browser UI with live SSE counter, entry/exit stats, and recent-events table.
10. **RTSP-ready** — Set `video_source` in `config.json` to an `rtsp://` URL to switch from file to live stream.

---

## Compute Load Estimate

| Component | CPU Load | GPU | Notes |
|---|---|---|---|
| YOLOv8n-face | 25–40% | 0 | CPUExecutionProvider |
| InsightFace (`buffalo_sc`) | 20–30% | 0 | Lightweight CPU model |
| ByteTrack | ~5% | 0 | Pure Python / NumPy |
| SQLite I/O | <2% | 0 | Disk-bound |
| OpenCV display | 5–10% | 0 | CPU decode |
| **Total (CPU only)** | **~55–80%** | **0** | ~10–15 FPS on modern CPU |

---

## Setup Instructions

### 1. Prerequisites

- Python 3.9 – 3.11 (recommended: 3.10)
- Windows / macOS / Linux

### 2. Clone / Navigate to the project

```bash
cd e:\katamoran\face_tracker
```

### 3. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: The first run will automatically download YOLOv8n-face weights and InsightFace model files (~200 MB total). Ensure you have an internet connection.

### 5. Download the sample video

```bash
python download_video.py
```

This downloads the video from Google Drive into `data/sample_video.mp4`.  
Alternatively, place any video file at `data/sample_video.mp4` manually.

---

## Sample `config.json`

```json
{
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
  "enable_dashboard": true,
  "show_preview": true,
  "save_output_video": false,
  "output_video_path": "data/output.avi",
  "insightface_model": "buffalo_sc",
  "insightface_provider": "CPUExecutionProvider"
}
```

| Key | Description |
|---|---|
| `detection_skip_frames` | Run YOLO every N frames; reuse last boxes in between |
| `similarity_threshold` | Cosine similarity cutoff for face re-identification |
| `bytetrack_track_thresh` | High-confidence threshold for ByteTrack first round |
| `bytetrack_track_buffer` | Frames to hold a lost track before removing it |
| `enable_dashboard` | Start Flask dashboard at `dashboard_port` |
| `insightface_provider` | `"CPUExecutionProvider"` or `"CUDAExecutionProvider"` |

---

## Running the System

```bash
# Default (uses config.json)
python main.py

# Custom video file
python main.py --source data/my_video.mp4

# Live RTSP stream
python main.py --source rtsp://192.168.1.100/stream

# Headless (no preview window — for servers)
python main.py --headless

# Custom config path
python main.py --config path/to/config.json
```

Press **Q** in the preview window to stop.

---

## Dashboard

When `enable_dashboard` is `true` in `config.json`, open your browser at:

```
http://localhost:5000
```

Features:
- **Live visitor counter** — updates every second via Server-Sent Events.
- **Stats sidebar** — total events, entries, exits.
- **Events table** — last 50 events, auto-refreshed every 5 seconds.

---

## Output Structure

```
face_tracker/
├── data/
│   ├── face_db.sqlite          # SQLite database
│   └── sample_video.mp4        # Input video
├── logs/
│   ├── events.log              # Mandatory text log
│   ├── entries/
│   │   └── 2026-03-21/
│   │       ├── abc123_2026-03-21T10-05-01.jpg
│   │       └── …
│   └── exits/
│       └── 2026-03-21/
│           ├── abc123_2026-03-21T10-05-30.jpg
│           └── …
```

### `events.log` sample

```
2026-03-21 10:05:01 | INFO     | ENTRY  | face_id=abc12345-... | ts=2026-03-21T10:05:01 | image=logs/entries/2026-03-21/abc12345_....jpg
2026-03-21 10:05:10 | INFO     | RECOGNITION | face_id=abc12345-... | score=0.8821
2026-03-21 10:05:30 | INFO     | EXIT   | face_id=abc12345-... | ts=2026-03-21T10:05:30 | image=logs/exits/2026-03-21/abc12345_....jpg
```

### Database tables

**`faces`**
| face_id | first_seen | last_seen | embedding (BLOB) | thumbnail_path |
|---|---|---|---|---|

**`events`**
| event_id | face_id | event_type | timestamp | image_path |
|---|---|---|---|---|

---

## Assumptions

1. **Single camera / single video file** — the system is designed for one input stream at a time.
2. **CPU-only** — all models use `CPUExecutionProvider`; no CUDA required.
3. **Frontal or near-frontal faces** — side-profile faces may not generate reliable embeddings.
4. **One face per person** — each unique UUID corresponds to one person; twins/lookalikes will share an ID if similarity exceeds the threshold.
5. **`similarity_threshold = 0.45`** — tunable; lower values are more permissive (more IDs merged), higher values are stricter (more IDs split).
6. **Entry = first time a new track is recognised**; **Exit = track removed by ByteTracker** (disappeared for `track_buffer` frames).
7. **Video file is placed at `data/sample_video.mp4`**; RTSP URL can be set in `config.json` for live deployment.
8. **`buffalo_sc` model** — chosen for CPU speed; swap to `buffalo_l` for higher accuracy at the cost of ~2× slower inference.

---

## AI Planning Document

### Planning Steps Followed

1. **Requirement analysis** — parsed the hackathon brief, identified all mandatory modules, constraints (no `face_recognition` library), and bonus items (frontend).
2. **Tech stack selection**
   - YOLO: `ultralytics` (easiest API, auto-downloads weights)
   - Recognition: InsightFace `buffalo_sc` (CPU-friendly ArcFace)
   - Tracker: ByteTrack (state-of-the-art, low computational cost)
   - DB: SQLite (zero config, portable)
   - Frontend: Flask + SSE (lightweight, no JS framework needed)
3. **Architecture design** — defined module boundaries and interfaces before writing any code; callbacks (`on_enter` / `on_exit`) decouple the tracker from the logger.
4. **Compute estimation** — profiled each component's expected CPU load on a modern laptop CPU to confirm the pipeline is viable without a GPU.
5. **Implementation** — modules written bottom-up: database → recogniser → detector → tracker → logger → orchestrator → dashboard.
6. **Configuration-first** — every tunable parameter surfaced in `config.json` so the system can be adjusted without code changes.

### Prompts Used with AI Tools
- *"Design a modular Python face tracking pipeline using YOLOv8, InsightFace, and ByteTrack with SQLite logging."*
- *"Implement ByteTrack from scratch in pure Python with Kalman filter and Hungarian matching — no GPU, no C extensions."*
- *"Write a Flask dashboard with Server-Sent Events to stream a live integer counter from SQLite."*
- *"What cosine similarity threshold is appropriate for ArcFace 512-d embeddings for visitor de-duplication?"*

---

> **Demo video** https://www.youtube.com/watch?v=m57ooCX497I

*
