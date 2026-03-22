"""
face_detector.py - YOLOv8-based face detection module.

Uses the yolov8l-face.pt model (Large) fine-tuned for face detection.
The 'large' model has 4x the parameters of 'nano' and dramatically improves
detection of small, tilted, and partially occluded faces.
Weights are downloaded automatically on first run into a local 'models/' folder.

"""

import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

# Local folder where weights are cached
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Use the NANO model — fast and CPU-friendly
YOLO_FACE_MODEL_NAME = "yolov8n-face.pt"

# Multiple candidate download URLs tried in order
YOLO_FACE_URLS = [
    "https://github.com/akanametov/yolo-face/releases/latest/download/yolov8n-face.pt",
    "https://github.com/akanametov/yolo-face/releases/download/v0.0.3/yolov8n-face.pt",
    "https://github.com/akanametov/yolo-face/releases/download/v0.0.2/yolov8n-face.pt",
    "https://github.com/akanametov/yolo-face/releases/download/v0.0.1/yolov8n-face.pt",
    "https://huggingface.co/Ultralytics/assets/resolve/main/yolov8n-face.pt",
]


def _ensure_model(model_name: str = YOLO_FACE_MODEL_NAME) -> str:
    """
    Return the local path to the YOLO face model weights, downloading
    from the first reachable URL if not already present.

    Args:
        model_name: Filename of the .pt weights file.

    Returns:
        Absolute path to the weights file.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    local_path = os.path.join(MODELS_DIR, model_name)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 1_000_000:
        logger.info("YOLO model found locally at: %s", local_path)
        return local_path

    logger.info("Model '%s' not found. Trying download sources...", model_name)

    try:
        import requests
    except ImportError:
        raise RuntimeError(
            "The 'requests' package is required for model download. "
            "Run: pip install requests"
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    last_error = None
    for url in YOLO_FACE_URLS:
        print(f"[FaceDetector] Trying: {url}")
        try:
            resp = requests.get(url, headers=headers, stream=True,
                                timeout=30, allow_redirects=True)
            if resp.status_code != 200:
                print(f"  ✗ HTTP {resp.status_code} — skipping.")
                last_error = f"HTTP {resp.status_code} from {url}"
                continue

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(local_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  Downloading... {pct:.1f}%  ", end="", flush=True)

            print()  # newline after progress

            # Sanity-check: file must be > 1 MB
            if os.path.getsize(local_path) < 1_000_000:
                os.remove(local_path)
                last_error = f"File too small from {url} (likely an error page)"
                print(f"  ✗ {last_error}")
                continue

            logger.info("Model saved to: %s", local_path)
            print(f"  ✓ Saved to: {local_path}")
            return local_path

        except Exception as exc:
            last_error = str(exc)
            print(f"  ✗ Error: {exc}")
            if os.path.exists(local_path):
                os.remove(local_path)
            continue

    # All URLs failed – give user a clear manual-download message
    raise RuntimeError(
        f"\n{'='*60}\n"
        f"Could not auto-download {model_name}.\n"
        f"Last error: {last_error}\n\n"
        f"Please download it MANUALLY:\n"
        f"  1. Open: https://github.com/akanametov/yolo-face/releases\n"
        f"  2. Download {model_name}\n"
        f"  3. Place it at: {local_path}\n"
        f"Then run python main.py again.\n"
        f"{'='*60}"
    )


class FaceDetector:
    """
    Wraps YOLOv8 for face detection with configurable frame-skip logic.

    Args:
        model_path: Filename of the YOLO weights (auto-downloaded if missing).
        min_confidence: Minimum detection confidence to accept a box.
        skip_frames: Detect every N frames; reuse last detections in between.
        device: 'cpu' or 'cuda' — always 'cpu' in our environment.
    """

    def __init__(
        self,
        model_path: str = YOLO_FACE_MODEL_NAME,
        min_confidence: float = 0.5,
        skip_frames: int = 5,
        device: str = "cpu",
    ) -> None:
        self.min_confidence = min_confidence
        self.skip_frames = skip_frames
        self.device = device
        self._frame_count = 0
        self._last_detections: List[Tuple[int, int, int, int, float]] = []

        # Resolve (and if necessary download) the model weights
        resolved_path = _ensure_model(model_path)
        logger.info("Loading YOLO model: %s (device=%s)", resolved_path, device)
        self.model = YOLO(resolved_path)
        logger.info("YOLO model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in a single BGR frame.

        Runs the YOLO model every `skip_frames` frames; returns cached
        detections for intermediate frames.

        Args:
            frame: BGR image as a NumPy array (H x W x 3).

        Returns:
            List of tuples (x1, y1, x2, y2, confidence) with integer pixel
            coordinates clipped to the frame boundaries.
        """
        self._frame_count += 1

        # Only run inference on every N-th frame
        if (self._frame_count - 1) % self.skip_frames != 0:
            return self._last_detections

        results = self.model(
            frame,
            verbose=False,
            device=self.device,
        )

        detections: List[Tuple[int, int, int, int, float]] = []
        h, w = frame.shape[:2]

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.min_confidence:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Clamp to frame boundaries
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w - 1, int(x2))
                y2 = min(h - 1, int(y2))
                # Discard degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append((x1, y1, x2, y2, conf))

        self._last_detections = detections
        return detections

    def track(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, int, float]]:
        """
        Detect AND track faces using Ultralytics' built-in ByteTrack tracker.
        Uses model.track(persist=True) — the same approach as the reference
        repos (mahdi-marjani, dyneth02).

        Args:
            frame: BGR image as a NumPy array (H x W x 3).

        Returns:
            List of tuples (track_id, x1, y1, x2, y2, confidence).
            track_id is a stable integer that persists across frames.
        """
        results = self.model.track(
            frame,
            persist=True,       # keep tracker state between frames
            verbose=False,
            conf=self.min_confidence,
            iou=0.45,
            device=self.device,
            tracker="bytetrack.yaml",  # lightweight ByteTrack
        )

        tracked: List[Tuple[int, int, int, int, int, float]] = []
        h, w = frame.shape[:2]

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if box.id is None:
                    continue  # box not yet assigned a track ID
                track_id = int(box.id[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w - 1, int(x2))
                y2 = min(h - 1, int(y2))
                if x2 > x1 and y2 > y1:
                    tracked.append((track_id, x1, y1, x2, y2, conf))

        return tracked

    def crop_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int, float],
        padding: float = 0.15,
    ) -> Optional[np.ndarray]:
        """
        Crop a face from the frame with optional padding.

        Args:
            frame: Full BGR frame.
            bbox: (x1, y1, x2, y2, conf) bounding box.
            padding: Fractional padding to add around the face box.

        Returns:
            Cropped BGR face image, or None if the crop is empty.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2, _ = bbox
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w - 1, x2 + pad_x)
        cy2 = min(h - 1, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        return crop

    def reset_frame_count(self) -> None:
        """Reset internal frame counter (useful when switching video sources)."""
        self._frame_count = 0
        self._last_detections = []
