"""
tracker.py - ByteTrack-inspired multi-object tracker for face bounding boxes.

ByteTrack overview:
  - High-confidence detections are matched to existing tracks via IoU.
  - Low-confidence detections are used as a second round to recover lost tracks.
  - Unmatched tracks are kept for `track_buffer` frames before being removed.
  - Each track gets a unique integer track_id.

Callbacks `on_enter` and `on_exit` are fired when tracks are created/lost,
allowing the pipeline to trigger logging and visitor counting.

Reference: ByteTrack (Zhang et al., 2022) – self-contained pure-Python/NumPy
implementation; no extra C extension required.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# Type aliases
BBox = Tuple[int, int, int, int, float]  # x1, y1, x2, y2, conf
OnEventCallback = Callable[[int, BBox], None]  # track_id, bbox


# ---------------------------------------------------------------------------
# Kalman Filter (lightweight 4-state: cx, cy, w, h and their velocities)
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """
    A simple Kalman filter for tracking a single bounding box.
    State: [cx, cy, w, h, vx, vy, vw, vh]
    """

    count = 0  # class-level counter for assigning track IDs

    def __init__(self, bbox: BBox) -> None:
        KalmanBoxTracker.count += 1
        self.track_id: int = KalmanBoxTracker.count
        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 0
        self.time_since_update: int = 0
        self.face_id: Optional[str] = None  # assigned after recognition

        # State and covariance matrices
        self._x = self._bbox_to_state(bbox)  # (8,)
        self._P = np.eye(8) * 10.0
        self._P[4:, 4:] *= 100.0  # high velocity uncertainty at start

        # Transition matrix (constant velocity model)
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = 1.0

        # Measurement matrix (observe only cx, cy, w, h)
        self._H = np.eye(4, 8)

        # Process noise
        self._Q = np.eye(8)
        self._Q[4:, 4:] *= 0.01

        # Measurement noise
        self._R = np.eye(4) * 1.0

        self._last_bbox = bbox

    # ---- Kalman predict ----
    def predict(self) -> BBox:
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.age += 1
        self.time_since_update += 1
        return self._state_to_bbox(self._x)

    # ---- Kalman update ----
    def update(self, bbox: BBox) -> None:
        z = self._bbox_to_state(bbox)[:4]  # observed cx, cy, w, h
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(8) - K @ self._H) @ self._P
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self._last_bbox = bbox

    def get_state(self) -> BBox:
        return self._state_to_bbox(self._x)

    # ---- Helpers ----
    @staticmethod
    def _bbox_to_state(bbox: BBox) -> np.ndarray:
        x1, y1, x2, y2, conf = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0])

    @staticmethod
    def _state_to_bbox(state: np.ndarray) -> BBox:
        cx, cy, w, h = state[:4]
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return (x1, y1, x2, y2, 1.0)


# ---------------------------------------------------------------------------
# IoU utility
# ---------------------------------------------------------------------------

def _iou(b1: BBox, b2: BBox) -> float:
    """Compute Intersection over Union between two bboxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def _iou_matrix(tracks: List[BBox], dets: List[BBox]) -> np.ndarray:
    """Return IoU matrix of shape (n_tracks, n_dets)."""
    mat = np.zeros((len(tracks), len(dets)), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(dets):
            mat[i, j] = _iou(t, d)
    return mat


def _match_hungarian(cost_matrix: np.ndarray, threshold: float):
    """
    Run Hungarian matching on a cost matrix (higher = better).
    Returns (matched_pairs, unmatched_track_idxs, unmatched_det_idxs).
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    matched, unmatched_t, unmatched_d = [], [], []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= threshold:
            matched.append((r, c))
        else:
            unmatched_t.append(r)
            unmatched_d.append(c)

    all_t = set(range(cost_matrix.shape[0]))
    all_d = set(range(cost_matrix.shape[1]))
    matched_t = {m[0] for m in matched}
    matched_d = {m[1] for m in matched}
    unmatched_t += list(all_t - matched_t - set(unmatched_t))
    unmatched_d += list(all_d - matched_d - set(unmatched_d))
    return matched, unmatched_t, unmatched_d


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """
    ByteTrack multi-object tracker adapted for face bounding boxes.

    Args:
        track_thresh: High-confidence threshold for first-round matching.
        track_buffer: Number of frames to keep a lost track before removing.
        match_thresh: Minimum IoU score to accept a match.
        on_enter: Callback(track_id, bbox) fired when a new track is created.
        on_exit: Callback(track_id, last_bbox) fired when a track is removed.
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        on_enter: Optional[OnEventCallback] = None,
        on_exit: Optional[OnEventCallback] = None,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.on_enter = on_enter
        self.on_exit = on_exit

        self._active_tracks: List[KalmanBoxTracker] = []
        self._lost_tracks: List[KalmanBoxTracker] = []

        # face_id → track_id mapping (populated by the pipeline)
        self._face_to_track: Dict[str, int] = {}
        self._track_to_face: Dict[int, str] = {}

        KalmanBoxTracker.count = 0  # reset for fresh session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: List[BBox]) -> List[KalmanBoxTracker]:
        """
        Update tracker with new detections and return active track list.

        Args:
            detections: List of (x1, y1, x2, y2, confidence) from detector.

        Returns:
            List of active KalmanBoxTracker objects (each has .track_id,
            .face_id, and .get_state()).
        """
        # Predict positions for all active and lost tracks
        predicted_active = [(t, t.predict()) for t in self._active_tracks]
        predicted_lost = [(t, t.predict()) for t in self._lost_tracks]

        # Split detections into high / low confidence
        high_dets = [d for d in detections if d[4] >= self.track_thresh]
        low_dets = [d for d in detections if d[4] < self.track_thresh]

        # --- Round 1: match high-conf dets to active tracks ---
        matched1, unmatched_tracks, unmatched_dets1 = self._hungarian_match(
            predicted_active, high_dets, self.match_thresh
        )
        for t_idx, d_idx in matched1:
            predicted_active[t_idx][0].update(high_dets[d_idx])

        # --- Round 2: match low-conf dets to unmatched active tracks ---
        remaining_tracks = [predicted_active[i] for i in unmatched_tracks]
        matched2, still_unmatched_tracks, unmatched_dets2 = self._hungarian_match(
            remaining_tracks, low_dets, 0.5
        )
        for t_idx, d_idx in matched2:
            remaining_tracks[t_idx][0].update(low_dets[d_idx])

        # Tracks that had no match → move to lost
        lost_indices = {remaining_tracks[i][0].track_id for i in still_unmatched_tracks}

        # --- Round 3: match unmatched high-conf dets to lost tracks ---
        remaining_high = [high_dets[i] for i in unmatched_dets1]
        matched3, _, unmatched_new_dets = self._hungarian_match(
            predicted_lost, remaining_high, 0.5
        )
        for t_idx, d_idx in matched3:
            track = predicted_lost[t_idx][0]
            track.update(remaining_high[d_idx])
            self._lost_tracks.remove(track)
            self._active_tracks.append(track)

        # --- Create new tracks for unmatched high-conf detections ---
        for i in unmatched_new_dets:
            new_track = KalmanBoxTracker(remaining_high[i])
            self._active_tracks.append(new_track)
            logger.debug("New track created: id=%d", new_track.track_id)
            if self.on_enter:
                self.on_enter(new_track.track_id, new_track.get_state())

        # --- Move lost tracks to _lost_tracks ---
        newly_lost = [t for t in self._active_tracks if t.track_id in lost_indices]
        for t in newly_lost:
            t.hit_streak = 0
            self._active_tracks.remove(t)
            self._lost_tracks.append(t)

        # --- Remove tracks that have been lost too long ---
        removed = [t for t in self._lost_tracks if t.time_since_update > self.track_buffer]
        for t in removed:
            self._lost_tracks.remove(t)
            logger.debug("Track removed (lost too long): id=%d", t.track_id)
            if self.on_exit:
                self.on_exit(t.track_id, t.get_state())

        return self._active_tracks

    def assign_face_id(self, track_id: int, face_id: str) -> None:
        """Link a recognised face_id to a track_id."""
        self._track_to_face[track_id] = face_id
        self._face_to_track[face_id] = track_id
        for t in self._active_tracks:
            if t.track_id == track_id:
                t.face_id = face_id
                break

    def get_face_id(self, track_id: int) -> Optional[str]:
        """Return the face_id mapped to a track_id, or None."""
        return self._track_to_face.get(track_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hungarian_match(
        predicted_tracks: List[Tuple[KalmanBoxTracker, BBox]],
        dets: List[BBox],
        threshold: float,
    ):
        if not predicted_tracks or not dets:
            return [], list(range(len(predicted_tracks))), list(range(len(dets)))
        track_boxes = [p[1] for p in predicted_tracks]
        iou_mat = _iou_matrix(track_boxes, dets)
        return _match_hungarian(iou_mat, threshold)
