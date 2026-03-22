"""
face_recognizer.py - InsightFace ArcFace embedding generation and matching.

Root cause of duplicate IDs (fixed here):
  - InsightFace.app.get() runs its OWN face detector on the input image.
    When we pass a tight YOLO crop, InsightFace's detector often fails to
    locate the face → returns [] → embedding is None → system registers a
    NEW face every time the same person appears.
  Fix: pass a padded, upscaled version of the crop so InsightFace's
       internal detector always has enough context and resolution.

  - The default threshold 0.45 was too strict for ArcFace cosine similarity.
    Same-person similarity across frames is typically 0.25–0.55.
  Fix: lowered default to 0.30.
"""

import logging
import uuid
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalised vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _pad_and_upscale(img: np.ndarray, pad_ratio: float = 0.4,
                     min_side: int = 160) -> np.ndarray:
    """
    Add border padding and ensure the image is at least `min_side` pixels
    on the shorter side.  Both operations give InsightFace's internal detector
    enough context and resolution to reliably locate the face.
    """
    h, w = img.shape[:2]

    # 1. Upscale if too small
    if min(h, w) < min_side:
        scale = min_side / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]

    # 2. Add padding (replicate border so the face detector isn't confused
    #    by an abrupt black border)
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)
    img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                             cv2.BORDER_REPLICATE)
    return img


class FaceRecognizer:
    """
    Generates ArcFace embeddings via InsightFace and matches them against
    a gallery of known faces stored in memory (mirroring the database).

    Args:
        model_name: InsightFace model pack name ('buffalo_sc' or 'buffalo_l').
        providers: ONNX Runtime execution providers list.
        threshold: Cosine similarity threshold for a positive match.
                   0.30 works well for ArcFace across varying angles/lighting.
    """

    def __init__(
        self,
        model_name: str = "buffalo_sc",
        providers: Optional[List[str]] = None,
        threshold: float = 0.30,
    ) -> None:
        self.threshold = threshold
        self._gallery: List[Tuple[str, np.ndarray]] = []

        if providers is None:
            providers = ["CPUExecutionProvider"]

        logger.info("Loading InsightFace model '%s' (threshold=%.2f)", model_name, threshold)
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(name=model_name, providers=providers)
            # ctx_id=-1 forces CPU; det_size=640 handles higher-res frames better
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("InsightFace model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load InsightFace: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Gallery management
    # ------------------------------------------------------------------

    def load_gallery(self, entries: List[Tuple[str, np.ndarray]]) -> None:
        """Replace in-memory gallery with entries from the database."""
        self._gallery = list(entries)
        logger.info("Gallery loaded: %d known faces.", len(self._gallery))

    def add_to_gallery(self, face_id: str, embedding: np.ndarray) -> None:
        """Add a single embedding to the in-memory gallery."""
        self._gallery.append((face_id, embedding))

    # ------------------------------------------------------------------
    # Embedding generation  (THE KEY FIX)
    # ------------------------------------------------------------------

    def get_embedding(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Fallback: generate embedding from a face crop (padded for reliability)."""
        if face_bgr is None or face_bgr.size == 0:
            return None
        h, w = face_bgr.shape[:2]
        if h < 20 or w < 20:
            return None
        img = _pad_and_upscale(face_bgr, pad_ratio=0.6, min_side=200)
        try:
            faces = self._app.get(img)
        except Exception as exc:
            logger.warning("InsightFace get_embedding error: %s", exc)
            return None
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return largest.normed_embedding.astype(np.float32)

    def get_embeddings_from_frame(
        self, frame_bgr: np.ndarray
    ) -> list:
        """
        PRIMARY recognition method: run InsightFace on the full frame.

        This is far more reliable than cropping because InsightFace's internal
        RetinaFace detector is designed to operate on full images.

        Returns:
            List of ((x1, y1, x2, y2), embedding_array) tuples for every
            detected face in the frame, sorted largest-face-first.
        """
        try:
            faces = self._app.get(frame_bgr)
        except Exception as exc:
            logger.warning("InsightFace full-frame error: %s", exc)
            return []

        results = []
        for face in faces:
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            emb = face.normed_embedding.astype(np.float32)
            area = (x2 - x1) * (y2 - y1)
            results.append(((x1, y1, x2, y2), emb, area))

        # Largest face first
        results.sort(key=lambda r: r[2], reverse=True)
        return [(bbox, emb) for bbox, emb, _ in results]

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, embedding: np.ndarray) -> tuple:
        """
        Find the best matching face_id in the gallery.

        Returns:
            (face_id, score) — face_id is None if best score < threshold.
        """
        if not self._gallery:
            return None, 0.0

        best_id = None
        best_score = -1.0

        for fid, stored_emb in self._gallery:
            score = _cosine_similarity(embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_id = fid

        if best_score >= self.threshold:
            logger.debug("Match: %s  score=%.3f", best_id, best_score)
            return best_id, best_score

        logger.debug("No match: best=%.3f < threshold=%.3f", best_score, self.threshold)
        return None, best_score

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def new_face_id() -> str:
        """Generate a new unique face identifier (UUID4)."""
        return str(uuid.uuid4())

