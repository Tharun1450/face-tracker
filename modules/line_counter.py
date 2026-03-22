"""
line_counter.py - Virtual line-crossing entry/exit counter.

Approach (same as mahdi-marjani/object-tracking-entry-exit and ultralytics
ObjectCounter):

  - A single virtual line divides the frame into two halves.
  - Each tracked object's centre is checked against the line every frame.
  - When the centre crosses from one side to the other, an ENTRY or EXIT
    event fires depending on the direction of crossing.

Direction convention (configurable):
  - "down"  → crossing from top-half to bottom-half  = ENTRY
  - "up"    → crossing from bottom-half to top-half  = EXIT
  - "right" → crossing from left-half to right-half  = ENTRY
  - "left"  → crossing from right-half to left-half  = EXIT

The line is defined by two points and can be horizontal or diagonal.
"""

import logging
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _sign(value: float) -> int:
    """Return +1, 0 or -1."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _side_of_line(px: float, py: float,
                  x1: float, y1: float,
                  x2: float, y2: float) -> int:
    """
    Return which side of the directed line (x1,y1)→(x2,y2) point (px,py) is on.
    Uses the cross-product sign:
      > 0  → left side
      < 0  → right side
      = 0  → on the line
    """
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return _sign(cross)


class LineCrossingCounter:
    """
    Detects when tracked objects cross a user-defined virtual line and
    fires entry/exit callbacks.

    Args:
        line_start: (x, y) start point of the line in pixel coords.
        line_end:   (x, y) end point of the line in pixel coords.
        entry_side: Which cross-product sign means ENTRY (+1 or -1).
                    Default +1 means crossing from right→left of the
                    directed line triggers ENTRY.
        on_entry: Callback(track_id, face_id, bbox).
        on_exit:  Callback(track_id, face_id, bbox).
    """

    def __init__(
        self,
        line_start: Tuple[int, int],
        line_end: Tuple[int, int],
        entry_side: int = 1,          # +1 or -1
        on_entry: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
    ) -> None:
        self.x1, self.y1 = line_start
        self.x2, self.y2 = line_end
        self.entry_side = entry_side
        self.on_entry = on_entry or (lambda tid, fid, bbox: None)
        self.on_exit = on_exit or (lambda tid, fid, bbox: None)

        # track_id → last known side (-1, 0, +1)
        self._prev_side: Dict[int, int] = {}
        self.entry_count = 0
        self.exit_count = 0

    def update(self, track_id: int, face_id: Optional[str],
               bbox: Tuple[int, int, int, int, float]) -> None:
        """
        Call this every frame for every active track.

        Args:
            track_id: ByteTrack track identifier.
            face_id:  Recognised face_id or None.
            bbox:     (x1, y1, x2, y2, conf) bounding box.
        """
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2          # use bottom centre for people

        side = _side_of_line(cx, cy, self.x1, self.y1, self.x2, self.y2)

        prev = self._prev_side.get(track_id)
        self._prev_side[track_id] = side

        if prev is None or prev == 0 or side == 0 or prev == side:
            return   # no crossing yet / on the line / same side

        # A crossing occurred!
        if side == self.entry_side:
            self.entry_count += 1
            logger.info("LINE CROSS → ENTRY | track=%d face=%s", track_id, face_id)
            self.on_entry(track_id, face_id, bbox)
        else:
            self.exit_count += 1
            logger.info("LINE CROSS → EXIT  | track=%d face=%s", track_id, face_id)
            self.on_exit(track_id, face_id, bbox)

    def remove_track(self, track_id: int) -> None:
        """Call when ByteTrack removes a track to free memory."""
        self._prev_side.pop(track_id, None)

    def draw(self, frame, color_entry=(0, 255, 0), color_exit=(0, 0, 255),
             thickness=3) -> None:
        """Draw the counting line on a frame (in-place)."""
        import cv2
        # Draw the line
        cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2),
                 (255, 255, 0), thickness)
        # Labels
        mid_x = (self.x1 + self.x2) // 2
        mid_y = (self.y1 + self.y2) // 2
        cv2.putText(frame, f"IN:{self.entry_count}",
                    (mid_x - 60, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_entry, 2)
        cv2.putText(frame, f"OUT:{self.exit_count}",
                    (mid_x + 10, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_exit, 2)
