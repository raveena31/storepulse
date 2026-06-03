"""
pipeline/edge_cases.py
──────────────────────
Detection-layer edge case guards (EC-1 through EC-9, EC-19, EC-20).

Every function here is a pure filter — it takes a detection or track state
and returns a corrected/filtered result with no side effects.
All functions are independently testable.
"""
from __future__ import annotations

import math
from typing import Optional


# ── EC-1  GROUP ENTRY ─────────────────────────────────────────────────────────

class EntryExitCounter:
    """
    Counts ENTRY / EXIT events per individual track using line-crossing.

    EC-1 guarantee: one event per track per direction (debounced).
    Hysteresis prevents oscillation around the line.
    """

    def __init__(self, line_y: float, hysteresis: int = 8):
        self.line_y = line_y
        self.hysteresis = hysteresis
        # track_id → last direction counted ("ENTRY" | "EXIT" | None)
        self._last_dir: dict[int, Optional[str]] = {}
        self._prev_y: dict[int, float] = {}

    def update(self, track_id: int, feet_y: float) -> Optional[str]:
        """
        Call once per frame per track.
        Returns "ENTRY", "EXIT", or None.

        ENTRY = crossing downward (outside → inside store)
        EXIT  = crossing upward  (inside → outside store)

        Hysteresis: require the feet to clear line_y ± hysteresis to
        prevent rapid toggling when a person hovers near the line.
        """
        prev = self._prev_y.get(track_id)
        self._prev_y[track_id] = feet_y

        if prev is None:
            return None

        last = self._last_dir.get(track_id)

        # Crossed downward (entry into store)
        if prev < self.line_y - self.hysteresis and feet_y >= self.line_y:
            if last != "ENTRY":
                self._last_dir[track_id] = "ENTRY"
                return "ENTRY"

        # Crossed upward (exit from store)
        elif prev > self.line_y + self.hysteresis and feet_y <= self.line_y:
            if last != "EXIT":
                self._last_dir[track_id] = "EXIT"
                return "EXIT"

        return None


# ── EC-2  TAILGATING / MERGED BOX ─────────────────────────────────────────────

def split_merged_box(
    box: list[float],
    single_person_w: float,
    second_box: Optional[list[float]] = None,
    iou_thresh: float = 0.55,
    width_ratio: float = 1.8,
) -> list[list[float]]:
    """
    Split a merged detection box into two individual person boxes.

    Triggers if:
      - box width > width_ratio × single_person_w  (two people merged into one blob)
      - OR IoU with a nearby suppressed box > iou_thresh (tailgating overlap)

    Returns: [box] unchanged if no split needed, or [box_left, box_right] if split.
    """
    x1, y1, x2, y2 = box
    box_w = x2 - x1

    should_split = False

    if box_w > width_ratio * single_person_w:
        should_split = True

    if not should_split and second_box is not None:
        iou = _iou(box, second_box)
        if iou > iou_thresh:
            should_split = True

    if not should_split:
        return [box]

    # Split evenly down the middle
    mid_x = (x1 + x2) / 2
    left  = [x1, y1, mid_x, y2]
    right = [mid_x, y1, x2, y2]
    return [left, right]


def _iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


# ── EC-3  DOORWAY LOITER ──────────────────────────────────────────────────────

def net_direction(y_history: list[float], line_y: float, min_net: float = 25.0) -> bool:
    """
    Returns True only if the y_history shows a genuine crossing — net
    displacement across line_y exceeds min_net pixels.

    Prevents people who hover/loiter in the doorway from generating phantom
    ENTRY/EXIT events due to small oscillations.

    Args:
        y_history: list of feet_y positions for this track (chronological)
        line_y:    the crossing threshold
        min_net:   minimum net pixel displacement required to count as a crossing

    Returns True if crossing is genuine, False if it's just loitering.
    """
    if len(y_history) < 2:
        return False
    start_y = y_history[0]
    end_y   = y_history[-1]
    net = abs(end_y - start_y)
    crosses = (start_y < line_y) != (end_y < line_y)  # actually crossed
    return crosses and net >= min_net


# ── EC-5  DOOR SWING PHANTOMS ─────────────────────────────────────────────────

def crossing_is_real(track_age_frames: int, min_age: int = 4) -> bool:
    """
    Gate crossings by track age to suppress door-swing reflection phantoms.

    Phantoms created by a swinging door die within 1-2 frames.
    Real people persist for ≥ min_age frames before crossing.

    Returns True if the track is old enough to be a real person.
    """
    return track_age_frames >= min_age


# ── EC-7  MANNEQUIN / STANDEE ─────────────────────────────────────────────────

def is_static_prop(
    positions: list[tuple[float, float]],
    px_thresh: float = 6.0,
    min_records: int = 30,
) -> bool:
    """
    Detect mannequins / standees / posters falsely classified as persons.

    A real person moves; a static prop stays within px_thresh pixels over
    min_records frames.

    Args:
        positions:   list of (x, y) centroid positions for this track
        px_thresh:   maximum total movement to be considered a prop (pixels)
        min_records: minimum observations required before flagging

    Returns True if the track looks like a static prop.
    """
    if len(positions) < min_records:
        return False  # not enough data to decide yet

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    movement = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
    return movement < px_thresh


# ── EC-8  GLASS REFLECTIONS ───────────────────────────────────────────────────

def drop_reflection(
    bbox: list[float],
    glass_masks: list[list[list[float]]],
) -> bool:
    """
    Suppress detections whose feet point falls inside a glass/mirror polygon.

    Feet point = bottom-centre of bounding box.
    Glass mask polygons are loaded from config/store_ST1008.yaml.

    Returns True if this detection should be dropped (it's a reflection).
    """
    x1, y1, x2, y2 = bbox
    feet_x = (x1 + x2) / 2
    feet_y = y2

    for polygon in glass_masks:
        if _point_in_polygon(feet_x, feet_y, polygon):
            return True
    return False


def _point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


# ── EC-9  SHADOWS ─────────────────────────────────────────────────────────────

def looks_like_shadow(
    bbox: list[float],
    conf: float,
    ar_min: float = 0.25,
    ar_max: float = 0.65,
    conf_floor: float = 0.20,
) -> bool:
    """
    Detect shadow detections: very wide, flat boxes with low confidence.

    Shadows from artificial store lighting appear as:
      - Unusually low aspect ratio (wide and flat: h/w < 0.65)
      - Low detection confidence (< conf_floor)

    Returns True if this detection is likely a shadow.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    ar = h / w  # aspect ratio: < 1 means wider than tall
    return ar_min <= ar <= ar_max and conf < conf_floor


# ── EC-19  BOUNDARY EXIT ──────────────────────────────────────────────────────

def is_boundary_exit(
    bbox: list[float],
    frame_w: float,
    frame_h: float,
    pad: int = 12,
) -> bool:
    """
    Detect when a track exits the frame at the image boundary rather than
    crossing the entry line.

    A person walking behind a display unit may leave through the frame edge.
    We should park them in LostTrackBuffer instead of emitting EXIT — they
    might reappear on another camera.

    Returns True if any edge of the bbox is within `pad` pixels of the frame edge.
    """
    x1, y1, x2, y2 = bbox
    return (
        x1 <= pad or
        y1 <= pad or
        x2 >= frame_w - pad or
        y2 >= frame_h - pad
    )


# ── EC-20  STICKY STAFF LABEL ─────────────────────────────────────────────────

def sticky_staff(
    track_state: dict,
    tid: int,
    frame_is_staff: bool,
    conf: float,
    lock_conf: float = 0.8,
) -> dict:
    """
    Once a track is confidently classified as staff, lock that label forever.

    Prevents flickering when a staff member briefly looks like a customer
    (e.g. sits down, removes uniform jacket).

    track_state: mutable dict keyed by track_id with sub-keys:
        "is_staff": bool
        "locked":   bool
        "staff_conf": float

    Modifies track_state in place and returns it.
    """
    state = track_state.setdefault(tid, {"is_staff": False, "locked": False, "staff_conf": 0.0})

    if state["locked"]:
        return track_state  # label is locked — never re-evaluate

    if frame_is_staff and conf >= lock_conf:
        state["is_staff"] = True
        state["locked"] = True
        state["staff_conf"] = conf
    elif frame_is_staff:
        state["is_staff"] = True  # tentative (not locked)

    return track_state
