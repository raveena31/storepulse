"""
app/stream.py
─────────────
MJPEG video stream endpoint.

Reads the CCTV video file for a given camera, runs YOLOv8n detection on each
frame, draws bounding boxes + track IDs + zone overlays, and streams the
annotated frames to the browser as a multipart/x-mixed-replace MJPEG stream.

This is the simplest possible approach for live video in a browser:
  - No WebSockets needed
  - Works in any browser with an <img src="/stream/CAM_01"> tag
  - Browser renders each JPEG frame as it arrives

Performance:
  - Processes every 2nd frame (configurable) to stay real-time on CPU
  - Resizes frame to 960x540 before encoding (half resolution = 4x faster JPEG encode)
  - YOLOv8n is fast enough for ~10fps on a modern CPU

Zone overlay:
  - Each zone polygon from config is drawn as a semi-transparent coloured region
  - Zone name is labelled in the centre of the polygon
  - Only zones owned by this camera are drawn

Bounding box colours:
  GREEN  → confident detection (confidence >= 0.6)
  YELLOW → medium confidence (0.4 – 0.6)
  RED    → low confidence (< 0.4)
  PURPLE → staff (is_staff from CAM_04)
"""

from __future__ import annotations

import os
import time
from typing import Generator, Optional

import cv2
import numpy as np

# Zone polygon colours (BGR) — one per zone, cycling if more zones than colours
ZONE_COLOURS = [
    (255, 180, 0),    # light blue
    (0, 200, 100),    # green
    (200, 0, 200),    # magenta
    (0, 180, 255),    # orange
    (150, 0, 255),    # purple
    (0, 255, 200),    # yellow-green
    (255, 100, 100),  # light blue 2
    (100, 255, 100),  # light green 2
]

# Bounding box colours by confidence (BGR)
COLOUR_HIGH   = (0, 220, 0)     # green
COLOUR_MED    = (0, 200, 220)   # yellow
COLOUR_LOW    = (0, 80, 220)    # red
COLOUR_STAFF  = (220, 60, 220)  # purple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_PATHS = {
    "CAM_01": os.path.join(BASE_DIR, "data", "cctv", "CCTV Footage", "CAM 1.mp4"),
    "CAM_02": os.path.join(BASE_DIR, "data", "cctv", "CCTV Footage", "CAM 2.mp4"),
    "CAM_03": os.path.join(BASE_DIR, "data", "cctv", "CCTV Footage", "CAM 3.mp4"),
    "CAM_04": os.path.join(BASE_DIR, "data", "cctv", "CCTV Footage", "CAM 4.mp4"),
    "CAM_05": os.path.join(BASE_DIR, "data", "cctv", "CCTV Footage", "CAM 5.mp4"),
}

_model = None
_config = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
    return _model


def _get_config():
    global _config
    if _config is None:
        import yaml
        cfg_path = os.path.join(os.getenv("CONFIG_DIR", "config"), "store_ST1008.yaml")
        with open(cfg_path) as f:
            _config = yaml.safe_load(f)
    return _config


def _draw_zone_overlays(frame: np.ndarray, camera_id: str) -> np.ndarray:
    """Draw semi-transparent zone polygons onto the frame."""
    try:
        cfg = _get_config()
        ownership = cfg.get("camera_zone_ownership", {})
        owned = set(ownership.get(camera_id, []))
        zones = cfg.get("zones", {})

        overlay = frame.copy()
        colour_idx = 0

        for zone_id, zone_data in zones.items():
            if zone_id not in owned:
                continue
            polygon = zone_data.get("polygon", [])
            if not polygon:
                continue

            pts = np.array(polygon, dtype=np.int32)
            colour = ZONE_COLOURS[colour_idx % len(ZONE_COLOURS)]
            colour_idx += 1

            # Fill polygon with 25% opacity
            cv2.fillPoly(overlay, [pts], colour)
            # Draw solid border
            cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2)

            # Label in centre of polygon
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            display = zone_data.get("display_name", zone_id)
            cv2.putText(frame, display, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Blend the filled overlay at 25% opacity
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    except Exception:
        pass  # don't crash the stream if config is missing

    return frame


def _draw_entry_line(frame: np.ndarray) -> np.ndarray:
    """Draw the entry/exit line on CAM_03 frames."""
    try:
        cfg = _get_config()
        line_y = cfg.get("entry_line_y", 520)
        h, w = frame.shape[:2]
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(frame, "ENTRY LINE", (10, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    except Exception:
        pass
    return frame


def generate_stream(camera_id: str, skip: int = 1) -> Generator[bytes, None, None]:
    """
    Generator that yields MJPEG frames for a given camera.

    Each yielded value is a complete JPEG frame wrapped in multipart headers.
    The browser's <img> tag updates automatically as new frames arrive.

    Args:
        camera_id: e.g. "CAM_01"
        skip:      Process every Nth frame. 1 = every frame, 2 = every other frame.
    """
    video_path = VIDEO_PATHS.get(camera_id)
    if not video_path or not os.path.exists(video_path):
        # Yield a black "camera not found" frame
        blank = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.putText(blank, f"Video not found: {camera_id}", (80, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 255), 2)
        _, jpeg = cv2.imencode(".jpg", blank)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        return

    cfg = _get_config()
    role = cfg.get("camera_roles", {}).get(camera_id, "product_zone")
    model = _get_model()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0

    # track_id → colour (so same person keeps same colour across frames)
    track_colours: dict[int, tuple] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue

        frame_idx += 1
        if frame_idx % (skip + 1) != 0:
            continue

        # ── Resize to 960×540 for faster processing + smaller stream ──────
        frame = cv2.resize(frame, (960, 540))

        # ── Draw zone overlays ─────────────────────────────────────────────
        frame = _draw_zone_overlays(frame, camera_id)
        if role == "entry_exit":
            frame = _draw_entry_line(frame)

        # ── Run detection + tracking ───────────────────────────────────────
        try:
            results = model.track(
                frame,
                persist=True,
                conf=0.25,
                classes=[0],
                tracker="bytetrack.yaml",
                verbose=False,
            )
        except Exception:
            results = []

        # ── Draw bounding boxes ────────────────────────────────────────────
        for result in (results or []):
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                if box.id is None:
                    continue
                tid = int(box.id[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                # Assign stable colour per track_id
                if tid not in track_colours:
                    # Cycle through a palette of distinct colours
                    palette = [
                        (0, 220, 0), (0, 200, 220), (220, 120, 0),
                        (220, 0, 220), (0, 120, 220), (220, 220, 0),
                        (0, 220, 180), (180, 0, 220),
                    ]
                    track_colours[tid] = palette[tid % len(palette)]

                # Override colour for staff camera
                if role == "staff_only":
                    colour = COLOUR_STAFF
                else:
                    colour = track_colours[tid]

                # Draw filled top bar + bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                label = f"#{tid} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Draw feet point
                feet_x = (x1 + x2) // 2
                cv2.circle(frame, (feet_x, y2), 4, colour, -1)

        # ── Camera role badge ──────────────────────────────────────────────
        role_colours = {
            "product_zone": (0, 180, 0),
            "entry_exit":   (0, 200, 220),
            "staff_only":   (200, 0, 200),
            "billing":      (0, 140, 255),
        }
        badge_colour = role_colours.get(role, (128, 128, 128))
        badge = f"{camera_id}  {role.upper().replace('_', ' ')}"
        cv2.rectangle(frame, (0, 0), (len(badge) * 10 + 10, 28), badge_colour, -1)
        cv2.putText(frame, badge, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ── Detection count ────────────────────────────────────────────────
        n_det = sum(
            len(r.boxes) for r in (results or [])
            if r.boxes is not None and r.boxes.id is not None
        )
        cv2.putText(frame, f"Detected: {n_det}", (6, 520),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # ── Encode as JPEG and yield ───────────────────────────────────────
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )

        # Throttle to roughly real-time (target ~15fps output)
        time.sleep(max(0, (skip + 1) / fps - 0.01))

    cap.release()
