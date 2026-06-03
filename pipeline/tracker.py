"""
pipeline/tracker.py
───────────────────
Multi-object tracking using ByteTrack via ultralytics model.track().

The tracking problem:
  YOLOv8 detects people independently in each frame.
  Without tracking, "person at position (500,300) in frame 100" and
  "person at position (520,310) in frame 101" look like different people.
  ByteTrack solves this by assigning a consistent track_id to the same person
  across frames, even through brief occlusions or low-confidence detections.

Why ByteTrack?
  - Pure motion-based tracking — no appearance model (re-ID) needed
  - Works well in retail stores: low crowd density, stable lighting
  - Bundled inside ultralytics — no extra model downloads
  - Paper: https://arxiv.org/abs/2110.06864

Why model.track() instead of BYTETracker() directly?
  - ultralytics 8.4+ changed the BYTETracker constructor signature
  - model.track() is the stable public API — version-safe
  - persist=True tells the tracker to keep state between frame calls
    (without this, track_ids reset every frame and tracking doesn't work)

One CameraTracker per camera:
  Each camera has its own tracker instance.
  This prevents track_ids from CAM_01 colliding with track_ids from CAM_02.
  The visitor_id format is "{store_id}_T{camera_id}_{track_id:04d}" to
  ensure global uniqueness when events from multiple cameras are merged.
"""

from __future__ import annotations

import numpy as np


class CameraTracker:
    """
    Per-camera ByteTrack wrapper.

    Usage:
        tracker = CameraTracker("CAM_01")
        while True:
            ret, frame = cap.read()
            tracks = tracker.track_frame(frame, model)
            for t in tracks:
                print(t["track_id"], t["bbox"], t["confidence"])
    """

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        # Note: no tracker object stored here.
        # ultralytics model.track(persist=True) manages its own internal state.

    def track_frame(
        self,
        frame: np.ndarray,
        model,
        conf_threshold: float = 0.25,
    ) -> list[dict]:
        """
        Run detection + ByteTrack on a single video frame.

        Args:
            frame:          BGR numpy array from cv2.VideoCapture.read()
            model:          Loaded ultralytics YOLO model (with persist=True tracking state)
            conf_threshold: Minimum detection confidence (default 0.25)

        Returns:
            List of tracked person dicts:
            [
                {
                    "track_id":         int,    # stable ID across frames for this camera
                    "bbox":             [x1, y1, x2, y2],
                    "confidence":       float,
                    "class_id":         0,      # always 0 (person)
                    "confidence_band":  str,    # "HIGH" | "MED" | "LOW"
                }
            ]

        Returns [] on any error (never raises) — missing one frame is fine.

        Note on track_id uniqueness:
            track_ids are unique WITHIN one camera's session.
            The emit.py module prefixes them with store_id + camera_id to create
            globally unique visitor_ids when multiple cameras' events are merged.
        """
        try:
            results = model.track(
                frame,
                persist=True,           # maintain track state between calls
                conf=conf_threshold,
                classes=[0],            # persons only (COCO class 0)
                tracker="bytetrack.yaml",  # use ByteTrack algorithm
                verbose=False,          # suppress per-frame console output
            )
        except Exception:
            # Silently return empty on any error — a missed frame is acceptable
            return []

        tracks = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                # box.id is None if the tracker couldn't assign an ID this frame
                if box.id is None:
                    continue
                tid = int(box.id[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])

                if cls != 0:
                    continue  # shouldn't happen since classes=[0], but safety check

                band = "HIGH" if conf >= 0.6 else ("MED" if conf >= 0.4 else "LOW")
                tracks.append({
                    "track_id": tid,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls,
                    "confidence_band": band,
                })

        return tracks
