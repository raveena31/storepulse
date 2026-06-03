"""
pipeline/detect.py
──────────────────
Person detection using YOLOv8n (the "nano" — smallest, fastest variant).

Why YOLOv8n?
  - Ships inside the `ultralytics` pip package — no separate weight download step
    (it auto-downloads yolov8n.pt ~6MB on first run)
  - Fast enough for real-time use on CPU (~15fps on modern laptop)
  - Class ID 0 = person in the COCO dataset used for training
  - Accurate enough for overhead/angled CCTV views at 1920×1080

Key design decisions:
  - Low-confidence detections are NEVER filtered out here.
    They are returned with confidence_band="LOW" so downstream analytics
    can flag uncertain data rather than silently drop it.
    (Requirement: "Never suppress low-confidence detections")
  - Model is loaded ONCE as a module-level singleton (_model).
    Loading takes ~1-2 seconds; reloading per frame would be unusable.
  - Only class_id == 0 (person) detections are returned.
    All other COCO classes (cars, furniture, etc.) are filtered here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Module-level model singleton — loaded once, reused for all frames
_model = None


def _get_model(weights: str = "yolov8n.pt"):
    """
    Load YOLOv8n model (lazy — only loads on first call).

    Args:
        weights: Path or model name. "yolov8n.pt" auto-downloads from GitHub
                 to the current directory on first use.

    Returns:
        ultralytics.YOLO model instance.
    """
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(weights)
    return _model


def process_frame(
    frame: np.ndarray,
    model=None,
    conf_threshold: float = 0.25,
) -> list[dict]:
    """
    Run YOLOv8n inference on a single video frame.

    Args:
        frame:          BGR numpy array (as returned by cv2.VideoCapture.read())
        model:          YOLO model instance. If None, uses the module singleton.
        conf_threshold: Minimum confidence to include detection (0.0–1.0).
                        Default 0.25 — low on purpose, we never suppress detections.

    Returns:
        List of detection dicts, filtered to persons only (class_id == 0):
        [
            {
                "bbox":             [x1, y1, x2, y2],  # pixel coordinates
                "confidence":       float,              # AI confidence score
                "class_id":         0,                  # always 0 (person)
                "confidence_band":  str,                # "HIGH" | "MED" | "LOW"
            },
            ...
        ]

    Notes:
        - confidence_band is a human-readable flag for downstream use:
            HIGH  → confidence >= 0.60 (reliable detection)
            MED   → confidence >= 0.40 (probably correct)
            LOW   → confidence  < 0.40 (uncertain — include but flag)
        - Returns [] (empty list) if no persons detected, never raises.
    """
    if model is None:
        model = _get_model()

    results = model(frame, conf=conf_threshold, verbose=False)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls[0])
            if cls != 0:  # skip non-person detections
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Assign human-readable confidence band
            if conf >= 0.6:
                band = "HIGH"
            elif conf >= 0.4:
                band = "MED"
            else:
                band = "LOW"

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls,
                "confidence_band": band,
            })

    return detections
