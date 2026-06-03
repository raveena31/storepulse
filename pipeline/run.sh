#!/usr/bin/env bash
# Usage: ./pipeline/run.sh <video_path> <store_id> [camera_id] [api_url]
# Processes a video file through the YOLOv8+ByteTrack pipeline
# and POSTs events to the Store Intelligence ingest API.

set -euo pipefail

VIDEO_PATH="${1:?Usage: run.sh <video_path> <store_id> [camera_id] [api_url]}"
STORE_ID="${2:?store_id required}"
CAMERA_ID="${3:-CAM_01}"
API_URL="${4:-http://localhost:8000/events/ingest}"

echo "=== Store Intelligence Pipeline ==="
echo "Video : $VIDEO_PATH"
echo "Store : $STORE_ID"
echo "Camera: $CAMERA_ID"
echo "API   : $API_URL"
echo ""

python - <<PYEOF
import sys
sys.path.insert(0, ".")
import cv2
from datetime import datetime, timezone
from pipeline.detect import process_frame, _get_model
from pipeline.tracker import ByteTrackWrapper
from pipeline.zones import assign_zone, _load_config
from pipeline.emit import build_event, post_events

VIDEO_PATH = "$VIDEO_PATH"
STORE_ID   = "$STORE_ID"
CAMERA_ID  = "$CAMERA_ID"
API_URL    = "$API_URL"
BATCH_SIZE = 100

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open video {VIDEO_PATH}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
model = _get_model()
tracker = ByteTrackWrapper()
config = _load_config()

events = []
frame_idx = 0
track_zones: dict = {}   # track_id -> current zone
track_zone_enter: dict = {} # track_id -> zone enter time
track_first_seen: dict = {} # track_id -> first seen time

print("Processing frames...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    ts = datetime.now(timezone.utc)

    dets = process_frame(frame, model)
    tracks = tracker.update(dets, frame)

    h, w = frame.shape[:2]
    for t in tracks:
        tid = t["track_id"]
        x1, y1, x2, y2 = t["bbox"]
        feet = ((x1 + x2) / 2, y2)

        if tid not in track_first_seen:
            track_first_seen[tid] = ts
            events.append(build_event(tid, "ENTRY", STORE_ID, CAMERA_ID, ts,
                                       confidence=t["confidence"]))

        zone = assign_zone(feet, config)
        prev_zone = track_zones.get(tid)

        if zone != prev_zone:
            if prev_zone and tid in track_zone_enter:
                dwell = int((ts - track_zone_enter[tid]).total_seconds() * 1000)
                events.append(build_event(tid, "ZONE_EXIT", STORE_ID, CAMERA_ID, ts,
                                           zone_id=prev_zone, dwell_ms=dwell,
                                           confidence=t["confidence"]))
            if zone:
                track_zone_enter[tid] = ts
                events.append(build_event(tid, "ZONE_ENTER", STORE_ID, CAMERA_ID, ts,
                                           zone_id=zone, confidence=t["confidence"]))
            track_zones[tid] = zone

    if len(events) >= BATCH_SIZE:
        resp = post_events(events, API_URL)
        print(f"  Batch posted: {resp}")
        events = []

# Flush remaining
if events:
    resp = post_events(events, API_URL)
    print(f"  Final batch: {resp}")

cap.release()
print(f"\nDone. Processed {frame_idx} frames, {len(track_first_seen)} unique tracks.")
PYEOF
