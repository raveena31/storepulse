"""
scripts/run_pipeline.py
-----------------------
Process all 5 CCTV cameras for store ST1008.
Extracts OSD timestamp from first frame, runs YOLOv8n + ByteTrack per camera,
assigns zones, fires correct event types per camera role, posts to ingest API.

Usage:
    python scripts/run_pipeline.py [--api http://localhost:8000] [--dry-run] [--skip-frames N]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.detect import _get_model
from pipeline.tracker import CameraTracker
from pipeline.zones import _load_config, assign_zone, get_camera_role, is_in_glass_mask
from pipeline.emit import build_event, post_events

STORE_ID = "ST1008"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEOS = {
    "CAM_01": os.path.join(BASE_DIR, "data", "cctv", "CCTV Footage", "CAM 1.mp4"),
    "CAM_02": os.path.join(BASE_DIR, "data", "cctv", "CCTV Footage", "CAM 2.mp4"),
    "CAM_03": os.path.join(BASE_DIR, "data", "cctv", "CCTV Footage", "CAM 3.mp4"),
    "CAM_04": os.path.join(BASE_DIR, "data", "cctv", "CCTV Footage", "CAM 4.mp4"),
    "CAM_05": os.path.join(BASE_DIR, "data", "cctv", "CCTV Footage", "CAM 5.mp4"),
}

IST_OFFSET = timedelta(hours=5, minutes=30)


# ── OSD timestamp extraction ──────────────────────────────────────────────────

def extract_osd_timestamp(frame: np.ndarray) -> Optional[datetime]:
    """Try pytesseract on the top-right OSD region. Returns UTC datetime or None."""
    try:
        import pytesseract
        h, w = frame.shape[:2]
        roi = frame[0:50, w - 400:w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config="--psm 7")
        m = re.search(r"(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2}):(\d{2})", text)
        if m:
            dd, mm, yyyy, hh, mi, ss = [int(x) for x in m.groups()]
            ist = datetime(yyyy, mm, dd, hh, mi, ss)
            return (ist - IST_OFFSET).replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return None


def hardcoded_base_ts() -> datetime:
    """
    Fallback: parsed visually from OSD in frames — 10/04/2026 20:10:00 IST.
    IST 20:10:00 → UTC 14:40:00.
    """
    ist = datetime(2026, 4, 10, 20, 10, 0)
    return (ist - IST_OFFSET).replace(tzinfo=timezone.utc)


# ── Entry/Exit line-crossing tracker ─────────────────────────────────────────

class LineCrossTracker:
    def __init__(self, line_y: int, hysteresis: int = 20):
        self.line_y = line_y
        self.hysteresis = hysteresis
        self._prev_y: dict[int, float] = {}
        self._state: dict[int, str] = {}  # "inside" | "outside"

    def update(self, tracks: list[dict]) -> list[tuple[int, str]]:
        """Returns [(track_id, 'ENTRY'|'EXIT'), ...]"""
        events = []
        for t in tracks:
            tid = t["track_id"]
            feet_y = t["bbox"][3]
            prev_y = self._prev_y.get(tid)

            if prev_y is not None:
                # Crossed downward (into store) = ENTRY
                if prev_y < self.line_y - self.hysteresis and feet_y >= self.line_y:
                    if self._state.get(tid) != "inside":
                        self._state[tid] = "inside"
                        events.append((tid, "ENTRY"))
                # Crossed upward (out of store) = EXIT
                elif prev_y > self.line_y + self.hysteresis and feet_y <= self.line_y:
                    if self._state.get(tid) != "outside":
                        self._state[tid] = "outside"
                        events.append((tid, "EXIT"))
            self._prev_y[tid] = feet_y
        return events


# ── Per-camera processing ─────────────────────────────────────────────────────

def process_camera(
    camera_id: str,
    video_path: str,
    model,
    config: dict,
    skip_frames: int = 5,
    dry_run: bool = False,
    api_url: str = "http://localhost:8000/events/ingest",
) -> dict:

    role = get_camera_role(camera_id, config)
    print(f"\n{'='*60}")
    print(f"Camera: {camera_id}  Role: {role}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return {"camera_id": camera_id, "error": "cannot_open"}

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Grab first frame for timestamp extraction
    ret0, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    base_ts = (extract_osd_timestamp(first_frame) if ret0 else None) or hardcoded_base_ts()
    print(f"  Base timestamp (UTC): {base_ts.isoformat()}")
    print(f"  Frames={total}  FPS={fps:.1f}  Duration={total/fps/60:.1f}min  skip_frames={skip_frames}")

    tracker    = CameraTracker(camera_id)
    line_cross = LineCrossTracker(config.get("entry_line_y", 520)) if role == "entry_exit" else None

    track_first_seen: dict[int, datetime] = {}
    track_zones: dict[int, Optional[str]] = {}
    track_zone_enter_ts: dict[int, datetime] = {}
    billing_seen: set[int] = set()

    events_batch: list[dict] = []
    frame_idx  = 0
    proc_count = 0
    total_events_posted = 0

    def flush(force=False):
        nonlocal total_events_posted
        if len(events_batch) >= 80 or force:
            if not dry_run:
                post_events(events_batch[:], api_url)
            total_events_posted += len(events_batch)
            events_batch.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % (skip_frames + 1) != 0:
            continue
        proc_count += 1
        frame_ts = base_ts + timedelta(seconds=frame_idx / fps)

        tracks = tracker.track_frame(frame, model, conf_threshold=0.25)

        # ── STAFF-ONLY (CAM_04 stockroom) ────────────────────────────────
        if role == "staff_only":
            for t in tracks:
                tid = t["track_id"]
                if tid not in track_first_seen:
                    track_first_seen[tid] = frame_ts
                    events_batch.append(build_event(
                        tid, "ENTRY", STORE_ID, camera_id, frame_ts,
                        is_staff=True, confidence=t["confidence"],
                        metadata={"source": "staff_only_camera", "confidence_band": t.get("confidence_band", "MED")},
                    ))
            flush()
            continue

        # ── ENTRY / EXIT (CAM_03) ─────────────────────────────────────────
        if role == "entry_exit":
            # Filter glass-mask reflections
            clean = [t for t in tracks if not is_in_glass_mask(
                (t["bbox"][0] + t["bbox"][2]) / 2, t["bbox"][3], config)]
            crossings = line_cross.update(clean)
            for tid, etype in crossings:
                conf = next((t["confidence"] for t in clean if t["track_id"] == tid), 0.5)
                if etype == "ENTRY":
                    track_first_seen[tid] = frame_ts
                events_batch.append(build_event(
                    tid, etype, STORE_ID, camera_id, frame_ts,
                    zone_id="ENTRY_ZONE", confidence=conf,
                ))
            # Zone dwell near entry
            for t in clean:
                tid = t["track_id"]
                x_c = (t["bbox"][0] + t["bbox"][2]) / 2
                zone = assign_zone((x_c, t["bbox"][3]), camera_id, config)
                prev_zone = track_zones.get(tid)
                if zone != prev_zone and zone is not None:
                    if prev_zone and tid in track_zone_enter_ts:
                        dwell = int((frame_ts - track_zone_enter_ts[tid]).total_seconds() * 1000)
                        events_batch.append(build_event(
                            tid, "ZONE_DWELL", STORE_ID, camera_id, frame_ts,
                            zone_id=prev_zone, dwell_ms=min(dwell, 600_000),
                            confidence=t["confidence"]))
                    track_zone_enter_ts[tid] = frame_ts
                    events_batch.append(build_event(
                        tid, "ZONE_ENTER", STORE_ID, camera_id, frame_ts,
                        zone_id=zone, confidence=t["confidence"]))
                track_zones[tid] = zone
            flush()
            continue

        # ── BILLING (CAM_05) ──────────────────────────────────────────────
        if role == "billing":
            current_ids = {t["track_id"] for t in tracks}
            for t in tracks:
                tid = t["track_id"]
                if tid not in track_first_seen:
                    track_first_seen[tid] = frame_ts
                if tid not in billing_seen:
                    billing_seen.add(tid)
                    # Heuristic: person close to counter edge (feet_y < 600) is likely staff
                    feet_y = t["bbox"][3]
                    is_staff_h = feet_y < 450
                    events_batch.append(build_event(
                        tid, "BILLING_QUEUE_JOIN", STORE_ID, camera_id, frame_ts,
                        zone_id="BILLING", is_staff=is_staff_h,
                        confidence=t["confidence"],
                        metadata={"queue_depth": len(billing_seen),
                                  "confidence_band": t.get("confidence_band", "MED")},
                    ))
            flush()
            continue

        # ── PRODUCT ZONE cameras (CAM_01, CAM_02) ────────────────────────
        for t in tracks:
            tid = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            feet = ((x1 + x2) / 2, y2)

            if tid not in track_first_seen:
                track_first_seen[tid] = frame_ts

            zone = assign_zone(feet, camera_id, config)
            prev_zone = track_zones.get(tid)

            if zone != prev_zone:
                if prev_zone and tid in track_zone_enter_ts:
                    dwell = int((frame_ts - track_zone_enter_ts[tid]).total_seconds() * 1000)
                    events_batch.append(build_event(
                        tid, "ZONE_DWELL", STORE_ID, camera_id, frame_ts,
                        zone_id=prev_zone, dwell_ms=min(dwell, 600_000),
                        confidence=t["confidence"]))
                    events_batch.append(build_event(
                        tid, "ZONE_EXIT", STORE_ID, camera_id, frame_ts,
                        zone_id=prev_zone, confidence=t["confidence"]))
                if zone is not None:
                    track_zone_enter_ts[tid] = frame_ts
                    events_batch.append(build_event(
                        tid, "ZONE_ENTER", STORE_ID, camera_id, frame_ts,
                        zone_id=zone, confidence=t["confidence"]))
                track_zones[tid] = zone
        flush()

    # ── Flush final dwell + EXIT ──────────────────────────────────────────
    clip_end = base_ts + timedelta(seconds=total / fps)
    for tid, zone in track_zones.items():
        if zone and tid in track_zone_enter_ts:
            dwell = int((clip_end - track_zone_enter_ts[tid]).total_seconds() * 1000)
            events_batch.append(build_event(
                tid, "ZONE_DWELL", STORE_ID, camera_id, clip_end,
                zone_id=zone, dwell_ms=min(dwell, 600_000), confidence=0.5))
    for tid in list(track_first_seen.keys()):
        events_batch.append(build_event(
            tid, "EXIT", STORE_ID, camera_id, clip_end,
            is_staff=(role == "staff_only"), confidence=0.5))
    flush(force=True)

    cap.release()
    stats = {
        "camera_id": camera_id,
        "role": role,
        "frames_processed": proc_count,
        "unique_tracks": len(track_first_seen),
        "total_events": total_events_posted,
    }
    print(f"  Done: {proc_count} frames  |  {len(track_first_seen)} tracks  |  "
          f"{total_events_posted} events {'(dry run)' if dry_run else 'posted'}")
    return stats


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Store Intelligence CCTV Pipeline")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-frames", type=int, default=5,
                        help="Process 1 in every (N+1) frames. 5 = ~5fps from 30fps.")
    parser.add_argument("--cameras", nargs="+",
                        default=["CAM_01", "CAM_02", "CAM_03", "CAM_04", "CAM_05"])
    args = parser.parse_args()

    api_url = args.api.rstrip("/") + "/events/ingest"
    print(f"\n{'='*60}")
    print(f"Store Intelligence Pipeline — {STORE_ID}")
    print(f"API: {api_url}  dry_run={args.dry_run}  skip={args.skip_frames}")

    config = _load_config()
    print("Loading YOLOv8n model...")
    t0 = time.time()
    model = _get_model()
    print(f"Model ready in {time.time()-t0:.1f}s")

    all_stats = []
    for cam_id in args.cameras:
        if cam_id not in VIDEOS:
            print(f"  Unknown camera: {cam_id}")
            continue
        stats = process_camera(
            cam_id, VIDEOS[cam_id], model, config,
            skip_frames=args.skip_frames,
            dry_run=args.dry_run,
            api_url=api_url,
        )
        all_stats.append(stats)

    print(f"\n{'='*60}")
    print("SUMMARY")
    total_tracks = total_events = 0
    for s in all_stats:
        if "error" in s:
            print(f"  {s['camera_id']}: ERROR — {s['error']}")
        else:
            print(f"  {s['camera_id']} ({s['role']}): {s['unique_tracks']} tracks, {s['total_events']} events")
            total_tracks += s["unique_tracks"]
            total_events += s["total_events"]
    print(f"\n  TOTAL: {total_tracks} tracks, {total_events} events {'(dry run)' if args.dry_run else 'posted'}")
    if not args.dry_run:
        base = args.api.rstrip("/")
        print(f"\n  Results:")
        print(f"    {base}/stores/{STORE_ID}/metrics")
        print(f"    {base}/stores/{STORE_ID}/funnel")
        print(f"    {base}/stores/{STORE_ID}/anomalies")


if __name__ == "__main__":
    main()
