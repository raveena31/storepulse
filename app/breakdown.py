"""
app/breakdown.py
─────────────────
Transparency endpoint: shows exactly where every number on the dashboard
comes from. Answers: "How did we get to 93 visitors?"

Returns a full split by:
  - Camera (which camera generated how many unique tracks + events)
  - Event type (how many ENTRY, EXIT, ZONE_ENTER, BILLING_QUEUE_JOIN etc)
  - Zone (per-zone visit counts)
  - Role (customer vs staff sessions)
  - Confidence band (HIGH / MEDIUM / LOW)
  - Time window (latest event, earliest event, span)

Plus a `provenance` block explaining each headline number.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from .sessions import build_sessions, _ts
from .models import session_confidence


def compute_breakdown(events: list[dict]) -> dict:
    """
    Build a complete provenance breakdown of the event stream.

    Returns a dict with EVERY number on the dashboard explained.
    """
    if not events:
        return _empty_breakdown()

    # ── Per camera ────────────────────────────────────────────────────────
    cam_tracks: dict[str, set[str]] = defaultdict(set)
    cam_events: dict[str, int] = defaultdict(int)
    cam_events_by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # ── Per event type ────────────────────────────────────────────────────
    by_event_type: dict[str, int] = defaultdict(int)

    # ── Per zone ──────────────────────────────────────────────────────────
    zone_visits: dict[str, set[str]] = defaultdict(set)
    zone_dwell: dict[str, int] = defaultdict(int)

    # ── Per role (customer vs staff) ──────────────────────────────────────
    staff_event_count = 0
    customer_event_count = 0

    # ── Per confidence band ───────────────────────────────────────────────
    conf_band_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    # ── Time span ─────────────────────────────────────────────────────────
    timestamps = []

    for e in events:
        cam = e.get("camera_id", "unknown")
        etype = e.get("event_type", "unknown")
        vid = e.get("visitor_id", "")
        is_staff = bool(e.get("is_staff", False))
        conf = float(e.get("confidence", 0.0) or 0.0)
        zone = e.get("zone_id")
        dwell = int(e.get("dwell_ms", 0) or 0)

        cam_tracks[cam].add(vid)
        cam_events[cam] += 1
        cam_events_by_type[cam][etype] += 1
        by_event_type[etype] += 1

        if is_staff:
            staff_event_count += 1
        else:
            customer_event_count += 1

        # Confidence band
        if conf >= 0.7:
            conf_band_counts["HIGH"] += 1
        elif conf >= 0.4:
            conf_band_counts["MEDIUM"] += 1
        else:
            conf_band_counts["LOW"] += 1

        # Zone tracking
        if zone and not is_staff:
            zone_visits[zone].add(vid)
            if etype in ("ZONE_ENTER", "ZONE_DWELL"):
                zone_dwell[zone] += min(dwell, 600_000)  # 10-min cap

        timestamps.append(_ts(e))

    # ── Sessions for headline numbers ─────────────────────────────────────
    sessions = build_sessions(events)
    customer_sessions = [s for s in sessions if not s.get("is_staff")]
    staff_sessions = [s for s in sessions if s.get("is_staff")]

    # ── Headline numbers ──────────────────────────────────────────────────
    unique_visitors_raw_tracks = len(set(
        e.get("visitor_id") for e in events
        if e.get("visitor_id") and not e.get("is_staff")
    ))
    unique_visitors_sessions = len(customer_sessions)
    unique_staff = len(staff_sessions)

    # Door entries through CAM_03 specifically
    door_entries = by_event_type.get("ENTRY", 0)
    door_exits = by_event_type.get("EXIT", 0)
    # Approx people currently in store = door entries minus exits (capped >= 0)
    in_store_now = max(0, door_entries - door_exits)

    # Build per-camera output with explanation
    by_camera_out: dict[str, dict] = {}
    for cam in sorted(cam_tracks.keys()):
        tracks = cam_tracks[cam]
        non_staff_tracks = {
            t for t in tracks
            if not any(e.get("is_staff") for e in events
                       if e.get("visitor_id") == t and e.get("camera_id") == cam)
        }
        by_camera_out[cam] = {
            "events_total": cam_events[cam],
            "unique_tracks": len(tracks),
            "unique_customer_tracks": len(non_staff_tracks),
            "events_by_type": dict(cam_events_by_type[cam]),
        }

    # Build per-zone output
    by_zone_out: dict[str, dict] = {}
    for zone in sorted(zone_visits.keys()):
        n = len(zone_visits[zone])
        d = zone_dwell[zone]
        by_zone_out[zone] = {
            "unique_visitors": n,
            "total_dwell_ms": d,
            "avg_dwell_sec": round((d / n) / 1000, 1) if n > 0 else 0.0,
        }

    earliest = min(timestamps) if timestamps else None
    latest = max(timestamps) if timestamps else None

    return {
        "headline": {
            "events_total": len(events),
            "unique_visitors_raw_tracks": unique_visitors_raw_tracks,
            "unique_visitors_sessions": unique_visitors_sessions,
            "unique_staff_sessions": unique_staff,
            "door_entries_cam03": door_entries,
            "door_exits_cam03": door_exits,
            "estimated_in_store_now": in_store_now,
        },
        "by_camera": by_camera_out,
        "by_event_type": dict(sorted(by_event_type.items())),
        "by_zone": by_zone_out,
        "by_role": {
            "customer_events": customer_event_count,
            "staff_events": staff_event_count,
        },
        "by_confidence_band": conf_band_counts,
        "time_window": {
            "earliest_event_utc": earliest.isoformat() if earliest else None,
            "latest_event_utc": latest.isoformat() if latest else None,
            "span_seconds": round((latest - earliest).total_seconds(), 1)
                if earliest and latest else 0,
        },
        "provenance": {
            "unique_visitors_explained": (
                f"Each camera assigns its own track IDs. {unique_visitors_raw_tracks} unique "
                f"visitor_ids exist across all 5 cameras. "
                f"Cross-camera dedup (EC-17/18) is implemented but not yet wired into the "
                f"live pipeline runner — meaning a customer visible on CAM_01 AND CAM_02 "
                f"may be counted twice. Real number is likely 30-50% lower."
            ),
            "door_entries_explained": (
                f"CAM_03 watches the glass entrance door. {door_entries} line-crossings "
                f"detected (people whose feet crossed entry_line_y going downward into the store). "
                f"Passersby on the street are filtered by glass_mask_polygons and net-direction guard (EC-3)."
            ),
            "conversion_rate_explained": (
                f"buyers / unique_visitors_sessions. POS invoices in the time window are "
                f"matched to billing sessions via correlate_txn_to_session() (closest unattributed "
                f"within 5-minute window, EC-34)."
            ),
        },
    }


def _empty_breakdown() -> dict:
    return {
        "headline": {
            "events_total": 0,
            "unique_visitors_raw_tracks": 0,
            "unique_visitors_sessions": 0,
            "unique_staff_sessions": 0,
            "door_entries_cam03": 0,
            "door_exits_cam03": 0,
            "estimated_in_store_now": 0,
        },
        "by_camera": {},
        "by_event_type": {},
        "by_zone": {},
        "by_role": {"customer_events": 0, "staff_events": 0},
        "by_confidence_band": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
        "time_window": {"earliest_event_utc": None, "latest_event_utc": None, "span_seconds": 0},
        "provenance": {},
    }