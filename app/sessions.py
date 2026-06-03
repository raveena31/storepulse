"""
app/sessions.py
───────────────
Session builder — the analytical core of the system.

EDGE CASES SOLVED HERE (Session/Funnel group, prompt 3):
  EC-41  close_dangling_sessions: any session without an EXIT event gets
         exit_ts = clip_end_ts and exit_inferred = True.
  EC-43  within_watermark: detect events that arrived late by more than `grace_s`
         so /metrics can flag "metrics_pending_late_events".
  EC-46/47  Empty store + all-staff: build_sessions returns [] safely;
            callers handle gracefully (covered by metrics.py).

A SESSION is the fundamental unit for all metrics. One session = one visitor's
entire store visit, regardless of how many individual events they generated.

Why sessions instead of raw event counts?
  - Raw count: "how many ENTRY events?" → wrong, counts re-entries twice
  - Session count: "how many unique visitors?" → correct

Key rules enforced here:
  1. One session per visitor_id (REENTRY merges into the existing session)
  2. Sticky staff: if ANY event has is_staff=True, the whole session is staff
  3. Dwell time capped at 10 minutes per event (prevents bad camera data inflating numbers)
  4. Dangling sessions (no EXIT seen) are closed at clip end with exit_inferred=True
  5. Staff sessions are INCLUDED in the output — callers filter them:
        customer_sessions = [s for s in sessions if not s["is_staff"]]

Session structure returned:
    {
        "visitor_id":     str,
        "store_id":       str,
        "entry_ts":       datetime | None,
        "exit_ts":        datetime | None,
        "exit_inferred":  bool,           # True if we closed the session without an EXIT event
        "zones":          {zone_id: total_dwell_ms},
        "billing":        {               # None if visitor never joined billing queue
            "join_ts":    datetime,
            "depth":      int | None,     # queue length at time of joining
            "abandoned":  bool,           # True if BILLING_QUEUE_ABANDON seen
            "attributed": bool,           # True after POS correlation marks this as a purchase
        } | None,
        "events":         [list of raw event dicts],
        "is_staff":       bool,
        "confs":          [list of confidence floats],
    }
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

# Cap dwell contribution per event at 10 minutes.
# Without this cap, a camera that loses track of a person and re-detects them
# 2 hours later would produce a single ZONE_DWELL with dwell_ms=7,200,000 —
# which would completely distort zone averages.
MAX_DWELL_PER_EVENT_MS = 600_000  # 10 minutes in milliseconds


def build_sessions(events: list[dict]) -> list[dict]:
    """
    Group a flat list of events into per-visitor sessions.

    Args:
        events: List of event dicts (from DB or ingestion). Must have at minimum:
                visitor_id, event_type, timestamp (or ts), is_staff, confidence.

    Returns:
        List of session dicts (one per unique visitor_id).
        Includes BOTH customer and staff sessions — callers filter by is_staff.
    """
    # Sort by timestamp ascending so events are processed in chronological order
    sorted_events = sorted(events, key=lambda e: _ts(e))

    # Determine the clip end time: the timestamp of the very last event.
    # Used to close dangling sessions that have no EXIT event.
    clip_end: Optional[datetime] = None
    for e in sorted_events:
        t = _ts(e)
        if clip_end is None or t > clip_end:
            clip_end = t

    sessions: dict[str, dict] = {}  # visitor_id → session dict

    for e in sorted_events:
        vid = e["visitor_id"]
        etype = e["event_type"]
        t = _ts(e)

        # ── Create session if first time we see this visitor ───────────────
        if vid not in sessions:
            sessions[vid] = {
                "visitor_id": vid,
                "store_id": e.get("store_id", ""),
                "entry_ts": None,
                "exit_ts": None,
                "exit_inferred": False,
                "zones": {},      # zone_id → accumulated dwell ms
                "billing": None,  # populated on BILLING_QUEUE_JOIN
                "events": [],
                "is_staff": False,
                "confs": [],
            }

        sess = sessions[vid]

        # ── Sticky staff rule ──────────────────────────────────────────────
        # Once is_staff=True appears for any event, the whole session is staff.
        # This prevents a partially-classified visitor from contaminating metrics.
        if e.get("is_staff"):
            sess["is_staff"] = True

        sess["confs"].append(e.get("confidence", 0.0))
        sess["events"].append(e)

        # ── Process each event type ────────────────────────────────────────

        if etype == "ENTRY":
            # Set entry_ts only once — first ENTRY wins
            if sess["entry_ts"] is None:
                sess["entry_ts"] = t

        elif etype == "REENTRY":
            # Same person came back — do NOT create a second session.
            # Only set entry_ts if it wasn't already set (shouldn't happen
            # if we saw the original ENTRY, but handle gracefully).
            if sess["entry_ts"] is None:
                sess["entry_ts"] = t

        elif etype == "EXIT":
            # Record the exit time (last EXIT wins if there are multiple)
            sess["exit_ts"] = t

        elif etype in ("ZONE_ENTER", "ZONE_DWELL"):
            # Accumulate dwell time for this zone, capped per event
            zone = e.get("zone_id")
            if zone:
                raw_dwell = e.get("dwell_ms", 0)
                capped_dwell = min(raw_dwell, MAX_DWELL_PER_EVENT_MS)
                sess["zones"][zone] = sess["zones"].get(zone, 0) + capped_dwell

        elif etype == "ZONE_EXIT":
            # Dwell already captured via ZONE_DWELL events — nothing to do here
            pass

        elif etype == "BILLING_QUEUE_JOIN":
            # Record billing engagement. First JOIN per session wins.
            if sess["billing"] is None:
                # queue_depth may be in metadata dict or metadata object
                meta = e.get("metadata", {})
                depth = meta.get("queue_depth") if isinstance(meta, dict) else None
                sess["billing"] = {
                    "join_ts": t,
                    "depth": depth,
                    "abandoned": False,
                    "attributed": False,  # set to True by POS correlator on purchase match
                }

        elif etype == "BILLING_QUEUE_ABANDON":
            # Mark this billing engagement as abandoned
            if sess["billing"] is not None:
                sess["billing"]["abandoned"] = True

    # ── Close dangling sessions (no EXIT event recorded) ──────────────────
    for sess in sessions.values():
        if sess["exit_ts"] is None:
            sess["exit_ts"] = clip_end
            sess["exit_inferred"] = True

        # Ensure entry_ts is set even for visitors we only saw in zone/dwell events
        if sess["entry_ts"] is None and sess["events"]:
            sess["entry_ts"] = _ts(sess["events"][0])

    return list(sessions.values())


def close_dangling_sessions(open_sessions: list[dict], clip_end_ts: datetime) -> list[dict]:
    """
    Close any session with no recorded EXIT event.

    Sets exit_ts = clip_end_ts and exit_inferred = True.
    These sessions are valid for metrics but their dwell time may be truncated.

    Args:
        open_sessions: list of session dicts (may include closed ones — they're left alone)
        clip_end_ts:   the cut-off timestamp (typically the last event ts in the clip)

    Returns the same list (mutated in place).

    EC-41 implementation.
    """
    if clip_end_ts is not None and clip_end_ts.tzinfo is None:
        clip_end_ts = clip_end_ts.replace(tzinfo=timezone.utc)
    for sess in open_sessions:
        if sess.get("exit_ts") is None:
            sess["exit_ts"] = clip_end_ts
            sess["exit_inferred"] = True
    return open_sessions


def within_watermark(event_ts: datetime, now: datetime, grace_s: int = 30) -> bool:
    """
    True iff this event arrived within the grace watermark.

    If an event's timestamp is more than `grace_s` seconds in the past relative
    to `now`, it's "late" and may cause metrics for that window to need
    recomputation.

    EC-43 implementation.
    """
    if event_ts is None or now is None:
        return False
    et = event_ts if event_ts.tzinfo else event_ts.replace(tzinfo=timezone.utc)
    n  = now      if now.tzinfo      else now.replace(tzinfo=timezone.utc)
    return (n - et).total_seconds() <= grace_s


def has_late_events(events: list[dict], grace_s: int = 5) -> bool:
    """
    True iff at least one event in this batch has an ingest timestamp gap
    greater than grace_s seconds relative to its event timestamp.

    Used by GET /metrics to set metrics_pending_late_events=True.
    We approximate "ingest_ts" with datetime.now(UTC) at read time —
    if events are mostly from the recent past (live feed), this is fine.
    For historical replays, this stays False (no real-time semantics).
    """
    if not events:
        return False
    now = datetime.now(timezone.utc)
    for e in events:
        et = _ts(e)
        # Late = event timestamp is in the past by more than grace_s seconds
        # BUT only relevant if data is supposed to be live (< 1 hour old)
        age = (now - et).total_seconds()
        if 0 <= age <= 3600 and age > grace_s:
            return True
    return False


def _ts(event: dict) -> datetime:
    """
    Extract a timezone-aware datetime from an event dict.

    Handles both "timestamp" (from Pydantic models) and "ts" (from DB rows).
    Falls back to datetime.now(UTC) if no timestamp found — this should never
    happen in production but prevents a crash in edge cases.
    """
    ts = event.get("timestamp") or event.get("ts")

    if isinstance(ts, datetime):
        # Already a datetime — make sure it's tz-aware
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts

    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            # Handle "Z" suffix which Python < 3.11 doesn't parse natively
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))

    # Absolute fallback — should never reach here
    return datetime.now(timezone.utc)
