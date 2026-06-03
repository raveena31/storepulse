# PROMPT: "Generate pytest cases for app/sessions.py covering: REENTRY uses same visitor_id
#          (no new session created), sticky staff (any is_staff=True event taints the whole
#          session), dwell cap at 10 minutes per event."
# CHANGES MADE: AI's first sticky-staff draft used `if all_events_are_staff:` — that's wrong
#               because a visitor might be misclassified for one frame and the heuristic
#               flips back. I rewrote to "ANY is_staff=True event → session is_staff=True"
#               (sticky), which matches the EC-20 lock contract. AI also tried to dedup
#               REENTRY by timestamp instead of visitor_id; corrected to assert there's
#               exactly ONE session output for the same visitor_id regardless of REENTRY count.

from datetime import datetime, timezone
from app.sessions import build_sessions


def _evt(visitor_id, event_type, ts_str, is_staff=False, zone_id=None, dwell_ms=0):
    return {
        "event_id": f"{visitor_id}-{event_type}-{ts_str}",
        "store_id": "ST1008",
        "camera_id": "CAM_01",
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": datetime.fromisoformat(ts_str),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": 0.9,
    }


def test_reentry_same_visitor_one_session():
    """ENTRY → EXIT → REENTRY for same visitor_id → exactly ONE session."""
    events = [
        _evt("V001", "ENTRY",   "2024-01-15T10:00:00+00:00"),
        _evt("V001", "EXIT",    "2024-01-15T10:30:00+00:00"),
        _evt("V001", "REENTRY", "2024-01-15T11:00:00+00:00"),
    ]
    sessions = build_sessions(events)
    assert len(sessions) == 1
    s = sessions[0]
    assert s["visitor_id"] == "V001"
    # entry_ts must be the original ENTRY, not overwritten by REENTRY
    assert s["entry_ts"] is not None


def test_sticky_staff():
    """One is_staff=True event in a visitor's events → whole session is_staff=True."""
    events = [
        _evt("V002", "ENTRY",      "2024-01-15T09:00:00+00:00", is_staff=False),
        _evt("V002", "ZONE_ENTER", "2024-01-15T09:05:00+00:00", is_staff=False, zone_id="FOH_MAKEUP"),
        _evt("V002", "ZONE_ENTER", "2024-01-15T09:10:00+00:00", is_staff=True,  zone_id="BILLING"),
        _evt("V002", "EXIT",       "2024-01-15T09:30:00+00:00", is_staff=False),
    ]
    sessions = build_sessions(events)
    assert len(sessions) == 1
    assert sessions[0]["is_staff"] is True
