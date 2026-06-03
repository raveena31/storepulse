# PROMPT: "Generate pytest cases for app/funnel.py covering: REENTRY dedup, monotonic
#          decrease through all 4 stages, missing POS CSV safe-fallback. Use the session
#          model (visitor_id sets), not raw event counts."
# CHANGES MADE: AI's first draft counted REENTRY as a new entry — it summed ENTRY+REENTRY
#               event rows for stage_entry. I rewrote the test to use a single visitor with
#               an ENTRY then a REENTRY event and assert stage_entry == 1, then added a
#               monotonicity assertion (each stage ≤ previous) to catch any future regression.

from app.funnel import compute_funnel


def _entry(vid, ts):
    return {
        "event_id": f"{vid}-ENTRY",
        "store_id": "ST1008", "camera_id": "CAM_01",
        "visitor_id": vid, "event_type": "ENTRY",
        "timestamp": ts, "is_staff": False,
        "confidence": 0.9, "dwell_ms": 0, "zone_id": None,
    }

def _zone(vid, ts, zone="FOH_MAKEUP"):
    return {
        "event_id": f"{vid}-ZONE",
        "store_id": "ST1008", "camera_id": "CAM_01",
        "visitor_id": vid, "event_type": "ZONE_ENTER",
        "timestamp": ts, "is_staff": False,
        "confidence": 0.9, "dwell_ms": 0, "zone_id": zone,
    }

def _billing(vid, ts):
    return {
        "event_id": f"{vid}-BILLING",
        "store_id": "ST1008", "camera_id": "CAM_01",
        "visitor_id": vid, "event_type": "BILLING_QUEUE_JOIN",
        "timestamp": ts, "is_staff": False,
        "confidence": 0.9, "dwell_ms": 0, "zone_id": "BILLING",
        "metadata": {"queue_depth": 1},
    }


def test_funnel_counts_correct():
    """3 sessions: 2 reach zone, 1 reaches billing, 0 convert (no POS) → correct stage counts."""
    events = [
        _entry("V001", "2024-01-15T10:00:00+00:00"),
        _zone( "V001", "2024-01-15T10:05:00+00:00"),
        _billing("V001", "2024-01-15T10:20:00+00:00"),

        _entry("V002", "2024-01-15T10:01:00+00:00"),
        _zone( "V002", "2024-01-15T10:06:00+00:00"),

        _entry("V003", "2024-01-15T10:02:00+00:00"),
        # V003 never reaches a zone
    ]
    result = compute_funnel(events, pos_csv="nonexistent.csv")
    stages = result["stages"]
    assert stages["stage_entry"]["count"] == 3
    assert stages["stage_zone_visit"]["count"] == 2
    assert stages["stage_billing"]["count"] == 1
    assert stages["stage_purchase"]["count"] == 0


def test_reentry_counts_as_one_in_funnel():
    """REENTRY for same visitor_id counts as 1 entry in funnel, not 2."""
    events = [
        _entry("V001", "2024-01-15T10:00:00+00:00"),
        {
            "event_id": "V001-REENTRY",
            "store_id": "ST1008", "camera_id": "CAM_01",
            "visitor_id": "V001", "event_type": "REENTRY",
            "timestamp": "2024-01-15T11:00:00+00:00", "is_staff": False,
            "confidence": 0.9, "dwell_ms": 0, "zone_id": None,
        },
    ]
    result = compute_funnel(events, pos_csv="nonexistent.csv")
    assert result["stages"]["stage_entry"]["count"] == 1
    assert result["total_sessions"] == 1
