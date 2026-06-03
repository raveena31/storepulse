# PROMPT: "Generate pytest cases for app/metrics.py covering: zero-event store, all-staff
#          clip, and division-by-zero guards. Must verify status=NO_TRAFFIC, conversion_rate=0.0,
#          never NaN."
# CHANGES MADE: AI returned the all-staff case as buyers=0 / unique_visitors=10 (treating staff
#               as visitors, then guarding the division). That's wrong — staff sessions must
#               be EXCLUDED from the denominator entirely, otherwise a busy stockroom drags
#               conversion to ~0. I corrected the assertion to unique_visitors=0 and
#               status=NO_TRAFFIC, which forced a real fix in compute_metrics (filter
#               customer_sessions before computing visitors, not after).

from app.metrics import compute_metrics


def test_zero_events_no_exception():
    """Zero events → conversion_rate=0.0, status=NO_TRAFFIC, no exception."""
    result = compute_metrics([], pos_csv="nonexistent.csv")
    assert result["conversion_rate"] == 0.0
    assert result["status"] == "NO_TRAFFIC"
    assert result["unique_visitors"] == 0
    assert result["buyers"] == 0


def test_all_staff_events():
    """All events have is_staff=True → unique_visitors=0, conversion_rate=0.0."""
    events = [
        {
            "event_id": "e1",
            "store_id": "ST1008",
            "camera_id": "CAM_01",
            "visitor_id": "STAFF_01",
            "event_type": "ENTRY",
            "timestamp": "2024-01-15T10:00:00+00:00",
            "is_staff": True,
            "confidence": 0.9,
            "dwell_ms": 0,
            "zone_id": None,
        },
        {
            "event_id": "e2",
            "store_id": "ST1008",
            "camera_id": "CAM_01",
            "visitor_id": "STAFF_02",
            "event_type": "ENTRY",
            "timestamp": "2024-01-15T10:01:00+00:00",
            "is_staff": True,
            "confidence": 0.9,
            "dwell_ms": 0,
            "zone_id": None,
        },
    ]
    result = compute_metrics(events, pos_csv="nonexistent.csv")
    assert result["unique_visitors"] == 0
    assert result["conversion_rate"] == 0.0
    assert result["status"] == "NO_TRAFFIC"
