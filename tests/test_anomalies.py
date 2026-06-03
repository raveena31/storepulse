# PROMPT: "Generate pytest cases for app/anomalies.py covering: queue spike CRITICAL at 8+
#          visitors, zero events no-crash, suggested_action is operational not placeholder."
# CHANGES MADE: AI used datetime(2024, 1, 15, ...) as test timestamps, but the queue-depth
#               window anchors to "last 5 minutes of the event stream" — those events were
#               two years old so the window contained nothing and the spike never fired.
#               Switched to datetime.now(UTC) - timedelta(seconds=i*5) so events fall inside
#               the 5-minute window. Also added an assertion that suggested_action contains
#               "URGENT" for CRITICAL severity, which forced the anomaly module to write
#               actionable messages instead of placeholders.

from datetime import datetime, timedelta, timezone
from app.anomalies import compute_anomalies


def _billing_join(vid, offset_seconds=0):
    ts = (datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)).isoformat()
    return {
        "event_id": f"{vid}-BQ",
        "store_id": "ST1008", "camera_id": "CAM_04",
        "visitor_id": vid, "event_type": "BILLING_QUEUE_JOIN",
        "timestamp": ts, "is_staff": False,
        "confidence": 0.8, "dwell_ms": 0, "zone_id": "BILLING",
        "metadata": {"queue_depth": 1},
    }


def test_billing_queue_spike_critical():
    """8 concurrent BILLING_QUEUE_JOIN events → BILLING_QUEUE_SPIKE with CRITICAL severity."""
    events = [_billing_join(f"V{i:03d}", offset_seconds=i * 5) for i in range(8)]
    anomalies = compute_anomalies(events, pos_csv="nonexistent.csv")
    spike = next((a for a in anomalies if a["type"] == "BILLING_QUEUE_SPIKE"), None)
    assert spike is not None
    assert spike["severity"] == "CRITICAL"
    assert spike["value"] >= 8


def test_zero_events_no_anomalies_no_crash():
    """Zero events → empty anomaly list, no exception."""
    anomalies = compute_anomalies([], pos_csv="nonexistent.csv")
    assert anomalies == []
