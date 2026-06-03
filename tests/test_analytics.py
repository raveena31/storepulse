# PROMPT: "Generate pytest cases for the analytics layer: brand heatmap attention_vs_sales,
#          real anomaly thresholds with 7-day baseline, structured JSON logging with trace_id,
#          STALE_FEED detection, confidence band calibration."
# CHANGES MADE: AI's CONVERSION_DROP test seeded daily_stats with rates of 0.20 for the SAME
#               date as today's metrics call, then asserted the anomaly fires. That's wrong —
#               you can't compare today against itself. I seeded 5 rows for the 5 prior days
#               (date - timedelta(days=i) for i in 1..5) and used today_date_str to exclude
#               today from the baseline lookup. This forced rolling_avg_conversion() to add
#               an exclude_date parameter — without that, the anomaly would never fire on
#               the first day of operation.
#               AI also wrote the structured-logging test to assert response body contains
#               trace_id. I corrected it to check X-Trace-Id response header AND parse the
#               captured log lines as JSON — the header alone could be lying.

import json
import logging
import os
import io
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.heatmap import (
    brand_key,
    attention_vs_sales,
    capped_zone_dwell,
    data_confidence,
    zone_status,
    compute_heatmap,
)
from app.anomalies import compute_anomalies, _current_queue_depth
from app.health import health_status, STALE_THRESHOLD_S
from app.models import confidence_band, session_confidence
from app.metrics import session_confidence_distribution


# ── Section 1 — Brand heatmap ─────────────────────────────────────────────────

class TestBrandKey:
    def test_basic_normalisation(self):
        assert brand_key("Faces Canada") == "faces canada"
        assert brand_key("NY Bae")       == "ny bae"

    def test_ampersand_to_and(self):
        assert brand_key("P&G Beauty") == "p and g beauty"

    def test_collapses_whitespace(self):
        assert brand_key("  Lakme   Absolute  ") == "lakme absolute"

    def test_none_safe(self):
        assert brand_key(None) == ""
        assert brand_key("") == ""


class TestAttentionVsSales:

    def test_high_attention_low_sales(self):
        """Faces Canada gets 60% of dwell but only 20% of revenue → HIGH_ATTENTION_LOW_SALES."""
        zone_dwell = {"FACES_CANADA": 600, "LAKME": 400}
        # Brand "Faces Canada" maps to FACES_CANADA per config; Lakme→LAKME
        pos_brand_rev = {"Faces Canada": 200.0, "Lakme": 800.0}
        avs = attention_vs_sales(zone_dwell, pos_brand_rev)
        assert avs["FACES_CANADA"]["interpretation"] == "HIGH_ATTENTION_LOW_SALES"
        assert avs["FACES_CANADA"]["gap"] > 0.10

    def test_balanced(self):
        zone_dwell = {"FACES_CANADA": 500, "LAKME": 500}
        pos_brand_rev = {"Faces Canada": 500.0, "Lakme": 500.0}
        avs = attention_vs_sales(zone_dwell, pos_brand_rev)
        assert avs["FACES_CANADA"]["interpretation"] == "BALANCED"

    def test_low_attention_high_sales(self):
        """LAKME gets 20% of dwell but 60% of revenue → LOW_ATTENTION_HIGH_SALES."""
        zone_dwell = {"FACES_CANADA": 800, "LAKME": 200}
        pos_brand_rev = {"Faces Canada": 400.0, "Lakme": 600.0}
        avs = attention_vs_sales(zone_dwell, pos_brand_rev)
        assert avs["LAKME"]["interpretation"] == "LOW_ATTENTION_HIGH_SALES"

    def test_zero_dwell_safe(self):
        avs = attention_vs_sales({}, {"Lakme": 100.0})
        assert avs["LAKME"]["dwell_share"] == 0.0

    def test_zero_revenue_safe(self):
        avs = attention_vs_sales({"LAKME": 1000}, {})
        assert avs["LAKME"]["sales_share"] == 0.0


class TestCappedDwell:
    def test_under_cap(self):
        assert capped_zone_dwell(60_000) == 60_000

    def test_over_cap(self):
        assert capped_zone_dwell(900_000) == 600_000  # capped at 10 min

    def test_negative_returns_zero(self):
        assert capped_zone_dwell(-5) == 0


class TestDataConfidence:
    def test_low_when_few_sessions(self):
        assert data_confidence(10) == "LOW"

    def test_ok_when_enough(self):
        assert data_confidence(50) == "OK"

    def test_boundary(self):
        assert data_confidence(20) == "OK"
        assert data_confidence(19) == "LOW"


class TestZoneStatus:
    def test_uncovered_zone(self):
        assert zone_status(visits=0, has_coverage=False, window_min=60) == "UNKNOWN_NO_COVERAGE"
        # Even with visits, no-coverage wins
        assert zone_status(visits=5, has_coverage=False, window_min=60) == "UNKNOWN_NO_COVERAGE"

    def test_dead_zone(self):
        assert zone_status(visits=0, has_coverage=True, window_min=60) == "DEAD_ZONE"

    def test_active(self):
        assert zone_status(visits=3, has_coverage=True, window_min=60) == "ACTIVE"

    def test_short_window_not_dead(self):
        # Less than 30 min of observation → can't declare dead
        assert zone_status(visits=0, has_coverage=True, window_min=10) == "ACTIVE"


# ── Section 2 — Real anomalies ────────────────────────────────────────────────

def _billing_join_event(vid, offset_seconds=0):
    """Build a BILLING_QUEUE_JOIN event close to now."""
    ts = (datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)).isoformat()
    return {
        "event_id": f"{vid}-bq-{offset_seconds}",
        "store_id": "ST1008",
        "camera_id": "CAM_05",
        "visitor_id": vid,
        "event_type": "BILLING_QUEUE_JOIN",
        "timestamp": ts,
        "is_staff": False,
        "confidence": 0.8,
        "dwell_ms": 0,
        "zone_id": "BILLING",
        "metadata": {"queue_depth": 1},
    }


class TestQueueSpike:

    def test_critical_at_8(self):
        events = [_billing_join_event(f"V{i:03d}", offset_seconds=i*5) for i in range(8)]
        anomalies = compute_anomalies(events, pos_csv="nonexistent.csv")
        spike = next((a for a in anomalies if a["type"] == "BILLING_QUEUE_SPIKE"), None)
        assert spike is not None
        assert spike["severity"] == "CRITICAL"
        assert "URGENT" in spike["suggested_action"]
        assert spike["value"] >= 8

    def test_warn_at_6(self):
        events = [_billing_join_event(f"V{i:03d}", offset_seconds=i*5) for i in range(6)]
        anomalies = compute_anomalies(events, pos_csv="nonexistent.csv")
        spike = next((a for a in anomalies if a["type"] == "BILLING_QUEUE_SPIKE"), None)
        assert spike is not None
        assert spike["severity"] == "WARN"

    def test_no_spike_below_threshold(self):
        events = [_billing_join_event(f"V{i:03d}", offset_seconds=i*5) for i in range(3)]
        anomalies = compute_anomalies(events, pos_csv="nonexistent.csv")
        spike = next((a for a in anomalies if a["type"] == "BILLING_QUEUE_SPIKE"), None)
        assert spike is None

    def test_subsequent_exit_reduces_count(self):
        """A visitor who joined queue then EXITed should NOT count in queue depth."""
        v_exit = _billing_join_event("V_EXIT", offset_seconds=60)
        exit_ev = {
            "event_id": "V_EXIT-exit",
            "store_id": "ST1008", "camera_id": "CAM_05",
            "visitor_id": "V_EXIT", "event_type": "EXIT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_staff": False, "confidence": 0.8, "dwell_ms": 0, "zone_id": None,
        }
        events = [v_exit, exit_ev]
        assert _current_queue_depth(events) == 0


class TestConversionDrop:

    def test_fires_with_baseline(self, tmp_path, monkeypatch):
        """7d avg = 0.20, today = 0.05, visitors = 25 → CONVERSION_DROP fires."""
        db_path = str(tmp_path / "drop.db")
        monkeypatch.setenv("DB_PATH", db_path)
        import app.db as appdb
        appdb._repo = None
        appdb.DB_PATH = db_path
        repo = appdb.get_repo()

        # Seed 5 days of 20% baseline (date != today)
        base = datetime.now(timezone.utc).date()
        for i in range(1, 6):  # yesterday → 5 days ago
            d = (base - timedelta(days=i)).isoformat()
            repo.upsert_daily_stats("ST1008", d, conversion_rate=0.20,
                                    unique_visitors=50, revenue_inr=10000.0)

        # Build events: 25 unique visitors, only 1 reaches billing (≈ low conversion)
        events = []
        for i in range(25):
            events.append({
                "event_id": f"V{i:03d}-entry",
                "store_id": "ST1008", "camera_id": "CAM_03",
                "visitor_id": f"V{i:03d}", "event_type": "ENTRY",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "is_staff": False, "confidence": 0.9, "dwell_ms": 0, "zone_id": None,
            })

        # POS missing → 0 buyers → today_rate=0.0 < 0.20 × 0.6 = 0.12
        anomalies = compute_anomalies(events, pos_csv="nonexistent.csv",
                                       store_id="ST1008",
                                       today_date_str=base.isoformat())
        drop = next((a for a in anomalies if a["type"] == "CONVERSION_DROP"), None)
        assert drop is not None
        assert drop["severity"] == "WARN"
        assert "Conversion" in drop["suggested_action"]
        assert "7d avg" in drop["suggested_action"]
        appdb._repo = None

    def test_no_fire_when_no_baseline(self, tmp_path, monkeypatch):
        """No daily_stats history → no CONVERSION_DROP can fire."""
        db_path = str(tmp_path / "nobase.db")
        monkeypatch.setenv("DB_PATH", db_path)
        import app.db as appdb
        appdb._repo = None
        appdb.DB_PATH = db_path
        appdb.get_repo()  # ensure table exists

        events = [{
            "event_id": f"V{i:03d}-entry",
            "store_id": "ST1008", "camera_id": "CAM_03",
            "visitor_id": f"V{i:03d}", "event_type": "ENTRY",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_staff": False, "confidence": 0.9, "dwell_ms": 0, "zone_id": None,
        } for i in range(25)]
        anomalies = compute_anomalies(events, pos_csv="nonexistent.csv", store_id="ST1008")
        drop = next((a for a in anomalies if a["type"] == "CONVERSION_DROP"), None)
        assert drop is None
        appdb._repo = None


class TestDeadZoneAndCoverageGap:

    def test_dead_zone_fires_for_covered_zone_with_no_visits(self):
        """Zone with camera coverage but zero visits in last 60 min → DEAD_ZONE."""
        # Inject an event so the function has at least one event to anchor latest_ts
        events = [{
            "event_id": "anchor",
            "store_id": "ST1008", "camera_id": "CAM_02",
            "visitor_id": "V001", "event_type": "ZONE_ENTER",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_staff": False, "confidence": 0.9, "dwell_ms": 0, "zone_id": "LAKME",
        }]
        anomalies = compute_anomalies(events, pos_csv="nonexistent.csv", store_id="ST1008")
        # MINIMALIST (covered) has no events → should be flagged DEAD_ZONE
        dead = [a for a in anomalies if a["type"] == "DEAD_ZONE"]
        assert any(d["value"] == "MINIMALIST" for d in dead)
        for d in dead:
            assert "re-merchandising" in d["suggested_action"]

    def test_coverage_gap_for_uncovered_zone(self):
        """Zone listed in zones_without_camera_coverage → COVERAGE_GAP, not DEAD_ZONE."""
        events = [{
            "event_id": "anchor",
            "store_id": "ST1008", "camera_id": "CAM_02",
            "visitor_id": "V001", "event_type": "ZONE_ENTER",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_staff": False, "confidence": 0.9, "dwell_ms": 0, "zone_id": "LAKME",
        }]
        anomalies = compute_anomalies(events, pos_csv="nonexistent.csv", store_id="ST1008")
        gaps = [a for a in anomalies if a["type"] == "COVERAGE_GAP"]
        # NY_BAE and DERMdoc are in zones_without_camera_coverage config
        gap_values = {g["value"] for g in gaps}
        assert "NY_BAE" in gap_values or "DERMdoc" in gap_values
        # Uncovered zones must NOT also appear as DEAD_ZONE
        dead_values = {a["value"] for a in anomalies if a["type"] == "DEAD_ZONE"}
        assert not (gap_values & dead_values)


# ── Section 3 — Structured logging ────────────────────────────────────────────

class TestStructuredLogging:

    def test_log_emits_json_with_trace_id(self, tmp_path, monkeypatch):
        """Spin up a TestClient, hit /health, capture stdout, parse JSON line."""
        from fastapi.testclient import TestClient

        db_path = str(tmp_path / "log.db")
        monkeypatch.setenv("DB_PATH", db_path)
        import app.db as appdb
        appdb._repo = None
        appdb.DB_PATH = db_path

        import importlib
        import app.main as main
        importlib.reload(main)

        # Capture log output
        captured = io.StringIO()
        handler = logging.StreamHandler(captured)
        handler.setFormatter(main._JsonFormatter())
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            client = TestClient(main.app)
            r = client.get("/health")
            assert r.status_code == 200
            assert "X-Trace-Id" in r.headers
            # Trace-Id header is a valid UUID
            tid = r.headers["X-Trace-Id"]
            uuid.UUID(tid)
        finally:
            root.removeHandler(handler)

        # Find the "request" log line emitted by the middleware
        lines = [ln for ln in captured.getvalue().splitlines() if ln.strip().startswith("{")]
        request_lines = [json.loads(ln) for ln in lines
                         if "endpoint" in json.loads(ln)]
        assert any(d.get("endpoint") == "/health"
                   and d.get("method") == "GET"
                   and "trace_id" in d
                   and "latency_ms" in d
                   and "status_code" in d
                   for d in request_lines)
        appdb._repo = None


# ── Section 4 — STALE_FEED health ─────────────────────────────────────────────

class TestStaleFeedHealth:

    def test_live_when_recent(self, tmp_path, monkeypatch):
        """Event 2 min ago → feed=LIVE."""
        db_path = str(tmp_path / "live.db")
        monkeypatch.setenv("DB_PATH", db_path)
        import app.db as appdb
        appdb._repo = None
        appdb.DB_PATH = db_path
        repo = appdb.SQLiteRepo(db_path)
        appdb._repo = repo

        recent_ts = datetime.now(timezone.utc) - timedelta(seconds=120)
        repo.insert_ignore({
            "event_id": "live-1", "store_id": "ST1008", "camera_id": "CAM_01",
            "visitor_id": "V1", "event_type": "ENTRY",
            "timestamp": recent_ts,
            "zone_id": None, "dwell_ms": 0, "is_staff": False,
            "confidence": 0.9, "metadata": {},
        })

        result = health_status(repo)
        assert result["stores"]["ST1008"]["feed"] == "LIVE"
        appdb._repo = None

    def test_stale_when_old(self, tmp_path, monkeypatch):
        """Event 15 min ago → feed=STALE_FEED."""
        db_path = str(tmp_path / "stale.db")
        monkeypatch.setenv("DB_PATH", db_path)
        import app.db as appdb
        appdb._repo = None
        appdb.DB_PATH = db_path
        repo = appdb.SQLiteRepo(db_path)
        appdb._repo = repo

        old_ts = datetime.now(timezone.utc) - timedelta(seconds=900)  # 15 min
        repo.insert_ignore({
            "event_id": "stale-1", "store_id": "ST1008", "camera_id": "CAM_01",
            "visitor_id": "V1", "event_type": "ENTRY",
            "timestamp": old_ts,
            "zone_id": None, "dwell_ms": 0, "is_staff": False,
            "confidence": 0.9, "metadata": {},
        })

        result = health_status(repo)
        assert result["stores"]["ST1008"]["feed"] == "STALE_FEED"
        # lag_seconds should be in the right ballpark
        assert 800 <= result["stores"]["ST1008"]["lag_seconds"] <= 1000
        appdb._repo = None

    def test_health_payload_shape(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "shape.db")
        monkeypatch.setenv("DB_PATH", db_path)
        import app.db as appdb
        appdb._repo = None
        appdb.DB_PATH = db_path
        repo = appdb.SQLiteRepo(db_path)
        appdb._repo = repo

        result = health_status(repo)
        assert result["service"] == "ok"
        assert "checked_at_utc" in result
        assert "stores" in result
        appdb._repo = None


# ── Section 5 — Confidence calibration ────────────────────────────────────────

class TestConfidenceBand:
    def test_high(self):
        assert confidence_band(0.85) == "HIGH"
        assert confidence_band(0.70) == "HIGH"

    def test_medium(self):
        assert confidence_band(0.55) == "MEDIUM"
        assert confidence_band(0.40) == "MEDIUM"

    def test_low(self):
        assert confidence_band(0.20) == "LOW"
        assert confidence_band(0.39) == "LOW"

    def test_invalid_returns_low(self):
        assert confidence_band("garbage") == "LOW"
        assert confidence_band(None) == "LOW"


class TestSessionConfidence:
    def test_high_avg(self):
        assert session_confidence([0.8, 0.75, 0.9]) == "HIGH"

    def test_low_avg(self):
        assert session_confidence([0.3, 0.35, 0.28]) == "LOW"

    def test_empty_is_low(self):
        assert session_confidence([]) == "LOW"


class TestConfidenceDistribution:
    def test_dist_shape(self):
        sessions = [
            {"confs": [0.8, 0.9]},   # HIGH
            {"confs": [0.5, 0.45]},  # MEDIUM
            {"confs": [0.3, 0.2]},   # LOW
            {"confs": [0.75]},        # HIGH
        ]
        d = session_confidence_distribution(sessions)
        assert d == {"HIGH": 2, "MEDIUM": 1, "LOW": 1}
