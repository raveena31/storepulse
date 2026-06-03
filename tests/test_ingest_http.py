# PROMPT: "Generate FastAPI TestClient cases for /events/ingest covering: EC-44 idempotency
#          at the HTTP layer (separate POST calls, not just batches), EC-45 partial-success
#          (mixed valid/invalid → HTTP 207 with rejected[index, error] entries)."
# CHANGES MADE: AI sent both POST calls in a single TestClient session and asserted the
#               second returned status_code=409 (Conflict). That's wrong on two counts:
#               our contract is HTTP 200 + duplicates=N (not 409), and the test never
#               touched the DB to verify the row count. I rewrote to (a) check the response
#               dict has duplicates==1, ingested==0, and (b) open a sqlite3 connection
#               directly and assert SELECT COUNT(*) WHERE event_id=? returns exactly 1.
#               That second check is what makes the test demo-proof — without it, idempotency
#               could be lying.

import os
import sqlite3

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Spin up the FastAPI app with a fresh tmp DB per test."""
    db_path = str(tmp_path / "edge_test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("POS_CSV_PATH", "nonexistent.csv")
    # Force the app.db singleton to use the tmp path
    import app.db as appdb
    appdb._repo = None
    appdb.DB_PATH = db_path
    # Re-import metric/funnel/etc modules so they pick up POS_CSV_PATH
    import importlib
    import app.pos as pos
    importlib.reload(pos)
    import app.main as main
    importlib.reload(main)
    yield TestClient(main.app), db_path
    appdb._repo = None


VALID_EVENT = {
    "event_id":   "ec44-fixed-id",
    "store_id":   "ST1008",
    "camera_id":  "CAM_01",
    "visitor_id": "V001",
    "event_type": "ENTRY",
    "timestamp":  "2026-04-10T10:00:00+00:00",
    "confidence": 0.9,
}


# ── EC-44  IDEMPOTENT INGEST AT HTTP LAYER ────────────────────────────────────

def test_ec44_double_post_same_event_id(client):
    """POST identical event_id twice in SEPARATE HTTP calls → only 1 row in DB."""
    c, db_path = client

    r1 = c.post("/events/ingest", json={"events": [VALID_EVENT]})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["ingested"] == 1
    assert d1["duplicates"] == 0

    # Second POST with identical event_id
    r2 = c.post("/events/ingest", json={"events": [VALID_EVENT]})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["ingested"] == 0
    assert d2["duplicates"] == 1

    # DB sanity: exactly one row for this event_id
    conn = sqlite3.connect(db_path)
    n = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_id = ?", (VALID_EVENT["event_id"],)
    ).fetchone()[0]
    assert n == 1


# ── EC-45  PARTIAL SUCCESS WITH STRUCTURED ERRORS ─────────────────────────────

def test_ec45_partial_success_structured(client):
    """5 events: index 2 bad confidence, index 4 missing store_id → 3 ingested, 2 rejected, HTTP 207."""
    c, _ = client

    events = [
        {**VALID_EVENT, "event_id": "ok-0", "visitor_id": "V0"},
        {**VALID_EVENT, "event_id": "ok-1", "visitor_id": "V1"},
        # index 2 — invalid confidence
        {**VALID_EVENT, "event_id": "bad-2", "visitor_id": "V2", "confidence": 1.5},
        {**VALID_EVENT, "event_id": "ok-3", "visitor_id": "V3"},
        # index 4 — missing store_id
        {"event_id": "bad-4", "camera_id": "CAM_01", "visitor_id": "V4",
         "event_type": "ENTRY", "timestamp": "2026-04-10T10:00:00+00:00",
         "confidence": 0.5},
    ]

    r = c.post("/events/ingest", json={"events": events})
    assert r.status_code == 207, f"Expected 207 Multi-Status, got {r.status_code}"

    d = r.json()
    assert d["ingested"] == 3
    assert len(d["rejected"]) == 2

    # Check the rejected indices and error contents
    rejected = {item["index"]: item["error"] for item in d["rejected"]}
    assert 2 in rejected
    assert "confidence" in rejected[2].lower()
    assert 4 in rejected
    assert "store_id" in rejected[4].lower()
