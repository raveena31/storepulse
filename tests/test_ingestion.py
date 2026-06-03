# PROMPT: "Generate pytest cases for app/ingestion.py covering: idempotent double-ingest
#          (same event_id twice → 1 row in DB), partial-success batch (valid + invalid mixed),
#          structured rejected list with index + error."
# CHANGES MADE: AI's first draft asserted the second double-ingest call returned 'ingested=2'
#               because the events had no explicit event_id — Pydantic was generating fresh
#               uuid4 each time, so they weren't actually duplicates. I added explicit
#               event_id values to the fixtures and re-asserted ingested=0, duplicates=2.
#               This caught a real demo-breaker: clients that don't set event_id will never
#               benefit from idempotency.

import os
import tempfile
import pytest
from datetime import datetime, timezone

from app.db import SQLiteRepo
from app.ingestion import ingest_events

VALID_EVENT = {
    "event_id": "fixed-evt-id-0001",
    "store_id": "ST1008",
    "camera_id": "CAM_01",
    "visitor_id": "V001",
    "event_type": "ENTRY",
    "timestamp": "2024-01-15T10:00:00+00:00",
    "confidence": 0.9,
}

VALID_EVENT_2 = {
    "event_id": "fixed-evt-id-0002",
    "store_id": "ST1008",
    "camera_id": "CAM_01",
    "visitor_id": "V002",
    "event_type": "EXIT",
    "timestamp": "2024-01-15T10:30:00+00:00",
    "confidence": 0.85,
}


@pytest.fixture
def repo(tmp_path):
    db_file = str(tmp_path / "test.db")
    return SQLiteRepo(db_path=db_file)


def test_double_ingest_produces_duplicates(repo):
    """Ingest same batch twice → second call has duplicates > 0, no new DB rows."""
    batch = [dict(VALID_EVENT), dict(VALID_EVENT_2)]

    r1 = ingest_events(batch, repo=repo)
    assert r1.ingested == 2
    assert r1.duplicates == 0
    assert r1.rejected == []

    r2 = ingest_events(batch, repo=repo)
    assert r2.ingested == 0
    assert r2.duplicates == 2
    assert r2.rejected == []

    rows = repo.events_for("ST1008")
    assert len(rows) == 2  # still exactly 2


def test_partial_rejection(repo):
    """Batch with 2 malformed events → rest ingested, 2 in rejected list."""
    bad1 = {"store_id": "ST1008"}  # missing required fields
    bad2 = {
        "store_id": "ST1008",
        "camera_id": "CAM_01",
        "visitor_id": "V003",
        "event_type": "ENTRY",
        "timestamp": "not-a-datetime",  # invalid
        "confidence": 0.9,
    }
    good = dict(VALID_EVENT)

    batch = [bad1, good, bad2]
    r = ingest_events(batch, repo=repo)

    assert r.ingested == 1
    assert len(r.rejected) == 2
    assert r.rejected[0].index == 0
    assert r.rejected[1].index == 2
