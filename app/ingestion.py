"""
app/ingestion.py
────────────────
Idempotent batch event ingestion logic.

This module is the single entry point for writing data into the system.
It sits between the HTTP layer (main.py) and the storage layer (db.py).

Contract:
  - Accepts up to 500 raw event dicts in one call
  - Validates each event with the Pydantic Event model
  - Inserts valid events via insert_ignore (idempotent by event_id)
  - Collects failures with index + error message — NEVER raises on bad data
  - Returns IngestResponse with ingested / duplicates / rejected counts

Why handle validation here instead of in FastAPI's request model?
Because a batch may contain MIXED good/bad events. We want partial success:
  "I ingested events 0,1,3 but rejected event 2 (bad confidence value)"
A single Pydantic model on the request body would reject the ENTIRE batch
if any one event is invalid. That's too strict for a real-time pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .db import SQLiteRepo, get_repo
from .models import Event, IngestResponse, RejectedEvent


def ingest_events(raw_events: list[Any], repo: SQLiteRepo | None = None) -> IngestResponse:
    """
    Validate and store a batch of raw event dicts.

    Args:
        raw_events: List of dicts (from JSON body). Each must match the Event schema.
        repo:       SQLiteRepo instance. Pass a test repo in unit tests; defaults to
                    the module-level singleton for production use.

    Returns:
        IngestResponse(
            ingested=N,          # events newly written to DB
            duplicates=N,        # events whose event_id already existed (silently ignored)
            rejected=[...]       # events that failed validation — list of {index, error, event_id}
        )

    Guarantees:
        - Never raises an exception (catches all validation and DB errors per-event)
        - Rejected events don't block valid events in the same batch
        - Sending the same batch twice → second call returns ingested=0, duplicates=N
    """
    if repo is None:
        repo = get_repo()

    ingested = 0
    duplicates = 0
    rejected: list[RejectedEvent] = []

    for i, raw in enumerate(raw_events):
        # Try to get event_id from the raw dict for error reporting,
        # even if the full event fails validation
        event_id_hint = raw.get("event_id") if isinstance(raw, dict) else None

        # ── Validate with Pydantic ─────────────────────────────────────────
        try:
            event = Event.model_validate(raw)
        except ValidationError as exc:
            # Pydantic gives a detailed error list — convert to a readable string
            rejected.append(RejectedEvent(
                index=i,
                error=str(exc),
                event_id=event_id_hint,
            ))
            continue
        except Exception as exc:
            # Catch-all for anything unexpected (e.g. raw is not a dict)
            rejected.append(RejectedEvent(
                index=i,
                error=f"Unexpected validation error: {exc}",
                event_id=event_id_hint,
            ))
            continue

        # ── Write to database ──────────────────────────────────────────────
        event_dict = event.model_dump()
        inserted = repo.insert_ignore(event_dict)

        if inserted:
            ingested += 1
        else:
            # insert_ignore returned False → event_id already in DB
            duplicates += 1

    return IngestResponse(ingested=ingested, duplicates=duplicates, rejected=rejected)
