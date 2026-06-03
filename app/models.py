"""
app/models.py
─────────────
Pydantic v2 schema definitions for the Store Intelligence API.

Every piece of data flowing through this system is validated here first.
If an event doesn't match this schema it is REJECTED at the ingest boundary —
it never touches the database, and the caller gets a clear error message.

Key design rules enforced here:
  - timestamp MUST be tz-aware UTC  (naive datetimes are rejected)
  - confidence MUST be in [0.0, 1.0]
  - Batch size is capped at 500 events per request
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Event metadata ─────────────────────────────────────────────────────────────

class EventMeta(BaseModel):
    """
    Optional extra fields attached to any event.
    All fields are optional so legacy producers don't break.
    """
    queue_depth: Optional[int] = None       # how many people in billing queue at this moment
    sku_zone: Optional[str] = None          # product zone the event relates to
    session_seq: Optional[int] = None       # sequence number within the visitor's session
    confidence_band: Optional[str] = None   # "HIGH" | "MED" | "LOW" — human-readable confidence
    inferred: bool = False                  # True if this event was inferred (e.g. dangling session close)


# ── Event type literal ─────────────────────────────────────────────────────────

# Only these 8 strings are valid event_type values.
# Anything else is rejected by Pydantic at validation time.
EventType = Literal[
    "ENTRY",                  # person crossed the store entry line (inward)
    "EXIT",                   # person crossed the store entry line (outward)
    "ZONE_ENTER",             # person stepped into a named product zone
    "ZONE_EXIT",              # person left a named product zone
    "ZONE_DWELL",             # person spent dwell_ms milliseconds inside a zone
    "BILLING_QUEUE_JOIN",     # person joined the billing counter queue
    "BILLING_QUEUE_ABANDON",  # person left the queue without completing a purchase
    "REENTRY",                # same visitor_id detected entering again — NOT a new visitor
]


# ── Core Event schema ──────────────────────────────────────────────────────────

class Event(BaseModel):
    """
    The atomic unit of data in this system.

    One event = one thing that happened to one person at one moment.
    Events are stored immutably; analytics are computed from them on read.

    Idempotency: event_id is the deduplication key.
    Sending the same event_id twice → second one is silently ignored.
    """

    # Identity
    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event ID. Auto-generated if not provided. Used for deduplication.",
    )
    store_id: str = Field(..., description="Store identifier, e.g. 'ST1008'")
    camera_id: str = Field(..., description="Camera that detected this event, e.g. 'CAM_01'")
    visitor_id: str = Field(
        ...,
        description="Unique person ID assigned by the tracker, e.g. 'ST1008_T0042'. "
                    "REENTRY events must reuse the same visitor_id — never create a new one.",
    )

    # What happened
    event_type: EventType
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp. MUST be tz-aware (tzinfo required). Raises if naive.",
    )
    zone_id: Optional[str] = Field(None, description="Zone where event occurred, e.g. 'LAKME'")
    dwell_ms: int = Field(0, description="Milliseconds spent in zone (for ZONE_DWELL events)")

    # Classification
    is_staff: bool = Field(
        False,
        description="True if this person is a staff member. "
                    "Sticky: if ANY event for a visitor_id is is_staff=True, "
                    "the entire session is excluded from customer metrics.",
    )
    confidence: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="AI detection confidence [0.0, 1.0]. "
                    "Low-confidence events are NEVER dropped — they are flagged "
                    "with confidence_band='LOW' and included in all calculations.",
    )
    metadata: EventMeta = Field(default_factory=EventMeta)

    @field_validator("timestamp")
    @classmethod
    def must_be_utc(cls, v: datetime) -> datetime:
        """Reject naive datetimes. All timestamps in this system are UTC."""
        if v.tzinfo is None:
            raise ValueError(
                "timestamp must be tz-aware UTC. "
                "Add '+00:00' or 'Z' suffix, e.g. '2026-04-10T14:40:00+00:00'"
            )
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        """Redundant with ge/le but gives a clearer error message."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v


# ── Ingest request / response ──────────────────────────────────────────────────

# ── Confidence calibration (EC-50) ────────────────────────────────────────────

def confidence_band(conf: float) -> str:
    """
    Map a confidence value to a coarse band.

    Bands:
      HIGH:   conf >= 0.70
      MEDIUM: 0.40 <= conf < 0.70
      LOW:    conf <  0.40

    Low-confidence detections are NEVER dropped — they're labelled instead.
    EC-50 implementation.
    """
    try:
        c = float(conf)
    except (TypeError, ValueError):
        return "LOW"
    if c >= 0.70:
        return "HIGH"
    if c >= 0.40:
        return "MEDIUM"
    return "LOW"


def session_confidence(event_confs: list[float]) -> str:
    """
    Average confidence across all events in a session, then band it.

    Empty list → LOW (no signal).
    EC-50 implementation.
    """
    if not event_confs:
        return "LOW"
    try:
        nums = [float(c) for c in event_confs]
    except (TypeError, ValueError):
        return "LOW"
    if not nums:
        return "LOW"
    return confidence_band(sum(nums) / len(nums))


class IngestRequest(BaseModel):
    """
    Wrapper for a batch of events sent to POST /events/ingest.
    Maximum 500 events per request (enforced by max_length).
    """
    events: list[Event] = Field(..., max_length=500)


class RejectedEvent(BaseModel):
    """Describes a single event that failed validation during ingest."""
    index: int = Field(..., description="Position of this event in the original batch (0-indexed)")
    error: str = Field(..., description="Validation error message")
    event_id: Optional[str] = Field(None, description="event_id if it was present in the bad payload")


class IngestResponse(BaseModel):
    """
    Response from POST /events/ingest.

    HTTP 200 → all events accepted (rejected list is empty)
    HTTP 207 → partial success (some events rejected, check rejected list)
    HTTP 400 → entire request body is invalid JSON (never returned for bad event data)
    HTTP 5xx → NEVER returned for bad event data (business errors are not server errors)
    """
    ingested: int = Field(..., description="Number of new events inserted into the database")
    duplicates: int = Field(..., description="Number of events skipped because event_id already existed")
    rejected: list[RejectedEvent] = Field(
        default_factory=list,
        description="Events that failed Pydantic validation. Includes index and error message.",
    )
