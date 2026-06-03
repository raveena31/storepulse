"""
pipeline/emit.py
────────────────
Build validated event dicts and POST them to the ingest API.

This module is the final step in the pipeline:
  detection → tracking → zone assignment → emit (this file)

build_event() constructs an event dict that matches the Pydantic Event schema.
post_events() sends a batch of events to POST /events/ingest.

visitor_id format:
  "{store_id}_T{track_id:04d}"
  e.g. "ST1008_T0042" for track_id=42 in store ST1008.

  The track_id is assigned by ByteTrack and is unique per camera per session.
  Since each camera has its own tracker, we don't need to include camera_id
  in the visitor_id — the pipeline uses one model.track() call per camera,
  so track IDs don't overlap between cameras for the same person.

  IMPORTANT: If the same person appears in CAM_01 (track 5) and CAM_02 (track 12),
  they'll get two different visitor_ids: ST1008_T0005 and ST1008_T0012.
  Cross-camera identity matching (re-identification) is a future enhancement.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx


def build_event(
    track_id: int,
    event_type: str,
    store_id: str,
    camera_id: str,
    timestamp: datetime,
    zone_id: Optional[str] = None,
    dwell_ms: int = 0,
    is_staff: bool = False,
    confidence: float = 0.0,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Build a schema-valid event dict ready to POST to /events/ingest.

    Args:
        track_id:   ByteTrack track ID (integer assigned by tracker)
        event_type: One of the 8 event types (ENTRY, EXIT, ZONE_ENTER, etc.)
        store_id:   Store identifier, e.g. "ST1008"
        camera_id:  Camera that generated this event, e.g. "CAM_01"
        timestamp:  UTC datetime. Naive datetimes are coerced to UTC.
        zone_id:    Zone name (e.g. "LAKME") or None for non-zone events
        dwell_ms:   Milliseconds spent in zone (for ZONE_DWELL events)
        is_staff:   True if this person is known to be staff
        confidence: YOLOv8 detection confidence [0.0, 1.0]
        metadata:   Optional extra fields (queue_depth, confidence_band, etc.)

    Returns:
        Dict matching the Event Pydantic schema.
        Ready to be included in {"events": [...]} body for POST /events/ingest.
    """
    # Ensure timestamp is tz-aware (the Event validator requires this)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    # Clamp confidence to [0,1] — emit even low-confidence detections (EC-50)
    safe_conf = max(0.0, min(1.0, confidence))

    # Use the canonical band function (EC-50)
    from app.models import confidence_band as _band
    band = _band(safe_conf)

    return {
        "event_id": str(uuid.uuid4()),          # unique ID for deduplication
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": f"{store_id}_T{track_id:04d}",  # globally unique per camera session
        "event_type": event_type,
        "timestamp": timestamp.isoformat(),     # ISO-8601 UTC string
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": safe_conf,
        "metadata": {
            **(metadata or {}),
            "confidence_band": band,
        },
    }


def post_events(
    events: list[dict],
    api_url: str = "http://localhost:8000/events/ingest",
) -> dict:
    """
    POST a batch of events to the ingest API.

    Args:
        events:   List of event dicts (as built by build_event())
        api_url:  Full URL of the /events/ingest endpoint

    Returns:
        Response dict from the API:
            {"ingested": N, "duplicates": N, "rejected": [...]}
        Or an error dict on network failure:
            {"error": "...", "status_code": N}

    Notes:
        - Sends up to 500 events in one request (API limit)
        - Never raises — returns error dict instead
        - Timeout is 30 seconds — increase if the API is under heavy load
    """
    payload = {"events": events}
    try:
        resp = httpx.post(api_url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "status_code": e.response.status_code}
    except Exception as e:
        return {"error": str(e)}
