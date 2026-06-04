"""
app/main.py
───────────
FastAPI application — all HTTP endpoints + frontend dashboard.

Endpoints:
    GET  /                                → Dashboard UI (HTML)
    GET  /stream/{camera_id}              → MJPEG video stream with detection overlay
    GET  /health                          → service alive + feed lag per store
    POST /events/ingest                   → idempotent batch event ingest (up to 500)
    GET  /events/recent                   → last N events for a store (for dashboard event feed)
    GET  /stores/{store_id}/metrics       → conversion rate, dwell, queue, revenue
    GET  /stores/{store_id}/funnel        → 4-stage funnel with drop-off %
    GET  /stores/{store_id}/heatmap       → zone visit counts, dwell, attention vs sales
    GET  /stores/{store_id}/anomalies     → queue spike, conversion drop, dead zone alerts

All store endpoints accept an optional ?date=YYYY-MM-DD query parameter.
  - Default: today UTC
  - Use footage date for historical runs (e.g. ?date=2026-04-10)
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .db import get_repo
from .health import get_health
from .ingestion import ingest_events
from .models import IngestRequest, IngestResponse
from .metrics import compute_metrics
from .funnel import compute_funnel
from .heatmap import compute_heatmap
from .anomalies import compute_anomalies
from .pos import POS_CSV_PATH

# ── Structured JSON logging ───────────────────────────────────────────────────
#
# All logs are emitted as one JSON object per line on stdout (Docker-friendly).
# Format: {"ts_utc","level","trace_id","store_id","endpoint","method",
#          "latency_ms","event_count","status_code","msg"}
# A trace_id is generated per request and attached to the request.state so it
# can be retrieved inside error responses.

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts_utc":   datetime.now(timezone.utc).isoformat(),
            "level":    record.levelname,
            "logger":   record.name,
            "msg":      record.getMessage(),
        }
        # Merge any extras attached via logger.info(..., extra={...})
        for key in ("trace_id", "store_id", "endpoint", "method",
                    "latency_ms", "event_count", "status_code"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        return json.dumps(payload, default=str)


def _configure_logging() -> logging.Logger:
    root = logging.getLogger()
    # Remove uvicorn's default handler so we control the format
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return logging.getLogger("store_intelligence")


logger = _configure_logging()


app = FastAPI(
    title="Store Intelligence API",
    description="Real-time retail analytics from CCTV footage.",
    version="1.0.0",
)

POS_CSV = os.getenv("POS_CSV_PATH", POS_CSV_PATH)

# Serve static files (dashboard HTML/CSS/JS)
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── Helper to extract store_id from path ──────────────────────────────────────

def _store_id_from_path(path: str) -> Optional[str]:
    """
    Extract store_id from URL paths like /stores/{store_id}/metrics.
    Returns None when the path has no store_id.
    """
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "stores":
        return parts[1]
    return None


# ── Middleware: structured request logging with trace_id ──────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        # Unhandled — log + return structured 503 (never expose stack trace)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.error(
            "unhandled exception",
            extra={
                "trace_id":    trace_id,
                "endpoint":    request.url.path,
                "method":      request.method,
                "store_id":    _store_id_from_path(request.url.path),
                "latency_ms":  latency_ms,
                "status_code": 503,
            },
            exc_info=exc,
        )
        return JSONResponse(
            status_code=503,
            content={
                "error":     "internal_server_error",
                "trace_id":  trace_id,
                "message":   "An unexpected error occurred. Check logs.",
            },
            headers={"X-Trace-Id": trace_id},
        )

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Trace-Id"] = trace_id

    # Pull event_count if it was set on request.state by ingest handler
    event_count = getattr(request.state, "event_count", None)

    logger.info(
        "request",
        extra={
            "trace_id":    trace_id,
            "endpoint":    request.url.path,
            "method":      request.method,
            "store_id":    _store_id_from_path(request.url.path),
            "latency_ms":  latency_ms,
            "event_count": event_count,
            "status_code": response.status_code,
        },
    )
    return response


# ── Global exception handler (HTTPException pass-through, others → 503) ──────

from fastapi.exceptions import HTTPException as FastAPIHTTPException

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    logger.error(
        "unhandled exception in handler",
        extra={
            "trace_id":    trace_id,
            "endpoint":    request.url.path,
            "method":      request.method,
            "store_id":    _store_id_from_path(request.url.path),
            "status_code": 503,
        },
        exc_info=exc,
    )
    return JSONResponse(
        status_code=503,
        content={
            "error":    "internal_server_error",
            "trace_id": trace_id,
            "message":  "An unexpected error occurred. Check logs.",
        },
        headers={"X-Trace-Id": trace_id},
    )


# ── Dashboard UI ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Serve the analytics dashboard at the root."""
    html_path = os.path.join(_static_dir, "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/dashboard/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_alias():
    """Dashboard alias — /dashboard/ serves the same page as /."""
    html_path = os.path.join(_static_dir, "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/dashboard/stream/{store_id}", include_in_schema=False)
async def dashboard_sse(store_id: str, date: Optional[str] = None):
    """
    Server-Sent Events stream for the dashboard at /dashboard/.

    Every 2 seconds emits a JSON payload:
        {conversion_rate, unique_visitors, buyers,
         current_queue_depth, revenue_inr, ts_utc}
    """
    async def event_generator():
        while True:
            try:
                events = _day_events(store_id, date)
                m = compute_metrics(events, POS_CSV)
                payload = {
                    "conversion_rate":     m.get("conversion_rate", 0.0),
                    "unique_visitors":     m.get("unique_visitors", 0),
                    "buyers":              m.get("buyers", 0),
                    "current_queue_depth": m.get("current_queue_depth", 0),
                    "revenue_inr":         m.get("revenue_inr", 0.0),
                    "ts_utc":              datetime.now(timezone.utc).isoformat(),
                }
                yield f"data: {json.dumps(payload)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Breakdown endpoint — transparency on every number ─────────────────────────

@app.get("/stores/{store_id}/breakdown")
def breakdown(store_id: str, date: Optional[str] = None):
    """
    Returns a complete split of every number on the dashboard:
    per-camera unique tracks, per-zone visits, per-event-type counts,
    confidence band distribution, and a provenance section explaining
    each headline metric.
    """
    from .breakdown import compute_breakdown
    events = _day_events(store_id, date)
    result = compute_breakdown(events)
    result["store_id"] = store_id
    result["date"] = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    result["as_of"] = datetime.now(timezone.utc).isoformat()
    return result


# ── Demo replay — plays back events chronologically at high speed ────────────

@app.get("/dashboard/replay/{store_id}", include_in_schema=False)
async def replay_sse(
    store_id: str,
    date: Optional[str] = None,
    speed: float = 30.0,
):
    """
    Replays events from the DB at `speed`× real-time speed.
    Each tick, recomputes metrics on the cumulative subset of events seen so far.

    Perfect for the demo video: visitor counter ticks from 0 -> 93 naturally.

    Query params:
        speed: playback multiplier (default 30x = 1 hour replayed in 2 minutes)
    """
    async def gen():
        all_events = _day_events(store_id, date)
        if not all_events:
            yield "data: " + json.dumps({"replay": "no_data"}) + "\n\n"
            return
        all_events.sort(key=lambda e: e.get("ts", ""))
        clip_start = datetime.fromisoformat(
            all_events[0]["ts"].replace("Z", "+00:00")
        )

        # Emit metrics every 1.5s wall-clock while marching through events
        wall_step = 1.5
        cursor_idx = 0
        clip_cursor = clip_start

        from .pos import POS_CSV_PATH as _POS
        pos = POS_CSV or _POS

        # First emit a zero baseline
        yield "data: " + json.dumps({
            "replay_progress": 0,
            "clip_time_utc": clip_cursor.isoformat(),
            "events_seen": 0,
            "events_total": len(all_events),
            "metrics": compute_metrics([], pos),
        }) + "\n\n"

        while cursor_idx < len(all_events):
            await asyncio.sleep(wall_step)
            # Advance clip cursor by wall_step * speed seconds
            from datetime import timedelta as _td
            clip_cursor += _td(seconds=wall_step * speed)

            # Include all events up to clip_cursor
            while cursor_idx < len(all_events):
                ts_str = all_events[cursor_idx]["ts"]
                evt_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if evt_ts > clip_cursor:
                    break
                cursor_idx += 1

            subset = all_events[:cursor_idx]
            m = compute_metrics(subset, pos)
            from .breakdown import compute_breakdown
            bd = compute_breakdown(subset)

            payload = {
                "replay_progress": round(cursor_idx / len(all_events) * 100, 1),
                "clip_time_utc": clip_cursor.isoformat(),
                "events_seen": cursor_idx,
                "events_total": len(all_events),
                "metrics": {
                    "unique_visitors":     m.get("unique_visitors", 0),
                    "buyers":              m.get("buyers", 0),
                    "conversion_rate":     m.get("conversion_rate", 0.0),
                    "current_queue_depth": m.get("current_queue_depth", 0),
                    "revenue_inr":         m.get("revenue_inr", 0.0),
                },
                "headline_breakdown": bd["headline"],
                "by_camera":  bd["by_camera"],
                "by_event_type": bd["by_event_type"],
                "latest_events": [
                    {
                        "ts": e["ts"], "camera_id": e["camera_id"],
                        "visitor_id": e["visitor_id"], "event_type": e["event_type"],
                        "zone_id": e.get("zone_id"), "is_staff": bool(e.get("is_staff")),
                    }
                    for e in all_events[max(0, cursor_idx - 8):cursor_idx]
                ][::-1],
            }
            yield "data: " + json.dumps(payload) + "\n\n"

        # Final state — held for 5s so the dashboard shows the end-state
        yield "data: " + json.dumps({"replay_progress": 100, "done": True}) + "\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── MJPEG video stream ─────────────────────────────────────────────────────────

@app.get("/stream/{camera_id}", include_in_schema=False)
async def video_stream(camera_id: str, skip: int = 2):
    """
    MJPEG stream for a camera.

    Runs YOLOv8n detection on each frame, draws bounding boxes + zone overlays,
    and streams annotated JPEG frames to the browser.

    Query params:
      skip: process every (skip+1)th frame. Default 2 = every 3rd frame (~10fps from 30fps).

    Use in HTML: <img src="/stream/CAM_01">
    """
    from .stream import generate_stream
    return StreamingResponse(
        generate_stream(camera_id, skip=skip),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Recent events (for dashboard event feed) ───────────────────────────────────

@app.get("/events/recent")
def recent_events(store_id: str, date: Optional[str] = None, limit: int = 40):
    """
    Return the most recent N events for a store on a given date.
    Used by the dashboard event-feed panel (polls every 5 seconds).
    """
    events = _day_events(store_id, date)
    # Return newest-first
    events_sorted = sorted(events, key=lambda e: e.get("ts", ""), reverse=True)
    return events_sorted[:limit]


# ── Helper: load events for a given date ──────────────────────────────────────

def _day_events(store_id: str, date: Optional[str] = None) -> list[dict]:
    repo = get_repo()
    if date:
        try:
            day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day.replace(hour=23, minute=59, second=59)
    return repo.events_for(store_id, start=day, end=day_end)


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", summary="Service health and feed lag per store")
def health():
    return get_health()


@app.post("/events/ingest", response_model=IngestResponse)
async def ingest(request: Request):
    """Idempotent batch event ingest. Returns 207 if any events rejected."""
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    if isinstance(body, dict):
        raw_events = body.get("events", [])
    elif isinstance(body, list):
        raw_events = body
    else:
        raise HTTPException(status_code=400, detail="Body must be JSON object or array")

    if len(raw_events) > 500:
        raise HTTPException(status_code=400, detail="Batch exceeds 500 events")

    # Stamp event_count for structured logging
    request.state.event_count = len(raw_events)

    result = ingest_events(raw_events, repo=get_repo())
    return JSONResponse(content=result.model_dump(), status_code=207 if result.rejected else 200)


@app.get("/stores/{store_id}/metrics")
def metrics(store_id: str, date: Optional[str] = None):
    events = _day_events(store_id, date)
    result = compute_metrics(events, POS_CSV)
    use_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    result["store_id"] = store_id
    result["date"] = use_date
    result["as_of"] = datetime.now(timezone.utc).isoformat()
    # EC-43: flag the response when late events were detected in this window
    from .sessions import has_late_events
    result["metrics_pending_late_events"] = has_late_events(events, grace_s=5)

    # Persist today's stats so the CONVERSION_DROP anomaly has a baseline
    from .metrics import upsert_today_stats
    upsert_today_stats(result, store_id=store_id, date_str=use_date)
    return result


@app.get("/stores/{store_id}/funnel")
def funnel(store_id: str, date: Optional[str] = None):
    events = _day_events(store_id, date)
    result = compute_funnel(events, POS_CSV)
    result["store_id"] = store_id
    result["date"] = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    result["as_of"] = datetime.now(timezone.utc).isoformat()
    return result


@app.get("/stores/{store_id}/heatmap")
def heatmap(store_id: str, date: Optional[str] = None):
    events = _day_events(store_id, date)
    result = compute_heatmap(events, POS_CSV)
    result["store_id"] = store_id
    result["date"] = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    result["as_of"] = datetime.now(timezone.utc).isoformat()
    return result


@app.get("/stores/{store_id}/anomalies")
def anomalies(store_id: str, date: Optional[str] = None):
    events = _day_events(store_id, date)
    use_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    items = compute_anomalies(
        events, pos_csv=POS_CSV, store_id=store_id, today_date_str=use_date,
    )
    return {
        "store_id": store_id,
        "anomalies": items,
        "count": len(items),
        "date": use_date,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


# ── SSE metrics stream (dashboard Band 1) ─────────────────────────────────────

@app.get("/api/live", include_in_schema=False)
async def stream_metrics(store_id: str = "ST1008", date: Optional[str] = None):
    """
    Server-Sent Events stream for the dashboard.
    Pushes metrics + anomalies + heatmap every 3 seconds.
    Dashboard connects with: new EventSource('/stream/metrics?store_id=ST1008&date=...')
    """
    async def event_generator():
        while True:
            try:
                events = _day_events(store_id, date)
                payload = {
                    "metrics":   compute_metrics(events, POS_CSV),
                    "anomalies": compute_anomalies(events, POS_CSV),
                    "heatmap":   compute_heatmap(events, POS_CSV),
                    "funnel":    compute_funnel(events, POS_CSV),
                    "ts_utc":    datetime.now(timezone.utc).isoformat(),
                }
                yield f"data: {json.dumps(payload)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            await asyncio.sleep(3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Report export ─────────────────────────────────────────────────────────────

@app.get("/reports/export", include_in_schema=False)
def report_export(
    store_id: str = "ST1008",
    date: Optional[str] = None,
    format: str = "json",
):
    """
    Export a store report in CSV or JSON format.
    Used by the dashboard Download buttons.
    """
    use_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    events = _day_events(store_id, date)

    m = compute_metrics(events, POS_CSV)
    f = compute_funnel(events, POS_CSV)
    h = compute_heatmap(events, POS_CSV)
    a = compute_anomalies(events, POS_CSV)

    report = {
        "store_id":   store_id,
        "date":       use_date,
        "metrics":    m,
        "funnel":     f.get("stages", {}),
        "heatmap":    h.get("zones", {}),
        "anomalies":  a,
        "generated":  datetime.now(timezone.utc).isoformat(),
    }

    if format == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Metrics section
        writer.writerow(["METRICS"])
        writer.writerow(["metric", "value"])
        for k, v in m.items():
            if not isinstance(v, dict):
                writer.writerow([k, v])

        # Funnel section
        writer.writerow([])
        writer.writerow(["FUNNEL"])
        writer.writerow(["stage", "count", "drop_off_pct"])
        for stage, data in f.get("stages", {}).items():
            writer.writerow([stage, data.get("count", 0), data.get("drop_off_from_previous_pct", 0)])

        # Heatmap section
        writer.writerow([])
        writer.writerow(["ZONE HEATMAP"])
        writer.writerow(["zone", "visit_count", "avg_dwell_ms", "normalised_score"])
        for zone, data in h.get("zones", {}).items():
            writer.writerow([zone, data.get("visit_count", 0),
                             data.get("avg_dwell_ms", 0), data.get("normalised_score", 0)])

        content = buf.getvalue()
        return StreamingResponse(
            iter([content]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=report_{store_id}_{use_date}.csv"},
        )

    # Default: JSON
    return JSONResponse(
        content=report,
        headers={"Content-Disposition": f"attachment; filename=report_{store_id}_{use_date}.json"},
    )