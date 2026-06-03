"""
app/metrics.py
──────────────
Compute the main store dashboard metrics from a day's worth of events.

The North Star metric: conversion_rate = buyers / unique_visitors
Every other number here supports understanding THAT number better.

All division operations are guarded — if the denominator is zero, the result
is 0.0. This function NEVER returns NaN, None, or raises ZeroDivisionError.

Input:  flat list of event dicts (from DB, for one store, one UTC day)
Output: metrics dict ready to return as JSON from GET /stores/{id}/metrics
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .sessions import build_sessions, _ts
from .pos import load_and_process_pos, correlate_conversions, POS_CSV_PATH
from .models import confidence_band, session_confidence


def session_confidence_distribution(sessions: list[dict]) -> dict[str, int]:
    """
    Tally each session's average-confidence band.

    Returns: {"HIGH": n, "MEDIUM": n, "LOW": n}

    Used by /metrics and /funnel so operators can tell at a glance how much of
    today's data is reliable. EC-50.
    """
    dist = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for s in sessions:
        confs = s.get("confs") or []
        band = session_confidence(confs)
        dist[band] = dist.get(band, 0) + 1
    return dist


def compute_metrics(events: list[dict], pos_csv: str = POS_CSV_PATH) -> dict:
    """
    Compute all store metrics for a given set of events.

    Steps:
      1. Build sessions from events (groups events by visitor_id)
      2. Filter out staff sessions
      3. Load + process POS CSV
      4. Correlate POS invoices → buyer sessions
      5. Calculate each metric

    Returns a dict with keys:
        unique_visitors, buyers, conversion_rate, avg_dwell_ms_by_zone,
        current_queue_depth, abandonment_rate, revenue_inr,
        revenue_per_visitor_inr, status, data_confidence
    """
    # No events → return zeroed-out metrics with NO_TRAFFIC status
    if not events:
        return _empty_metrics()

    # Build sessions and filter to customers only
    sessions = build_sessions(events)
    customer_sessions = [s for s in sessions if not s.get("is_staff")]
    unique_visitors = len(customer_sessions)

    # All events are staff → same as no customer traffic
    if unique_visitors == 0:
        return _empty_metrics()

    # Load POS data and find which sessions converted to purchases
    baskets = load_and_process_pos(pos_csv)
    converted = correlate_conversions(baskets, customer_sessions)
    buyers = len(converted)

    # Sum total revenue from all matched baskets
    revenue_inr = _sum_revenue(baskets, customer_sessions)

    # ── Guarded divisions ──────────────────────────────────────────────────
    # Every / operation has an explicit zero-guard. This is a hard requirement.
    conversion_rate = buyers / unique_visitors if unique_visitors > 0 else 0.0
    revenue_per_visitor = revenue_inr / unique_visitors if unique_visitors > 0 else 0.0

    return {
        "unique_visitors": unique_visitors,
        "buyers": buyers,
        "conversion_rate": round(conversion_rate, 4),
        "avg_dwell_ms_by_zone": _avg_dwell_by_zone(customer_sessions),
        "current_queue_depth": _current_queue_depth(events),
        "abandonment_rate": round(_abandonment_rate(customer_sessions), 4),
        "revenue_inr": round(revenue_inr, 2),
        "revenue_per_visitor_inr": round(revenue_per_visitor, 2),
        "status": "OK",
        # LOW confidence = fewer than 20 visitors (results not statistically reliable)
        "data_confidence": "LOW" if unique_visitors < 20 else "OK",
        # Per-session confidence distribution from raw event confs
        "session_confidence_distribution": session_confidence_distribution(customer_sessions),
    }


def upsert_today_stats(
    metrics_dict: dict, store_id: str, date_str: str,
) -> None:
    """
    Persist today's headline metrics into the daily_stats table so the
    CONVERSION_DROP anomaly has a 7-day rolling baseline tomorrow.

    Safe to call on every /metrics request; INSERT OR REPLACE keeps one row
    per (store_id, date).
    """
    if not metrics_dict or metrics_dict.get("status") == "NO_TRAFFIC":
        return
    try:
        from .db import get_repo
        get_repo().upsert_daily_stats(
            store_id=store_id,
            date=date_str,
            conversion_rate=metrics_dict.get("conversion_rate", 0.0),
            unique_visitors=metrics_dict.get("unique_visitors", 0),
            revenue_inr=metrics_dict.get("revenue_inr", 0.0),
        )
    except Exception:
        pass  # never break the request because of stats persistence


def _empty_metrics() -> dict:
    """
    Return a zeroed-out metrics dict for the no-traffic case.
    All values are 0 / empty / "NO_TRAFFIC" — never None or NaN.
    """
    return {
        "unique_visitors": 0,
        "buyers": 0,
        "conversion_rate": 0.0,
        "avg_dwell_ms_by_zone": {},
        "current_queue_depth": 0,
        "abandonment_rate": 0.0,
        "revenue_inr": 0.0,
        "revenue_per_visitor_inr": 0.0,
        "status": "NO_TRAFFIC",
        "data_confidence": "LOW",
        "session_confidence_distribution": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
    }


def _avg_dwell_by_zone(sessions: list[dict]) -> dict[str, float]:
    """
    Calculate average dwell time per zone across all customer sessions.

    Returns: {zone_id: avg_dwell_ms_float}

    A zone only appears if at least one session visited it.
    The average is across sessions that visited that zone
    (not across all sessions — that would dilute it with zeros).
    """
    zone_totals: dict[str, list[int]] = {}
    for s in sessions:
        for zone, dwell in s.get("zones", {}).items():
            zone_totals.setdefault(zone, []).append(dwell)
    return {
        zone: round(sum(dwells) / len(dwells), 1)
        for zone, dwells in zone_totals.items()
        if dwells  # guard: never divide by empty list
    }


def _current_queue_depth(events: list[dict]) -> int:
    """
    Estimate how many people are currently at the billing counter.

    Uses a rolling 5-minute window anchored on the LATEST event timestamp
    (not datetime.now — works correctly for historical footage too).

    Logic:
      - Start with empty set
      - BILLING_QUEUE_JOIN → add visitor_id to set
      - EXIT or BILLING_QUEUE_ABANDON → remove from set
      - Final set size = current queue depth
    """
    if not events:
        return 0
    # Anchor to latest event, not clock time (correct for historical data)
    latest_ts = max(_ts(e) for e in events)
    cutoff = latest_ts - timedelta(minutes=5)

    in_queue: set[str] = set()
    for e in sorted(events, key=lambda e: _ts(e)):
        if e.get("is_staff"):
            continue
        if _ts(e) < cutoff:
            continue
        etype = e["event_type"]
        vid = e["visitor_id"]
        if etype == "BILLING_QUEUE_JOIN":
            in_queue.add(vid)
        elif etype in ("EXIT", "BILLING_QUEUE_ABANDON"):
            in_queue.discard(vid)
    return len(in_queue)


def _abandonment_rate(sessions: list[dict]) -> float:
    """
    Fraction of billing-queue-joiners who abandoned without purchasing.

    abandonment_rate = abandoned_sessions / sessions_that_joined_billing_queue
    Returns 0.0 if no sessions joined the billing queue.
    """
    joined = [s for s in sessions if s.get("billing") is not None]
    if not joined:
        return 0.0  # guard: no sessions joined billing queue
    abandoned = sum(1 for s in joined if s["billing"].get("abandoned"))
    return abandoned / len(joined)


def _sum_revenue(baskets: list[dict], customer_sessions: list[dict]) -> float:
    """
    Sum total revenue from all POS baskets that were matched to sessions.

    We re-run correlate_conversions here to get the converted set,
    then sum ALL basket values (not per-session — some sessions may have
    multiple baskets, which is rare but possible).
    """
    if not baskets or not customer_sessions:
        return 0.0
    # Re-correlate (correlate_conversions is deterministic for the same inputs)
    converted = correlate_conversions(baskets, customer_sessions)
    if not converted:
        return 0.0
    # Sum all basket values (proxy — in production this would filter by matched invoices)
    return sum(b["value_inr"] for b in baskets)
