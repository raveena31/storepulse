"""
app/anomalies.py
────────────────
Real anomaly detection with operational suggested_actions.

Three anomaly families:

1. BILLING_QUEUE_SPIKE   — operational: open more counters NOW
2. CONVERSION_DROP       — managerial: today vs 7-day rolling baseline
3. DEAD_ZONE             — merchandising: zone unused despite coverage
   COVERAGE_GAP          — telemetry: zone has no camera, no signal possible

The 7-day rolling baseline is stored in the `daily_stats` table (populated by
compute_metrics on every call). This makes CONVERSION_DROP a real signal,
not a placeholder.

Every anomaly's `suggested_action` is a complete, operational instruction —
a regional manager should be able to act on it without further interpretation.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import yaml

from .sessions import build_sessions, _ts


# ── Thresholds ────────────────────────────────────────────────────────────────

# Billing queue
QUEUE_WARN_THRESHOLD     = 6
QUEUE_CRITICAL_THRESHOLD = 8

# Conversion drop
CONVERSION_DROP_MIN_VISITORS = 20
CONVERSION_DROP_RATIO        = 0.6  # today < 60% of 7d avg → fire

# Dead zone
DEAD_ZONE_WINDOW_MIN = 60


# ── Top-level orchestration ───────────────────────────────────────────────────

def compute_anomalies(
    events: list[dict],
    pos_csv: Optional[str] = None,
    store_id: str = "ST1008",
    today_date_str: Optional[str] = None,
) -> list[dict]:
    """
    Run all anomaly checks. Returns list of {type, severity, value,
    threshold, suggested_action, detected_at_utc} dicts.

    Empty input → empty output, no exceptions.

    Args:
        events:          raw events for the current window
        pos_csv:         POS CSV path (only needed for CONVERSION_DROP)
        store_id:        store id (for daily_stats lookup)
        today_date_str:  YYYY-MM-DD for daily_stats lookup; default = today UTC
    """
    if not events:
        return []

    anomalies: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    # ── 1. Billing queue spike ────────────────────────────────────────────
    spike = _check_queue_spike(events, now_iso)
    if spike:
        anomalies.append(spike)

    # ── 2. Conversion drop (uses 7-day rolling baseline from DB) ─────────
    drop = _check_conversion_drop(events, pos_csv, store_id, today_date_str, now_iso)
    if drop:
        anomalies.append(drop)

    # ── 3. Dead zones + coverage gaps ─────────────────────────────────────
    anomalies.extend(_check_zones(events, now_iso))

    return anomalies


# ── Billing queue spike ───────────────────────────────────────────────────────

def _check_queue_spike(events: list[dict], now_iso: str) -> Optional[dict]:
    """
    Count distinct non-staff visitor_ids whose LATEST event in the last 5 min
    is BILLING_QUEUE_JOIN with no subsequent EXIT or BILLING_QUEUE_ABANDON.
    """
    depth = _current_queue_depth(events)
    if depth < QUEUE_WARN_THRESHOLD:
        return None

    if depth >= QUEUE_CRITICAL_THRESHOLD:
        return {
            "type": "BILLING_QUEUE_SPIKE",
            "severity": "CRITICAL",
            "value": depth,
            "threshold": QUEUE_CRITICAL_THRESHOLD,
            "suggested_action": (
                f"URGENT: {depth} in billing queue. "
                f"Open counter 2 immediately and alert floor manager."
            ),
            "detected_at_utc": now_iso,
        }

    return {
        "type": "BILLING_QUEUE_SPIKE",
        "severity": "WARN",
        "value": depth,
        "threshold": QUEUE_WARN_THRESHOLD,
        "suggested_action": (
            f"Open a second counter - {depth} in queue, "
            f"estimated abandonment risk HIGH"
        ),
        "detected_at_utc": now_iso,
    }


def _current_queue_depth(events: list[dict]) -> int:
    """
    Visitors with a latest BILLING_QUEUE_JOIN event in the last 5 min
    and no subsequent EXIT or BILLING_QUEUE_ABANDON.

    Anchored to the latest event timestamp (not datetime.now) — correct for
    both live streams and historical footage replays.
    """
    if not events:
        return 0
    latest_ts = max(_ts(e) for e in events)
    cutoff = latest_ts - timedelta(minutes=5)

    # Per visitor, find their latest relevant event in the window
    latest_per_visitor: dict[str, tuple[datetime, str]] = {}
    for e in events:
        if e.get("is_staff"):
            continue
        et = e.get("event_type")
        if et not in ("BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "EXIT"):
            continue
        t = _ts(e)
        if t < cutoff:
            continue
        vid = e["visitor_id"]
        prev = latest_per_visitor.get(vid)
        if prev is None or t > prev[0]:
            latest_per_visitor[vid] = (t, et)

    return sum(1 for _, (_, et) in latest_per_visitor.items()
               if et == "BILLING_QUEUE_JOIN")


# ── Conversion drop ───────────────────────────────────────────────────────────

def _check_conversion_drop(
    events: list[dict],
    pos_csv: Optional[str],
    store_id: str,
    today_date_str: Optional[str],
    now_iso: str,
) -> Optional[dict]:
    """
    Compare today's conversion to the 7-day rolling average from daily_stats.

    Fires WARN if:
        today's conversion < (7d avg × CONVERSION_DROP_RATIO)
        AND today's unique_visitors ≥ CONVERSION_DROP_MIN_VISITORS
        AND 7d avg is available (n >= 1)
    """
    from .metrics import compute_metrics  # local to avoid circular import
    from .db import get_repo

    metrics = compute_metrics(events, pos_csv or "nonexistent.csv")
    today_visitors = metrics.get("unique_visitors", 0)
    today_rate     = metrics.get("conversion_rate", 0.0)

    if today_visitors < CONVERSION_DROP_MIN_VISITORS:
        return None

    repo = get_repo()
    if today_date_str is None:
        today_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        avg_rate, n_days = repo.rolling_avg_conversion(
            store_id, days=7, exclude_date=today_date_str
        )
    except Exception:
        return None

    if n_days == 0 or avg_rate <= 0:
        return None  # no baseline → no anomaly

    if today_rate >= avg_rate * CONVERSION_DROP_RATIO:
        return None

    today_pct = round(today_rate * 100, 1)
    avg_pct   = round(avg_rate   * 100, 1)
    return {
        "type": "CONVERSION_DROP",
        "severity": "WARN",
        "value": round(today_rate, 4),
        "threshold": round(avg_rate * CONVERSION_DROP_RATIO, 4),
        "suggested_action": (
            f"Conversion {today_pct}% vs 7d avg {avg_pct}%. "
            f"Check: staffing levels, stock availability, AC/environment."
        ),
        "detected_at_utc": now_iso,
    }


# ── Dead zones + coverage gaps ────────────────────────────────────────────────

def _check_zones(events: list[dict], now_iso: str) -> list[dict]:
    """
    Two related zone-level anomalies:

    DEAD_ZONE:    has camera coverage but zero visits in last DEAD_ZONE_WINDOW_MIN
    COVERAGE_GAP: no camera covers this zone — no signal can be inferred
    """
    out: list[dict] = []
    if not events:
        return out

    cfg = _load_config()
    all_zones = set(cfg.get("zones", {}).keys())
    no_coverage = set(cfg.get("zones_without_camera_coverage", []) or [])

    if not all_zones:
        return out

    # 1. COVERAGE_GAP — emitted once per uncovered zone
    for zone in sorted(no_coverage):
        out.append({
            "type": "COVERAGE_GAP",
            "severity": "INFO",
            "value": zone,
            "threshold": None,
            "suggested_action": (
                f"Zone '{zone}' has no camera coverage - "
                f"dwell data unavailable."
            ),
            "detected_at_utc": now_iso,
        })

    # 2. DEAD_ZONE — only check zones WITH coverage
    covered_zones = all_zones - no_coverage
    latest_ts = max(_ts(e) for e in events)
    cutoff = latest_ts - timedelta(minutes=DEAD_ZONE_WINDOW_MIN)

    recently_visited: set[str] = set()
    for e in events:
        if e.get("is_staff"):
            continue
        if _ts(e) >= cutoff and e.get("zone_id"):
            recently_visited.add(e["zone_id"])

    dead = sorted(covered_zones - recently_visited)
    for zone in dead:
        out.append({
            "type": "DEAD_ZONE",
            "severity": "INFO",
            "value": zone,
            "threshold": DEAD_ZONE_WINDOW_MIN,
            "suggested_action": (
                f"Zone '{zone}' has zero visitors for {DEAD_ZONE_WINDOW_MIN}+ min. "
                f"Action: check camera view is unobstructed; "
                f"if confirmed dead -> consider re-merchandising."
            ),
            "detected_at_utc": now_iso,
        })

    return out


# ── Config loader ─────────────────────────────────────────────────────────────

_cfg_cache: Optional[dict] = None


def _load_config() -> dict:
    global _cfg_cache
    if _cfg_cache is None:
        path = os.path.join(os.getenv("CONFIG_DIR", "config"), "store_ST1008.yaml")
        try:
            with open(path) as f:
                _cfg_cache = yaml.safe_load(f) or {}
        except Exception:
            _cfg_cache = {}
    return _cfg_cache