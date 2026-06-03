"""
app/health.py
─────────────
GET /health endpoint logic.

Reports per-store feed status with accurate STALE_FEED detection.
Critical at scale (40+ stores) — operators need to know which feeds are dark.

Logic:
  - "service": "ok" always (service can respond even when stores are stale)
  - Per-store: last_event_utc, lag_seconds, feed
       feed = "STALE_FEED" if lag > lag_s, else "LIVE"
  - "checked_at_utc" is included for audit trails
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

# Default stale threshold — 10 minutes. Tunable per deploy.
STALE_THRESHOLD_S = 600


def health_status(repo, now: Optional[datetime] = None, lag_s: int = STALE_THRESHOLD_S) -> dict:
    """
    Build the full health response payload.

    Args:
        repo:  SQLiteRepo (or any object exposing _conn() and last_event_ts())
        now:   reference timestamp (defaults to datetime.now UTC)
        lag_s: stale threshold in seconds

    Returns:
        {
            "service": "ok",
            "checked_at_utc": "<ISO>",
            "stores": {
                store_id: {
                    "last_event_utc": "<ISO|None>",
                    "lag_seconds":    float,
                    "feed":           "LIVE" | "STALE_FEED",
                }
            }
        }
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Discover all store_ids that have events in the database
    try:
        with repo._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT store_id FROM events"
            ).fetchall()
        store_ids = [r["store_id"] for r in rows]
    except Exception:
        store_ids = []

    stores: dict[str, dict] = {}
    for sid in store_ids:
        last_ts = repo.last_event_ts(sid)
        if last_ts is None:
            stores[sid] = {
                "last_event_utc": None,
                "lag_seconds": None,
                "feed": "STALE_FEED",
            }
            continue

        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        lag = (now - last_ts).total_seconds()

        stores[sid] = {
            "last_event_utc": last_ts.isoformat(),
            "lag_seconds": round(lag, 1),
            "feed": "STALE_FEED" if lag > lag_s else "LIVE",
        }

    return {
        "service": "ok",
        "checked_at_utc": now.isoformat(),
        "stores": stores,
    }


# Backwards-compatible thin wrapper used by main.py
def get_health() -> dict:
    """Backwards-compat: get_health() uses the module-level repo."""
    from .db import get_repo
    return health_status(get_repo())
