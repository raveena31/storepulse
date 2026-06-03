"""
app/db.py
─────────
SQLite storage layer — the only file that touches the database.

Design principles:
  - All database logic lives HERE. No other file imports sqlite3.
  - SQLiteRepo exposes a narrow interface (3 methods). Swap to Postgres
    by writing a PostgresRepo with the same 3 methods. Nothing else changes.
  - insert_ignore() uses SQLite's INSERT OR IGNORE for idempotency —
    duplicate event_id = silent no-op, returns False.
  - All timestamps stored as ISO-8601 strings in UTC.
  - DB file is created automatically on first run (no manual setup needed).
  - get_repo() returns a module-level singleton — one connection pool per process.

Database schema (single table):

    events
    ├── event_id   TEXT PRIMARY KEY    ← deduplication key
    ├── store_id   TEXT                ← which store
    ├── camera_id  TEXT                ← which camera
    ├── visitor_id TEXT                ← which person (tracker ID)
    ├── event_type TEXT                ← ENTRY | EXIT | ZONE_ENTER | ...
    ├── ts         TEXT                ← ISO-8601 UTC timestamp
    ├── zone_id    TEXT nullable       ← which zone (null for ENTRY/EXIT)
    ├── dwell_ms   INTEGER             ← milliseconds in zone (0 for non-dwell events)
    ├── is_staff   INTEGER (0/1)       ← boolean stored as integer
    ├── confidence REAL                ← [0.0, 1.0]
    └── payload    TEXT                ← JSON blob of EventMeta fields

Indexes:
    idx_store_ts       on (store_id, ts)           ← used by events_for() time-range queries
    idx_store_visitor  on (store_id, visitor_id)   ← used by session builder lookups
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

# Read DB path from environment variable (set in docker-compose.yml or shell).
# Falls back to a local path for development without Docker.
DB_PATH = os.getenv("DB_PATH", "data/store_intelligence.db")


class SQLiteRepo:
    """
    Repository class for the events table.

    Usage:
        repo = SQLiteRepo()                    # uses DB_PATH env var
        repo = SQLiteRepo("path/to/test.db")   # for tests — inject a tmp path

    All public methods open + close their own connection. This is intentional:
    SQLite's WAL mode handles concurrent reads fine; we don't need a connection pool.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # Create the data directory if it doesn't exist (e.g. first Docker run)
        dir_name = os.path.dirname(db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        """Open a new SQLite connection with Row factory for dict-like access."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the events table and indexes if they don't exist yet."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id   TEXT PRIMARY KEY,
                    store_id   TEXT NOT NULL,
                    camera_id  TEXT NOT NULL,
                    visitor_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    ts         TEXT NOT NULL,
                    zone_id    TEXT,
                    dwell_ms   INTEGER DEFAULT 0,
                    is_staff   INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.0,
                    payload    TEXT
                )
            """)
            # Composite index for the most common query pattern:
            # "give me all events for store X between time A and time B"
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_store_ts ON events(store_id, ts)"
            )
            # Index for session building — looking up all events for one visitor
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_store_visitor ON events(store_id, visitor_id)"
            )

            # Daily aggregate stats used by CONVERSION_DROP anomaly (7-day rolling avg).
            # One row per (store_id, date). Upserted from compute_metrics on each call.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    store_id        TEXT NOT NULL,
                    date            TEXT NOT NULL,
                    conversion_rate REAL NOT NULL,
                    unique_visitors INTEGER NOT NULL,
                    revenue_inr     REAL NOT NULL,
                    updated_at      TEXT NOT NULL,
                    PRIMARY KEY (store_id, date)
                )
            """)
            conn.commit()

    def upsert_daily_stats(
        self, store_id: str, date: str,
        conversion_rate: float, unique_visitors: int, revenue_inr: float,
    ) -> None:
        """Insert or replace today's stats for a store."""
        from datetime import datetime, timezone
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO daily_stats
                    (store_id, date, conversion_rate, unique_visitors, revenue_inr, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(store_id, date) DO UPDATE SET
                    conversion_rate = excluded.conversion_rate,
                    unique_visitors = excluded.unique_visitors,
                    revenue_inr     = excluded.revenue_inr,
                    updated_at      = excluded.updated_at
            """, (
                store_id, date,
                float(conversion_rate), int(unique_visitors), float(revenue_inr),
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()

    def rolling_avg_conversion(
        self, store_id: str, days: int = 7, exclude_date: str = None,
    ) -> tuple[float, int]:
        """
        Average conversion_rate over the last N days for a store, optionally
        excluding a specific date (typically today, to avoid self-comparison).

        Returns (avg_rate, n_days_used). avg_rate is 0.0 if no data.
        Only includes rows where unique_visitors >= 20 (statistically reliable).
        """
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT conversion_rate
                FROM daily_stats
                WHERE store_id = ?
                  AND unique_visitors >= 20
                  AND (? IS NULL OR date != ?)
                ORDER BY date DESC
                LIMIT ?
            """, (store_id, exclude_date, exclude_date, days))
            rates = [row[0] for row in cur.fetchall()]
        if not rates:
            return 0.0, 0
        return sum(rates) / len(rates), len(rates)

    def insert_ignore(self, event_dict: dict) -> bool:
        """
        Insert one event. Silently ignores duplicates (same event_id).

        Returns:
            True  → event was new and inserted
            False → event_id already existed (duplicate, no-op)

        This is the idempotency guarantee:
        The pipeline can safely resend events on retry without creating duplicates.
        """
        # Serialize EventMeta to JSON for the payload column
        payload = json.dumps(event_dict.get("metadata", {}))

        # Coerce timestamp to ISO string if it's still a datetime object
        ts = event_dict.get("timestamp") or event_dict.get("ts")
        if isinstance(ts, datetime):
            ts = ts.isoformat()

        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO events
                       (event_id, store_id, camera_id, visitor_id, event_type, ts,
                        zone_id, dwell_ms, is_staff, confidence, payload)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        event_dict["event_id"],
                        event_dict["store_id"],
                        event_dict["camera_id"],
                        event_dict["visitor_id"],
                        event_dict["event_type"],
                        ts,
                        event_dict.get("zone_id"),
                        event_dict.get("dwell_ms", 0),
                        1 if event_dict.get("is_staff") else 0,
                        event_dict.get("confidence", 0.0),
                        payload,
                    ),
                )
                conn.commit()
                # total_changes == 0 means INSERT OR IGNORE hit a duplicate
                return conn.total_changes > 0
        except sqlite3.IntegrityError:
            return False

    def events_for(
        self,
        store_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Fetch all events for a store, optionally filtered by UTC time window.

        Args:
            store_id: Store identifier, e.g. "ST1008"
            start:    Inclusive lower bound (UTC datetime). None = no lower bound.
            end:      Inclusive upper bound (UTC datetime). None = no upper bound.

        Returns:
            List of event dicts ordered by timestamp ascending.
            Each dict has a "timestamp" key (datetime) and "metadata" key (dict).
        """
        query = "SELECT * FROM events WHERE store_id = ?"
        params: list = [store_id]

        if start:
            query += " AND ts >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND ts <= ?"
            params.append(end.isoformat())

        query += " ORDER BY ts"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            # Convert SQLite INTEGER (0/1) back to Python bool
            d["is_staff"] = bool(d["is_staff"])
            # Deserialize the JSON metadata payload
            try:
                d["metadata"] = json.loads(d.get("payload") or "{}")
            except Exception:
                d["metadata"] = {}
            # Parse the ISO timestamp string back to datetime
            ts_str = d["ts"]
            if isinstance(ts_str, str):
                try:
                    d["timestamp"] = datetime.fromisoformat(ts_str)
                except ValueError:
                    d["timestamp"] = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            result.append(d)

        return result

    def last_event_ts(self, store_id: str) -> Optional[datetime]:
        """
        Return the timestamp of the most recent event for a store.
        Used by GET /health to calculate feed lag.
        Returns None if the store has no events yet.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT ts FROM events WHERE store_id = ? ORDER BY ts DESC LIMIT 1",
                (store_id,),
            ).fetchone()

        if not row:
            return None

        ts_str = row["ts"]
        try:
            return datetime.fromisoformat(ts_str)
        except ValueError:
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


# ── Module-level singleton ─────────────────────────────────────────────────────

_repo: Optional[SQLiteRepo] = None


def get_repo() -> SQLiteRepo:
    """
    Return the module-level SQLiteRepo singleton.

    Why a singleton? We want one DB connection pool per process,
    not one per request. FastAPI is async but SQLite is synchronous —
    this pattern is safe for our read/write patterns.

    Tests bypass this by injecting their own SQLiteRepo(tmp_path) directly.
    """
    global _repo
    if _repo is None:
        _repo = SQLiteRepo(DB_PATH)
    return _repo
