# Architectural Choices

Three load-bearing decisions, each structured as: options considered → what AI suggested →
what I chose → why the rationale ties to the North Star.

---

## Decision 1: YOLOv8n + ByteTrack vs alternatives

**Options considered**
- YOLOv8n + ByteTrack (via ultralytics, single pip install)
- YOLOv5 + DeepSORT (separate trackers package, requires re-ID weights)
- Detectron2 + StrongSORT (heavy GPU stack, best accuracy on benchmark sets)
- MMDetection + OC-SORT (academic-grade, complex install)

**What AI suggested**
> Claude initially suggested DeepSORT, citing better cross-camera re-identification accuracy
> on the MOT17 benchmark and standard usage in retail analytics papers. It noted DeepSORT's
> appearance-based association handles occlusion in crowded scenes more gracefully than pure
> motion trackers.

**What I chose**
YOLOv8n + ByteTrack — bundled, lightweight, with cross-camera re-ID handled separately by
`ReIDGallery` (EC-12) outside the tracker.

**Why**
The acceptance gate is the load-bearing constraint here: `docker compose up` with zero manual
steps. DeepSORT needs a separate appearance model weight file (~150MB) that's not in the pip
package, plus an additional torch dependency that bloats the image by 800MB. ByteTrack ships
inside ultralytics, so a single `pip install ultralytics` gives both the detector and the
tracker — and ultralytics auto-downloads the 6MB YOLOv8n weights on first invocation, well
within Docker's first-run budget.

The cross-camera REENTRY argument that pushed AI toward DeepSORT doesn't actually apply to our
use case: in a retail store, a customer who leaves and returns 10 minutes later (phone call,
forgot wallet) needs an *appearance* match, not motion continuity — motion-based association
has already given up by then. I built that as a separate `ReIDGallery` with a configurable
15-minute window and 0.62 cosine threshold (EC-13), so the tracker only has to handle short-term
occlusion (a customer ducking behind a display), which ByteTrack does well via its two-stage
matching of high+low confidence detections.

Tie to the North Star: this choice keeps the conversion *denominator* honest by maintaining
track IDs through brief occlusions without inflating it via tracker fragmentation. Every
fragmented track would otherwise become a phantom new visitor.

---

## Decision 2: Session-centric schema vs event-log schema

**Options considered**
- Pure event log — write every detection event to DB, derive everything on read
- Pre-aggregated sessions — collapse events to one session row per visitor at write time
- Hybrid — event log + materialised sessions view, refreshed periodically

**What AI suggested**
> Claude suggested a pure event-log schema for "maximum flexibility — you can re-derive any
> metric later by replaying events." It quoted Kappa architecture papers and CQRS patterns,
> and warned that pre-aggregation locks you into today's metric definitions.

**What I chose**
Pure event log on disk (`events` table), with sessions built on demand in `build_sessions()`
and a small `daily_stats` aggregate table for cross-day baselines (CONVERSION_DROP anomaly).
The session is a *derived* object — there's no `sessions` table — but every metric reads
through it.

**Why**
The event log is the source of truth (AI was right about that), but every consumer needs the
*session* abstraction to get the right answer. Counting raw ENTRY events double-counts REENTRY.
Counting unique visitor_ids in the events table forgets to filter staff. The session is where
both rules apply — REENTRY merges, sticky_staff filters — and it has to apply *consistently*
across `metrics`, `funnel`, `heatmap`, and `anomalies`. Materialising sessions means writing
that logic in one place (sessions.py:build_sessions) and every analytics module reads through it.

Keeping the events table as the on-disk truth means we can reprocess a day if the session logic
changes (which it did three times during development — EC-20 sticky staff lock added two days
in). Pre-aggregating sessions to disk would have meant a backfill every time. The cost is one
function call per query that walks ~10k events for a busy day — measured at 12ms p99 on the
real Brigade footage.

`daily_stats` is the only intentional pre-aggregation, and only because the CONVERSION_DROP
anomaly needs a 7-day baseline that survives container restarts and works across worker
processes.

Tie to the North Star: this choice means the same `unique_visitors` number appears in every
endpoint — metrics, funnel, heatmap. A CEO comparing tabs on the dashboard cannot get two
different answers to "how many people visited today."

---

## Decision 3: SQLite-first behind Repo interface vs Postgres-first

**Options considered**
- SQLite directly, no abstraction
- SQLite behind a `Repo` interface, ready for Postgres swap
- Postgres from day one (via SQLAlchemy or asyncpg)
- DuckDB for analytical queries + SQLite for OLTP

**What AI suggested**
> Claude suggested PostgreSQL from day one, citing production-readiness, concurrent writes for
> multi-worker uvicorn deployments, and standard practice for any service that might cross
> 10qps. It said SQLite "limits scalability."

**What I chose**
SQLite behind a `SQLiteRepo` class with three methods (`insert_ignore`, `events_for`,
`last_event_ts`) plus two `daily_stats` helpers. The interface is narrow enough that a
`PostgresRepo` is a 60-line drop-in.

**Why**
The acceptance gate is `docker compose up` with no manual steps. Postgres means a second
container in `docker-compose.yml`, a healthcheck dependency between API and DB, a connection
string env var that must match between services, and a one-shot migration to create the events
table on first boot. Each of those is a place where a reviewer running `docker compose up` on a
fresh clone might see an error and stop reading. SQLite has none of them: the file is created
on first write, the API container ships with sqlite3 in stdlib, and there's no networking to
mis-configure.

The scalability argument is real but not load-bearing for the timeline of this challenge.
SQLite in WAL mode handles thousands of writes per second from a single process — well above
the 380-events-per-clip load we measured on the Brigade footage. The known limit is
write-concurrency across processes, which only matters if we scale to multiple uvicorn workers
or multiple POS ingest streams — and that scaling decision should bring its own DB review.

The Repo interface (visible in `app/db.py`) means the swap, when it happens, is one file. Tests
inject `SQLiteRepo(tmp_path)` directly, so a `PostgresRepo` can be substituted without touching
a single test.

Tie to the North Star: SQLite means the demo machine — the reviewer's laptop — runs the system
identically to production. Same correctness, same idempotency, same `INSERT OR IGNORE`
semantics. The CEO who looks at the dashboard at our pitch meeting sees the actual system,
not a simulation.
