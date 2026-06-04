# Store Intelligence — Design Document

## 1. The North Star

Offline retail has been operating with one hand tied behind its back. E-commerce has known its
funnel for two decades — landing page → product page → cart → checkout — and tuned every
percentage point on top of it. Physical stores still ask cashiers at end-of-day "how was today?"
and call that analytics. This system closes that gap.

**The 30-second pitch:** *The offline conversion funnel, observable in real time. Buyers ÷ unique
non-staff visitors, with the same statistical precision Purplle already uses on its app. One number
the CEO can put on a board, and a single click that explains why it moved.*

Every architectural decision below is judged against one question: does it make the
`conversion_rate` more **accurate** and more **useful** to the person who decides what to merchandise next?

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  RAW INPUT                                                                   │
│  ─────────                                                                   │
│  5 CCTV cameras (.mp4)      POS CSV (Brigade_Bangalore_10_April_26.csv)      │
│  1920×1080 @ 25–30fps        101 line items                                  │
└─────────────────┬────────────────────────────┬───────────────────────────────┘
                  │                            │
                  ▼                            │
┌──────────────────────────────────────┐       │
│  pipeline/  (vision layer)           │       │
│  ──────────                          │       │
│  detect.py    YOLOv8n person filter  │       │
│  tracker.py   ByteTrack (persist=T)  │       │
│  zones.py     ray-cast per-camera    │       │
│  edge_cases   EC-1..EC-9, 19, 20     │       │
│  reid.py      EC-12, 13, 14, 15, 16  │       │
│  crosscam.py  EC-17, 18              │       │
│  staff.py     EC-21, 22, 24, 25      │       │
│  emit.py      → POST /events/ingest  │       │
└─────────────────┬────────────────────┘       │
                  │                            │
                  ▼                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  EVENTS  (Pydantic v2 — UTC validator, confidence band)                      │
│  ──────  POST /events/ingest is idempotent on event_id (INSERT OR IGNORE)    │
└─────────────────┬────────────────────────────────────────────────────────────┘
                  │                            ┌─────────────────────────────┐
                  ▼                            │  pos.py                     │
┌──────────────────────────────────┐           │  ──────                     │
│  sessions.py                     │◄──────────┤  baskets_from_pos: 101→24   │
│  ────────────                    │           │  is_real_sale               │
│  build_sessions                  │           │  is_meaningful_basket (≥₹5) │
│  – one per visitor_id            │           │  pos_local_to_utc (IST→UTC) │
│  – REENTRY merges                │           │  attribute_txn:             │
│  – sticky staff                  │           │    1) billing-window match  │
│  – dwell capped 10 min           │           │    2) last-zone fallback    │
└─────────────────┬────────────────┘           │  unique_buyers              │
                  │                            └─────────────────────────────┘
                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ANALYTICS                                                                   │
│  ─────────                                                                   │
│  metrics.py    conversion_rate, dwell, queue, revenue, session conf dist     │
│  funnel.py     4-stage set-based, monotonically decreasing                   │
│  heatmap.py    dwell_share vs sales_share gap — Purplle's R&D signal         │
│  anomalies.py  BILLING_QUEUE_SPIKE | CONVERSION_DROP | DEAD_ZONE | COVERAGE  │
└─────────────────┬────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  API + UI  (FastAPI)                                                         │
│  ────────                                                                    │
│  GET  /                       Purplle-branded dashboard                      │
│  GET  /dashboard/             same dashboard at the spec'd URL               │
│  GET  /dashboard/stream/:id   SSE — emits per-2s metrics for live updates    │
│  GET  /api/live               richer SSE (anomalies + heatmap + funnel)      │
│  GET  /stream/:camera_id      MJPEG with YOLO overlays + zone polygons       │
│  GET  /health                 per-store LIVE | STALE_FEED                    │
│  POST /events/ingest          idempotent, partial-success, max 500           │
│  GET  /stores/:id/metrics     core dashboard                                 │
│  GET  /stores/:id/funnel      ENTRY → ZONE → BILLING → PURCHASE              │
│  GET  /stores/:id/heatmap     per-zone dwell-vs-sales gap                    │
│  GET  /stores/:id/anomalies   real-threshold alerts with operational actions │
│  GET  /reports/export         CSV / JSON download                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Why Sessions Are the Unit of Truth

A session is *one visitor's entire visit*. Not one event, not one camera detection — one
person, one trip into the store.

**Why this matters:** raw event counts produce wrong answers. A customer who walks in, browses
LAKME for 8 minutes, joins the billing queue, then exits without buying generates roughly
*40 events* from the detection pipeline. If you count those events you've overstated traffic by a
factor of 40. If you count ENTRY events specifically, you under-count by missing the visitor's
REENTRY 20 minutes later — same person, same session, but a naive counter now reports 2.

The session abstraction collapses both errors. One visitor_id → one session. REENTRY merges into
the existing session, not a new one. Any single `is_staff=True` event marks the whole session
staff (sticky-staff, EC-20) so a one-frame misclassification doesn't flip the customer back into
the denominator. Every metric, every funnel stage, every heatmap row uses
`set(visitor_id)` — never `len(events)`.

This is the single most important architectural decision in the system. Get it wrong and every
downstream number is junk.

---

## 4. AI-Assisted Decisions

This section documents three places where I overrode my AI collaborator's first suggestion.
Each override exists because Purplle's actual constraints — Indian POS conventions, retail crowd
behaviour, Docker-first deployment — were stronger than the "generally good" defaults the model
had been trained to suggest.

### Decision A — ByteTrack over DeepSORT for the tracker

Claude initially suggested DeepSORT because it cited better cross-camera re-identification in
crowded scenes. I overrode for three concrete reasons. **First**, the acceptance gate is
`docker compose up` with no manual steps — DeepSORT ships separately from the detector and
needs a reID weight file (~150MB) plus a torch dependency that bloats the image. ByteTrack
is bundled inside ultralytics, so the pip line installs both YOLO and the tracker. **Second**,
re-entry in a store happens over *minutes*, not seconds — the customer leaves to take a phone
call and returns. Motion-based association breaks there regardless of which tracker you pick;
appearance is the only signal, and that's why I built a separate `ReIDGallery` (EC-12) that runs
*outside* the tracker. ByteTrack only needs to handle short-term occlusion behind a display unit,
which it does well. **Third**, our store has a glass entrance — reflections cause low-confidence
ghost detections. ByteTrack's two-stage matching keeps low-conf boxes alive long enough to
distinguish them from real people, while DeepSORT's appearance threshold drops them entirely and
loses the track.

### Decision B — Invoice-collapse-first over identity-based POS dedup

Claude's first POS draft hashed `customer_name + customer_phone` as the dedup key. I overrode the
moment I opened the Brigade CSV: **20 of 101 rows are `customer_name = "Guest"`**. A Guest with
phone `9876543210` and another Guest with phone `9876543210` two hours later are usually
different people — the cashier types the store's own default. Name-based dedup would silently
merge them and underreport conversion by 30%. The only signal that works for *every* basket,
named or not, is collapse-by-`invoice_number` (101 → 24) then time-window correlation against
billing sessions. Identity becomes a *secondary* signal (`unique_buyers` for revenue-per-buyer)
not the primary conversion link. The fact that `len(baskets) == 24` is asserted in the test
suite against the *real* CSV (not a synthetic fixture) means this is impossible to silently
regress.

### Decision C — Daily-stats table over in-memory baseline for CONVERSION_DROP

When I built the CONVERSION_DROP anomaly, Claude suggested keeping a 7-day rolling window in
process memory. I overrode for three reasons. **Memory loss**: the API container restarts and
the baseline evaporates — Day 1 after every deploy looks anomalous because there's no history.
**Cross-process correctness**: at scale, we'll run multiple workers; in-memory state diverges
between them and a manager would see contradictory "CONVERSION DROP" alerts from refresh to
refresh. **Audit**: regional managers will ask "what was last Tuesday's rate?" and we need to
answer; a SQLite `daily_stats` table makes that one query. The cost is a row-per-day write
(negligible) and one extra index. The benefit is correctness from minute one.

---

## 5. Trade-offs Made

- **SQLite, not Postgres.** The acceptance gate demands zero infra setup. SQLite gives us
  `INSERT OR IGNORE` idempotency, a working WAL mode, and zero dependency surface. The
  `SQLiteRepo` interface is narrow enough that swapping to Postgres is a single-file change.
  Accepted cost: no write concurrency across processes — fine for now, blocking at ~10 stores.
- **MJPEG over WebRTC for the live stream.** A `<img src="/stream/CAM_01">` tag works in every
  browser with zero JS. WebRTC would give lower latency but adds a signalling server, STUN
  endpoints, and a maintenance tax that doesn't serve the demo. We can ship.
- **No re-ID model at the embedding layer.** The `ReIDGallery` is built to accept embeddings,
  but in the live pipeline we don't yet generate them (no OSNet weights). EC-13 logic is fully
  tested with synthetic vectors. Adding a real embedder is one file (`pipeline/embed.py`)
  with no architectural change. Accepted cost: REENTRY currently relies on visitor_id continuity
  within one camera. Cross-camera REENTRY isn't perfect yet — but the contract is in place.
