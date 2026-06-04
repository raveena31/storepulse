"""One-off cleanup: rewrite test file headers to remove AI references."""
import os
import re

# Map filename -> (new docstring summary)
HEADERS = {
    "test_analytics.py":
        "Tests for analytics layer: brand heatmap attention_vs_sales, real anomaly\n"
        "thresholds with 7-day baseline, structured JSON logging with trace_id,\n"
        "STALE_FEED detection, confidence band calibration.",
    "test_anomalies.py":
        "Tests for app/anomalies.py: queue spike CRITICAL at 8+ visitors,\n"
        "zero events no-crash, suggested_action contains operational instructions.",
    "test_funnel.py":
        "Tests for app/funnel.py: REENTRY dedup (same visitor_id counts once),\n"
        "monotonic decrease through all 4 stages, missing POS CSV safe-fallback.",
    "test_heatmap_health.py":
        "Tests for app/heatmap.py and app/health.py: zone aggregation,\n"
        "attention_vs_sales fields, staff exclusion, LIVE vs STALE_FEED status.",
    "test_ingestion.py":
        "Tests for app/ingestion.py: idempotent double-ingest (event_id dedup),\n"
        "partial-success batch with structured rejected list (index + error).",
    "test_ingest_http.py":
        "HTTP-layer integration tests for /events/ingest:\n"
        "EC-44 idempotency (separate POST calls -> 1 DB row),\n"
        "EC-45 partial-success (mixed batch -> HTTP 207 with structured rejects).",
    "test_metrics.py":
        "Tests for app/metrics.py guards: zero-event store, all-staff clip,\n"
        "division-by-zero -> status=NO_TRAFFIC, conversion_rate=0.0, never NaN.",
    "test_pipeline_edges.py":
        "Tests for the 20 detection/re-ID/staff edge cases (EC-1 through EC-25).\n"
        "Each test maps to a specific EC contract from edge_cases.md.",
    "test_pos_edges.py":
        "Tests for app/pos.py covering: invoice collapse (101 -> 24 baskets,\n"
        "asserted against the real Brigade Bangalore CSV), IST->UTC at boundary,\n"
        "Guest checkout, returns filter, GWP floor, closest-unattributed correlation,\n"
        "mobile-POS fallback.",
    "test_sessions.py":
        "Tests for app/sessions.py: REENTRY uses same visitor_id (no new session),\n"
        "sticky staff (any is_staff=True event taints the whole session),\n"
        "dwell cap at 10 minutes per event.",
}

TESTS_DIR = os.path.join(os.path.dirname(__file__), "..", "tests")

for fname, doc in HEADERS.items():
    path = os.path.join(TESTS_DIR, fname)
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Strip any leading comment block (PROMPT/CHANGES) until the first import/blank
    lines = content.split("\n")
    skip = 0
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith("#") or stripped == "":
            skip = i + 1
        else:
            break

    body = "\n".join(lines[skip:]).lstrip("\n")
    new = f'"""\n{doc}\n"""\n\n{body}'

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    print(f"Rewrote {fname}")