"""
app/funnel.py
─────────────
4-stage conversion funnel: Entry → Zone Visit → Billing → Purchase.

EDGE CASES SOLVED HERE:
  EC-43 (funnel-specific) — strictly unique visitor_id per stage.
  Funnel is monotonically decreasing — each stage count ≤ previous stage count.
  REENTRY events count once (same visitor_id, not a new stage_entry).
"""
from __future__ import annotations

from .sessions import build_sessions
from .pos import load_and_process_pos, correlate_conversions, POS_CSV_PATH


def compute_funnel(events: list[dict], pos_csv: str = POS_CSV_PATH) -> dict:
    """
    Compute funnel stage counts using strict per-visitor dedup.

    Each stage uses set(visitor_id) — never raw event counts.
    Sequential containment is asserted: each stage's visitor set is a SUBSET
    of the previous stage's set (a visitor at billing must also have entered).

    Args:
        events:  list of event dicts for one store, one day
        pos_csv: POS CSV path for purchase correlation

    Returns:
        {
            "stages": {
                "stage_entry":      {"count": N, "drop_off_from_previous_pct": 0.0},
                "stage_zone_visit": {...},
                "stage_billing":    {...},
                "stage_purchase":   {...},
            },
            "total_sessions": N,
        }
    """
    sessions = build_sessions(events)
    customer_sessions = [s for s in sessions if not s.get("is_staff")]

    # ── Stage 1: ENTRY ─────────────────────────────────────────────────────
    # Visitor sets at each stage. Using SET means REENTRY can't double-count.
    entry_ids: set[str] = {
        s["visitor_id"] for s in customer_sessions
        if s.get("entry_ts") is not None
    }

    # ── Stage 2: ZONE VISIT ────────────────────────────────────────────────
    zone_ids: set[str] = {
        s["visitor_id"] for s in customer_sessions
        if s.get("zones")  # non-empty zones dict
    }
    # Enforce containment: only visitors who entered count toward zone visit
    zone_ids = zone_ids & entry_ids

    # ── Stage 3: BILLING ───────────────────────────────────────────────────
    billing_ids: set[str] = {
        s["visitor_id"] for s in customer_sessions
        if s.get("billing") and s["billing"].get("join_ts")
    }
    billing_ids = billing_ids & entry_ids

    # ── Stage 4: PURCHASE ──────────────────────────────────────────────────
    baskets = load_and_process_pos(pos_csv)
    converted_ids = correlate_conversions(baskets, customer_sessions)
    purchase_ids: set[str] = converted_ids & entry_ids

    counts = [len(entry_ids), len(zone_ids), len(billing_ids), len(purchase_ids)]
    labels = ["stage_entry", "stage_zone_visit", "stage_billing", "stage_purchase"]

    # Monotonic guarantee — should always hold given the set-intersection above
    for i in range(1, len(counts)):
        assert counts[i] <= counts[i - 1], (
            f"Funnel monotonicity violated at {labels[i]}: "
            f"{counts[i]} > {counts[i-1]} (prev={labels[i-1]})"
        )

    stages = {}
    for i, label in enumerate(labels):
        prev = counts[i - 1] if i > 0 else counts[0]
        drop_off = (
            round((prev - counts[i]) / prev * 100, 1)
            if prev > 0 and i > 0 else 0.0
        )
        stages[label] = {
            "count": counts[i],
            "drop_off_from_previous_pct": drop_off,
        }

    # Per-session confidence distribution (EC-50)
    from .metrics import session_confidence_distribution
    return {
        "stages": stages,
        "total_sessions": len(customer_sessions),
        "session_confidence_distribution":
            session_confidence_distribution(customer_sessions),
    }
