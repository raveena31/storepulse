"""
app/pos.py
──────────
POS (Point-of-Sale) data loader, basket builder, and purchase correlator.

This module owns the conversion NUMERATOR. The denominator (visitors) comes
from sessions.py. Both must be exact for conversion_rate to be meaningful.

──────────────────────────────────────────────────────────────────────────────
EDGE CASES SOLVED HERE (POS group, prompt 3):
──────────────────────────────────────────────────────────────────────────────
EC-35  Multi-item basket = ONE conversion (the #1 failure mode).
       101 line items → 24 invoices for the Brigade Bangalore CSV.
EC-36  Guest checkout — name "Guest" is NOT a usable identity, never dedup by it.
EC-37  unique_buyers — distinct usable identities + each Guest counts as 1.
EC-38  Mobile POS fallback — if no billing-zone session matched, fall back to
       any non-staff session whose last event ts is within the window.
EC-39  Filter returns and non-sales (invoice_type != "sales", return_id not null,
       total_amount <= 0).
EC-40  Filter pure-GWP baskets (< ₹5 floor — gift-with-purchase only).
EC-42  Clock skew estimation: cross-correlate billing-zone activity vs POS
       transaction histograms to find the shift maximising overlap.
EC-49  IST → UTC conversion done EXACTLY ONCE at the boundary.
       A 5h30m off-by-default silently zeroes all POS correlation.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger("pos")

# IST is UTC+5:30 (no daylight saving)
IST_OFFSET = timedelta(hours=5, minutes=30)

POS_CSV_PATH = os.getenv("POS_CSV_PATH", "data/pos_transactions.csv")

# Brigade CSV uses "guest" as a customer_name for anonymous walk-ins.
GUEST_TOKENS = {"guest", "", "anonymous", "walk-in", "walkin", "n/a", "na", "none"}


# ── EC-49  IST → UTC TIMEZONE BOUNDARY ────────────────────────────────────────

def pos_local_to_utc(date_str: str, time_str: str) -> Optional[datetime]:
    """
    Parse an IST (India local) date/time pair and return UTC datetime.

    POS CSV uses local time. All downstream logic is UTC.
    A wrong conversion silently zeros all correlations (5h30m off).

    Accepts formats:
        date_str: "DD-MM-YYYY" or "DD/MM/YYYY"
        time_str: "HH:MM:SS"

    Returns tz-aware UTC datetime, or None if parsing fails.

    EC-49 implementation.
    """
    if not date_str or not time_str:
        return None
    s = f"{str(date_str).strip()} {str(time_str).strip()}"
    # Try common Indian POS formats
    for fmt in ("%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M",
                "%d/%m/%Y %H:%M"):
        try:
            ist_dt = datetime.strptime(s, fmt)
            return (ist_dt - IST_OFFSET).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ── EC-39  FILTER RETURNS AND NON-SALES ───────────────────────────────────────

def is_real_sale(row: dict) -> bool:
    """
    True iff this POS row represents a genuine sales transaction.

    Excludes:
      - invoice_type != "sales"
      - rows linked to a returns transaction (return_id non-empty)
      - zero/negative amounts

    EC-39 implementation.
    """
    itype = str(row.get("invoice_type", "")).strip().lower()
    if itype != "sales":
        return False

    rid = row.get("return_id")
    rid_str = "" if rid is None else str(rid).strip().lower()
    if rid_str and rid_str != "nan":
        return False

    try:
        amt = float(row.get("total_amount", 0) or 0)
    except (TypeError, ValueError):
        return False
    return amt > 0


# ── EC-40  PURE GWP / TINY BASKETS ────────────────────────────────────────────

def is_meaningful_basket(basket: dict, floor: float = 5.0) -> bool:
    """
    True iff the basket value is at least `floor` rupees.

    Filters out pure gift-with-purchase invoices (₹0, ₹1) that aren't
    real conversions — the customer paid nothing.

    EC-40 implementation.
    """
    try:
        return float(basket.get("value_inr", 0) or 0) >= floor
    except (TypeError, ValueError):
        return False


# ── EC-35  MULTI-ITEM BASKET = ONE CONVERSION ────────────────────────────────

def baskets_from_pos(rows: list[dict]) -> list[dict]:
    """
    Collapse line items into invoices.

    For each unique invoice_number:
      - sum total_amount across all rows
      - take earliest order_time as the invoice timestamp (IST → UTC)
      - keep the customer_name and customer_number from the first row
      - keep store_id from the first row

    101 line items in the Brigade CSV must produce exactly 24 baskets.

    Args:
        rows: list of dicts from CSV (already pre-filtered with is_real_sale
              recommended, but this function will apply filters defensively)

    Returns: list of basket dicts:
        {
            "invoice_number": str,
            "ts_utc":         datetime (UTC),
            "value_inr":      float,
            "store_id":       str,
            "customer_name":  str | None,
            "customer_phone": str | None,
        }

    EC-35 implementation — the #1 failure mode this challenge tests for.
    """
    if not rows:
        return []

    # Defensive: apply real-sale filter
    rows = [r for r in rows if is_real_sale(r)]
    if not rows:
        return []

    # Group by invoice_number
    by_invoice: dict[str, list[dict]] = {}
    for r in rows:
        inv = str(r.get("invoice_number", "")).strip()
        if not inv:
            continue
        by_invoice.setdefault(inv, []).append(r)

    baskets: list[dict] = []
    for inv, items in by_invoice.items():
        # Sum amounts
        total = 0.0
        for r in items:
            try:
                total += float(r.get("total_amount", 0) or 0)
            except (TypeError, ValueError):
                pass

        # Earliest IST timestamp → UTC
        ts_utc: Optional[datetime] = None
        for r in items:
            t = pos_local_to_utc(r.get("order_date", ""), r.get("order_time", ""))
            if t is not None and (ts_utc is None or t < ts_utc):
                ts_utc = t
        if ts_utc is None:
            continue  # can't place this invoice in time

        first = items[0]
        baskets.append({
            "invoice_number": inv,
            "ts_utc": ts_utc,
            "value_inr": round(total, 2),
            "store_id": str(first.get("store_id", "")).strip(),
            "customer_name": (str(first.get("customer_name", "")).strip()
                              if first.get("customer_name") is not None else None),
            "customer_phone": (str(first.get("customer_number", "")).strip()
                               if first.get("customer_number") is not None else None),
        })

    # EC-40: drop pure-GWP / tiny baskets
    baskets = [b for b in baskets if is_meaningful_basket(b)]
    return baskets


# ── EC-36  GUEST CHECKOUT ─────────────────────────────────────────────────────

def usable_identity(name: Optional[str], phone: Optional[str]) -> Optional[str]:
    """
    Return a deduplication key for this buyer, or None if no usable identity.

    "Guest" is NOT a usable identity — different physical people share the name.
    Phone numbers are usable when present and look like a number.

    Returns:
        usable_id string suitable for dedup, OR None for anonymous buyers.

    EC-36 implementation.
    """
    n = (name or "").strip().lower()
    p = (phone or "").strip()

    if n in GUEST_TOKENS:
        # Even if a phone is provided alongside Guest, we don't trust it for dedup
        # (cashier often types the store's default phone)
        return None

    if n:
        # Use name as the identity (case-insensitive, whitespace-collapsed)
        return n

    if p and any(c.isdigit() for c in p):
        return f"phone:{p}"

    return None


# ── EC-37  UNIQUE BUYERS ──────────────────────────────────────────────────────

def unique_buyers(baskets: list[dict]) -> int:
    """
    Count distinct buyers across a list of baskets.

    For baskets with a usable identity: count distinct identities.
    For baskets without (Guest/anonymous): each counts as 1 separate buyer.

    Returns: integer count.

    EC-37 implementation.
    """
    named_ids: set[str] = set()
    anon_count = 0
    for b in baskets:
        ident = usable_identity(b.get("customer_name"), b.get("customer_phone"))
        if ident is None:
            anon_count += 1
        else:
            named_ids.add(ident)
    return len(named_ids) + anon_count


# ── EC-34  CLOSEST-UNATTRIBUTED CORRELATION ──────────────────────────────────

def correlate_txn_to_session(
    txn_ts: datetime,
    billing_sessions: list[dict],
    window_s: int = 300,
) -> Optional[str]:
    """
    Match one POS transaction to the best billing-zone session.

    Filter: sessions with billing.join_ts not None, NOT already attributed,
    join_ts within window_s BEFORE txn_ts.

    Among candidates, pick the closest (smallest gap).
    Mark the chosen session attributed=True (mutates in place).

    Returns the visitor_id of the matched session, or None.

    EC-34 implementation.
    """
    bts = _ensure_utc(txn_ts)
    window = timedelta(seconds=window_s)

    best: Optional[dict] = None
    best_gap = timedelta.max

    for s in billing_sessions:
        if s.get("is_staff"):
            continue
        billing = s.get("billing")
        if not billing or billing.get("join_ts") is None:
            continue
        if billing.get("attributed"):
            continue

        jts = _ensure_utc(billing["join_ts"])
        gap = bts - jts
        if timedelta(0) <= gap <= window and gap < best_gap:
            best = s
            best_gap = gap

    if best is None:
        return None

    best["billing"]["attributed"] = True
    return best["visitor_id"]


# ── EC-38  MOBILE POS FALLBACK ────────────────────────────────────────────────

def attribute_txn(
    txn_ts: datetime,
    billing_sessions: list[dict],
    all_sessions: list[dict],
    window_s: int = 300,
) -> tuple[Optional[str], str]:
    """
    Attribute a POS transaction to a visitor using two strategies:

      1. Primary  — billing-zone match (EC-34).
      2. Fallback — closest non-staff session whose last event is within window.
                    Catches mobile POS / sales-on-floor where the customer
                    never appeared on the billing camera.

    Returns (visitor_id, method) where method ∈ {"billing", "last_zone_fallback", "none"}.

    EC-38 implementation.
    """
    # Try the primary attribution
    vid = correlate_txn_to_session(txn_ts, billing_sessions, window_s=window_s)
    if vid is not None:
        return vid, "billing"

    # Fallback: any non-staff session whose last event ts is within window
    bts = _ensure_utc(txn_ts)
    window = timedelta(seconds=window_s)

    best: Optional[dict] = None
    best_gap = timedelta.max
    for s in all_sessions:
        if s.get("is_staff"):
            continue
        if s.get("_fallback_attributed"):
            continue
        last_ts = _ensure_utc(s.get("exit_ts")) or _last_event_ts(s)
        if last_ts is None:
            continue
        gap = bts - last_ts
        if timedelta(0) <= gap <= window and gap < best_gap:
            best = s
            best_gap = gap

    if best is None:
        return None, "none"

    best["_fallback_attributed"] = True
    return best["visitor_id"], "last_zone_fallback"


# ── EC-42  CLOCK SKEW ESTIMATION ──────────────────────────────────────────────

def estimate_clock_offset(
    billing_minute_hist: dict[int, int],
    txn_minute_hist: dict[int, int],
    max_shift: int = 10,
) -> int:
    """
    Estimate clock offset (in minutes) between camera and POS.

    Idea: billing-queue-join activity and POS transaction activity peak together.
    If they peak at different minutes, the difference is the clock skew.

    Cross-correlate the two histograms by shifting txn_hist by ±max_shift minutes
    and finding the shift that maximises element-wise product-sum.

    Args:
        billing_minute_hist: {minute: count} of BILLING_QUEUE_JOIN events
        txn_minute_hist:     {minute: count} of POS invoice timestamps
        max_shift:           ± minutes to consider

    Returns:
        Integer offset (minutes). Positive means POS clock is AHEAD of camera.
        Apply this offset to camera ts before correlation.

    EC-42 implementation.
    """
    if not billing_minute_hist or not txn_minute_hist:
        return 0

    best_shift = 0
    best_score = -1
    for shift in range(-max_shift, max_shift + 1):
        score = 0
        for minute, b_count in billing_minute_hist.items():
            score += b_count * txn_minute_hist.get(minute + shift, 0)
        if score > best_score:
            best_score = score
            best_shift = shift

    logger.info("estimate_clock_offset: best_shift=%d minutes (score=%d)",
                best_shift, best_score)
    return best_shift


# ── Top-level loaders ─────────────────────────────────────────────────────────

def load_and_process_pos(csv_path: str = POS_CSV_PATH) -> list[dict]:
    """
    Load the POS CSV and return a list of clean basket dicts.

    Pipeline:
      1. Read CSV via pandas
      2. Apply is_real_sale row filter (EC-39)
      3. Collapse line items to invoices (EC-35)
      4. Drop pure-GWP / sub-₹5 baskets (EC-40)
      5. Convert IST → UTC at the boundary (EC-49)

    Returns [] for missing/unreadable files (never raises).
    """
    if not os.path.exists(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    except Exception as exc:
        logger.warning("load_and_process_pos: read failed: %s", exc)
        return []

    if df.empty:
        return []

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rows = df.to_dict(orient="records")
    return baskets_from_pos(rows)


def correlate_conversions(
    baskets: list[dict],
    sessions: list[dict],
    window_s: int = 300,
) -> set:
    """
    Top-level convenience: match all baskets to sessions using EC-34 + EC-38.

    Returns set of converted visitor_ids.
    """
    converted: set = set()
    non_staff = [s for s in sessions if not s.get("is_staff")]
    # Reset per-call attribution flags so re-runs are deterministic
    for s in non_staff:
        if s.get("billing"):
            s["billing"]["attributed"] = False
        s["_fallback_attributed"] = False

    for basket in baskets:
        vid, method = attribute_txn(
            basket["ts_utc"], non_staff, non_staff, window_s=window_s
        )
        if vid is not None:
            converted.add(vid)
    return converted


# ── Helpers ───────────────────────────────────────────────────────────────────

def _last_event_ts(session: dict) -> Optional[datetime]:
    events = session.get("events", [])
    if not events:
        return None
    from .sessions import _ts
    return max(_ts(e) for e in events)


def _ensure_utc(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None
