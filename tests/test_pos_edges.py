# PROMPT: "Generate pytest cases for app/pos.py covering: invoice collapse (101 → 24),
#          IST→UTC, Guest checkout, returns filter, GWP floor, closest-unattributed
#          correlation, mobile-POS fallback. Use the real Brigade Bangalore CSV for EC-35
#          so the test fails loudly if anyone introduces a regression."
# CHANGES MADE: AI's EC-35 test counted len(df) (101) instead of len(unique invoices) (24)
#               — it asserted the wrong contract. That's the #1 failure mode this challenge
#               tests for, so I replaced it with `assert len(baskets) == 24` against the
#               real CSV (NOT a synthetic fixture — a synthetic test would pass while the
#               production loader silently broke). AI also wrote EC-49 to return a naive
#               datetime; I corrected to assert `utc.tzinfo == timezone.utc` because a naive
#               datetime would silently zero all POS correlation (5h30m off-by-default).
# Other tests are pure unit tests — no API, no DB.

import os
from datetime import datetime, timedelta, timezone

import pytest

from app.pos import (
    pos_local_to_utc,
    is_real_sale,
    is_meaningful_basket,
    baskets_from_pos,
    usable_identity,
    unique_buyers,
    correlate_txn_to_session,
    attribute_txn,
    estimate_clock_offset,
    load_and_process_pos,
    IST_OFFSET,
)
from app.sessions import (
    build_sessions,
    close_dangling_sessions,
    within_watermark,
)
from app.funnel import compute_funnel
from app.metrics import compute_metrics


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIGADE_CSV = os.path.join(BASE_DIR, "data", "Brigade_Bangalore_10_April_26.csv")

NOW_UTC = datetime(2026, 4, 10, 14, 40, 0, tzinfo=timezone.utc)


# ── EC-49  IST → UTC ──────────────────────────────────────────────────────────

class TestEC49ISTtoUTC:
    """A 5h30m off-by-default would silently zero all POS correlation."""

    def test_basic_conversion(self):
        """16:55:36 IST → 11:25:36 UTC"""
        utc = pos_local_to_utc("10-04-2026", "16:55:36")
        assert utc is not None
        assert utc.tzinfo == timezone.utc
        assert utc.year == 2026 and utc.month == 4 and utc.day == 10
        assert utc.hour == 11 and utc.minute == 25 and utc.second == 36

    def test_handles_slash_separator(self):
        utc = pos_local_to_utc("10/04/2026", "16:55:36")
        assert utc is not None
        assert utc.hour == 11

    def test_handles_iso_date(self):
        utc = pos_local_to_utc("2026-04-10", "16:55:36")
        assert utc is not None
        assert utc.hour == 11

    def test_malformed_returns_none(self):
        assert pos_local_to_utc("garbage", "garbage") is None
        assert pos_local_to_utc("", "") is None

    def test_midnight_rollover(self):
        """11-04 00:50 IST → 10-04 19:20 UTC (crosses midnight backwards)."""
        utc = pos_local_to_utc("11-04-2026", "00:50:00")
        assert utc is not None
        assert utc.day == 10 and utc.hour == 19 and utc.minute == 20


# ── EC-39  FILTER RETURNS AND NON-SALES ───────────────────────────────────────

class TestEC39FilterReturns:

    def test_sales_kept(self):
        row = {"invoice_type": "sales", "return_id": "", "total_amount": 100}
        assert is_real_sale(row) is True

    def test_returns_excluded(self):
        row = {"invoice_type": "sales", "return_id": "RET001", "total_amount": 100}
        assert is_real_sale(row) is False

    def test_non_sales_excluded(self):
        row = {"invoice_type": "RETURNS", "return_id": "", "total_amount": 100}
        assert is_real_sale(row) is False

    def test_zero_amount_excluded(self):
        row = {"invoice_type": "sales", "return_id": "", "total_amount": 0}
        assert is_real_sale(row) is False

    def test_negative_amount_excluded(self):
        row = {"invoice_type": "sales", "return_id": "", "total_amount": -50}
        assert is_real_sale(row) is False


# ── EC-40  PURE GWP / TINY BASKETS ────────────────────────────────────────────

class TestEC40PureGWP:

    def test_one_rupee_basket_excluded(self):
        assert is_meaningful_basket({"value_inr": 1.0}, floor=5.0) is False

    def test_meaningful_basket_kept(self):
        assert is_meaningful_basket({"value_inr": 200.0}, floor=5.0) is True

    def test_zero_basket_excluded(self):
        assert is_meaningful_basket({"value_inr": 0.0}, floor=5.0) is False


# ── EC-35  MULTI-ITEM BASKET = ONE CONVERSION (CRITICAL) ──────────────────────

class TestEC35MultiItemBasket:
    """The #1 failure mode — 101 line items must become exactly 24 baskets."""

    @pytest.mark.skipif(not os.path.exists(BRIGADE_CSV), reason="Brigade CSV not in repo")
    def test_brigade_csv_exactly_24_baskets(self):
        """REAL CSV check — must produce exactly 24 baskets from 101 line items."""
        baskets = load_and_process_pos(BRIGADE_CSV)
        assert len(baskets) == 24, (
            f"Expected 24 baskets from Brigade CSV, got {len(baskets)}. "
            f"This is the #1 failure mode this challenge tests for."
        )

    def test_synthetic_collapse(self):
        rows = [
            {"invoice_number": "INV-A", "invoice_type": "sales", "return_id": "",
             "order_date": "10-04-2026", "order_time": "10:00:00", "total_amount": 100,
             "store_id": "ST1008", "customer_name": "Anita", "customer_number": "999"},
            {"invoice_number": "INV-A", "invoice_type": "sales", "return_id": "",
             "order_date": "10-04-2026", "order_time": "10:00:00", "total_amount": 200,
             "store_id": "ST1008", "customer_name": "Anita", "customer_number": "999"},
            {"invoice_number": "INV-B", "invoice_type": "sales", "return_id": "",
             "order_date": "10-04-2026", "order_time": "10:05:00", "total_amount": 50,
             "store_id": "ST1008", "customer_name": "Guest", "customer_number": ""},
        ]
        baskets = baskets_from_pos(rows)
        assert len(baskets) == 2
        inv_a = next(b for b in baskets if b["invoice_number"] == "INV-A")
        assert inv_a["value_inr"] == 300.0  # 100 + 200 summed

    def test_earliest_time_picked(self):
        """When multiple line items have different times, basket ts = earliest."""
        rows = [
            {"invoice_number": "INV-X", "invoice_type": "sales", "return_id": "",
             "order_date": "10-04-2026", "order_time": "10:05:00", "total_amount": 100,
             "store_id": "ST1008"},
            {"invoice_number": "INV-X", "invoice_type": "sales", "return_id": "",
             "order_date": "10-04-2026", "order_time": "10:00:00", "total_amount": 50,
             "store_id": "ST1008"},
        ]
        baskets = baskets_from_pos(rows)
        assert len(baskets) == 1
        # 10:00 IST = 04:30 UTC
        assert baskets[0]["ts_utc"].hour == 4 and baskets[0]["ts_utc"].minute == 30


# ── EC-36  GUEST CHECKOUT ─────────────────────────────────────────────────────

class TestEC36GuestCheckout:

    def test_guest_returns_none(self):
        assert usable_identity("Guest", "9876543210") is None

    def test_guest_lowercase_returns_none(self):
        assert usable_identity("guest", "9876543210") is None

    def test_named_customer_returns_id(self):
        assert usable_identity("Anita Verma", "9876543210") == "anita verma"

    def test_empty_name_returns_phone_or_none(self):
        # No name → look at phone
        result = usable_identity("", "9876543210")
        # Could be phone-based or None — either is acceptable per spec
        assert result is None or "9876543210" in result

    def test_completely_empty_returns_none(self):
        assert usable_identity(None, None) is None
        assert usable_identity("", "") is None


# ── EC-37  UNIQUE BUYERS ──────────────────────────────────────────────────────

class TestEC37UniqueBuyers:

    def test_named_dedup_plus_guest_each_one(self):
        baskets = [
            {"customer_name": "Anita V", "customer_phone": "999"},
            {"customer_name": "Anita V", "customer_phone": "999"},  # same → dedup
            {"customer_name": "Guest",   "customer_phone": "888"},  # anon → +1
        ]
        # Anita = 1, Guest = 1 → total 2
        assert unique_buyers(baskets) == 2

    def test_two_guests_count_as_two(self):
        baskets = [
            {"customer_name": "Guest", "customer_phone": ""},
            {"customer_name": "Guest", "customer_phone": ""},
        ]
        # Each Guest is a separate buyer (we don't merge anonymous)
        assert unique_buyers(baskets) == 2

    def test_three_distinct_named(self):
        baskets = [
            {"customer_name": "A", "customer_phone": "1"},
            {"customer_name": "B", "customer_phone": "2"},
            {"customer_name": "C", "customer_phone": "3"},
        ]
        assert unique_buyers(baskets) == 3


# ── EC-34  CLOSEST UNATTRIBUTED CORRELATION ───────────────────────────────────

class TestEC34ClosestUnattributed:

    def _session(self, vid, join_offset_s):
        return {
            "visitor_id": vid,
            "is_staff": False,
            "billing": {
                "join_ts": NOW_UTC - timedelta(seconds=join_offset_s),
                "depth": 1,
                "abandoned": False,
                "attributed": False,
            },
        }

    def test_picks_closest(self):
        sessions = [
            self._session("V1", join_offset_s=180),  # 3 min before
            self._session("V2", join_offset_s=60),   # 1 min before (closest)
            self._session("V3", join_offset_s=240),  # 4 min before
        ]
        vid = correlate_txn_to_session(NOW_UTC, sessions, window_s=300)
        assert vid == "V2"

    def test_no_double_attribution(self):
        sessions = [self._session("V1", join_offset_s=60)]
        first = correlate_txn_to_session(NOW_UTC, sessions, window_s=300)
        second = correlate_txn_to_session(NOW_UTC, sessions, window_s=300)
        assert first == "V1"
        assert second is None  # already attributed

    def test_outside_window_no_match(self):
        sessions = [self._session("V1", join_offset_s=600)]  # 10 min before, window=300
        assert correlate_txn_to_session(NOW_UTC, sessions, window_s=300) is None

    def test_staff_ignored(self):
        s = self._session("STAFF_1", join_offset_s=60)
        s["is_staff"] = True
        assert correlate_txn_to_session(NOW_UTC, [s], window_s=300) is None


# ── EC-38  MOBILE POS FALLBACK ────────────────────────────────────────────────

class TestEC38MobilePOSFallback:

    def test_falls_back_to_last_zone(self):
        # No billing session, but a non-staff session active 90s ago
        sess = {
            "visitor_id": "VFLOOR",
            "is_staff": False,
            "billing": None,
            "exit_ts": NOW_UTC - timedelta(seconds=90),
            "events": [],
        }
        vid, method = attribute_txn(NOW_UTC, [], [sess], window_s=300)
        assert vid == "VFLOOR"
        assert method == "last_zone_fallback"

    def test_billing_match_preferred_over_fallback(self):
        billing_sess = {
            "visitor_id": "VBILL",
            "is_staff": False,
            "billing": {"join_ts": NOW_UTC - timedelta(seconds=60),
                        "depth": 1, "abandoned": False, "attributed": False},
            "exit_ts": None, "events": [],
        }
        floor_sess = {
            "visitor_id": "VFLOOR",
            "is_staff": False,
            "billing": None,
            "exit_ts": NOW_UTC - timedelta(seconds=30),
            "events": [],
        }
        vid, method = attribute_txn(NOW_UTC, [billing_sess], [billing_sess, floor_sess], window_s=300)
        assert method == "billing"
        assert vid == "VBILL"

    def test_no_match_at_all(self):
        vid, method = attribute_txn(NOW_UTC, [], [], window_s=300)
        assert vid is None
        assert method == "none"


# ── EC-42  CLOCK SKEW ESTIMATION ──────────────────────────────────────────────

class TestEC42ClockSkew:

    def test_positive_offset_detected(self):
        """Billing peak at minute 30, txn peak at minute 32 → offset = +2."""
        billing = {28: 1, 29: 2, 30: 5, 31: 3, 32: 1}
        txn     = {30: 1, 31: 2, 32: 5, 33: 3, 34: 1}
        offset = estimate_clock_offset(billing, txn, max_shift=10)
        assert offset == 2

    def test_no_offset(self):
        billing = {30: 5, 31: 3}
        txn     = {30: 5, 31: 3}
        offset = estimate_clock_offset(billing, txn, max_shift=10)
        assert offset == 0

    def test_empty_returns_zero(self):
        assert estimate_clock_offset({}, {30: 5}) == 0
        assert estimate_clock_offset({30: 5}, {}) == 0


# ── EC-41  DANGLING SESSIONS ──────────────────────────────────────────────────

class TestEC41DanglingSessions:

    def test_no_exit_inferred_at_clip_end(self):
        clip_end = datetime(2026, 4, 10, 22, 0, 0, tzinfo=timezone.utc)
        sessions = [
            {"visitor_id": "V1", "exit_ts": None, "exit_inferred": False},
            {"visitor_id": "V2",
             "exit_ts": datetime(2026, 4, 10, 21, 0, 0, tzinfo=timezone.utc),
             "exit_inferred": False},
        ]
        close_dangling_sessions(sessions, clip_end)
        assert sessions[0]["exit_ts"] == clip_end
        assert sessions[0]["exit_inferred"] is True
        # Already-closed session is untouched
        assert sessions[1]["exit_inferred"] is False


# ── EC-43  OUT-OF-ORDER / LATE EVENTS ─────────────────────────────────────────

class TestEC43Watermark:

    def test_within_watermark_recent(self):
        now = datetime.now(timezone.utc)
        recent = now - timedelta(seconds=10)
        assert within_watermark(recent, now, grace_s=30) is True

    def test_outside_watermark_late(self):
        now = datetime.now(timezone.utc)
        late = now - timedelta(seconds=60)
        assert within_watermark(late, now, grace_s=30) is False

    def test_events_sorted_in_build_sessions(self):
        """Ensure build_sessions handles unsorted input correctly."""
        events = [
            {"event_id": "e2", "store_id": "S", "camera_id": "C", "visitor_id": "V1",
             "event_type": "EXIT", "timestamp": "2026-04-10T10:30:00+00:00",
             "is_staff": False, "confidence": 0.9, "zone_id": None, "dwell_ms": 0},
            {"event_id": "e1", "store_id": "S", "camera_id": "C", "visitor_id": "V1",
             "event_type": "ENTRY", "timestamp": "2026-04-10T10:00:00+00:00",
             "is_staff": False, "confidence": 0.9, "zone_id": None, "dwell_ms": 0},
        ]
        sessions = build_sessions(events)
        assert len(sessions) == 1
        # entry_ts should be the earlier ts despite input order
        assert sessions[0]["entry_ts"].hour == 10 and sessions[0]["entry_ts"].minute == 0


# ── EC-46/47  EMPTY STORE + ALL-STAFF GUARDS ──────────────────────────────────

class TestEC4647EdgeGuards:

    def test_empty_store_safe_defaults(self):
        m = compute_metrics([], pos_csv="nonexistent.csv")
        assert m["unique_visitors"] == 0
        assert m["conversion_rate"] == 0.0
        assert m["status"] == "NO_TRAFFIC"
        # No NaN anywhere
        import math
        for k, v in m.items():
            if isinstance(v, float):
                assert not math.isnan(v), f"{k} is NaN"

    def test_all_staff_clip_safe_defaults(self):
        events = [
            {"event_id": f"e{i}", "store_id": "ST1008", "camera_id": "CAM_04",
             "visitor_id": f"STAFF_{i}", "event_type": "ENTRY",
             "timestamp": "2026-04-10T10:00:00+00:00",
             "is_staff": True, "confidence": 0.9, "zone_id": None, "dwell_ms": 0}
            for i in range(10)
        ]
        m = compute_metrics(events, pos_csv="nonexistent.csv")
        assert m["unique_visitors"] == 0
        assert m["conversion_rate"] == 0.0
        assert m["status"] == "NO_TRAFFIC"


# ── FUNNEL DEDUP ──────────────────────────────────────────────────────────────

class TestFunnelDedup:
    """Funnel must count strictly unique visitor_ids per stage. Monotonic."""

    def test_reentry_counts_as_one_entry(self):
        """1 re-entering visitor (REENTRY event same visitor_id) → 1, not 2."""
        events = [
            {"event_id": "e1", "store_id": "S", "camera_id": "C", "visitor_id": "V1",
             "event_type": "ENTRY", "timestamp": "2026-04-10T10:00:00+00:00",
             "is_staff": False, "confidence": 0.9, "zone_id": None, "dwell_ms": 0},
            {"event_id": "e2", "store_id": "S", "camera_id": "C", "visitor_id": "V1",
             "event_type": "REENTRY", "timestamp": "2026-04-10T11:00:00+00:00",
             "is_staff": False, "confidence": 0.9, "zone_id": None, "dwell_ms": 0},
        ]
        f = compute_funnel(events, pos_csv="nonexistent.csv")
        assert f["stages"]["stage_entry"]["count"] == 1

    def test_monotonic_decrease(self):
        """Each stage ≤ previous stage."""
        events = [
            {"event_id": f"v{i}-e", "store_id": "S", "camera_id": "C",
             "visitor_id": f"V{i}", "event_type": "ENTRY",
             "timestamp": "2026-04-10T10:00:00+00:00",
             "is_staff": False, "confidence": 0.9, "zone_id": None, "dwell_ms": 0}
            for i in range(5)
        ] + [
            {"event_id": f"v{i}-z", "store_id": "S", "camera_id": "C",
             "visitor_id": f"V{i}", "event_type": "ZONE_ENTER",
             "timestamp": "2026-04-10T10:05:00+00:00",
             "is_staff": False, "confidence": 0.9, "zone_id": "LAKME", "dwell_ms": 0}
            for i in range(3)
        ]
        f = compute_funnel(events, pos_csv="nonexistent.csv")
        c = f["stages"]
        assert c["stage_zone_visit"]["count"] <= c["stage_entry"]["count"]
        assert c["stage_billing"]["count"]    <= c["stage_zone_visit"]["count"]
        assert c["stage_purchase"]["count"]   <= c["stage_billing"]["count"]
