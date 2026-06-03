"""
app/heatmap.py
──────────────
Brand-level zone heatmap with attention-vs-sales reconciliation.

This is the unique merchandising insight Purplle's regional managers need:
where do customers spend time vs where do they actually buy?

Outputs per zone:
  - visit_count, avg_dwell_ms, normalised_score (0-100)
  - attention_vs_sales:
      dwell_share:     this zone's share of total dwell time
      sales_share:     this zone's brand share of total POS revenue
      gap:             dwell_share - sales_share
      interpretation:  HIGH_ATTENTION_LOW_SALES | BALANCED | LOW_ATTENTION_HIGH_SALES
  - status:           ACTIVE | DEAD_ZONE | UNKNOWN_NO_COVERAGE
  - data_confidence:  OK if ≥20 sessions, else LOW

Brand → zone mapping is config-driven from config/store_ST1008.yaml
under `zone_brand_map`. POS brand_name strings are normalised via brand_key()
before matching.

Single-session zone dwell is capped at 10 minutes (capped_zone_dwell) so a
single makeup trial doesn't dominate the heatmap.
"""
from __future__ import annotations

import os
from typing import Optional

import yaml

from .sessions import build_sessions
from .pos import load_and_process_pos, POS_CSV_PATH


# Single-session dwell cap to prevent one stationary customer dominating
SINGLE_SESSION_DWELL_CAP_MS = 600_000  # 10 minutes


# ── Normalisation ─────────────────────────────────────────────────────────────

def brand_key(s: str) -> str:
    """
    Normalise a brand-name string for matching.

    Steps:
      1. strip whitespace
      2. lowercase
      3. replace "&" with "and"
      4. collapse runs of whitespace into one space

    Examples:
        "Faces Canada"   → "faces canada"
        "NY Bae"         → "ny bae"
        "P&G Beauty"     → "p and g beauty"
        "  Lakme   "     → "lakme"
    """
    if s is None:
        return ""
    s = str(s).strip().lower().replace("&", " and ")
    # Collapse whitespace
    return " ".join(s.split())


# ── Dwell capping ─────────────────────────────────────────────────────────────

def capped_zone_dwell(dwell_ms: int, cap_ms: int = SINGLE_SESSION_DWELL_CAP_MS) -> int:
    """
    Cap single-session zone dwell at cap_ms milliseconds.

    Why: a customer who tries a makeup product for 30 minutes shouldn't
    dominate the zone's dwell-share calculation and trigger false
    HIGH_ATTENTION_LOW_SALES interpretations.
    """
    if dwell_ms is None or dwell_ms < 0:
        return 0
    return min(int(dwell_ms), int(cap_ms))


# ── Confidence calibration ────────────────────────────────────────────────────

def data_confidence(n_sessions: int, min_n: int = 20) -> str:
    """
    "OK" when we have at least min_n sessions, else "LOW".

    Low-sample heatmaps are statistically unreliable — surface this to the
    user rather than silently presenting noise as insight.
    """
    return "OK" if n_sessions >= min_n else "LOW"


# ── Zone status ───────────────────────────────────────────────────────────────

def zone_status(visits: int, has_coverage: bool, window_min: int) -> str:
    """
    Classify a zone's current state.

    UNKNOWN_NO_COVERAGE  — no camera covers this zone (config-declared)
                           Cannot say whether it's dead or busy.
    DEAD_ZONE            — has coverage but zero visits over a meaningful window
                           (default ≥30 min). Real merchandising concern.
    ACTIVE               — has visits in the window.
    """
    if not has_coverage:
        return "UNKNOWN_NO_COVERAGE"
    if visits == 0 and window_min >= 30:
        return "DEAD_ZONE"
    return "ACTIVE"


# ── Brand → zone mapping (config-driven) ──────────────────────────────────────

_config_cache: Optional[dict] = None


def _load_config() -> dict:
    global _config_cache
    if _config_cache is None:
        cfg_path = os.path.join(
            os.getenv("CONFIG_DIR", "config"), "store_ST1008.yaml"
        )
        try:
            with open(cfg_path) as f:
                _config_cache = yaml.safe_load(f) or {}
        except Exception:
            _config_cache = {}
    return _config_cache


def _brand_to_zone() -> dict[str, str]:
    """
    Invert the zone_brand_map config block into {normalised_brand_key: zone_id}.
    """
    cfg = _load_config()
    zone_brand_map = cfg.get("zone_brand_map", {})
    result: dict[str, str] = {}
    for zone_id, brand_names in zone_brand_map.items():
        for brand_name in (brand_names or []):
            key = brand_key(brand_name)
            if key:
                result[key] = zone_id
    return result


def _zones_with_camera_coverage() -> set[str]:
    """
    Set of zone_ids that DO have camera coverage.
    Computed as: all zones MINUS zones_without_camera_coverage list.
    """
    cfg = _load_config()
    all_zones = set(cfg.get("zones", {}).keys())
    no_cov = set(cfg.get("zones_without_camera_coverage", []) or [])
    return all_zones - no_cov


# ── Attention vs sales (the core insight) ─────────────────────────────────────

def attention_vs_sales(
    zone_dwell: dict[str, float],
    pos_brand_rev: dict[str, float],
    gap_threshold: float = 0.10,
) -> dict[str, dict]:
    """
    Per-zone reconciliation of dwell share vs revenue share.

    Args:
        zone_dwell:    {zone_id: total_capped_dwell_ms} (typically from compute_heatmap)
        pos_brand_rev: {brand_name: revenue_inr} (raw — function will normalise)
        gap_threshold: |gap| > this triggers HIGH_/LOW_ interpretation

    Returns:
        {
            zone_id: {
                "dwell_share":   float in [0, 1],
                "sales_share":   float in [0, 1],
                "gap":           float (dwell - sales),
                "interpretation": "HIGH_ATTENTION_LOW_SALES"
                                 | "BALANCED"
                                 | "LOW_ATTENTION_HIGH_SALES",
            }
        }

    All divisions guarded — zero total dwell or zero total revenue → 0.0 shares.

    Interpretation rules:
        gap > +gap_threshold       → HIGH_ATTENTION_LOW_SALES (re-merchandise)
        gap < -gap_threshold       → LOW_ATTENTION_HIGH_SALES (efficient zone)
        otherwise                  → BALANCED
    """
    brand_to_zone = _brand_to_zone()

    total_dwell = sum(v for v in zone_dwell.values() if v) or 0.0
    # Aggregate POS revenue per ZONE (via brand_to_zone mapping)
    zone_revenue: dict[str, float] = {}
    for brand_name, rev in pos_brand_rev.items():
        z = brand_to_zone.get(brand_key(brand_name))
        if z is None:
            continue
        zone_revenue[z] = zone_revenue.get(z, 0.0) + float(rev or 0.0)
    total_revenue = sum(zone_revenue.values()) or 0.0

    result: dict[str, dict] = {}
    # All zones that appear in either dwell or revenue get a row
    all_zones = set(zone_dwell.keys()) | set(zone_revenue.keys())
    for zone in all_zones:
        d = zone_dwell.get(zone, 0.0)
        r = zone_revenue.get(zone, 0.0)
        d_share = (d / total_dwell)   if total_dwell   > 0 else 0.0
        r_share = (r / total_revenue) if total_revenue > 0 else 0.0
        gap = d_share - r_share

        if gap > gap_threshold:
            interp = "HIGH_ATTENTION_LOW_SALES"
        elif gap < -gap_threshold:
            interp = "LOW_ATTENTION_HIGH_SALES"
        else:
            interp = "BALANCED"

        result[zone] = {
            "dwell_share":    round(d_share, 4),
            "sales_share":    round(r_share, 4),
            "gap":            round(gap, 4),
            "interpretation": interp,
        }
    return result


# ── POS revenue by brand ──────────────────────────────────────────────────────

def _pos_brand_revenue(csv_path: str = POS_CSV_PATH) -> dict[str, float]:
    """
    Aggregate POS revenue per brand_name from the raw CSV.

    Returns: {brand_name_raw: total_inr}
    """
    if not os.path.exists(csv_path):
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Filter sales only
        if "invoice_type" in df.columns:
            df = df[df["invoice_type"].astype(str).str.strip().str.lower() == "sales"]
        if "return_id" in df.columns:
            df = df[df["return_id"].isna() | (df["return_id"].astype(str).str.strip() == "")]

        # Find brand column
        brand_col = next((c for c in df.columns if c in ("brand_name", "brand")), None)
        if brand_col is None:
            return {}
        amount_col = next((c for c in df.columns if c in ("total_amount", "amount", "nmv", "gmv")), None)
        if amount_col is None:
            return {}

        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
        agg = df.groupby(brand_col)[amount_col].sum().to_dict()
        return {str(k): float(v) for k, v in agg.items()}
    except Exception:
        return {}


# ── Top-level: full heatmap payload ───────────────────────────────────────────

def compute_heatmap(events: list[dict], pos_csv: str = POS_CSV_PATH) -> dict:
    """
    Build the full heatmap payload returned by GET /stores/{id}/heatmap.

    For each zone with activity OR with a configured brand mapping:
      visit_count, avg_dwell_ms, normalised_score (0-100),
      attention_vs_sales (dwell_share, sales_share, gap, interpretation),
      status (ACTIVE | DEAD_ZONE | UNKNOWN_NO_COVERAGE),
      data_confidence (OK | LOW).

    Single-session dwells are capped via capped_zone_dwell before aggregation.
    """
    sessions = build_sessions(events)
    customer_sessions = [s for s in sessions if not s.get("is_staff")]
    n_sessions = len(customer_sessions)

    # Aggregate visits + capped dwell per zone
    zone_visits: dict[str, int] = {}
    zone_dwell_capped: dict[str, float] = {}
    for s in customer_sessions:
        for zone, dwell in s.get("zones", {}).items():
            zone_visits[zone] = zone_visits.get(zone, 0) + 1
            zone_dwell_capped[zone] = (
                zone_dwell_capped.get(zone, 0.0) + capped_zone_dwell(dwell)
            )

    # Load POS brand revenue
    pos_brand_rev = _pos_brand_revenue(pos_csv)

    # Compute attention-vs-sales for all relevant zones
    avs = attention_vs_sales(zone_dwell_capped, pos_brand_rev)

    # Determine coverage and union of all zones to display
    zones_with_cov = _zones_with_camera_coverage()
    cfg_zones = set(_load_config().get("zones", {}).keys())
    union_zones = zone_visits.keys() | avs.keys() | cfg_zones

    if not union_zones:
        return {
            "zones": {},
            "data_confidence": data_confidence(n_sessions),
            "attention_vs_sales": {},
        }

    max_visits = max(zone_visits.values()) if zone_visits else 1

    zones_out: dict[str, dict] = {}
    for zone in union_zones:
        visits = zone_visits.get(zone, 0)
        dwell_total = zone_dwell_capped.get(zone, 0.0)
        avg_dwell = (dwell_total / visits) if visits > 0 else 0.0
        normalised = round((visits / max_visits) * 100, 1) if max_visits > 0 else 0.0
        has_cov = zone in zones_with_cov

        zones_out[zone] = {
            "visit_count": visits,
            "avg_dwell_ms": round(avg_dwell, 1),
            "normalised_score": normalised,
            "status": zone_status(visits, has_cov, window_min=60),
            "data_confidence": data_confidence(n_sessions),
        }

    return {
        "zones": zones_out,
        "data_confidence": data_confidence(n_sessions),
        "attention_vs_sales": avs,
    }
