"""
pipeline/staff.py
─────────────────
Staff detection functions (EC-21, EC-22, EC-24, EC-25).

Staff exclusion is the single most important data quality guarantee:
  conversion_rate = buyers / unique_NON_STAFF_visitors

Every staff detection mechanism here feeds into the is_staff flag on events.
"""
from __future__ import annotations

import os
from typing import Optional

import yaml

_roster_cache: Optional[list[str]] = None


def _load_roster(path: Optional[str] = None) -> list[str]:
    """Load staff names from YAML. Cached after first call."""
    global _roster_cache
    if _roster_cache is not None:
        return _roster_cache
    cfg_path = path or os.path.join(
        os.getenv("CONFIG_DIR", "config"), "staff_roster.yaml"
    )
    try:
        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        _roster_cache = [s["name"].lower().strip() for s in data.get("staff", [])]
    except Exception:
        _roster_cache = []
    return _roster_cache


# ── EC-24  ROSTER CROSS-CHECK ────────────────────────────────────────────────

def roster_is_staff(name: str, roster_path: Optional[str] = None) -> bool:
    """
    Check if a detected name matches a staff roster entry.

    Normalises case and whitespace before comparing.
    Roster is loaded from config/staff_roster.yaml (not hardcoded).

    Staff roster: kasthuri v, zufishan khazra, shashikala, naziya begum, priya v

    EC-24 implementation.
    """
    roster = _load_roster(roster_path)
    normalised = name.lower().strip()
    return normalised in roster


# ── EC-25  BEHAVIOUR SCORE ────────────────────────────────────────────────────

def staff_behaviour_score(
    zones_visited: int,
    total_dwell_min: float,
    distinct_visits: int,
) -> int:
    """
    Score a track's behaviour on staff-like signals. Score ≥ 2 → treat as staff.

    Signal 1: zones_visited ≥ 6  (staff roam everywhere; customers usually 1-3)
    Signal 2: total_dwell_min ≥ 120  (staff work full shift; customers 10-45 min)
    Signal 3: distinct_visits ≥ 4  (staff appear in many clips; customers usually 1)

    Returns the score (0, 1, 2, or 3). Caller checks score >= 2.

    EC-25 implementation.
    """
    score = 0
    if zones_visited >= 6:
        score += 1
    if total_dwell_min >= 120:
        score += 1
    if distinct_visits >= 4:
        score += 1
    return score


def is_staff_by_behaviour(
    zones_visited: int,
    total_dwell_min: float,
    distinct_appearances: int,
) -> bool:
    """Convenience wrapper: True if behaviour score ≥ 2."""
    return staff_behaviour_score(zones_visited, total_dwell_min, distinct_appearances) >= 2


# ── EC-21  UNIFORM MATCH ─────────────────────────────────────────────────────

def uniform_match(
    torso_hist: list[float],
    uniform_ref_hist: list[float],
    thr: float = 0.7,
) -> bool:
    """
    Colour histogram intersection to detect staff uniforms.

    Histogram intersection: sum(min(a,b)) / sum(b)
    A score ≥ thr means the torso colour matches the reference uniform.

    Args:
        torso_hist:      colour histogram of the detected person's torso region
        uniform_ref_hist: reference uniform histogram (loaded from config)
        thr:             match threshold (0.7 = 70% overlap)

    Returns True if the torso colour matches the staff uniform.

    EC-21 implementation.
    """
    if not uniform_ref_hist or sum(uniform_ref_hist) == 0:
        return False
    intersection = sum(min(a, b) for a, b in zip(torso_hist, uniform_ref_hist))
    score = intersection / (sum(uniform_ref_hist) + 1e-9)
    return score >= thr


# ── EC-22  CASHIER DETECTION ─────────────────────────────────────────────────

def is_cashier(
    zone_id: str,
    dwell_ms: int,
    behind_counter: bool,
    billing_zone_id: str = "BILLING",
    long_ms: int = 20 * 60 * 1000,
) -> bool:
    """
    Detect the cashier — a staff member permanently stationed at the billing counter.

    Cashier criteria:
      - Located in the BILLING zone
      - Dwell time ≥ 50% of long_ms (default 20 minutes = they've been there a long time)
      - behind_counter = True (their position is behind the counter polygon)

    Why this matters: the cashier would otherwise be counted as being "in the billing
    queue" — inflating queue depth and distorting abandonment_rate.

    Returns True if this track is the cashier (exclude from queue depth).

    EC-22 implementation.
    """
    return (
        zone_id == billing_zone_id and
        dwell_ms >= long_ms * 0.5 and
        behind_counter
    )
