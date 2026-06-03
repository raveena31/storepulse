"""
pipeline/reid.py
────────────────
Re-identification (Re-ID) — recognising returning visitors across sessions.

EC-12  ReIDGallery       — stores exit embeddings; matches on re-entry
EC-13  reentry_decision  — THE arbiter: appearance beats timing
EC-14  update_running_embed — exponential moving average of embeddings
EC-15  feasible_match    — physics-based speed feasibility check
EC-16  LostTrackBuffer   — short-term buffer for occlusion stitching

──────────────────────────────────────────────────────────────────────
THE 3-SECOND TRAP (EC-13) — critical design decision
──────────────────────────────────────────────────────────────────────

Problem: A door guard or bouncer leaves and re-enters in 3 seconds.
Naive approach: "gap < 5s → must be same person" → WRONG. Two different
people can enter 3 seconds apart. The timing tells us nothing about identity.

Solution: Appearance (embedding cosine similarity) is the SOLE arbiter.
Timing is only a hint for pruning the gallery (don't match a 2-hour-old exit).
If sim ≥ sim_thr → REENTRY regardless of gap.
If sim < sim_thr → NEW_VISITOR regardless of gap.

Embeddings in this system:
  We use a lightweight 128-d embedding derived from the person crop.
  In the real deployment this comes from a re-ID model (e.g. OSNet, FastReID).
  In tests we pass synthetic numpy-like lists directly.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger("reid")


# ── EC-13  REENTRY DECISION ──────────────────────────────────────────────────

def reentry_decision(
    exit_embed: list[float],
    entry_embed: list[float],
    gap_s: float,
    sim_thr: float = 0.62,
    min_gap_s: float = 2.0,
) -> tuple[str, str]:
    """
    Decide whether an entry is a re-entry or a new visitor.

    THIS IS THE CRITICAL FUNCTION — appearance similarity is the sole arbiter.
    Timing is only a guard for physically impossible scenarios.

    Args:
        exit_embed:  embedding from the visitor's last exit crop
        entry_embed: embedding from the new entry crop
        gap_s:       seconds between exit and new entry
        sim_thr:     cosine similarity threshold for "same person" (0.62)
        min_gap_s:   minimum gap — if less than this AND low sim → definitely different

    Returns:
        (decision, reason) where decision ∈ {"REENTRY", "NEW_VISITOR"}

    Decision logic:
        1. If gap < min_gap_s AND sim < sim_thr → NEW_VISITOR
           (too fast for re-entry AND appearance doesn't match → different person)
        2. If sim ≥ sim_thr → REENTRY
           (appearance matches → same person regardless of timing)
        3. Otherwise → NEW_VISITOR

    The reason string is logged on every call for auditability.
    """
    sim = cosine_sim(exit_embed, entry_embed)

    if gap_s < min_gap_s and sim < sim_thr:
        reason = f"too-fast ({gap_s:.1f}s) and low-sim ({sim:.2f}) → different person"
        decision = "NEW_VISITOR"
    elif sim >= sim_thr:
        reason = f"appearance match sim={sim:.2f} ≥ {sim_thr} → same person"
        decision = "REENTRY"
    else:
        reason = f"low-sim={sim:.2f} < {sim_thr} → different person"
        decision = "NEW_VISITOR"

    logger.debug("reentry_decision: gap=%.1fs sim=%.3f → %s (%s)", gap_s, sim, decision, reason)
    return decision, reason


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) + 1e-9
    norm_b = math.sqrt(sum(y * y for y in b)) + 1e-9
    return dot / (norm_a * norm_b)


# ── EC-12  RE-ID GALLERY ─────────────────────────────────────────────────────

class ReIDGallery:
    """
    Short-term gallery of recent visitor embeddings for re-identification.

    When a visitor exits, their embedding is stored here for `window_s` seconds.
    When a new entry arrives, the gallery is searched for the best cosine match.
    If found above sim_thr → it's a re-entry (same visitor_id reused).
    If not found → it's a new visitor.

    EC-12 implementation.
    """

    def __init__(self, window_s: int = 900, sim_thr: float = 0.62):
        """
        Args:
            window_s: gallery retention window in seconds (default 15 minutes)
            sim_thr:  cosine similarity threshold for a match
        """
        self.window_s = window_s
        self.sim_thr = sim_thr
        # entries: list of {"visitor_id", "embed", "exit_ts"}
        self._entries: list[dict] = []

    def on_exit(self, visitor_id: str, embed: list[float], ts: datetime) -> None:
        """
        Register a visitor's exit embedding into the gallery.

        Call this when a visitor's EXIT event is fired.
        """
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        self._entries.append({
            "visitor_id": visitor_id,
            "embed": list(embed),
            "exit_ts": ts,
        })

    def match_on_entry(
        self, embed: list[float], ts: datetime
    ) -> Optional[str]:
        """
        Try to match a new entry against the gallery.

        Steps:
          1. Prune entries older than window_s
          2. Find the entry with the highest cosine similarity ≥ sim_thr
          3. If found: remove it from gallery (one-to-one matching) and return visitor_id
          4. If not found: return None (new visitor)

        Returns: visitor_id if re-entry matched, None if new visitor.
        """
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        self._prune(ts)

        best_sim = -1.0
        best_idx = None

        for i, entry in enumerate(self._entries):
            sim = cosine_sim(embed, entry["embed"])
            if sim >= self.sim_thr and sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx is not None:
            matched = self._entries.pop(best_idx)
            return matched["visitor_id"]

        return None

    def _prune(self, now: datetime) -> None:
        """Remove gallery entries older than window_s seconds."""
        cutoff = now - timedelta(seconds=self.window_s)
        self._entries = [e for e in self._entries if e["exit_ts"] >= cutoff]


# ── EC-14  EMBED DRIFT ───────────────────────────────────────────────────────

def update_running_embed(
    running: list[float],
    new_embed: list[float],
    alpha: float = 0.1,
) -> list[float]:
    """
    Exponential moving average of embeddings to handle gradual appearance drift.

    As lighting changes across the day, or a customer removes a layer of clothing,
    their embedding drifts. A static stored embedding would stop matching.
    EMA with small alpha keeps the gallery fresh without losing identity.

    running_new = (1 - alpha) * running + alpha * new_embed

    EC-14 implementation.
    """
    return [
        (1 - alpha) * r + alpha * n
        for r, n in zip(running, new_embed)
    ]


# ── EC-15  CLOTHING COLLISION ────────────────────────────────────────────────

def feasible_match(
    last_pos: tuple[float, float],
    last_ts: float,
    cand_pos: tuple[float, float],
    cand_ts: float,
    max_speed_px_s: float = 600.0,
) -> bool:
    """
    Physics-based speed feasibility check for re-ID matching.

    Reject a match if the implied travel speed exceeds max_speed_px_s.
    Prevents two people wearing similar colours from being merged
    when they're physically too far apart in too short a time.

    Args:
        last_pos:       (x, y) of the stored track's last known position
        last_ts:        timestamp of last_pos (seconds, e.g. time.time())
        cand_pos:       (x, y) of the candidate new detection
        cand_ts:        timestamp of candidate
        max_speed_px_s: maximum plausible human speed in pixels/second

    Returns True if the match is physically feasible.

    EC-15 implementation.
    """
    dt = abs(cand_ts - last_ts)
    if dt <= 0:
        return True  # same timestamp → no speed constraint applies

    dist = math.sqrt(
        (cand_pos[0] - last_pos[0]) ** 2 +
        (cand_pos[1] - last_pos[1]) ** 2
    )
    implied_speed = dist / dt
    return implied_speed <= max_speed_px_s


# ── EC-16  LOST TRACK BUFFER ─────────────────────────────────────────────────

class LostTrackBuffer:
    """
    Short-term buffer for tracks that disappeared behind a display unit or
    walked outside the camera's field of view at a frame boundary.

    When a track vanishes without an EXIT event (boundary exit), it is
    parked here. If it reappears within ttl_frames with a similar embedding,
    the original track_id is reclaimed (track stitching).

    This prevents the tracker from assigning a new ID every time a person
    ducks behind a shelf and re-emerges.

    EC-16 implementation.
    """

    def __init__(self, ttl_frames: int = 45):
        self.ttl_frames = ttl_frames
        # tid → {"embed", "pos", "age_frames"}
        self._buffer: dict[int, dict] = {}

    def park(
        self,
        tid: int,
        embed: list[float],
        pos: tuple[float, float],
    ) -> None:
        """Park a lost track in the buffer."""
        self._buffer[tid] = {"embed": list(embed), "pos": pos, "age_frames": 0}

    def tick(self) -> None:
        """Age all parked tracks by one frame. Prune expired ones."""
        expired = [
            tid for tid, state in self._buffer.items()
            if state["age_frames"] >= self.ttl_frames
        ]
        for tid in expired:
            del self._buffer[tid]
        for state in self._buffer.values():
            state["age_frames"] += 1

    def reclaim(
        self,
        embed: list[float],
        pos: tuple[float, float],
        sim_thr: float = 0.6,
    ) -> Optional[int]:
        """
        Try to reclaim a parked track for a new detection.

        Returns the original track_id if a match is found and removes it
        from the buffer. Returns None if no match.
        """
        best_sim = -1.0
        best_tid = None

        for tid, state in self._buffer.items():
            sim = cosine_sim(embed, state["embed"])
            if sim >= sim_thr and sim > best_sim:
                best_sim = sim
                best_tid = tid

        if best_tid is not None:
            del self._buffer[best_tid]
            return best_tid

        return None
