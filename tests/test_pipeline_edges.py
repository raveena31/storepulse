# PROMPT: "Generate pytest cases for the 20 detection/re-ID/staff edge cases (EC-1 through
#          EC-25). Each test must map to a specific EC and assert the contract from
#          edge_cases.md, not the implementation."
# CHANGES MADE: AI's EC-13 (3-second trap) test asserted that gap<3s → NEW_VISITOR regardless
#               of similarity. That's the OPPOSITE of the contract — appearance is the
#               arbiter, timing is only a hint. I rewrote the test fixtures: (a) same person
#               with sim>0.7 and gap=30s → REENTRY; (b) different people with sim<0.3 and
#               gap=3s → NEW_VISITOR. Same gap, opposite outcomes — proves appearance wins.
#               AI's EC-7 mannequin test used random jitter that occasionally exceeded the
#               px_thresh; I replaced with deterministic alternating jitter ≤2px so the test
#               is reproducible across CI runs.
# Each test maps to a specific EC number in the playbook.

import math
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.edge_cases import (
    EntryExitCounter,
    split_merged_box,
    net_direction,
    crossing_is_real,
    is_static_prop,
    drop_reflection,
    looks_like_shadow,
    is_boundary_exit,
    sticky_staff,
)
from pipeline.reid import (
    reentry_decision,
    cosine_sim,
    ReIDGallery,
    update_running_embed,
    feasible_match,
    LostTrackBuffer,
)
from pipeline.crosscam import owns_detection, crosscam_inherit, build_zone_camera_map
from pipeline.staff import (
    staff_behaviour_score,
    is_staff_by_behaviour,
    uniform_match,
    is_cashier,
    roster_is_staff,
)


NOW = datetime(2026, 4, 10, 14, 40, 0, tzinfo=timezone.utc)

# ── GROUP A: Detection & Counting ─────────────────────────────────────────────

class TestEC1GroupEntry:
    """EC-1: 3 simultaneous tracks crossing downward → exactly 3 ENTRY events."""

    def test_three_tracks_three_entries(self):
        counter = EntryExitCounter(line_y=500)
        # Simulate each track crossing from y=480 (above line) to y=520 (below)
        results = []
        for tid in [1, 2, 3]:
            counter._prev_y[tid] = 480.0  # above line
        for tid in [1, 2, 3]:
            r = counter.update(tid, 520.0)  # cross downward
            results.append(r)
        assert results == ["ENTRY", "ENTRY", "ENTRY"]

    def test_same_track_not_counted_twice(self):
        counter = EntryExitCounter(line_y=500)
        counter._prev_y[1] = 480.0
        r1 = counter.update(1, 520.0)   # ENTRY
        counter._prev_y[1] = 518.0
        r2 = counter.update(1, 522.0)   # still below line — no re-count
        assert r1 == "ENTRY"
        assert r2 is None

    def test_hysteresis_prevents_false_trigger(self):
        counter = EntryExitCounter(line_y=500, hysteresis=8)
        counter._prev_y[1] = 494.0  # within hysteresis band above line
        r = counter.update(1, 502.0)   # crosses but not by full hysteresis
        # 494 is NOT < 500-8=492, so condition prev < line_y - hysteresis is False
        assert r is None


class TestEC2TailgatingMergedBox:
    """EC-2: Wide box → splits into 2."""

    def test_wide_box_splits(self):
        normal_w = 80
        wide_box = [100, 200, 100 + 2.1 * normal_w, 400]  # 2.1x normal width
        result = split_merged_box(wide_box, normal_w)
        assert len(result) == 2

    def test_normal_box_not_split(self):
        normal_box = [100, 200, 180, 400]  # width=80 = 1.0x
        result = split_merged_box(normal_box, 80)
        assert len(result) == 1

    def test_high_iou_triggers_split(self):
        box = [100, 200, 180, 400]
        overlapping = [105, 200, 185, 400]  # nearly identical → high IoU
        result = split_merged_box(box, 200, second_box=overlapping, iou_thresh=0.5)
        assert len(result) == 2


class TestEC3DoorwayLoiter:
    """EC-3: Oscillating y_history never emits a crossing."""

    def test_oscillation_not_counted(self):
        # Person hovers around line_y=500, never moves more than 10px net
        history = [498, 502, 497, 503, 499, 501, 498]
        result = net_direction(history, line_y=500, min_net=25.0)
        assert result is False

    def test_genuine_crossing_counted(self):
        # Person cleanly crosses from 470 to 530 — net 60px
        history = [470, 480, 490, 500, 510, 520, 530]
        result = net_direction(history, line_y=500, min_net=25.0)
        assert result is True


class TestEC5DoorSwingPhantoms:
    """EC-5: Young tracks (< 4 frames) are suppressed."""

    def test_young_track_suppressed(self):
        assert crossing_is_real(track_age_frames=2, min_age=4) is False

    def test_old_track_allowed(self):
        assert crossing_is_real(track_age_frames=4, min_age=4) is True
        assert crossing_is_real(track_age_frames=10, min_age=4) is True


class TestEC7MannequinStandee:
    """EC-7: 50-frame track with ≤3px jitter → is_static_prop=True."""

    def test_static_prop_detected(self):
        # 50 positions alternating within ±2px — total diagonal movement < 6px
        positions = [(500 + (i % 2) * 2, 400 + (i % 3) * 1) for i in range(50)]
        # max x spread = 2, max y spread = 2 → diagonal ≈ 2.8 < 6
        assert is_static_prop(positions, px_thresh=6.0) is True

    def test_moving_person_not_prop(self):
        # Person walks across 200px
        positions = [(i * 4, 400) for i in range(50)]
        assert is_static_prop(positions, px_thresh=6.0) is False

    def test_insufficient_records_returns_false(self):
        # Only 10 positions — not enough to decide
        positions = [(500, 400)] * 10
        assert is_static_prop(positions, min_records=30) is False


class TestEC8GlassReflections:
    """EC-8: Bbox whose feet fall inside glass mask → dropped."""

    MASK = [[[400, 380], [800, 380], [800, 560], [400, 560]]]

    def test_feet_inside_mask_dropped(self):
        bbox = [580, 300, 620, 540]  # feet at (600, 540) → inside mask
        assert drop_reflection(bbox, self.MASK) is True

    def test_feet_outside_mask_kept(self):
        bbox = [100, 300, 150, 400]  # feet at (125, 400) → outside mask
        assert drop_reflection(bbox, self.MASK) is False


class TestEC9Shadows:
    """EC-9: Wide flat box with low confidence → looks_like_shadow=True."""

    def test_shadow_detected(self):
        bbox = [100, 350, 300, 400]  # w=200, h=50, ar=0.25, very flat
        assert looks_like_shadow(bbox, conf=0.15) is True

    def test_normal_person_not_shadow(self):
        bbox = [100, 100, 180, 400]  # w=80, h=300, ar=3.75, tall person
        assert looks_like_shadow(bbox, conf=0.80) is False


# ── GROUP B: Re-ID ────────────────────────────────────────────────────────────

class TestEC12ReIDGallery:
    """EC-12: Exit → re-entry gallery match."""

    def test_same_embed_matches(self):
        gallery = ReIDGallery(window_s=900, sim_thr=0.62)
        embed = [1.0, 0.0, 0.0, 0.0]
        gallery.on_exit("VIS_001", embed, NOW)
        matched = gallery.match_on_entry(embed, NOW + timedelta(minutes=2))
        assert matched == "VIS_001"

    def test_dissimilar_embed_no_match(self):
        gallery = ReIDGallery(window_s=900, sim_thr=0.62)
        gallery.on_exit("VIS_001", [1.0, 0.0, 0.0, 0.0], NOW)
        different = [0.0, 1.0, 0.0, 0.0]  # orthogonal → sim=0
        matched = gallery.match_on_entry(different, NOW + timedelta(minutes=2))
        assert matched is None

    def test_expired_entry_not_matched(self):
        gallery = ReIDGallery(window_s=60, sim_thr=0.62)  # 60s window
        embed = [1.0, 0.0, 0.0, 0.0]
        gallery.on_exit("VIS_001", embed, NOW)
        # Query 2 minutes later — entry expired
        matched = gallery.match_on_entry(embed, NOW + timedelta(minutes=2))
        assert matched is None


class TestEC13ThreeSecondTrap:
    """
    EC-13: THE CRITICAL TEST — appearance is the arbiter, not timing.

    Two fixtures:
      (a) same_person:          high sim (>0.7), gap=30s  → REENTRY
      (b) different_person_3s:  low sim (<0.3),  gap=3s   → NEW_VISITOR
    """

    def test_same_person_high_sim_reentry(self):
        """EC-13(a): same person, high similarity → REENTRY regardless of timing."""
        # High-similarity embeddings — same person at slightly different angle
        exit_embed  = [0.9, 0.4, 0.1, 0.0]
        entry_embed = [0.88, 0.42, 0.09, 0.01]
        decision, reason = reentry_decision(exit_embed, entry_embed, gap_s=30.0)
        assert decision == "REENTRY", f"Expected REENTRY, got {decision}: {reason}"
        assert "appearance match" in reason.lower() or "sim=" in reason

    def test_different_person_three_second_gap_new_visitor(self):
        """EC-13(b): different person, low similarity, 3s gap → NEW_VISITOR."""
        exit_embed  = [1.0, 0.0, 0.0, 0.0]
        entry_embed = [0.0, 1.0, 0.0, 0.0]  # orthogonal → sim ≈ 0
        decision, reason = reentry_decision(exit_embed, entry_embed, gap_s=3.0)
        assert decision == "NEW_VISITOR", f"Expected NEW_VISITOR, got {decision}: {reason}"

    def test_fast_reentry_low_sim_new_visitor(self):
        """EC-13: gap < min_gap AND low sim → NEW_VISITOR (too fast + different look)."""
        exit_embed  = [1.0, 0.0, 0.0, 0.0]
        entry_embed = [0.0, 1.0, 0.0, 0.0]
        decision, _ = reentry_decision(exit_embed, entry_embed, gap_s=1.0, min_gap_s=2.0)
        assert decision == "NEW_VISITOR"

    def test_high_sim_overrides_fast_gap(self):
        """EC-13: high sim wins even if gap is very fast — appearance is the arbiter."""
        exit_embed  = [1.0, 0.0, 0.0, 0.0]
        entry_embed = [1.0, 0.0, 0.0, 0.0]  # identical → sim=1.0
        decision, _ = reentry_decision(exit_embed, entry_embed, gap_s=1.0, min_gap_s=2.0)
        assert decision == "REENTRY"


class TestEC14EmbedDrift:
    """EC-14: Running embed converges toward new after 10 updates."""

    def test_ema_converges(self):
        running = [1.0, 0.0, 0.0, 0.0]
        target  = [0.0, 1.0, 0.0, 0.0]
        for _ in range(10):
            running = update_running_embed(running, target, alpha=0.1)
        # After 10 updates with alpha=0.1: running[0] = 0.9^10 ≈ 0.349
        assert running[0] < 0.5   # moved toward target
        assert running[1] > 0.5   # target dimension increased


class TestEC15ClothingCollision:
    """EC-15: 1000px in 0.5s → infeasible match."""

    def test_infeasible_speed(self):
        assert feasible_match(
            last_pos=(0, 0), last_ts=0.0,
            cand_pos=(1000, 0), cand_ts=0.5,
            max_speed_px_s=600.0,
        ) is False   # 2000 px/s > 600

    def test_feasible_speed(self):
        assert feasible_match(
            last_pos=(0, 0), last_ts=0.0,
            cand_pos=(100, 0), cand_ts=1.0,
            max_speed_px_s=600.0,
        ) is True    # 100 px/s << 600


class TestEC16LostTrackBuffer:
    """EC-16: Park → reclaim within TTL → works. Past TTL → None."""

    def test_reclaim_within_ttl(self):
        buf = LostTrackBuffer(ttl_frames=45)
        embed = [1.0, 0.0, 0.0, 0.0]
        buf.park(tid=42, embed=embed, pos=(100, 200))
        for _ in range(40):
            buf.tick()
        result = buf.reclaim(embed=[0.99, 0.01, 0.0, 0.0], pos=(105, 205))
        assert result == 42

    def test_expired_track_not_reclaimed(self):
        buf = LostTrackBuffer(ttl_frames=45)
        embed = [1.0, 0.0, 0.0, 0.0]
        buf.park(tid=42, embed=embed, pos=(100, 200))
        for _ in range(50):  # past TTL
            buf.tick()
        result = buf.reclaim(embed=embed, pos=(100, 200))
        assert result is None


class TestEC1718CrossCamera:
    """EC-17/18: Entry cam emits exit for VIS_01, floor cam inherits it."""

    def test_crosscam_inherit(self):
        gallery = ReIDGallery(window_s=900, sim_thr=0.62)
        embed = [1.0, 0.0, 0.0, 0.0]
        # Entry cam registers exit
        gallery.on_exit("VIS_001", embed, NOW)
        # Floor cam sees similar embed 10s later
        result = crosscam_inherit(gallery, embed, NOW + timedelta(seconds=10))
        assert result == "VIS_001"

    def test_owns_detection_correct_camera(self):
        zone_map = {"LAKME": "CAM_02", "BILLING": "CAM_05"}
        assert owns_detection("CAM_02", "LAKME", zone_map) is True
        assert owns_detection("CAM_01", "LAKME", zone_map) is False

    def test_no_ownership_allows_all(self):
        zone_map = {}
        assert owns_detection("CAM_01", "UNKNOWN_ZONE", zone_map) is True


class TestEC19BoundaryExit:
    """EC-19: Bbox touching frame edge → is_boundary_exit=True."""

    def test_left_edge_exit(self):
        bbox = [5, 200, 100, 400]  # x1=5 ≤ pad=12
        assert is_boundary_exit(bbox, frame_w=1920, frame_h=1080, pad=12) is True

    def test_right_edge_exit(self):
        bbox = [1800, 200, 1915, 400]  # x2=1915 ≥ 1920-12=1908
        assert is_boundary_exit(bbox, frame_w=1920, frame_h=1080, pad=12) is True

    def test_interior_not_boundary(self):
        bbox = [100, 200, 400, 600]  # well inside frame
        assert is_boundary_exit(bbox, frame_w=1920, frame_h=1080, pad=12) is False


class TestEC20StickyStaff:
    """EC-20: Locked staff label never reverts to non-staff."""

    def test_high_conf_locks_label(self):
        state = {}
        sticky_staff(state, tid=1, frame_is_staff=True, conf=0.85)
        assert state[1]["locked"] is True
        assert state[1]["is_staff"] is True

    def test_locked_label_does_not_revert(self):
        state = {}
        sticky_staff(state, tid=1, frame_is_staff=True, conf=0.85)
        # Subsequent frame says not staff — should be ignored
        sticky_staff(state, tid=1, frame_is_staff=False, conf=0.9)
        assert state[1]["is_staff"] is True  # still staff

    def test_low_conf_does_not_lock(self):
        state = {}
        sticky_staff(state, tid=1, frame_is_staff=True, conf=0.6, lock_conf=0.8)
        assert state[1]["locked"] is False  # tentative, not locked


# ── GROUP C: Staff Detection ───────────────────────────────────────────────────

class TestEC21UniformMatch:
    """EC-21: Histogram intersection ≥ 0.7 → uniform match."""

    def test_matching_histogram(self):
        ref  = [0.5, 0.3, 0.2]
        same = [0.5, 0.3, 0.2]  # identical → intersection=1.0
        assert uniform_match(same, ref, thr=0.7) is True

    def test_non_matching_histogram(self):
        ref   = [0.8, 0.1, 0.1]
        other = [0.1, 0.1, 0.8]  # very different colours
        assert uniform_match(other, ref, thr=0.7) is False


class TestEC22CashierDetection:
    """EC-22: Person in BILLING zone for ≥50% of 20 min behind counter = cashier."""

    def test_cashier_detected(self):
        assert is_cashier(
            zone_id="BILLING",
            dwell_ms=12 * 60 * 1000,  # 12 minutes ≥ 50% of 20 min
            behind_counter=True,
        ) is True

    def test_customer_not_cashier(self):
        assert is_cashier(
            zone_id="BILLING",
            dwell_ms=2 * 60 * 1000,  # 2 minutes — too short
            behind_counter=False,
        ) is False


class TestEC25BehaviourScore:
    """EC-25: zones=7, dwell=130min, visits=5 → score=3 → staff."""

    def test_full_staff_score(self):
        score = staff_behaviour_score(zones_visited=7, total_dwell_min=130, distinct_visits=5)
        assert score == 3
        assert is_staff_by_behaviour(7, 130, 5) is True

    def test_borderline_score(self):
        # Only zones and dwell → score=2 → still staff
        score = staff_behaviour_score(zones_visited=6, total_dwell_min=120, distinct_visits=1)
        assert score == 2
        assert is_staff_by_behaviour(6, 120, 1) is True

    def test_customer_score(self):
        # Short visit, few zones → score=0
        score = staff_behaviour_score(zones_visited=2, total_dwell_min=20, distinct_visits=1)
        assert score == 0
        assert is_staff_by_behaviour(2, 20, 1) is False
