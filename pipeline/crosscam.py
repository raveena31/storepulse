"""
pipeline/crosscam.py
────────────────────
Cross-camera identity continuity (EC-17, EC-18).

Problem: The same person walks from CAM_01's field of view into CAM_02's.
Without cross-camera Re-ID, they get a new visitor_id on CAM_02 —
double-counting them in the denominator.

Solution:
  EC-17 crosscam_inherit — when a new entry is detected on any camera,
        check the ReIDGallery (which stores exits from ALL cameras).
        If a match is found, inherit the existing visitor_id and emit REENTRY.

  EC-18 owns_detection — each zone is owned by exactly one camera.
        A person detected simultaneously in an overlap zone by two cameras
        only generates events from the owning camera.
        This prevents double-counting dwell time.
"""
from __future__ import annotations

from typing import Optional

from .reid import ReIDGallery


def crosscam_inherit(
    gallery: ReIDGallery,
    embed: list[float],
    ts,
) -> Optional[str]:
    """
    Attempt to inherit a visitor_id from another camera's recent exit.

    This is called in the pipeline when a new entry is detected on any camera,
    BEFORE assigning a new visitor_id.

    Args:
        gallery: The shared ReIDGallery (same instance across all cameras)
        embed:   embedding of the newly detected person
        ts:      UTC timestamp of the detection

    Returns:
        Existing visitor_id if match found (emit REENTRY event).
        None if no match (assign new visitor_id, emit ENTRY event).

    EC-17 implementation.
    """
    return gallery.match_on_entry(embed, ts)


def owns_detection(
    camera_id: str,
    zone_id: str,
    zone_camera_map: dict[str, str],
) -> bool:
    """
    Check whether this camera is the authoritative owner of a zone.

    In overlap areas where two cameras can see the same physical space,
    only the owning camera should emit zone events. The other camera's
    detections in that zone are silently dropped.

    Args:
        camera_id:       the camera processing this detection ("CAM_01")
        zone_id:         the zone being checked ("LAKME")
        zone_camera_map: {zone_id: owning_camera_id} — loaded from config

    Returns True if this camera should emit events for this zone.

    EC-18 implementation.
    """
    owner = zone_camera_map.get(zone_id)
    if owner is None:
        return True  # no ownership defined → allow all cameras (safe default)
    return owner == camera_id


def build_zone_camera_map(config: dict) -> dict[str, str]:
    """
    Build the zone→camera ownership map from store config.

    Inverts camera_zone_ownership: {camera_id: [zone_ids]}
    into: {zone_id: camera_id}

    Used by owns_detection().
    """
    ownership = config.get("camera_zone_ownership", {})
    zone_map: dict[str, str] = {}
    for camera_id, zones in ownership.items():
        for zone_id in zones:
            zone_map[zone_id] = camera_id
    return zone_map
