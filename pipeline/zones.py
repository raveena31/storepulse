"""
pipeline/zones.py
─────────────────
Zone assignment — maps a person's feet position to a named store zone.

How zones work:
  Each store zone (e.g. "LAKME", "BILLING") is defined as a polygon in the
  coordinate space of the camera that owns it. The polygon is specified as a
  list of [x, y] pixel coordinates in config/store_ST1008.yaml.

  Example config:
      zones:
        LAKME:
          camera: CAM_02
          polygon: [[400, 200], [800, 200], [800, 900], [400, 900]]
          display_name: "Lakme / Swiss Beauty"

  "Feet point" = the bottom-center of the person's bounding box.
  Using feet (not center of body) gives a more accurate floor position for
  overhead/angled cameras.

Point-in-polygon algorithm:
  We use the classic "ray casting" method:
    1. Cast a ray from the test point horizontally to the right
    2. Count how many polygon edges the ray crosses
    3. Odd count → inside | Even count → outside

  Reference: https://en.wikipedia.org/wiki/Point_in_polygon

Per-camera zone lookup:
  assign_zone() only checks zones owned by the given camera_id.
  This avoids false matches when two cameras have overlapping frame coverage:
  a person detected in CAM_01's frame should only match CAM_01's zones.

Camera roles (from config):
  product_zone  → CAM_01, CAM_02 — standard zone assignment
  entry_exit    → CAM_03 — used for line-crossing entry/exit detection
  staff_only    → CAM_04 — all detections = is_staff=True
  billing       → CAM_05 — BILLING_QUEUE_JOIN events
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

# Simple path-keyed cache so we don't re-read the YAML on every frame
_config_cache: dict = {}


def _load_config(config_path: Optional[str] = None) -> dict:
    """
    Load store config YAML. Cached after first load.

    Args:
        config_path: Path to YAML file. Defaults to CONFIG_DIR env var + store_ST1008.yaml.
    """
    path = config_path or os.path.join(
        os.getenv("CONFIG_DIR", "config"), "store_ST1008.yaml"
    )
    if path not in _config_cache:
        with open(path) as f:
            _config_cache[path] = yaml.safe_load(f)
    return _config_cache[path]


def _point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """
    Ray-casting point-in-polygon test.

    Args:
        x, y:    Test point coordinates
        polygon: List of [px, py] vertices. At least 3 required.

    Returns:
        True if (x, y) is inside the polygon, False otherwise.
        Points exactly on the boundary may return either result — acceptable for our use.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Check if the ray from (x,y) going right crosses this edge
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def assign_zone(
    feet_point: tuple[float, float],
    camera_id: str,
    config: Optional[dict] = None,
    config_path: Optional[str] = None,
) -> Optional[str]:
    """
    Find which zone a person is standing in, based on their feet position.

    Only zones owned by camera_id are checked (prevents cross-camera false matches).

    Args:
        feet_point:  (x, y) pixel coordinates of the person's feet
                     (typically bottom-center of bounding box: ((x1+x2)/2, y2))
        camera_id:   Which camera this detection comes from (e.g. "CAM_01")
        config:      Pre-loaded config dict (optional — avoids repeated file reads)
        config_path: Config file path override (optional)

    Returns:
        zone_id string (e.g. "LAKME") if inside a zone, or None if in an unzoned area.
    """
    if config is None:
        config = _load_config(config_path)

    x, y = feet_point
    # Only check zones that this camera is responsible for
    ownership = config.get("camera_zone_ownership", {})
    owned_zones = set(ownership.get(camera_id, []))

    for zone_id, zone_data in config.get("zones", {}).items():
        if zone_id not in owned_zones:
            continue
        polygon = zone_data.get("polygon", [])
        if polygon and _point_in_polygon(x, y, polygon):
            return zone_id

    return None  # person is in an unzoned area (e.g., aisle, middle of floor)


def camera_owns_zone(camera_id: str, zone_id: str, config: Optional[dict] = None) -> bool:
    """Check if a camera is responsible for a given zone."""
    if config is None:
        config = _load_config()
    ownership = config.get("camera_zone_ownership", {})
    return zone_id in ownership.get(camera_id, [])


def get_camera_role(camera_id: str, config: Optional[dict] = None) -> str:
    """
    Get the processing role for a camera.

    Returns one of:
        "product_zone"  → standard zone detection (CAM_01, CAM_02)
        "entry_exit"    → line-crossing detection (CAM_03)
        "staff_only"    → all detections flagged as staff (CAM_04)
        "billing"       → BILLING_QUEUE_JOIN events (CAM_05)
    """
    if config is None:
        config = _load_config()
    return config.get("camera_roles", {}).get(camera_id, "product_zone")


def is_in_glass_mask(x: float, y: float, config: Optional[dict] = None) -> bool:
    """
    Check if a point falls inside a glass/mirror exclusion polygon.

    Glass masks are defined in config to exclude reflections from the store's
    glass doors or mirrors. Detections inside these masks are dropped because
    they may be reflections of people outside the store.

    Used only for CAM_03 (entry camera which faces glass doors).
    """
    if config is None:
        config = _load_config()
    for poly in config.get("glass_mask_polygons", []):
        if _point_in_polygon(x, y, poly):
            return True
    return False
