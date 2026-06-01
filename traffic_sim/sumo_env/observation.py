# traffic_sim/sumo_env/observation.py
"""
ObservationBuilder
==================
Converts the running SUMO simulation into the RGB image observation
that is sent to the AI agent every step.

The image mirrors ``image_construct1()`` from the teacher's reference:

    - Shape    : (n x n x 3)  uint8  ndarray
    - One pixel per vehicle currently in the network.
    - Intensity -> current speed   (dim = stopped, bright = fast).
    - Colour    -> outgoing edge / intended direction at the intersection.
    - White     -> vehicle already past the intersection (arrived / clearing).

Color scheme (consistent for ALL network types)
-----------------------------------------------
The dest_colors table maps EXIT edge IDs to RGB colors.  For single-
intersection networks these are "C_to_N", "C_to_E" etc.  For multi-
intersection networks the exits are "J0_to_N0", "J_0_0_to_N_0_0" etc.

Rather than maintaining a static table that breaks on new topologies,
_normalize_dest_edge() does a two-step lookup:
  1. Direct match / prefix match against self.dest_colors (fast path, works
     for any user-supplied table and for the default "C_to_*" single-int table).
  2. Fallback: parse the DESTINATION NODE of the edge ("C_to_E" -> "E",
     "J0_to_N0" -> "N0", "J_0_0_to_N_0_0" -> "N_0_0") and map its FIRST
     LETTER to a canonical dest_colors key ("N"->"C_to_N" etc.).
     This covers multi-intersection exits without any extra configuration.

Outgoing-edge detection (vehicles drawn white)
----------------------------------------------
intersection_outgoing is the user-supplied set of outgoing edge IDs.
For multi-intersection networks we additionally test whether the CURRENT
EDGE name ends with a border-node suffix (N*, S*, E*, W*) after the last
"_to_".  This is done inside _is_outgoing() so no static set is needed.

TraCI calls used
----------------
    traci.vehicle.getIDList()
    traci.vehicle.getPosition(vid)
    traci.vehicle.getSpeed(vid)
    traci.vehicle.getRoute(vid)
    traci.vehicle.getRoadID(vid)
    traci.vehicle.getLaneID(vid)
    traci.simulation.getNetBoundary()
    traci.junction.getPosition(jid)   <- for crop bbox
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import numpy as np

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

MAX_SPEED_MS   = 14.0   # m/s -> pixel intensity normalisation
_CROP_RADIUS_M = 110.0  # metres around junction centre for cropped image

# Canonical single-intersection dest_colors keys, one per direction letter
_DIRECTION_TO_COLOR_KEY: Dict[str, str] = {
    "N": "C_to_N",
    "S": "C_to_S",
    "E": "C_to_E",
    "W": "C_to_W",
}


class ObservationBuilder:
    """
    Builds the RGB image observation from the running SUMO simulation.

    Parameters
    ----------
    dest_colors : dict {edge_id: (R, G, B)} | None
        Color table for destination edges.  Default covers all single-
        intersection types (four_way, t_junction, complex).
    intersection_outgoing : set[str] | None
        Edge IDs drawn white (vehicle has cleared the intersection).
        Multi-intersection outgoing edges are detected automatically.
    bbox : ((xmin,ymin),(xmax,ymax)) | None
        Fixed world bounding box for build_image().
    """

    _DEFAULT_DEST_COLORS: Dict[str, Tuple[float, float, float]] = {
        "C_to_N":  (1.0, 0.0, 0.0),   # red
        "C_to_S":  (0.0, 1.0, 0.0),   # green
        "C_to_E":  (0.0, 0.0, 1.0),   # blue
        "C_to_W":  (1.0, 1.0, 0.0),   # yellow
        "C_to_NE": (0.0, 1.0, 1.0),   # cyan
        "C_to_NW": (1.0, 0.0, 1.0),   # magenta
    }

    _DEFAULT_OUTGOING: set = {
        "C_to_N", "C_to_S", "C_to_E", "C_to_W", "C_to_NE", "C_to_NW"
    }

    def __init__(
        self,
        dest_colors:           Optional[Dict[str, Tuple[float, float, float]]] = None,
        intersection_outgoing: Optional[set]  = None,
        bbox:                  Optional[Tuple] = None,
    ):
        self.dest_colors           = dest_colors           or self._DEFAULT_DEST_COLORS
        self.intersection_outgoing = intersection_outgoing or self._DEFAULT_OUTGOING
        self._bbox                 = bbox

        # Cache: intersection_id -> (xmin, ymin, xmax, ymax)
        self._intersection_bboxes: Dict[int, Tuple[float, float, float, float]] = {}

    # ------------------------------------------------------------------ #
    #  Public: full-network image                                          #
    # ------------------------------------------------------------------ #

    def build_image(self, n: int = 50) -> np.ndarray:
        """Build an n×n RGB image of the full network."""
        (xmin, ymin), (xmax, ymax) = self._get_bbox()
        return self._render(xmin, ymin, xmax, ymax, n)

    # ------------------------------------------------------------------ #
    #  Public: per-intersection cropped image                              #
    # ------------------------------------------------------------------ #

    def build_image_for_intersection(
        self,
        intersection_id: int,
        junction_id: str,
        n: int = 50,
    ) -> np.ndarray:
        """
        Build an n×n image cropped to one intersection's region.

        Bbox is centred on the junction's world position with radius
        _CROP_RADIUS_M metres.  Computed once per episode and cached.

        Parameters
        ----------
        intersection_id : int   cache key
        junction_id     : str   SUMO node ID ("C", "J0", "J_0_1", …)
        n               : int   output side length in pixels
        """
        if intersection_id not in self._intersection_bboxes:
            self._intersection_bboxes[intersection_id] = (
                self._bbox_from_junction(junction_id)
            )
        xmin, ymin, xmax, ymax = self._intersection_bboxes[intersection_id]
        return self._render(xmin, ymin, xmax, ymax, n)

    def reset_bbox_cache(self) -> None:
        """Clear per-episode bbox cache.  Call after env.reset()."""
        self._intersection_bboxes.clear()

    # ------------------------------------------------------------------ #
    #  Core rendering engine                                               #
    # ------------------------------------------------------------------ #

    def _render(
        self,
        xmin: float, ymin: float,
        xmax: float, ymax: float,
        n: int,
    ) -> np.ndarray:
        """
        Render all vehicles inside the bounding box into an n×n RGB image.

        Per vehicle:
          - outgoing edge  -> white  (already cleared intersection)
          - known dest     -> dest color × speed intensity
          - unknown dest   -> grey
        """
        image = np.zeros((n, n, 3), dtype=np.uint8)
        dx    = max(1e-6, xmax - xmin)
        dy    = max(1e-6, ymax - ymin)

        for vid in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vid)
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue

            px, py = self._world_to_pixel(x, y, xmin, xmax, ymin, ymax, dx, dy, n)
            if not (0 <= px < n and 0 <= py < n):
                continue

            speed     = traci.vehicle.getSpeed(vid)
            intensity = self._speed_to_intensity(speed)

            # ---- Resolve current edge ----
            # SUMO internal edges start with ':' — in that case read the
            # actual edge from the lane ID by stripping the lane index.
            road_id = traci.vehicle.getRoadID(vid)
            if road_id.startswith(":"):
                lane_id   = traci.vehicle.getLaneID(vid)
                # Lane ID format: "<edge_id>_<lane_index>"
                # Strip the last "_<digit>" to get the edge ID.
                # e.g. "C_to_N_0" -> "C_to_N",  "J0_to_N0_1" -> "J0_to_N0"
                parts     = lane_id.rsplit("_", 1)
                curr_edge = parts[0] if len(parts) == 2 and parts[1].isdigit() else lane_id
            else:
                curr_edge = road_id

            # ---- White: already past the intersection ----
            if self._is_outgoing(curr_edge):
                image[py, px] = (255, 255, 255)
                continue

            # ---- Color: find destination from route ----
            route       = traci.vehicle.getRoute(vid) or []
            route_clean = [e for e in route if not e.startswith(":")]

            if not route_clean:
                image[py, px] = (intensity, intensity, intensity)
                continue

            # Search route edges for a recognisable color key
            edge_for_color = None
            for e in route_clean:
                edge_for_color = self._normalize_dest_edge(e)
                if edge_for_color is not None:
                    break

            if edge_for_color is None:
                # Last resort: try the last edge in the route
                edge_for_color = self._normalize_dest_edge(route_clean[-1])

            if edge_for_color is not None and edge_for_color in self.dest_colors:
                cr, cg, cb = self.dest_colors[edge_for_color]
                image[py, px] = (
                    int(cr * intensity),
                    int(cg * intensity),
                    int(cb * intensity),
                )
            else:
                image[py, px] = (intensity, intensity, intensity)

        return image

    # ------------------------------------------------------------------ #
    #  Outgoing-edge detection                                             #
    # ------------------------------------------------------------------ #

    def _is_outgoing(self, edge_id: str) -> bool:
        """
        Return True if edge_id is an outgoing edge (vehicle has cleared
        the intersection and is heading toward a border node).

        Two tests — either one is sufficient:
          1. edge_id is in self.intersection_outgoing  (user-supplied or
             the default "C_to_*" set for single intersections).
          2. The destination node in the edge name starts with a border
             letter (N, S, E, W).  This covers multi-intersection exits:
               "J0_to_N0"       -> dest "N0" -> starts with N -> outgoing
               "J_0_0_to_S_0_0" -> dest "S_0_0" -> starts with S -> outgoing
             It also handles "C_to_N", "C_to_NE", "C_to_NW" automatically.
        """
        if edge_id in self.intersection_outgoing:
            return True

        # Dynamic detection: edge goes toward a border node
        if "_to_" in edge_id:
            dest_node = edge_id.split("_to_")[-1]
            if dest_node and dest_node[0] in ("N", "S", "E", "W"):
                return True

        return False

    # ------------------------------------------------------------------ #
    #  Color key normalisation                                             #
    # ------------------------------------------------------------------ #

    def _normalize_dest_edge(self, edge_id: str) -> Optional[str]:
        """
        Map any route edge ID to a key that exists in self.dest_colors.

        Step 1 — exact match or prefix match (works for single-intersection
                  edge names like "C_to_N", "C_to_NE", and lane variants
                  like "C_to_N_0"):
            "C_to_N"    -> "C_to_N"
            "C_to_NE_0" -> "C_to_NE"

        Step 2 — destination-node fallback (works for multi-intersection):
            "J0_to_N0"       -> dest_node "N0" -> first letter "N" -> "C_to_N"
            "J_0_0_to_S_1_2" -> dest_node "S_1_2" -> first letter "S" -> "C_to_S"
            "J0_to_J1"       -> dest_node "J1" -> first letter "J" -> None
                                 (internal road, not a border edge — skip)

        Returns the matched dest_colors key, or None if no match.
        """
        # Step 1: direct / prefix match against the configured color table
        if edge_id in self.dest_colors:
            return edge_id
        for dest_key in self.dest_colors:
            if edge_id.startswith(f"{dest_key}_"):
                return dest_key

        # Step 2: destination-node heuristic for multi-intersection exits
        if "_to_" in edge_id:
            dest_node = edge_id.split("_to_")[-1]
            first_letter = dest_node[0] if dest_node else ""
            color_key = _DIRECTION_TO_COLOR_KEY.get(first_letter)
            # Only return it if that key actually exists in our color table
            if color_key is not None and color_key in self.dest_colors:
                return color_key

        return None

    # ------------------------------------------------------------------ #
    #  Bbox helpers                                                        #
    # ------------------------------------------------------------------ #

    def _bbox_from_junction(
        self, junction_id: str
    ) -> Tuple[float, float, float, float]:
        """Square bbox centred on junction_id ± _CROP_RADIUS_M metres."""
        try:
            jx, jy = traci.junction.getPosition(junction_id)
            r = _CROP_RADIUS_M
            return jx - r, jy - r, jx + r, jy + r
        except Exception:
            (xmin, ymin), (xmax, ymax) = traci.simulation.getNetBoundary()
            return xmin, ymin, xmax, ymax

    def _get_bbox(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if self._bbox is not None:
            return self._bbox
        (xmin, ymin), (xmax, ymax) = traci.simulation.getNetBoundary()
        return (xmin, ymin), (xmax, ymax)

    # ------------------------------------------------------------------ #
    #  Static pixel helpers                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _speed_to_intensity(v: float, v_max: float = MAX_SPEED_MS) -> int:
        """Map speed [0, v_max] to pixel intensity [40, 255]."""
        v = max(0.0, v)
        return int(np.clip(40 + round(215 * (v / max(v_max, 1e-6))), 0, 255))

    @staticmethod
    def _world_to_pixel(
        x: float, y: float,
        xmin: float, xmax: float,
        ymin: float, ymax: float,
        dx: float,   dy: float,
        n: int,
    ) -> Tuple[int, int]:
        """World (x, y) -> image pixel (px, py).  Origin = top-left."""
        px = int(round((x - xmin) / dx * (n - 1)))
        py = int(round((ymax - y) / dy * (n - 1)))
        return px, py