# traffic_sim/env/observation.py
"""
ObservationBuilder
==================
Converts the running SUMO simulation into the RGB image observation
that is sent to the AI agent every step.

The generated image structure:
    - Shape: (n x n x 3) uint8 ndarray
    - One pixel represents one vehicle currently in the network.
    - Intensity: indicates the current speed (dim = stopped, bright = fast).
    - Color: indicates the outgoing edge or intended direction at the intersection.
    - White: indicates the vehicle has already passed the intersection (clearing).

Color scheme (consistent for ALL network types):
-----------------------------------------------
The dest_colors table maps EXIT edge IDs to RGB colors. For single intersections, 
these are "C_to_N", "C_to_E", etc. For multi-intersection networks, they are "J0_to_N0", etc.
Instead of maintaining a static table that breaks on new topologies, _normalize_dest_edge() 
does a two-step lookup to automatically assign colors based on the destination node's prefix.

Outgoing-edge detection (vehicles drawn white):
----------------------------------------------
intersection_outgoing is the set of outgoing edge IDs. For multi-intersection networks, 
we dynamically check if the current edge name ends with a border-node suffix (N, S, E, W). 
This allows the environment to scale without needing manually updated static sets.

TraCI calls used:
----------------------------------------------
    traci.vehicle.getIDList()
    traci.vehicle.getPosition(vid)
    traci.vehicle.getSpeed(vid)
    traci.vehicle.getRoute(vid)
    traci.vehicle.getRoadID(vid)
    traci.vehicle.getLaneID(vid)
    traci.simulation.getNetBoundary()
    traci.junction.getPosition(jid) -> used to calculate the crop bounding box
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import numpy as np

# traci may not be installed if the project is imported outside a SUMO environment.
try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

# Constants
MAX_SPEED_MS   = 13.89  # m/s -> used for pixel intensity normalization
_CROP_RADIUS_M = 100.0  # meters around junction center for cropped image (increased for multi-intersection layouts)

# Canonical single-intersection dest_colors keys, mapping a direction letter to an edge
_DIRECTION_TO_COLOR_KEY: Dict[str, str] = {
    "N": "C_to_N",
    "S": "C_to_S",
    "E": "C_to_E",
    "W": "C_to_W",
}


class ObservationBuilder:
    """
    Builds the RGB image observation from the running SUMO simulation.

    Args:
        dest_colors (dict, optional): Color table for destination edges. 
                                      Defaults to single-intersection types.
        intersection_outgoing (set, optional): Edge IDs to be drawn white (cleared vehicles).
        bbox (tuple, optional): Fixed world bounding box for full network renders.
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
        # Prevents recalculating the bounding box at every single step
        self._intersection_bboxes: Dict[int, Tuple[float, float, float, float]] = {}

    def build_image(self, n: int = 50) -> np.ndarray:
        """
        Builds an n x n RGB image of the entire network.
        """
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
        Builds an n x n image cropped to a specific intersection's region.

        The bounding box is centered on the junction's world position with a 
        radius of _CROP_RADIUS_M. We compute this once per episode and cache it.

        Args:
            intersection_id (int): Used as the cache key.
            junction_id (str): SUMO node ID (e.g., "C", "J0", "J_0_1").
            n (int): Output side length in pixels.
            
        Returns:
            np.ndarray: The cropped RGB observation.
        """
        if intersection_id not in self._intersection_bboxes:
            self._intersection_bboxes[intersection_id] = self._bbox_from_junction(junction_id)
            
        xmin, ymin, xmax, ymax = self._intersection_bboxes[intersection_id]
        return self._render(xmin, ymin, xmax, ymax, n)

    def reset_bbox_cache(self) -> None:
        """
        Clears the per-episode bounding box cache. 
        Should be called right after env.reset().
        """
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
        Core rendering engine. Scans all vehicles inside the bounding box 
        and maps them onto an n x n RGB matrix.

        Rendering logic per vehicle:
          - Outgoing edge -> drawn white (already cleared intersection)
          - Known destination -> base color * speed intensity
          - Unknown destination -> grey scale based on speed
        """
        image = np.zeros((n, n, 3), dtype=np.uint8)
        
        # Prevent division by zero if the network bounding box is flat
        dx = max(1e-6, xmax - xmin)
        dy = max(1e-6, ymax - ymin)

        for vid in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vid)
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue

            px, py = self._world_to_pixel(x, y, xmin, xmax, ymin, ymax, dx, dy, n)
            if not (0 <= px < n and 0 <= py < n):
                continue

            speed = traci.vehicle.getSpeed(vid)
            intensity = self._speed_to_intensity(speed)

            # Resolve the current edge the vehicle is on.
            # SUMO internal edges (inside junctions) start with ':'.
            # If so, we extract the actual edge from the lane ID by stripping the lane index.
            road_id = traci.vehicle.getRoadID(vid)
            if road_id.startswith(":"):
                lane_id = traci.vehicle.getLaneID(vid)
                # Strip the last "_<digit>" to get the parent edge ID.
                # Example: "C_to_N_0" -> "C_to_N"
                parts = lane_id.rsplit("_", 1)
                curr_edge = parts[0] if len(parts) == 2 and parts[1].isdigit() else lane_id
            else:
                curr_edge = road_id

            # White pixel: vehicle has already cleared the intersection
            if self._is_outgoing(curr_edge):
                image[py, px] = (255, 255, 255)
                continue

            # Color pixel: figure out the destination from the vehicle's route
            route = traci.vehicle.getRoute(vid) or []
            route_clean = [e for e in route if not e.startswith(":")]

            if not route_clean:
                image[py, px] = (intensity, intensity, intensity)
                continue

            # Look ahead in the route for a recognizable edge we can assign a color to
            edge_for_color = None
            for e in route_clean:
                edge_for_color = self._normalize_dest_edge(e)
                if edge_for_color is not None:
                    break

            # Fallback to the very last edge in the route if nothing matched
            if edge_for_color is None:
                edge_for_color = self._normalize_dest_edge(route_clean[-1])

            # Apply the color scaled by the speed intensity
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
        Checks if a vehicle is currently on an outgoing edge heading away 
        from the intersection towards a network boundary.

        Uses two strategies:
          1. Direct match with self.intersection_outgoing.
          2. Dynamic parsing for multi-intersection layouts (checks if the 
          destination node starts with N, S, E, or W).
                "J0_to_N0"       -> dest "N0" -> starts with N -> outgoing
                "J_0_0_to_S_0_0" -> dest "S_0_0" -> starts with S -> outgoing
                It also handles "C_to_N", "C_to_NE", "C_to_NW" automatically.
        """
        if edge_id in self.intersection_outgoing:
            return True

        if "_to_" in edge_id:
            dest_node = edge_id.split("_to_")[-1]
            if dest_node and dest_node[0] in ("N", "S", "E", "W"):
                return True

        return False

    def _normalize_dest_edge(self, edge_id: str) -> Optional[str]:
        """
        Maps any route edge ID to a canonical key that exists in self.dest_colors.

        Step 1: Direct match or prefix match. 
                e.g., "C_to_N_0" -> "C_to_N"
        Step 2: Destination-node fallback for multi-intersections.
                e.g., "J_0_0_to_S_1_2" -> dest node is "S_1_2" -> starts with "S" -> maps to "C_to_S".

        Returns:
            str: The matched color key, or None if no match is found.
        """
        # Step 1: Direct or prefix match
        if edge_id in self.dest_colors:
            return edge_id
            
        for dest_key in self.dest_colors:
            if edge_id.startswith(f"{dest_key}_"):
                return dest_key

        # Step 2: Extract destination node and map based on its first letter
        if "_to_" in edge_id:
            dest_node = edge_id.split("_to_")[-1]
            first_letter = dest_node[0] if dest_node else ""
            color_key = _DIRECTION_TO_COLOR_KEY.get(first_letter)
            
            if color_key is not None and color_key in self.dest_colors:
                return color_key

        return None

    # ------------------------------------------------------------------ #
    #  Bbox helpers                                                        #
    # ------------------------------------------------------------------ #
    def _bbox_from_junction(self, junction_id: str) -> Tuple[float, float, float, float]:
        """
        Calculates a square bounding box centered around a specific junction.
        Falls back to the entire network boundary if the junction ID is invalid.
        """
        try:
            jx, jy = traci.junction.getPosition(junction_id)
            r = _CROP_RADIUS_M
            return jx - r, jy - r, jx + r, jy + r
        except Exception:
            (xmin, ymin), (xmax, ymax) = traci.simulation.getNetBoundary()
            return xmin, ymin, xmax, ymax

    def _get_bbox(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Returns the fixed bbox if defined, otherwise fetches the full network boundary."""
        if self._bbox is not None:
            return self._bbox
        (xmin, ymin), (xmax, ymax) = traci.simulation.getNetBoundary()
        return (xmin, ymin), (xmax, ymax)

    # ------------------------------------------------------------------ #
    #  Static pixel helpers                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _speed_to_intensity(v: float, v_max: float = MAX_SPEED_MS) -> int:
        """
        Maps a real-world speed [0, v_max] to an 8-bit pixel intensity [40, 255].
        Stopped vehicles are drawn at 40 (dim but visible), fast vehicles are brighter.
        """
        v = max(0.0, v)
        return int(np.clip(40 + round(215 * (v / max(v_max, 1e-6))), 0, 255))

    @staticmethod
    def _world_to_pixel(
        x: float, y: float,
        xmin: float, xmax: float,
        ymin: float, ymax: float,
        dx: float, dy: float,
        n: int,
    ) -> Tuple[int, int]:
        """
        Projects world coordinates (x, y) onto the n x n image grid (px, py).
        Origin (0,0) is at the top-left of the image.
        """
        px = int(round((x - xmin) / dx * (n - 1)))
        py = int(round((ymax - y) / dy * (n - 1)))
        return px, py