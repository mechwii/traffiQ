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
 
build_image_for_intersection()
--------------------------------------
For multi-intersection networks, each agent should see only its own
intersection's region rather than the whole network crammed into 50×50 pixels.
 
``build_image_for_intersection(intersection_id, n)`` crops the world
bounding box to the region around that intersection's incoming edges,
then renders only the vehicles in that region into an n×n image.
 
The crop boundaries are derived at runtime from the lane geometry stored in
TraCI, so no manual coordinates need to be specifie
 
TraCI calls used
----------------
    traci.vehicle.getIDList()
    traci.vehicle.getPosition(vid)
    traci.vehicle.getSpeed(vid)
    traci.vehicle.getRoute(vid)
    traci.vehicle.getRoadID(vid)
    traci.vehicle.getLaneID(vid)
    traci.simulation.getNetBoundary()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
 
import numpy as np
 
try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

MAX_SPEED_MS = 14.0   # m/s -> used for pixel intensity normalisation

# Extra padding (metres) added around each intersection's crop region so that
# vehicles that are close to but not yet on the intersection edge are visible.
_CROP_PADDING_M = 20.0

class ObservationBuilder:
    """
    Builds the RGB image observation from the running SUMO simulation.
 
    Parameters
    ----------
    dest_colors : dict {edge_id: (R, G, B)} | None
        Maps outgoing edge IDs to an (R, G, B) colour in [0.0, 1.0].
        Vehicles heading toward that exit are drawn in this colour.
        If None, the default table matching the teacher's reference is used.
 
        To match YOUR network, pass this from ``SumoEnvironment.__init__``:
            dest_colors = {
                "C_to_N": (1.0, 0.0, 0.0),   # red    -> north exit
                "C_to_S": (0.0, 1.0, 0.0),   # green  -> south exit
                "C_to_E": (0.0, 0.0, 1.0),   # blue   -> east exit
                "C_to_W": (1.0, 1.0, 0.0),   # yellow -> west exit
            }
 
    intersection_outgoing : set[str] | None
        Edge IDs that are *outgoing* from the main intersection.
        Vehicles already on one of these edges are drawn **white**
        (they have cleared the intersection).
        If None, the default set from the teacher's reference is used.
 
        To match YOUR network:
            intersection_outgoing = {"C_to_N", "C_to_S", "C_to_E", "C_to_W"}
 
    bbox : ((xmin, ymin), (xmax, ymax)) | None
        World-coordinate bounding box for the image.
        If None, the full network boundary is queried from TraCI at runtime.
    """

    # --------- Default color / outgoing tables ---------    
    
    """
    _DEFAULT_DEST_COLORS: Dict[str, Tuple[float, float, float]] = {
        "1to2": (1.0, 0.0, 0.0),   # red
        "1to6": (0.0, 1.0, 0.0),   # green
        "1to3": (0.0, 0.0, 1.0),   # blue
        "1to5": (1.0, 1.0, 0.0),   # yellow
    }

    _DEFAULT_OUTGOING: set = {"1to5", "1to3", "1to2", "1to6"}
    """

    # Be aware here cause depending on the configuration this might change so we have to give
    # a configuration for each config that we generate, here is a universal list, but not enough
    # Cause for many lanes we will have number in the name
    _DEFAULT_DEST_COLORS: Dict[str, Tuple[float, float, float]] = {
        "C_to_N":  (1.0, 0.0, 0.0),  # red   -> North (four_way, t_junction)
        "C_to_S":  (0.0, 1.0, 0.0),  # green    -> South (four_way, complex)
        "C_to_E":  (0.0, 0.0, 1.0),  # blue    -> East (tous)
        "C_to_W":  (1.0, 1.0, 0.0),  # yellow   -> West (tous)
        "C_to_NE": (0.0, 1.0, 1.0),  # cyan    -> North-East (complex)
        "C_to_NW": (1.0, 0.0, 1.0),  # magenta -> North-west (complex)
    }

    _DEFAULT_OUTGOING: set = {
        "C_to_N", "C_to_S", "C_to_E", "C_to_W", "C_to_NE", "C_to_NW"
    }
 
    def __init__(
        self,
        dest_colors:            Optional[Dict[str, Tuple[float, float, float]]] = None,
        intersection_outgoing:  Optional[set]   = None,
        bbox:                   Optional[Tuple]  = None,
    ):
        self.dest_colors           = dest_colors           or self._DEFAULT_DEST_COLORS
        self.intersection_outgoing = intersection_outgoing or self._DEFAULT_OUTGOING
        self._bbox                 = bbox   # None -> queried from TraCI at runtime

        # Cache: intersection_id -> crop bbox computed once per episode
        self._intersection_bboxes: Dict[int, Tuple[float, float, float, float]] = {}


    # --------- PUBLIC : full-network image  ---------    
    def build_image(self, n: int = 50) -> np.ndarray:
        """
        Build the (n x n x 3) uint8 RGB image sent to the AI agent.
  
        Parameters
        ----------
        n : int
            Image side length in pixels.  Default 50.
 
        Returns
        -------
        image : ndarray, shape (n, n, 3), dtype uint8
        """
        (xmin, ymin), (xmax, ymax) = self._get_bbox()
        return self._render(xmin, ymin, xmax, ymax, n)
        """ OLD WAY (BEFORE CROPPING IMAGE PER INTERSECTIONS)
        image = np.zeros((n, n, 3), dtype=np.uint8)
 
        (xmin, ymin), (xmax, ymax) = self._get_bbox()

        # We protect from the division by 0
        dx = max(1e-6, xmax - xmin)
        dy = max(1e-6, ymax - ymin)
 
        for vid in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vid)
 
            # Skip vehicles outside the bounding box
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue
 
            px, py = self._world_to_pixel(x, y, xmin, xmax, ymin, ymax, dx, dy, n)
            if not (0 <= px < n and 0 <= py < n):
                continue
 
            speed     = traci.vehicle.getSpeed(vid)
            intensity = self._speed_to_intensity(speed)
 
            # Build a clean route (strip SUMO internal edges starting with ':')
            route       = traci.vehicle.getRoute(vid) or []
            route_clean = [e for e in route if not e.startswith(":")]
 
            # Grey pixel when the route has no meaningful edge
            if not route_clean:
                image[py, px] = (intensity, intensity, intensity)
                continue
 
            # Find the first route edge that maps to a configured destination.
            # This supports exact edge IDs like "C_to_N" and lane-specific
            # variants such as "C_to_N_0" or "C_to_NE_1".
            edge_for_color = None
            for e in route_clean:
                edge_for_color = self._normalize_dest_edge(e)
                if edge_for_color is not None:
                    break
            if edge_for_color is None:
                edge_for_color = self._normalize_dest_edge(route_clean[-1]) or route_clean[-1]
 
            # Resolve current edge (SUMO internal edges start with ':')
            road_id = traci.vehicle.getRoadID(vid)
            if road_id.startswith(":"):
                lane_id   = traci.vehicle.getLaneID(vid)
                curr_edge = lane_id.split("_")[0] if lane_id else road_id
            else:
                curr_edge = road_id
 
            # Vehicles that have already cleared the intersection -> white
            if curr_edge in self.intersection_outgoing:
                image[py, px] = (255, 255, 255)
            elif edge_for_color in self.dest_colors:
                cr, cg, cb = self.dest_colors[edge_for_color]
                image[py, px] = (
                    int(cr * intensity),
                    int(cg * intensity),
                    int(cb * intensity),
                )
            else:
                # Unknown destination → grey
                image[py, px] = (intensity, intensity, intensity)
 
        return image
        """
    
    # --------- PUBLIC : per-intersection cropped image ---------    

    def build_image_for_intersection(
        self,
        intersection_id: int,
        incoming_lane_ids: List[str],
        n: int = 50,
    ) -> np.ndarray:
        """
        Build an n x n image cropped to a single intersection's region.
 
        The crop bounding box is derived from the world coordinates of the
        ``incoming_lane_ids`` (the lanes feeding into this intersection).
        A padding of ``_CROP_PADDING_M`` metres is added on all sides.
 
        On the first call for a given intersection_id the bbox is computed
        and cached.  Subsequent calls reuse the cache.
 
        Parameters
        ----------
        intersection_id : int
            Index of the intersection (0-based).
        incoming_lane_ids : list[str]
            All lane IDs that feed into this intersection.
            Typically the keys of the per-intersection leaders dict.
        n : int
            Output image side length in pixels.  Default 50.
 
        Returns
        -------
        ndarray, shape (n, n, 3), dtype uint8
        """
        if intersection_id not in self._intersection_bboxes:
            bbox = self._compute_intersection_bbox(incoming_lane_ids)
            self._intersection_bboxes[intersection_id] = bbox
 
        xmin, ymin, xmax, ymax = self._intersection_bboxes[intersection_id]
        return self._render(xmin, ymin, xmax, ymax, n)
 
    def reset_bbox_cache(self) -> None:
        """
        Clear the cached intersection bounding boxes.
        Call this at the start of each episode (after env.reset()) so that
        new lane geometries are picked up correctly.
        """
        self._intersection_bboxes.clear()

    # --------- Internal rendering engine ---------

    def _render(
        self,
        xmin: float, ymin: float,
        xmax: float, ymax: float,
        n: int,
    ) -> np.ndarray:
        """
        Render all vehicles inside the bounding box into an n x n image.
        This is the shared core used by both build_image() and
        build_image_for_intersection().
        """
        image = np.zeros((n, n, 3), dtype=np.uint8)
 
        dx = max(1e-6, xmax - xmin)
        dy = max(1e-6, ymax - ymin)
 
        for vid in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vid)
 
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue
 
            px, py = self._world_to_pixel(x, y, xmin, xmax, ymin, ymax, dx, dy, n)
            if not (0 <= px < n and 0 <= py < n):
                continue
 
            speed     = traci.vehicle.getSpeed(vid)
            intensity = self._speed_to_intensity(speed)
 
            route       = traci.vehicle.getRoute(vid) or []
            route_clean = [e for e in route if not e.startswith(":")]
 
            if not route_clean:
                image[py, px] = (intensity, intensity, intensity)
                continue
 
            # Find the first route edge that maps to a known destination
            edge_for_color = None
            for e in route_clean:
                edge_for_color = self._normalize_dest_edge(e)
                if edge_for_color is not None:
                    break
            if edge_for_color is None:
                edge_for_color = (
                    self._normalize_dest_edge(route_clean[-1]) or route_clean[-1]
                )
 
            # Resolve current edge
            road_id = traci.vehicle.getRoadID(vid)
            if road_id.startswith(":"):
                lane_id  = traci.vehicle.getLaneID(vid)
                curr_edge = lane_id.split("_")[0] if lane_id else road_id
            else:
                curr_edge = road_id
 
            if curr_edge in self.intersection_outgoing:
                image[py, px] = (255, 255, 255)
            elif edge_for_color in self.dest_colors:
                cr, cg, cb = self.dest_colors[edge_for_color]
                image[py, px] = (
                    int(cr * intensity),
                    int(cg * intensity),
                    int(cb * intensity),
                )
            else:
                image[py, px] = (intensity, intensity, intensity)
 
        return image    
    
    # ---- Internal crop bbox computation ----
    def _compute_intersection_bbox(
        self,
        incoming_lane_ids: List[str],
    ) -> Tuple[float, float, float, float]:
        """
        Compute the world bounding box that covers all incoming lanes
        for one intersection, extended by _CROP_PADDING_M on every side.
 
        Uses ``traci.lane.getShape(lane_id)`` which returns a list of
        (x, y) polyline points for the lane's centre line.
 
        Falls back to the full network boundary if no lane shapes are found.
        """
        all_x: List[float] = []
        all_y: List[float] = []
 
        for lane_id in incoming_lane_ids:
            try:
                shape = traci.lane.getShape(lane_id)  # [(x0,y0), (x1,y1), ...]
                for x, y in shape:
                    all_x.append(x)
                    all_y.append(y)
            except Exception:
                pass  # lane might be internal / not available
 
        if not all_x:
            # Fallback: full network bbox
            (xmin, ymin), (xmax, ymax) = traci.simulation.getNetBoundary()
            return xmin, ymin, xmax, ymax
 
        xmin = min(all_x) - _CROP_PADDING_M
        ymin = min(all_y) - _CROP_PADDING_M
        xmax = max(all_x) + _CROP_PADDING_M
        ymax = max(all_y) + _CROP_PADDING_M
        return xmin, ymin, xmax, ymax


    # ---- Internal Helpers ----

    def _normalize_dest_edge(self, edge_id: str) -> Optional[str]:
        """Map a route edge ID to a configured destination color key."""
        if edge_id in self.dest_colors:
            return edge_id
        for dest_key in self.dest_colors:
            if edge_id == dest_key or edge_id.startswith(f"{dest_key}_"):
                return dest_key
        return None
 
    def _get_bbox(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return the full-network bounding box."""
        if self._bbox is not None:
            return self._bbox
        (xmin, ymin), (xmax, ymax) = traci.simulation.getNetBoundary()
        return (xmin, ymin), (xmax, ymax)
 
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
        """Convert world (x, y) -> image pixel (px, py). Origin = top-left."""
        px = int(round((x - xmin) / dx * (n - 1)))
        py = int(round((ymax - y) / dy * (n - 1)))  # y-axis inverted
        return px, py





