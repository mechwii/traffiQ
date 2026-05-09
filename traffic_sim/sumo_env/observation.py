# traffic_sim/sumo_env/observation.py
"""
ObservationBuilder
==================
Converts the running SUMO simulation into the RGB image observation
that is sent to the AI agent every step.
 
The image mirrors ``image_construct1()`` from the teacher's reference:
 
    - Shape    : (n x n x 3)  uint8  ndarray
    - One pixel per vehicle currently in the network.
    - Intensity → current speed   (dim = stopped, bright = fast).
    - Colour    → outgoing edge / intended direction at the intersection.
    - White     → vehicle already past the intersection (arrived / clearing).
 
Extending to other observation types
-------------------------------------
If you later want to feed a numeric vector to the agent instead of (or in
addition to) an image, add a ``build_vector()`` method here and call it
from ``SumoEnvironment._build_observation()``.  The rest of the code does
not need to change.
 
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

from typing import Dict, Optional, Tuple
 
import numpy as np
 
try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

MAX_SPEED_MS = 14.0   # m/s -> used for pixel intensity normalisation

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
