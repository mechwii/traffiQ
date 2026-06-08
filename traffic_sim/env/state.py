# traffic_sim/env/state.py
"""
StateExtractor

Reads the current simulation state from SUMO via TraCI and packages it into
plain Python dicts that the rest of the project can use.

Implements the two state functions from the project specification:
    create_state()  -> full per-vehicle and per-edge snapshot of the simulation
    get_leaders()   -> frontmost vehicle per incoming lane

Why we filter out internal lanes in get_leaders():
    SUMO generates a large number of "internal" lanes (IDs starting with ':')
    inside every junction box. Including them would flood the returned dict with
    hundreds of mostly-None entries that the agent can't act on anyway. Filtering
    them here means callers only see the lanes that actually matter for intersection
    control decisions.

TraCI calls used:
    traci.simulation.getTime()               -> current simulation clock (s)
    traci.vehicle.getIDList()                -> all active vehicle IDs
    traci.vehicle.getSpeed(vid)              -> current speed (m/s)
    traci.vehicle.getPosition(vid)           -> (x, y) world coordinates
    traci.vehicle.getRoadID(vid)             -> edge the vehicle is currently on
    traci.vehicle.getLaneID(vid)             -> specific lane the vehicle is on
    traci.vehicle.getLanePosition(vid)       -> distance along the lane (m)
    traci.vehicle.getAccumulatedWaitingTime(vid) -> total wait time since entry (s)
    traci.edge.getLastStepOccupancy(eid)     -> fraction of edge occupied [0, 1]
    traci.edge.getLastStepMeanSpeed(eid)     -> mean vehicle speed on the edge (m/s)
    traci.lane.getIDList()                   -> all lane IDs in the network
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False


class StateExtractor:
    """
    Extracts and packages the current simulation state from TraCI.

    Created by SumoEnvironment once TraCI has connected. Has no internal
    state of its own — every method queries TraCI fresh each time it's called.
    """

    def create_state(self) -> Dict[str, Any]:
        """
        Build a full snapshot of the current simulation state.

        Queries TraCI for every active vehicle and every occupied edge, then
        bundles everything into a single dict. This is the authoritative state
        object consumed by the observation builder, the statistics collector,
        and the main training loop.

        Returns:
            dict with the following keys:
                simulation_time   (float) -> current clock in seconds
                vehicle_ids       (list)  -> IDs of all active vehicles
                vehicle_speeds    (dict)  -> {vid: speed in m/s}
                vehicle_positions (dict)  -> {vid: (x, y) world coordinates}
                vehicle_edges     (dict)  -> {vid: edge_id the vehicle is on}
                vehicle_lanes     (dict)  -> {vid: lane_id the vehicle is on}
                waiting_times     (dict)  -> {vid: accumulated wait in seconds}
                edge_occupancies  (dict)  -> {edge_id: occupancy fraction [0,1]}
                edge_mean_speeds  (dict)  -> {edge_id: mean speed in m/s}
                leaders           (dict)  -> {lane_id: leader_vid | None}
        """
        state: Dict[str, Any] = {}

        state["simulation_time"] = traci.simulation.getTime()

        vehicle_ids          = list(traci.vehicle.getIDList())
        state["vehicle_ids"] = vehicle_ids

        # Collect all per-vehicle data in a single pass to minimise TraCI calls.
        speeds:     Dict[str, float]               = {}
        positions:  Dict[str, Tuple[float, float]] = {}
        edges:      Dict[str, str]                 = {}
        lanes:      Dict[str, str]                 = {}
        wait_times: Dict[str, float]               = {}

        for vid in vehicle_ids:
            speeds[vid]     = traci.vehicle.getSpeed(vid)
            positions[vid]  = traci.vehicle.getPosition(vid)
            edges[vid]      = traci.vehicle.getRoadID(vid)
            lanes[vid]      = traci.vehicle.getLaneID(vid)
            wait_times[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)

        state["vehicle_speeds"]    = speeds
        state["vehicle_positions"] = positions
        state["vehicle_edges"]     = edges
        state["vehicle_lanes"]     = lanes
        state["waiting_times"]     = wait_times

        # Edge-level data only for real (non-internal) edges that have at least
        # one vehicle. Internal SUMO edges start with ':' and don't carry
        # meaningful occupancy or speed information.
        edge_occupancies: Dict[str, float] = {}
        edge_mean_speeds: Dict[str, float] = {}
        active_edges = {e for e in edges.values() if not e.startswith(":")}

        for eid in active_edges:
            edge_occupancies[eid] = traci.edge.getLastStepOccupancy(eid)
            edge_mean_speeds[eid] = traci.edge.getLastStepMeanSpeed(eid)

        state["edge_occupancies"] = edge_occupancies
        state["edge_mean_speeds"] = edge_mean_speeds

        # Attach leaders so callers don't need a separate round-trip to get them.
        state["leaders"] = self.get_leaders()

        return state

    def get_leaders(self) -> Dict[str, Optional[str]]:
        """
        Find the frontmost vehicle (leader) on every real incoming lane.

        "Leader" means the vehicle with the highest lane position, i.e. the one
        closest to the end of the lane and therefore about to enter the junction
        next. The AI agent uses this to decide who gets a green light.

        Internal SUMO lanes (IDs starting with ':') are excluded because they
        exist inside junction boxes and can't be meaningfully controlled.

        Algorithm:
            1. Get all lane IDs and discard internal ones.
            2. For each active vehicle, record which lane it's on and its position.
            3. For each lane, pick the vehicle with the highest lane position.

        Returns:
            dict {lane_id: vehicle_id | None}
            None means the lane is currently empty.
        """
        all_lanes  = list(traci.lane.getIDList())
        real_lanes = [lid for lid in all_lanes if not lid.startswith(":")]

        # Pre-allocate containers for every real lane.
        lane_vehicles:  Dict[str, List[str]]        = {lid: [] for lid in real_lanes}
        lane_positions: Dict[str, Dict[str, float]] = {lid: {} for lid in real_lanes}

        # Single pass over all active vehicles to group them by lane.
        for vid in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(vid)
            if lane_id in lane_vehicles:
                pos = traci.vehicle.getLanePosition(vid)
                lane_vehicles[lane_id].append(vid)
                lane_positions[lane_id][vid] = pos

        # For each lane, the leader is the vehicle with the highest position
        # (furthest along the lane = closest to the intersection).
        leaders: Dict[str, Optional[str]] = {}
        for lane_id in real_lanes:
            vehicles_on_lane = lane_vehicles[lane_id]
            if not vehicles_on_lane:
                leaders[lane_id] = None
            else:
                leaders[lane_id] = max(
                    vehicles_on_lane,
                    key=lambda v: lane_positions[lane_id][v],
                )

        return leaders

    @staticmethod
    def state_summary(state: Dict[str, Any]) -> str:
        """
        Return a compact one-line summary of a state dict.

        Useful for quick sanity checks during development -> call it whenever
        you want to log what's happening without dumping the full state object.
        """
        n_vehicles = len(state.get("vehicle_ids", []))
        sim_time   = state.get("simulation_time", 0.0)
        speeds     = list(state.get("vehicle_speeds", {}).values())
        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        waiting    = sum(1 for w in state.get("waiting_times", {}).values() if w > 0)
        n_leaders  = sum(1 for v in state.get("leaders", {}).values() if v is not None)

        return (
            f"t={sim_time:.1f}s | "
            f"vehicles={n_vehicles} | "
            f"mean_speed={mean_speed:.2f} m/s | "
            f"waiting={waiting} | "
            f"leaders={n_leaders}"
        )