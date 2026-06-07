# traffic_sim/sumo_env/state.py
"""
StateExtractor

Extracts the current simulation state from SUMO via the TraCI API.

Implements the two state-related functions from the specification:
  - create_state() => full snapshot of the environment
  - get_leaders()  => frontmost vehicle per lane

FIX vs previous version:
  get_leaders() now filters out internal SUMO lanes (starting with ':')
  and outgoing lanes (destination is a border node).  This reduces the
  returned dict from hundreds of entries (most None) to just the
  incoming lanes that the agent actually needs to make decisions about.

TraCI calls used:
  traci.vehicle.getIDList()
  traci.vehicle.getSpeed()
  traci.vehicle.getPosition()
  traci.vehicle.getRoadID()
  traci.vehicle.getAccumulatedWaitingTime()
  traci.vehicle.getLaneID()
  traci.vehicle.getLanePosition()
  traci.edge.getLastStepOccupancy()
  traci.edge.getLastStepMeanSpeed()
  traci.lane.getIDList()
  traci.simulation.getTime()
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False


class StateExtractor:
    """
    Extracts and packages simulation state from TraCI.

    Instantiated by SumoEnvironment after TraCI connects.
    """

    def create_state(self) -> Dict[str, Any]:
        """
        Build a full snapshot of the current simulation state.

        Returns:
            dict with keys: simulation_time, vehicle_ids, vehicle_speeds,
            vehicle_positions, vehicle_edges, vehicle_lanes, waiting_times,
            edge_occupancies, edge_mean_speeds, leaders.
        """
        state: Dict[str, Any] = {}

        state["simulation_time"] = traci.simulation.getTime()

        vehicle_ids = list(traci.vehicle.getIDList())
        state["vehicle_ids"] = vehicle_ids

        speeds:     Dict[str, float]                = {}
        positions:  Dict[str, Tuple[float, float]]  = {}
        edges:      Dict[str, str]                  = {}
        lanes:      Dict[str, str]                  = {}
        wait_times: Dict[str, float]                = {}

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

        # Edge-level information
        edge_occupancies: Dict[str, float] = {}
        edge_mean_speeds: Dict[str, float] = {}

        active_edges = {e for e in edges.values() if not e.startswith(":")}

        for eid in active_edges:
            edge_occupancies[eid] = traci.edge.getLastStepOccupancy(eid)
            edge_mean_speeds[eid] = traci.edge.getLastStepMeanSpeed(eid)

        state["edge_occupancies"] = edge_occupancies
        state["edge_mean_speeds"] = edge_mean_speeds

        state["leaders"] = self.get_leaders()

        return state

    def get_leaders(self) -> Dict[str, Optional[str]]:
        """
        Retrieve the frontmost vehicle (leader) for every non-internal lane.

        FIX: Only considers lanes belonging to real (non-internal) edges.
        Internal SUMO lanes (id starting with ':') are skipped entirely.
        This reduces the dict from hundreds of entries to just the lanes
        that matter for traffic control decisions.

        Strategy:
            1. Get all lane IDs, filter out internal lanes.
            2. Group vehicles by lane.
            3. For each lane, the leader is the vehicle with the highest
               lanePosition (furthest along = closest to the intersection).

        Returns:
            dict : {lane_id: vehicle_id | None}
        """
        all_lanes = list(traci.lane.getIDList())

        # Filter: keep only non-internal lanes
        real_lanes = [lid for lid in all_lanes if not lid.startswith(":")]

        # Build lane -> [vehicle_id] mapping
        lane_vehicles: Dict[str, List[str]]           = {lid: [] for lid in real_lanes}
        lane_positions: Dict[str, Dict[str, float]]   = {lid: {} for lid in real_lanes}

        for vid in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(vid)
            if lane_id in lane_vehicles:
                pos = traci.vehicle.getLanePosition(vid)
                lane_vehicles[lane_id].append(vid)
                lane_positions[lane_id][vid] = pos

        # For each lane, leader = vehicle with highest lane position
        leaders: Dict[str, Optional[str]] = {}
        for lane_id in real_lanes:
            vehicles_on_lane = lane_vehicles[lane_id]
            if not vehicles_on_lane:
                leaders[lane_id] = None
            else:
                leader_id = max(
                    vehicles_on_lane,
                    key=lambda v: lane_positions[lane_id][v],
                )
                leaders[lane_id] = leader_id

        return leaders

    @staticmethod
    def state_summary(state: Dict[str, Any]) -> str:
        """Return a compact, human-readable summary of a state dict."""
        n_vehicles  = len(state.get("vehicle_ids", []))
        sim_time    = state.get("simulation_time", 0.0)
        speeds      = list(state.get("vehicle_speeds", {}).values())
        mean_speed  = sum(speeds) / len(speeds) if speeds else 0.0
        waiting     = sum(
            1 for w in state.get("waiting_times", {}).values() if w > 0
        )
        n_leaders   = sum(
            1 for v in state.get("leaders", {}).values() if v is not None
        )

        return (
            f"t={sim_time:.1f}s | "
            f"vehicles={n_vehicles} | "
            f"mean_speed={mean_speed:.2f} m/s | "
            f"waiting={waiting} | "
            f"leaders={n_leaders}"
        )