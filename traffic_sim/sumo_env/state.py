# traffic_sim/sumo_env/state.py
"""
StateExtractor

Extracts the current simulation state from SUMO via the TraCI API.

Implements the two state-related functions from the specification:
  - create_state() => full snapshot of the environment
  - get_leaders()  => frontmost vehicle per lane


TraCI calls used :

  traci.vehicle.getIDList()          - all active vehicle IDs
  traci.vehicle.getSpeed()           — speed in m/s
  traci.vehicle.getPosition()        — (x, y) in metres
  traci.vehicle.getRoadID()          — current edge ID
  traci.vehicle.getAccumulatedWaitingTime() — total waiting time in s
  traci.vehicle.getLaneID()          — current lane ID
  traci.vehicle.getLeader()          — (leader_id, gap) ahead in same lane
  traci.edge.getLastStepOccupancy()  — lane occupancy 0–100 %
  traci.edge.getLastStepMeanSpeed()  — mean speed of last step (m/s)
  traci.lane.getIDList()             — all lane IDs in the network
  traci.simulation.getTime()         — current simulation clock

  """

from typing import Any, Dict, List, Optional, Tuple

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

# ============ MAIN CLASS ============

class StateExtractor:
    """
    Extracts and packages simulation state from TraCI.

    This class is instantiated by SumoEnvironment after TraCI connects,
    so it can safely call traci.* immediately.
    """

    def create_state(self) -> Dict[str, Any]:
        """
        Build a full snapshot of the current simulation state.

        Returns :
            dict with the following keys:
                simulation_time : float
                    Current simulation clock in seconds.

                vehicle_ids : list[str]
                    IDs of all vehicles currently in the network.

                vehicle_speeds : dict {vehicle_id: float}
                    Speed of each vehicle in m/s.

                vehicle_positions : dict {vehicle_id: tuple(float, float)}
                    (x, y) position of each vehicle in metres.

                vehicle_edges : dict {vehicle_id: str}
                    ID of the road edge each vehicle is currently on.

                vehicle_lanes : dict {vehicle_id: str}
                    ID of the lane each vehicle is currently on.

                waiting_times : dict {vehicle_id: float}
                    Accumulated waiting time for each vehicle in seconds.
                    A vehicle is "waiting" when its speed < 0.1 m/s.

                edge_occupancies : dict {edge_id: float}
                    Fraction of the edge occupied by vehicles (0–100 %).

                edge_mean_speeds : dict {edge_id: float}
                    Mean speed of vehicles on each edge (m/s).
                    Returns the road's max speed if the edge is empty.

                leaders : dict {lane_id: str | None}
                    Leader vehicle for each lane (see get_leaders()).
        """
        state: Dict[str, Any] = {}

        # Simulation time
        state["simulation_time"] = traci.simulation.getTime()

        # Vehicle information
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

        # Collect unique edge IDs from current vehicles
        # (avoids iterating the full network edge list every step)
        active_edges = set(edges.values())

        # Filter out internal SUMO edges (they start with ':')
        active_edges = {e for e in active_edges if not e.startswith(":")}

        for eid in active_edges:
            edge_occupancies[eid] = traci.edge.getLastStepOccupancy(eid)
            edge_mean_speeds[eid] = traci.edge.getLastStepMeanSpeed(eid)

        state["edge_occupancies"] = edge_occupancies
        state["edge_mean_speeds"] = edge_mean_speeds

        # Leaders
        state["leaders"] = self.get_leaders()

        return state

    def get_leaders(self) -> Dict[str, Optional[str]]:
        """
        Retrieve the frontmost vehicle (leader) for every lane.

        Strategy :
            For each lane in the network, find the vehicle with the lowest
            lanePosition value (closest to the lane end / intersection).

        TraCI does not have a direct "get lane leader" call, so we:
          1. Group vehicles by lane.
          2. For each lane, find the vehicle with the highest lanePosition
             (furthest along the lane = closest to its end = the leader).

        Returns :
            dict : {lane_id: vehicle_id | None}
                Maps each occupied lane to its current leader vehicle.
                Unoccupied lanes are included with value None.

        Alternative method :
            traci.vehicle.getLeader(vid, dist) returns the vehicle directly
            ahead of vid within *dist* metres.  This gives the leader
            relative to a specific vehicle, not the absolute lane leader.
            Both results are included in the return value.
        """
        # Get all lane IDs from the network
        all_lanes = list(traci.lane.getIDList())

        # Build lane -> [vehicle_id] mapping and lane positions
        lane_vehicles: Dict[str, List[str]] = {lid: [] for lid in all_lanes}
        lane_positions: Dict[str, Dict[str, float]] = {lid: {} for lid in all_lanes}

        for vid in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(vid)
            if lane_id in lane_vehicles:
                pos = traci.vehicle.getLanePosition(vid)
                lane_vehicles[lane_id].append(vid)
                lane_positions[lane_id][vid] = pos

        # For each lane, the leader is the vehicle with the HIGHEST
        # lane position (furthest along the lane)
        leaders: Dict[str, Optional[str]] = {}
        for lane_id in all_lanes:
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

    # ---- Utility ----

    @staticmethod
    def state_summary(state: Dict[str, Any]) -> str:
        """
        Return a compact, human-readable summary of a state dict.

        Example output:
            t=12.0s | vehicles=8 | mean_speed=8.3 m/s | waiting=2
        """
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
