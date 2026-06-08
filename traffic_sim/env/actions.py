# traffic_sim/env/actions.py
"""
ActionHandler

Single point of truth for all vehicle control in the simulation.
Every TraCI command that touches a vehicle goes through this class.
SumoEnvironment never calls TraCI vehicle functions directly, which keeps
the environment logic clean and makes control strategies easy to swap out.

The class implements the three action primitives from the project spec:
    set_action()        -> apply speed commands to arbitrary vehicles
    set_speedMode()     -> configure SUMO's internal safety checks per vehicle
    set_loaded_veh()    -> enforce the leader/follower hierarchy

On top of those, it provides the DQN-specific pipeline called each step:
    set_speed_modes_loaded()     -> set mode 55 for vehicles that just entered
    stop_non_leaders()           -> hold followers behind the intersection
    apply_binary_action()        -> go / stop leaders based on the AI decision
    stop_departed_non_leaders()  -> catch vehicles that spawned mid-step

TraCI calls used:
    traci.vehicle.setSpeed(vid, speed)
        Force a vehicle to a specific speed (m/s). Pass -1 to hand control
        back to SUMO's car-following model.

    traci.vehicle.setSpeedMode(vid, mode)
        Integer bitmask controlling which SUMO speed safety checks are active.
        See set_speedMode() for a full breakdown.

    traci.vehicle.setMaxSpeed(vid, speed)
        Cap the vehicle's maximum allowed speed.

    traci.vehicle.getLaneID(vid)
        Return the lane the vehicle is currently on.

    traci.vehicle.getIDList()
        Return all vehicle IDs currently active in the network.

    traci.vehicle.setStop(vid, edgeID, pos, duration=...)
        Queue a planned stop at a given position on an edge.
        Setting duration=0 on an existing stop releases the vehicle immediately.

    traci.vehicle.getNextStops(vid)
        Return the list of upcoming stops for this vehicle.

    traci.vehicle.setColor(vid, (R, G, B, A))
        Change the vehicle's color in sumo-gui (visual debug only).
"""

from typing import Any, Dict, List, Optional, Union

# traci may not be installed if the project is run outside a SUMO environment.
# We use this flag so callers can check availability without crashing at import time.
try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False


# Full stop target speed (m/s). SUMO keeps the vehicle "loaded" in the network
# at 0.0 so it still physically occupies space and blocks followers naturally.
STOP_SPEED = 0.0

# Speed mode 0 = binary 00000: every SUMO safety check disabled.
# Useful for pure external control, but collisions can go undetected.
SPEED_MODE_FULL_CONTROL = 0

# Speed mode 31 = binary 11111: all five SUMO safety checks active.
# This is SUMO's standard default behavior.
SPEED_MODE_DEFAULT = 31

# Speed mode 55 = binary 110111: all checks active except bit 3 (junction
# right-of-way). This is the mode the teacher's reference uses — it lets the
# AI decide who crosses the intersection instead of SUMO's built-in priority system.
SPEED_MODE_AI_CONTROL = 55


class ActionHandler:
    """
    Centralizes all TraCI vehicle commands for a running SUMO simulation.

    SumoEnvironment creates one instance of this class after TraCI connects
    and delegates all vehicle interactions here. Nothing outside this class
    should call traci.vehicle.set* directly.

    Args:
        step_length (float): Duration of one simulation step in seconds.
                             Stored in case we need acceleration-based speed
                             ramping later.
    """

    def __init__(self, step_length: float = 1.0):
        self.step_length = step_length

        # Snapshot of active vehicle IDs, refreshed once per step via refresh_cache().
        # We use a set so membership checks are O(1) instead of O(n),
        # which matters when hundreds of vehicles are in the network simultaneously.
        self._vehicle_cache: Optional[set] = None

    # ================================================================== #
    #  Cache management                                                    #
    # ================================================================== #

    def refresh_cache(self) -> None:
        """
        Takes a snapshot of the currently active vehicle IDs from TraCI.

        Call this exactly once at the start of each simulation step, before
        any other method in this class. All subsequent _vehicle_exists() calls
        within that step reuse this snapshot instead of hitting the TraCI API
        repeatedly.
        """
        self._vehicle_cache = set(traci.vehicle.getIDList())

    def _vehicle_exists(self, vehicle_id: str) -> bool:
        """
        Check whether a vehicle is still active in the simulation.

        Uses the cached snapshot when available. Falls back to a live TraCI
        call if refresh_cache() was not called this step (e.g. during setup).
        """
        if self._vehicle_cache is not None:
            return vehicle_id in self._vehicle_cache
        return vehicle_id in traci.vehicle.getIDList()

    # ================================================================== #
    #  Spec functions: set_action, set_speedMode, set_loaded_veh           #
    # ================================================================== #

    def set_action(
        self,
        actions: Union[Dict[str, float], List[Dict[str, Any]], Any]
    ) -> None:
        """
        Apply speed commands to one or more vehicles.

        We support two formats here to keep things flexible whether we are
        running scripted tests, RL agents, or rule-based controllers.

        Format 1: speed dictionary {vehicle_id: speed_m_s}
            Each key is a vehicle ID, the value is the target speed in m/s.
            Pass -1.0 to give control back to SUMO's car-following model.
            Example: {"veh_0": 8.0, "veh_1": 0.0, "veh_2": -1.0}

        Format 2: list of command dictionaries
            More expressive format that lets you mix action types.
            Supported types:
              "speed"     -> set exact speed (m/s)
              "max_speed" -> change the vehicle's speed cap
              "reset"     -> hand back control to SUMO
            Example:
                [
                    {"vehicle_id": "veh_0", "type": "speed",     "value": 8.0},
                    {"vehicle_id": "veh_1", "type": "max_speed", "value": 10.0},
                    {"vehicle_id": "veh_2", "type": "reset"},
                ]

        Args:
            actions: Commands to apply. Silently ignored if None.
        """
        if actions is None:
            return

        # Format 1 : {vehicle_id : speed}
        if isinstance(actions, dict):
            for vehicle_id, speed in actions.items():
                self._apply_speed(vehicle_id, speed)
            return

        # Format 2 : List of action dicts
        if isinstance(actions, list):
            for action in actions:
                vehicle_id  = action.get("vehicle_id")
                action_type = action.get("type", "speed")
                value       = action.get("value", 0.0)

                if vehicle_id is None:
                    continue

                if action_type == "speed":
                    self._apply_speed(vehicle_id, float(value))
                elif action_type == "max_speed":
                    self._apply_max_speed(vehicle_id, float(value))
                elif action_type == "reset":
                    self._reset_vehicle(vehicle_id)
                else:
                    print(
                        f"[ActionHandler] Unknown action type: '{action_type}' "
                        f"for vehicle '{vehicle_id}' - skipped."
                    )
            return

        print(
            f"[ActionHandler] Unrecognised action format: {type(actions)} - "
            "no actions applied."
        )

    def set_speedMode(
        self,
        vehicle_ids: Optional[List[str]] = None,
        mode: int = SPEED_MODE_DEFAULT,
    ) -> None:
        """
        Update the speed mode bitmask for one or more vehicles.

        SUMO uses this bitmask to know which safety rules to enforce.
        Each bit enables or disables a specific check:
            Bit 0 (value  1): Safe speed according to the car-following model
            Bit 1 (value  2): Maximum acceleration limit
            Bit 2 (value  4): Maximum deceleration limit
            Bit 3 (value  8): Junction right-of-way rules
            Bit 4 (value 16): Braking at red lights and stop lines

        Common presets used in this project:
            31  (11111) -> all checks active, default SUMO behavior
             0  (00000) -> full external control, no checks at all
            55 (110111) -> everything active except junction right-of-way (bit 3),
                           which is what we use so the AI controls who crosses

        Args:
            vehicle_ids: Vehicles to update. If None, all active vehicles are updated.
            mode: Bitmask to apply. Defaults to 31 (full SUMO safety).
        """
        targets = vehicle_ids or list(traci.vehicle.getIDList())
        for vehicle_id in targets:
            if self._vehicle_exists(vehicle_id):
                traci.vehicle.setSpeedMode(vehicle_id, mode)

    def set_loaded_veh(
        self,
        leaders: Dict[str, Optional[str]],
    ) -> None:
        """
        Enforce the leader/follower hierarchy for all active vehicles.

        Leaders (frontmost vehicle per lane) are released to SUMO's normal
        car-following model. Every other vehicle is commanded to stop at 0 m/s.
        This prevents followers from pushing unsupervised into the junction
        while the AI is still deciding who gets to go.

        Args:
            leaders: Map of {lane_id: vehicle_id | None} from StateExtractor.get_leaders().
        """
        leader_ids = {vid for vid in leaders.values() if vid is not None}

        for vehicle_id in traci.vehicle.getIDList():
            if vehicle_id in leader_ids:
                # Give leaders back to SUMO -> apply_binary_action() will then
                # queue a stop or release them depending on the AI decision.
                self._reset_vehicle(vehicle_id)
            else:
                self._apply_speed(vehicle_id, STOP_SPEED)

    # ================================================================== #
    #  Binary action methods (DQN agent pipeline)                          #
    # ================================================================== #

    def apply_binary_action(
        self,
        action: Dict[str, int],
        leaders: Dict[str, Optional[str]],
    ) -> None:
        """
        Translate a {lane_id: 0|1} decision into TraCI stop or release commands.

        This mirrors the teacher's set_leaders_actions2():
            action == 1 -> cancel any planned stop, let SUMO drive the car (green)
            action == 0 -> queue a stop 10m before the junction (red)

        The stop position is computed as (lane_length - 10m) so vehicles queue
        just short of the intersection box without overlapping it.
        setStop() with duration=0 cancels a previously queued stop without
        physically stopping the car.

        Args:
            action: Per-lane go/stop decision from the AI agent.
            leaders: Pre-fetched leader snapshot. We do NOT re-query TraCI here
                     to avoid inconsistencies if the simulation moved between calls.
        """
        for lane_id, go in action.items():
            vehicle_id = leaders.get(lane_id)
            if vehicle_id is None:
                continue

            try:
                road_id = traci.vehicle.getRoadID(vehicle_id)
                if road_id.startswith(":"):
                    # Vehicle is currently inside a junction (internal edge).
                    # Calling setStop on an internal edge raises a TraCIException, so we skip it.
                    continue

                lane     = traci.vehicle.getLaneID(vehicle_id)
                length   = traci.lane.getLength(lane)
                # Stop 10m before the edge ends so the car body doesn't overlap the junction box.
                stop_pos = max(1.0, length - 10.0)

                if go == 1:
                    # Green: cancel any planned stop and let SUMO take over.
                    traci.vehicle.setColor(vehicle_id, (0, 255, 0))
                    if traci.vehicle.getNextStops(vehicle_id):
                        # duration=0 instantly clears the queued stop without halting the car.
                        traci.vehicle.setStop(vehicle_id, road_id, stop_pos, duration=0)
                    traci.vehicle.setSpeed(vehicle_id, -1)
                else:
                    # Red: queue a stop before the junction if it's still reachable.
                    traci.vehicle.setColor(vehicle_id, (255, 0, 0))
                    self._safe_set_stop(vehicle_id, road_id, stop_pos)

            except traci.TraCIException:
                # The vehicle may have left the network between the leaders query
                # and this loop. Silently skip it.
                pass

    def stop_non_leaders(
        self,
        incoming_leaders: Dict[str, Optional[str]],
    ) -> None:
        """
        Force all follower vehicles to stop behind their lane's stop line.

        Called once per step BEFORE apply_binary_action() so only the
        designated leader for each lane is free to move. Vehicles that are
        inside the junction box or have already crossed it are skipped
        because they are no longer under intersection control.

        Matches the teacher's set_loaded_vehicle().

        Args:
            incoming_leaders: Current leader per lane. Every vehicle not in
                               this dict's values will be stopped.
        """
        leader_vids = {vid for vid in incoming_leaders.values() if vid is not None}

        for vid in traci.vehicle.getIDList():
            if vid in leader_vids:
                continue

            try:
                road_id = traci.vehicle.getRoadID(vid)

                if road_id.startswith(":"):
                    continue  # inside the junction, leave it alone

                if self._is_outgoing_edge(road_id):
                    continue  # already crossed, don't interfere

                lane     = traci.vehicle.getLaneID(vid)
                length   = traci.lane.getLength(lane)
                stop_pos = max(1.0, length - 10.0)

                if self._safe_set_stop(vid, road_id, stop_pos):
                    # Orange color so it's easy to spot waiting vehicles in the GUI.
                    traci.vehicle.setColor(vid, (255, 200, 0))

            except traci.TraCIException:
                pass

    def stop_departed_non_leaders(
        self,
        incoming_leaders: Dict[str, Optional[str]],
    ) -> None:
        """
        Stop vehicles that spawned during the current simulation step.

        SUMO can insert vehicles mid-step, so they won't be in the
        incoming_leaders snapshot taken before simulationStep(). This method
        runs AFTER simulationStep() and stops any new arrivals that aren't
        leaders, preventing them from free-rolling into the junction.

        Args:
            incoming_leaders: The same leader snapshot used before the step.
        """
        leader_vids = {vid for vid in incoming_leaders.values() if vid is not None}

        for vid in traci.simulation.getDepartedIDList():
            if vid in leader_vids:
                continue

            try:
                road_id = traci.vehicle.getRoadID(vid)
                if road_id.startswith(":"):
                    continue

                lane     = traci.vehicle.getLaneID(vid)
                length   = traci.lane.getLength(lane)
                stop_pos = max(1.0, length - 10.0)

                if self._safe_set_stop(vid, road_id, stop_pos):
                    traci.vehicle.setColor(vid, (255, 200, 0))

            except traci.TraCIException:
                pass

    def set_speed_modes_loaded(
        self,
        mode: int = SPEED_MODE_AI_CONTROL,
    ) -> None:
        """
        Apply the AI speed mode to all vehicles that were just loaded this step.

        "Loaded" means the vehicle was scheduled into the network but may not
        have actually departed yet. Setting the mode at load time (before departure)
        ensures that no vehicle ever drives even a single step under SUMO's default
        junction right-of-way rules.

        Args:
            mode: Speed mode bitmask. Defaults to 55 (AI control mode).
        """
        for vid in traci.simulation.getLoadedIDList():
            try:
                traci.vehicle.setSpeedMode(vid, mode)
            except traci.TraCIException:
                # Can happen if a vehicle was loaded but immediately discarded
                # by SUMO because of a routing error.
                pass

    # ================================================================== #
    #  Internal helpers                                                    #
    # ================================================================== #

    def _apply_speed(self, vehicle_id: str, speed: float) -> None:
        """
        Set a vehicle's target speed in m/s.

        -1.0 is SUMO's sentinel value meaning "resume normal car-following".
        Any other negative value is clamped to 0 to avoid undefined behavior.
        """
        if not self._vehicle_exists(vehicle_id):
            return
        if speed < 0 and speed != -1.0:
            speed = 0.0
        traci.vehicle.setSpeed(vehicle_id, speed)

    def _apply_max_speed(self, vehicle_id: str, speed: float) -> None:
        """Cap the vehicle's maximum allowed speed (m/s). Negative values are clamped to 0."""
        if not self._vehicle_exists(vehicle_id):
            return
        if speed < 0:
            speed = 0.0
        traci.vehicle.setMaxSpeed(vehicle_id, speed)

    def _reset_vehicle(self, vehicle_id: str) -> None:
        """Hand the vehicle back to SUMO's car-following model by setting speed to -1."""
        if not self._vehicle_exists(vehicle_id):
            return
        traci.vehicle.setSpeed(vehicle_id, -1.0)

    def _safe_set_stop(self, vid: str, road_id: str, stop_pos: float) -> bool:
        """
        Queue a stop for a vehicle only if it is safe and not redundant.

        Two guard conditions prevent unnecessary or conflicting stop calls:
            1. The vehicle must not have already passed stop_pos on its lane.
            2. There must not already be a stop queued on this same edge,
               to avoid sending duplicate setStop calls that confuse SUMO.

        Returns True if a stop was actually issued, False otherwise.
        """
        try:
            current_pos = traci.vehicle.getLanePosition(vid)
            if current_pos >= stop_pos:
                return False  # already past the stop line, too late

            for stop in traci.vehicle.getNextStops(vid):
                if stop[0].startswith(road_id):
                    return False  # stop already queued on this edge

            traci.vehicle.setStop(vid, road_id, stop_pos)
            return True

        except traci.TraCIException:
            return False

    @staticmethod
    def _is_outgoing_edge(edge_id: str) -> bool:
        """
        Check if an edge leads outward toward a network border node.

        In our naming convention, border (exit) nodes start with N, S, E, or W.
        We parse the destination part of the edge ID to check this.

        Examples:
            "C_to_N"         -> destination "N"     -> outgoing (True)
            "J0_to_N0"       -> destination "N0"    -> outgoing (True)
            "J_0_0_to_S_0_0" -> destination "S_0_0" -> outgoing (True)
            "J0_to_J1"       -> destination "J1"    -> internal road (False)
        """
        if "_to_" not in edge_id:
            return False
        dest = edge_id.split("_to_")[-1]
        return bool(dest) and dest[0] in ("N", "S", "E", "W")