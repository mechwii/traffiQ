# traffic_sim/env/actions.py
"""
ActionHandler

This class applies traffic control actions to vehicles in the SUMO simulation via the TraCi API.

It implements the three action functions from the specification : 
    - set_action(actions) => apply speed / movement commands to vehicles
    - set_speedMode(mode) => configure how SUMO enforces speed limits
    - set_loaded_veh(leaders) => stop all non-leader vehicles (hold them back)

It also provides the binary go/stop action logic used by the DQN agent:
    - apply_binary_action(action, leaders) => setStop-based go/stop for leaders
    - stop_non_leaders(incoming_leaders)    => hold followers behind intersection
    - set_speed_modes_loaded(mode)         => speed mode for newly loaded vehicles
    - stop_departed_non_leaders(leaders)    => catch vehicles entering mid-step

TraCi calls used (with explanations) :

    traci.vehicle.setSpeed(vid, speed)
      Force a vehicle to a specific speed (m/s).
      Use -1 to hand control back to SUMO's default car-following model.

    traci.vehicle.setSpeedMode(vid, mode):
        Integer bitmask that controls which SUMO speed checks are active.
        See set_speedMode() docstring for bit definitions.

    traci.vehicle.setMaxSpeed(vid, speed)
        Override the vehicle's maximum allowed speed.
    
    traci.vehicle.getLaneID(vid)
        Get the lane a vehicle is currently on.
    
    traci.vehicle.getIDList()
        All vehicles currently in the simulation.

    traci.vehicle.setStop(vid, edgeID, pos, ...)
        Queue a stop for a vehicle on a given edge at a given position.

    traci.vehicle.getNextStops(vid)
        Return the list of upcoming stops for a vehicle.

    traci.vehicle.setColor(vid, (R, G, B, A))
        Change the visual colour of a vehicle in sumo-gui.
"""

from typing import Any, Dict, List, Optional, Union  # for type management

# We safely import traci depending on if the environment has been rightly set up
try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

# Default speed to assign when stopping a vehicle (m/s)
# 0.0 = full stop -> use a very small positive value to keep the vehicle
# "loaded" (in the network) but stationary.
STOP_SPEED = 0.0

# Speed mode that gives full control to our code (disables all SUMO checks)
SPEED_MODE_FULL_CONTROL = 0

# Default speed mode that re-enables all SUMO safety checks
SPEED_MODE_DEFAULT = 31   # binary 11111: all checks active

# Speed mode used by the AI agent: 55 = binary 110111
# Disables junction right-of-way (bit 3) so the AI controls who crosses.
SPEED_MODE_AI_CONTROL = 55


class ActionHandler:
    """
    Applies control actions to vehicles in the running SUMO simulation.

    This is the SINGLE place where vehicle actions are applied.
    SumoEnvironment delegates ALL action logic here.

    Parameters :
        step_length : float
            Duration of one simulation step (seconds). Used when computing
            acceleration limits for smooth speed changes.
    """

    def __init__(self, step_length: float = 1.0):
        self.step_length = step_length
        # Cached vehicle ID set — call refresh_cache() once per step
        self._vehicle_cache: Optional[set] = None

    # ================================================================== #
    #  Cache management                                                    #
    # ================================================================== #

    def refresh_cache(self) -> None:
        """
        Refresh the cached set of active vehicle IDs.

        Must be called ONCE at the start of each step, before any other
        method.  This avoids calling traci.vehicle.getIDList() on every
        single _vehicle_exists() check (previously O(n) per check).
        """
        self._vehicle_cache = set(traci.vehicle.getIDList())

    def _vehicle_exists(self, vehicle_id: str) -> bool:
        """
        Return True if the vehicle is currently in the simulation.

        Uses the cached vehicle set if available; falls back to a live
        TraCI query if refresh_cache() has not been called this step.
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
        Apply control actions to one or more vehicles.

        The method accepts three formats so it can work with different
        control paradigms (manual scripts, RL agents, rule-based logic):

        Format 1 - Speed dict (most common):
            actions = {"veh_0": 8.0, "veh_1": 5.5, "veh_3": 0.0}
            Each key is a vehicle ID; the value is the target speed in m/s.
            Use -1.0 as the speed to return control to SUMO's default model.

        Format 2 - List of actions dicts:
            actions = [
                {"vehicle_id": "veh_0", "type": "speed",     "value": 8.0},
                {"vehicle_id": "veh_1", "type": "max_speed", "value": 10.0},
                {"vehicle_id": "veh_2", "type": "reset"},
            ]
            Supported types :
              - "speed" => set exact speed (m/s)
              - "max_speed" => change maximum allowed speed
              - "reset" => hand control back to SUMO car-following model

        Parameters :
            actions : dict | list | None
                Control commands (see formats above).
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
        Set the speed mode for one or more vehicles.

        SUMO's speed mode is an integer bitmask.  Each bit enables or
        disables a specific speed-safety check:

          Bit 0 (value  1): Regard safe speed (car-following model limit)
          Bit 1 (value  2): Regard maximum acceleration
          Bit 2 (value  4): Regard maximum deceleration
          Bit 3 (value  8): Regard right-of-way at junctions
          Bit 4 (value 16): Brake hard at red lights / stop lines

        Common presets:
          31  (binary 11111) - default: all checks active
           0  (binary 00000) - full external control: no SUMO checks
          55  (binary 110111) - disable junction right-of-way (AI control)

        Parameters :
            vehicle_ids : list[str] | None
                IDs of vehicles to update.  If None, all vehicles currently
                in the simulation are updated.

            mode : int
                Speed mode bitmask (default 31 = all SUMO checks active).
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
        Stop all vehicles that are NOT the lane leader.

        This function implements a "leader-follower" control strategy:
          - Leader vehicles (the frontmost in their lane) are given back
            control to SUMO's car-following model (speed = -1).
          - All other vehicles are commanded to stop (speed = 0).

        Parameters:
            leaders -> dict {lane_id: vehicle_id | None}
                Output of StateExtractor.get_leaders().  Maps lane IDs to
                the ID of the lead vehicle on that lane (or None).
        """
        leaders_ids = {vid for vid in leaders.values() if vid is not None}

        all_vehicles = list(traci.vehicle.getIDList())

        for vehicle_id in all_vehicles:
            if vehicle_id in leaders_ids:
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
        Apply a discrete {lane_id: 0|1} action to leader vehicles.

        Uses traci.vehicle.setStop() to match the teacher's reference:
          - action=0 -> queue a stop 10 m before edge end
          - action=1 -> release any existing stop, return to SUMO control

        Parameters
        ----------
        action  : {lane_id: 0|1}
        leaders : {lane_id: vehicle_id|None}  (pre-queried, NOT re-fetched)
        """
        for lane_id, go in action.items():
            vehicle_id = leaders.get(lane_id)
            if vehicle_id is None:
                continue

            try:
                road_id = traci.vehicle.getRoadID(vehicle_id)
                if road_id.startswith(":"):
                    continue

                lane   = traci.vehicle.getLaneID(vehicle_id)
                length = traci.lane.getLength(lane)
                stop_pos = max(1.0, length - 10.0)

                if go == 1:
                    # Release: remove existing stop, hand back to SUMO
                    traci.vehicle.setColor(vehicle_id, (0, 255, 0))
                    if traci.vehicle.getNextStops(vehicle_id):
                        traci.vehicle.setStop(
                            vehicle_id, road_id, stop_pos, duration=0
                        )
                    traci.vehicle.setSpeed(vehicle_id, -1)
                else:
                    # Stop before intersection
                    traci.vehicle.setColor(vehicle_id, (255, 0, 0))
                    self._safe_set_stop(vehicle_id, road_id, stop_pos)

            except traci.TraCIException:
                pass

    def stop_non_leaders(
        self,
        incoming_leaders: Dict[str, Optional[str]],
    ) -> None:
        """
        Stop ALL active non-leader vehicles on incoming edges.

        Called once per step before actions are applied.  Non-leaders are
        stopped 10 m before the edge end via setStop().  Vehicles on
        internal edges (inside junction) or outgoing edges (past the
        intersection) are left alone.

        Matches the teacher's set_loaded_vehicle().
        """
        leader_vids = {vid for vid in incoming_leaders.values()
                       if vid is not None}

        for vid in traci.vehicle.getIDList():
            if vid in leader_vids:
                continue
            try:
                road_id = traci.vehicle.getRoadID(vid)
                if road_id.startswith(":"):
                    continue
                if self._is_outgoing_edge(road_id):
                    continue

                lane   = traci.vehicle.getLaneID(vid)
                length = traci.lane.getLength(lane)
                stop_pos = max(1.0, length - 10.0)
                if self._safe_set_stop(vid, road_id, stop_pos):
                    traci.vehicle.setColor(vid, (255, 200, 0))
            except traci.TraCIException:
                pass

    def stop_departed_non_leaders(
        self,
        incoming_leaders: Dict[str, Optional[str]],
    ) -> None:
        """
        Stop vehicles that just departed this step and are not leaders.

        Called AFTER simulationStep() to catch new arrivals that entered
        the network during this step.
        """
        leader_vids = {vid for vid in incoming_leaders.values()
                       if vid is not None}

        for vid in traci.simulation.getDepartedIDList():
            if vid in leader_vids:
                continue
            try:
                road_id = traci.vehicle.getRoadID(vid)
                if road_id.startswith(":"):
                    continue
                lane   = traci.vehicle.getLaneID(vid)
                length = traci.lane.getLength(lane)
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
        Set speed mode for all newly loaded vehicles.

        Mode 55 = binary 110111 = disable junction right-of-way (bit 3).
        This gives the AI agent full control over who crosses.
        Matches the teacher's set_speed_mode().
        """
        for vid in traci.simulation.getLoadedIDList():
            try:
                traci.vehicle.setSpeedMode(vid, mode)
            except traci.TraCIException:
                pass

    # ================================================================== #
    #  Internal helpers                                                    #
    # ================================================================== #

    def _apply_speed(self, vehicle_id: str, speed: float) -> None:
        """
        Command a vehicle (with vehicle_id) to travel at speed m/s.

        Passing speed = -1.0 returns control to SUMO's car-following model.
        Negative speeds other than -1 are clamped to 0.
        """
        if not self._vehicle_exists(vehicle_id):
            return
        if speed < 0 and speed != -1.0:
            speed = 0.0
        traci.vehicle.setSpeed(vehicle_id, speed)

    def _apply_max_speed(self, vehicle_id: str, speed: float) -> None:
        """Set the maximum speed for vehicle_id."""
        if not self._vehicle_exists(vehicle_id):
            return
        if speed < 0:
            speed = 0.0
        traci.vehicle.setMaxSpeed(vehicle_id, speed)

    def _reset_vehicle(self, vehicle_id: str) -> None:
        """
        Return vehicle_id to SUMO's default car-following model
        by setting its speed to -1 (SUMO special value for "auto").
        """
        if not self._vehicle_exists(vehicle_id):
            return
        traci.vehicle.setSpeed(vehicle_id, -1.0)

    def _safe_set_stop(
        self, vid: str, road_id: str, stop_pos: float
    ) -> bool:
        """
        Set a stop for *vid* at *stop_pos* on *road_id*, but ONLY if:
          1. The vehicle hasn't already passed that position.
          2. The vehicle doesn't already have a stop queued on this edge.

        Returns True if the stop was successfully set.
        """
        try:
            current_pos = traci.vehicle.getLanePosition(vid)
            if current_pos >= stop_pos:
                return False
            
            # --- AJOUT : Calcul du point de non-retour ---
            speed = traci.vehicle.getSpeed(vid)
            # Distance de freinage = v^2 / (2 * decel). 
            # Decel est de 4.5 dans ton demand_generator.
            braking_distance = (speed ** 2) / (2 * 4.5)
            
            # Si la distance nécessaire pour freiner dépasse l'espace disponible,
            # on refuse l'ordre de stop (la voiture passe "au orange")
            if current_pos + braking_distance > stop_pos:
                return False

            for stop in traci.vehicle.getNextStops(vid):
                if stop[0].startswith(road_id):
                    return False

            traci.vehicle.setStop(vid, road_id, stop_pos)
            return True

        except traci.TraCIException:
            return False

    @staticmethod
    def _is_outgoing_edge(edge_id: str) -> bool:
        """
        Return True if edge_id goes toward a border node.

        Border nodes start with N/S/E/W.  Outgoing edges have one of
        these as their destination:
            "C_to_N"          -> dest "N"       -> outgoing
            "J0_to_N0"        -> dest "N0"      -> outgoing
            "J_0_0_to_S_0_0"  -> dest "S_0_0"   -> outgoing
            "J0_to_J1"        -> dest "J1"      -> NOT outgoing
        """
        if "_to_" not in edge_id:
            return False
        dest = edge_id.split("_to_")[-1]
        return bool(dest) and dest[0] in ("N", "S", "E", "W")