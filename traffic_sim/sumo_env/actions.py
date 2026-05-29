# traffic_sim/sumo_env/actions.py
"""
ActionHandler

This class applies traffic control actions to vehicles in the SUMO simulation via the TraCi API.

It implements the three action functions from the specification : 
    - set_action(actions) => apply speed / movement commands to vehicles
    - set_speedMode(mode) => configure how SUMO enforces speed limits
    - set_loaded_veh(leaders) => stop all non-leader vehicles (hold them back)

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
"""

from typing import Any, Dict, List, Optional, Union # for type management

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

class ActionHandler:
    """
    Applies control actions to vehicles in the running SUMO simulation

    Parameters :
        step_length : float
            Duration of one simulation step (seconds). Used when computing accelation limits for smooth speed changes.
    """

    def __init__(self, step_length: float = 1.0):
        self.step_length = step_length

    def set_action(
        self,
        actions: Union[Dict[str, float], List[Dict[str, Any]], Any]
    ) -> None :
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
                action_type = action.get("type", "speed") # We put speed as a default value
                value       = action.get("value", 0.0) # We put 0.0 as default speed

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
    ) -> None :
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
          22  (binary 10110) - disable safe-speed and right-of-way
                               (useful for RL agents that set precise speeds)
        
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
          - Leader vehicles (the frontmost in their lane) are given back control to SUMO's car-following model (speed = -1).
          - All other vehicles are commanded to stop (speed = 0).

        This is useful when a traffic management algorithm only needs to control leader vehicles, the rest wait behind them naturally.

        Parameters:
            leaders -> dict {lane_id: vehicle_id | None}
                Output of StateExtractor.get_leaders().  Maps lane IDs to
                the ID of the lead vehicle on that lane (or None).

        Example :       
            leaders = env.get_leaders()
            env.action_handler.set_loaded_veh(leaders)
        """

        # Firslty we build a set of vehicles IDs that are leaders
        leaders_ids = {vid for vid in leaders.values() if vid is not None}

        all_vehicles = list(traci.vehicle.getIDList())

        for vehicle_id in all_vehicles:
            if vehicle_id in leaders_ids:
                # We restore default car-following model for the leader
                self._reset_vehicle(vehicle_id)
            else:
                # Non-leader: hold in place
                self._apply_speed(vehicle_id, STOP_SPEED)    

    # ============ INTERNAL METHODS ============

    def _apply_speed(self, vehicle_id: str, speed: float) -> None :
        """
        Command a vehicule (with vehicle_id) to travel at speed m/s.

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

    @staticmethod
    def _vehicle_exists(vehicle_id: str) -> bool:
        """
        Return True if the vehicle is currently in the simulation.
        This method does not use any attribute of the class (and self) this is why this method is static
        """
        return vehicle_id in traci.vehicle.getIDList()