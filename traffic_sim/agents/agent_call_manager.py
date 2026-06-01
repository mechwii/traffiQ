from typing import Dict, Optional


class AgentCallManager:
    """
    Decides whether an agent needs to produce a new action this step.
 
    An agent is re-triggered when:
      (a) A lane that was empty last step now has a leader -> a new vehicle
          has arrived at the intersection and needs a decision.
      (b) A lane whose leader was told GO (action=1) now has a DIFFERENT
          vehicle as its leader  -> the previous vehicle cleared the
          intersection, so we reassess.
 
    The current action is held (replayed) on every step where neither
    condition fires, so the vehicles keep their current go/stop command.
 
    Parameters
    ----------
    intersection_id : int
        For logging only.
    """
 
    def __init__(self, intersection_id: int = 0):
        self.intersection_id   = intersection_id
        # lane_id -> vehicle_id that was the leader last time we evaluated
        self._prev_leaders:    Dict[str, Optional[str]] = {}
        # lane_id -> 0|1  last decision
        self._current_action:  Dict[str, int]           = {}

        # To avoid the simulation block on a step
        self._steps_since_last_call: int = 0
 
    def needs_new_decision(
        self,
        current_leaders: Dict[str, Optional[str]],
    ) -> bool:
        """
        Return True if the agent should be called this step.
 
        Trigger conditions
        ------------------
        1. A lane that was None (empty) now has a vehicle.
        2. A lane that had a vehicle the agent allowed to GO now has a
           DIFFERENT vehicle (the old one crossed; new one needs a decision).
        """
        self._steps_since_last_call += 1 

        # Condition 0: Security anti blocking (each 15 secondes)
        if self._steps_since_last_call >= 15:
            return True
        
        for lane_id, vid in current_leaders.items():
            prev_vid    = self._prev_leaders.get(lane_id)
            prev_action = self._current_action.get(lane_id, 0)
 
            # Condition 1: new arrival on a previously empty lane
            if vid is not None and prev_vid is None:
                return True
 
            # Condition 2: vehicle changed on a lane that was set to GO
            if vid is not None and prev_vid is not None:
                if vid != prev_vid and prev_action == 1:
                    return True
 
        return False
 
    def update(
        self,
        leaders:        Dict[str, Optional[str]],
        action:         Dict[str, int],
    ) -> None:
        """Record the leaders and action decided this step."""
        self.reset_step_since_last_call()
        self._prev_leaders   = dict(leaders)
        self._current_action = dict(action)
 
    def current_action(self) -> Dict[str, int]:
        """Return the last decided action (held between re-triggers)."""
        return dict(self._current_action)
 
    def reset_step_since_last_call(self):
        self._steps_since_last_call = 0

    def reset(self) -> None:
        """Clear state at the start of a new episode."""
        self._prev_leaders.clear()
        self._current_action.clear()
        self._steps_since_last_call = 0