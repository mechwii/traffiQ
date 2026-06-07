# traffic_sim/agents/agent_call_manager.py
"""
AgentCallManager
================
Decides whether an agent needs to produce a new action this step.

An agent is re-triggered when:
  (a) A lane that was empty now has a leader -> new arrival.
  (b) A lane whose leader was GO now has a DIFFERENT vehicle ->
      the previous vehicle cleared; new one needs a decision.
  (c) The last action was all-stop -> re-evaluate every 3 steps
      so the deadlock guard can release vehicles quickly.
  (d) Anti-blocking: no decision in the last 15 steps (safety net).

Parameters
----------
intersection_id : int
    For logging only.
"""

from typing import Dict, Optional


class AgentCallManager:

    def __init__(self, intersection_id: int = 0):
        self.intersection_id   = intersection_id
        self._prev_leaders:    Dict[str, Optional[str]] = {}
        self._current_action:  Dict[str, int]           = {}
        self._steps_since_last_call: int = 0

    def needs_new_decision(
        self,
        current_leaders: Dict[str, Optional[str]],
    ) -> bool:
        """
        Return True if the agent should be called this step.

        Trigger conditions:
          0. Anti-blocking: no decision in the last 15 steps.
          0b. Deadlock fast-path: if the last action was all-stop,
              re-evaluate every 3 steps so the deadlock guard can
              release vehicles without waiting 15 seconds.
          1. A lane that was empty now has a vehicle.
          2. A lane that had a GO vehicle now has a DIFFERENT vehicle.
        """
        self._steps_since_last_call += 1

        # Condition 0: Safety anti-blocking
        if self._steps_since_last_call >= 15:
            return True

        # Condition 0b: all-stop deadlock fast-path
        # When the AI outputs all-zero, the state won't change on its own
        # (everything is frozen).  Re-evaluate every 3 steps so the
        # deadlock guard gets a chance to release one vehicle.
        if (self._current_action
                and all(v == 0 for v in self._current_action.values())
                and self._steps_since_last_call >= 3):
            return True

        # First call of the episode
        if not self._prev_leaders:
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
        leaders: Dict[str, Optional[str]],
        action:  Dict[str, int],
    ) -> None:
        """Record the leaders and action decided this step."""
        self._steps_since_last_call = 0
        self._prev_leaders   = dict(leaders)
        self._current_action = dict(action)

    def current_action(self) -> Dict[str, int]:
        """Return the last decided action (held between re-triggers)."""
        return dict(self._current_action)

    def reset(self) -> None:
        """Clear state at the start of a new episode."""
        self._prev_leaders.clear()
        self._current_action.clear()
        self._steps_since_last_call = 0