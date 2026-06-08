# traffic_sim/agents/agent_call_manager.py
"""
AgentCallManager

Decides whether an agent needs to produce a new action this step or can
simply replay the decision it already made.

The idea is to avoid calling the CNN every single step (which would be wasteful
and cause stop/go oscillation). Instead, we only re-trigger the agent when
something meaningful has actually changed at its intersection.

An agent is re-triggered when any of these conditions are met:
    (a) A lane that was empty now has a leader -> a new vehicle arrived and
        needs a decision.
    (b) A lane whose leader was set to GO now has a DIFFERENT vehicle -> the
        previous vehicle cleared the intersection, a new one is waiting.
    (c) The last action was all-stop -> we re-evaluate every 3 steps so the
        deadlock guard inside IntersectionAgent gets a chance to release a vehicle.
    (d) No decision has been made in the last 15 steps -> safety net to prevent
        the intersection from being completely frozen by an edge case.

Args:
    intersection_id (int): Used only for log messages.
"""

from typing import Dict, Optional


class AgentCallManager:

    def __init__(self, intersection_id: int = 0):
        self.intersection_id         = intersection_id
        self._prev_leaders:          Dict[str, Optional[str]] = {}
        self._current_action:        Dict[str, int]           = {}
        self._steps_since_last_call: int                      = 0

    def needs_new_decision(
        self,
        current_leaders: Dict[str, Optional[str]],
    ) -> bool:
        """
        Return True if the agent should be called this step.

        Checks all trigger conditions in priority order and returns True
        on the first one that fires. If none fire, the current action is
        replayed without touching the model.

        Args:
            current_leaders: The latest {lane_id: vehicle_id | None} snapshot.

        Returns:
            True if the agent should produce a new decision, False to replay.
        """
        self._steps_since_last_call += 1

        # Condition (d): anti-blocking safety net.
        # If 15 steps have passed without a new decision, something probably went
        # wrong (e.g. a TraCI exception silently swallowed a trigger). Force a call.
        if self._steps_since_last_call >= 15:
            return True

        # Condition (c): deadlock fast-path.
        # When the AI outputs all-zero, the intersection is frozen and the state
        # will never change on its own. Re-evaluate every 3 steps so the deadlock
        # guard inside IntersectionAgent gets a chance to release a vehicle quickly.
        if (self._current_action
                and all(v == 0 for v in self._current_action.values())
                and self._steps_since_last_call >= 3):
            return True

        # Always trigger on the very first call of the episode (no previous state).
        if not self._prev_leaders:
            return True

        for lane_id, vid in current_leaders.items():
            prev_vid    = self._prev_leaders.get(lane_id)
            prev_action = self._current_action.get(lane_id, 0)

            # Condition (a): a new vehicle arrived on a previously empty lane.
            if vid is not None and prev_vid is None:
                return True

            # Condition (b): the vehicle on a GO lane changed, meaning the
            # previous leader cleared the intersection and a new one is waiting.
            if vid is not None and prev_vid is not None:
                if vid != prev_vid and prev_action == 1:
                    return True

        return False

    def update(
        self,
        leaders: Dict[str, Optional[str]],
        action:  Dict[str, int],
    ) -> None:
        """
        Record the leaders and action from the current step.

        Called right after the agent produces a new decision so the manager
        has a fresh baseline to compare against next step.
        """
        self._steps_since_last_call = 0
        self._prev_leaders          = dict(leaders)
        self._current_action        = dict(action)

    def current_action(self) -> Dict[str, int]:
        """Return a copy of the last decided action to replay for this step."""
        return dict(self._current_action)

    def reset(self) -> None:
        """Clear all state at the start of a new episode."""
        self._prev_leaders.clear()
        self._current_action.clear()
        self._steps_since_last_call = 0