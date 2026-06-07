# traffic_sim/agents/intersection_agent.py
"""
IntersectionAgent
=================
Wraps the pre-trained CNN (DQN) model and adapts its output to the
{lane_id: 0 | 1} action format used by SumoEnvironment.

One agent is created per intersection (complex intersections are excluded).

How the model works
-------------------
  Input  : (image_size, image_size, 3)  uint8 RGB image
  Output : 256 Q-values   (one per discrete action 0..255)
  Action : argmax over Q-values  ->  integer 0..255
           converted to 8-bit binary -> 8 bits, each bit = go/stop for one slot

Bit layout (teacher's training order, mapped to our directions)
---------------------------------------------------------------
  bits[0] -> N leader    bits[1] -> N follower (ignored)
  bits[2] -> E leader    bits[3] -> E follower (ignored)
  bits[4] -> W leader    bits[5] -> W follower (ignored)
  bits[6] -> S leader    bits[7] -> S follower (ignored)

  If the network has fewer than 4 incoming directions (e.g. T-junction
  with 3 arms), the extra bits are silently ignored.

Safety filter (route-aware)
---------------------------
  After the model produces go/stop bits per direction, _safety_filter()
  removes conflicting simultaneous "go" decisions.

  The filter is ROUTE-AWARE: it queries each vehicle's route via TraCI
  to determine whether two vehicles' paths actually cross inside the
  junction.  This is less restrictive than the old direction-only check
  (which blocked ALL perpendicular pairs) and closer to the teacher's
  security_matrice.npy behaviour.

  Conflict rules (standard right-hand traffic):
    - Parallel directions (N+S or E+W) NEVER conflict when both go
      straight or both turn right.
    - Right turns generally don't conflict with perpendicular traffic
      (they merge rather than cross).
    - Left turns conflict with oncoming straight and perpendicular
      through-traffic.
    - When route information is unavailable, the filter falls back to
      the conservative direction-only check.

  Priority order: N > E > S > W  (matches teacher's convention).

Model sharing
-------------
  For multi-intersection networks (2, 4, 8 junctions), each junction
  gets its own IntersectionAgent, but they all share the SAME Keras
  model instance.  Use the class method ``create_agents()`` to build
  all agents with a single model load.

Usage
-----
    from traffic_sim.agents.intersection_agent import IntersectionAgent

    # Single intersection
    agent = IntersectionAgent(model_path="./save_model", image_size=50)

    # Multiple intersections — one model, shared
    agents = IntersectionAgent.create_agents(
        count=4, model_path="./save_model", image_size=50
    )
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    import tensorflow as tf
    import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

# Cardinal directions the model knows about
_DIRECTIONS = ["N", "S", "E", "W"]

# Bit index in the 8-bit decoded action for each direction's leader decision
_DIRECTION_BIT_INDEX: Dict[str, int] = {
    "N": 0,
    "E": 2,
    "W": 4,
    "S": 6,
}

# Direction pairs that can ALWAYS safely cross (parallel traffic, same axis)
_ALWAYS_SAFE_PAIRS = {
    frozenset({"N", "S"}),
    frozenset({"E", "W"}),
}

# Priority order for resolving conflicts (higher priority kept first)
_PRIORITY_ORDER = ["N", "E", "S", "W"]

# Opposite directions
_OPPOSITE: Dict[str, str] = {"N": "S", "S": "N", "E": "W", "W": "E"}

# Turn types: (entry_direction, exit_direction) -> "straight" | "left" | "right"
# For right-hand traffic (driving on the right side of the road):
#   Entering from N, exiting to S = straight
#   Entering from N, exiting to W = right turn
#   Entering from N, exiting to E = left turn
_TURN_TYPE: Dict[Tuple[str, str], str] = {
    ("N", "S"): "straight", ("N", "W"): "right", ("N", "E"): "left",
    ("S", "N"): "straight", ("S", "E"): "right", ("S", "W"): "left",
    ("E", "W"): "straight", ("E", "S"): "right", ("E", "N"): "left",
    ("W", "E"): "straight", ("W", "N"): "right", ("W", "S"): "left",
}

# Route-aware conflict table.
# Key: (turn_type_A, turn_type_B)  Value: True if they conflict.
# Two RIGHT turns never conflict (they merge into different lanes).
# A RIGHT + STRAIGHT is safe if they're not from the same axis.
# LEFT turns conflict with almost everything from perpendicular dirs.
_PERPENDICULAR_CONFLICT: Dict[Tuple[str, str], bool] = {
    ("right",    "right"):    False,  # both merging, no crossing
    ("right",    "straight"): False,  # right turn merges, doesn't cross
    ("right",    "left"):     True,   # left turn crosses right's exit path
    ("straight", "right"):    False,  # symmetric
    ("straight", "straight"): True,   # perpendicular straights cross
    ("straight", "left"):     True,   # left turn crosses straight
    ("left",     "right"):    True,   # symmetric
    ("left",     "straight"): True,   # symmetric
    ("left",     "left"):     True,   # both crossing, conflict
}

# Regex to strip the trailing lane index from a lane ID
_LANE_INDEX_RE = re.compile(r"_\d+$")


def _strip_lane_index(lane_or_edge: str) -> str:
    """Strip the trailing lane index from a lane ID to recover the edge ID."""
    return _LANE_INDEX_RE.sub("", lane_or_edge)


def _lane_to_direction(lane_id: str) -> Optional[str]:
    """
    Infer the cardinal direction (N/S/E/W) of an INCOMING lane.

    Returns None for outgoing lanes, internal lanes, or unrecognised formats.
    Handles single-intersection, linear-chain, and grid topologies.
    """
    if "_to_" not in lane_id:
        return None

    origin, dest_raw = lane_id.split("_to_", 1)
    dest_edge = _strip_lane_index(dest_raw)

    # 1. Border node origin (single intersection: "N_to_C", multi: "N0_to_J0")
    for d in _DIRECTIONS:
        if origin == d:
            return d
        if len(origin) > 1 and origin[0] == d and (origin[1].isdigit() or origin[1] == "_"):
            return d

    # 2. Internal road between two junctions
    if not (origin.startswith("J") and dest_edge.startswith("J")):
        return None

    def _extract_nums(jid: str) -> List[int]:
        inner = jid[1:].lstrip("_")
        return [int(s) for s in inner.replace("_", " ").split() if s.isdigit()]

    o_nums = _extract_nums(origin)
    d_nums = _extract_nums(dest_edge)

    # Linear chain: J0 -> J1  (single index)
    if len(o_nums) == 1 and len(d_nums) == 1:
        if o_nums[0] < d_nums[0]: return "W"
        if o_nums[0] > d_nums[0]: return "E"

    # Grid: J_0_0 -> J_0_1  (row, col indices)
    elif len(o_nums) == 2 and len(d_nums) == 2:
        o_row, o_col = o_nums
        d_row, d_col = d_nums
        if o_col < d_col: return "W"
        if o_col > d_col: return "E"
        if o_row < d_row: return "S"
        if o_row > d_row: return "N"

    # Fallback: could not determine direction
    return None


def _get_vehicle_exit_direction(vid: str) -> Optional[str]:
    """
    Determine the exit direction of a vehicle from its route.

    Looks at the LAST edge in the vehicle's route and infers the
    border-node direction from its name.

    Returns one of "N", "S", "E", "W" or None if undetermined.
    """
    if not TRACI_AVAILABLE:
        return None
    try:
        route = traci.vehicle.getRoute(vid)
        if not route:
            return None
        last_edge = route[-1]
        # Last edge goes toward a border node: "C_to_N", "J0_to_S0", etc.
        if "_to_" in last_edge:
            dest_node = last_edge.split("_to_")[-1]
            if dest_node and dest_node[0] in ("N", "S", "E", "W"):
                return dest_node[0]
    except Exception:
        pass
    return None


def _int_to_bits(action_int: int, n_bits: int = 8) -> List[int]:
    """
    Convert integer 0..255 to a list of n_bits bits, MSB first.
    Mirrors the teacher's trad_action().
    """
    bits = []
    for _ in range(n_bits):
        bits.append(action_int % 2)
        action_int //= 2
    return list(reversed(bits))


class IntersectionAgent:
    """
    One DQN agent controlling a single (non-complex) intersection.

    Parameters
    ----------
    model_path      : str        path to SavedModel directory
    image_size      : int        input image side length in pixels
    intersection_id : int        index of this intersection (logging only)
    shared_model    : keras.Model | None
        If provided, this model is used directly instead of loading from
        disk.  Use create_agents() to build multiple agents that share
        one model.
    """

    def __init__(
        self,
        model_path: str = "./save_model",
        image_size: int = 50,
        intersection_id: int = 0,
        shared_model=None,
    ):
        if not TF_AVAILABLE:
            raise EnvironmentError(
                "TensorFlow / Keras is required for IntersectionAgent.\n"
                "Install with:  pip install tensorflow"
            )

        self.image_size      = image_size
        self.intersection_id = intersection_id
        self.model_path      = model_path

        if shared_model is not None:
            self.model = shared_model
            print(f"[IntersectionAgent #{intersection_id}] Using shared model.")
        else:
            print(f"[IntersectionAgent #{intersection_id}] Loading model from '{model_path}' ...")
            self.model = keras.models.load_model(model_path)
            self.model.summary()
            print(f"[IntersectionAgent #{intersection_id}] Model loaded.")

    # ------------------------------------------------------------------ #
    #  Factory method for multi-agent setups (shared model)                #
    # ------------------------------------------------------------------ #

    @classmethod
    def create_agents(
        cls,
        count: int,
        model_path: str = "./save_model",
        image_size: int = 50,
    ) -> List["IntersectionAgent"]:
        """
        Create *count* agents that share a single model instance.

        This avoids loading the same weights from disk N times and
        uses less memory.
        """
        print(f"[IntersectionAgent] Loading shared model from '{model_path}' ...")
        shared = keras.models.load_model(model_path)
        shared.summary()
        print(f"[IntersectionAgent] Creating {count} agent(s) with shared weights.")

        return [
            cls(
                model_path=model_path,
                image_size=image_size,
                intersection_id=i,
                shared_model=shared,
            )
            for i in range(count)
        ]

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Dict[str, int]:
        """
        Choose a go/stop action for every incoming lane at this intersection.

        Returns {lane_id: 0 | 1}  (1=go, 0=stop).

        Processing pipeline:
          1. CNN forward pass -> integer action 0..255
          2. Decode to 8-bit binary -> per-direction go/stop
          3. Map direction bits to lane IDs
          4. Safety filter: remove conflicting "go" pairs (route-aware)
          5. Deadlock guard: if all lanes stopped for too long, release
             the longest-waiting leader
        """
        image   = obs["image"]
        leaders = obs["leaders"]

        # 1. CNN inference
        action_int = self._predict(image)
        bits       = _int_to_bits(action_int, n_bits=8)

        # 2. Direction bits (only leader bits, indices 0,2,4,6)
        direction_bits: Dict[str, int] = {
            d: bits[idx] for d, idx in _DIRECTION_BIT_INDEX.items()
        }

        # 3. Map to lane IDs
        action: Dict[str, int] = {}
        for lane_id, vehicle_id in leaders.items():
            if vehicle_id is None:
                continue
            direction = _lane_to_direction(lane_id)
            if direction is None:
                continue
            action[lane_id] = direction_bits[direction]

        # 4. Safety filter: prevent conflicting "go" pairs (route-aware)
        action = self._safety_filter(action, leaders)

        # 5. Deadlock guard: if ALL are stopped and someone is waiting
        #    too long, release the longest-waiting leader
        if action and all(v == 0 for v in action.values()):
            action = self._deadlock_guard(action, leaders)

        return action

    # ------------------------------------------------------------------ #
    #  Route-aware safety filter                                           #
    # ------------------------------------------------------------------ #

    def _safety_filter(
        self,
        action: Dict[str, int],
        leaders: Dict[str, Optional[str]],
    ) -> Dict[str, int]:
        """
        Remove conflicting go-actions using route-aware conflict detection.

        For each pair of "go" directions, the filter:
          1. Checks if they are parallel (N+S or E+W) -> always safe.
          2. For perpendicular pairs, queries each vehicle's route via
             TraCI to determine turn type (straight, left, right).
          3. Uses the conflict table to decide if the pair is safe.
          4. Falls back to conservative blocking if route info is
             unavailable.

        This is less restrictive than the old direction-only check
        (which blocked ALL perpendicular pairs) and closer to the
        teacher's security_matrice.npy behaviour.
        """
        # Group "go" lanes by direction, with their vehicle IDs
        go_by_dir: Dict[str, List[Tuple[str, Optional[str]]]] = {}
        for lane_id, go_val in action.items():
            if go_val == 1:
                d = _lane_to_direction(lane_id)
                if d:
                    vid = leaders.get(lane_id)
                    go_by_dir.setdefault(d, []).append((lane_id, vid))

        if len(go_by_dir) <= 1:
            return action   # 0 or 1 direction active -> no conflict

        # Greedily accept directions in priority order
        accepted: Dict[str, str] = {}  # direction -> turn_type

        for d in _PRIORITY_ORDER:
            if d not in go_by_dir:
                continue

            # Determine turn type for this direction's leader
            turn_type = self._get_turn_type(d, go_by_dir[d])

            # Check against all already-accepted directions
            is_safe = True
            for accepted_d, accepted_turn in accepted.items():
                if not self._pair_is_safe(d, turn_type, accepted_d, accepted_turn):
                    is_safe = False
                    break

            if is_safe:
                accepted[d] = turn_type

        # Set rejected directions to STOP
        for d, lane_entries in go_by_dir.items():
            if d not in accepted:
                for lid, _ in lane_entries:
                    action[lid] = 0

        return action

    def _get_turn_type(
        self,
        entry_dir: str,
        lane_entries: List[Tuple[str, Optional[str]]],
    ) -> str:
        """
        Determine the turn type for a direction's leader vehicle.

        Queries TraCI for the vehicle's route to find the exit direction.
        Falls back to "straight" (most conservative for conflict checking)
        if route info is unavailable.
        """
        for _, vid in lane_entries:
            if vid is None:
                continue
            exit_dir = _get_vehicle_exit_direction(vid)
            if exit_dir is not None and exit_dir != entry_dir:
                turn = _TURN_TYPE.get((entry_dir, exit_dir))
                if turn is not None:
                    return turn
        # Fallback: assume straight (most likely to conflict)
        return "straight"

    @staticmethod
    def _pair_is_safe(
        dir_a: str, turn_a: str,
        dir_b: str, turn_b: str,
    ) -> bool:
        """
        Return True if two directions can safely cross simultaneously.

        Parallel directions (N+S, E+W) are always safe.
        For perpendicular directions, uses the route-aware conflict table.
        """
        pair = frozenset({dir_a, dir_b})

        # Parallel directions: always safe
        if pair in _ALWAYS_SAFE_PAIRS:
            return True

        # Perpendicular directions: check turn-type conflict table
        conflicts = _PERPENDICULAR_CONFLICT.get((turn_a, turn_b))
        if conflicts is not None:
            return not conflicts

        # Unknown turn combo — block to be safe
        return False

    # ------------------------------------------------------------------ #
    #  Deadlock guard                                                      #
    # ------------------------------------------------------------------ #

    def _deadlock_guard(
        self,
        action: Dict[str, int],
        leaders: Dict[str, Optional[str]],
    ) -> Dict[str, int]:
        """
        If ALL lanes are stopped (all-zero action), immediately release
        the leader that has been waiting the longest.

        Why no threshold: when vehicles are stopped via setStop(), SUMO
        treats it as a "planned stop" and getAccumulatedWaitingTime()
        may not increase.  So we use getSpeed() < 0.1 to confirm the
        vehicle is actually stopped, and release unconditionally.

        This fires every time the AI outputs all-zero, which is fine:
        the AgentCallManager will re-trigger more frequently during
        deadlocks (every 3 steps) so the released vehicle has time
        to cross before the next decision.
        """
        best_lane      = None
        best_wait      = -1.0
        any_stopped    = False

        for lid in action:
            vid = leaders.get(lid)
            if vid is None:
                continue
            try:
                speed = traci.vehicle.getSpeed(vid)
                if speed < 0.1:
                    any_stopped = True
                wait = traci.vehicle.getAccumulatedWaitingTime(vid)
                if wait < 0.1:
                    wait = 1.0   # setStop vehicles may report 0
            except Exception:
                wait = 1.0

            if wait > best_wait:
                best_wait = wait
                best_lane = lid

        if any_stopped and best_lane is not None:
            print(
                f"[DEADLOCK GUARD #{self.intersection_id}] "
                f"All stopped — releasing {best_lane}."
            )
            action[best_lane] = 1

        return action

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _predict(self, image: np.ndarray) -> int:
        """Run CNN forward pass, return greedy action integer 0-255."""
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        state_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        return int(tf.argmax(action_probs[0]).numpy())

    def __repr__(self) -> str:
        return (
            f"IntersectionAgent(id={self.intersection_id}, "
            f"model='{self.model_path}', image_size={self.image_size})"
        )