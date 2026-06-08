# traffic_sim/agents/intersection_agent.py
"""
IntersectionAgent

Wraps the pre-trained CNN (DQN) model and adapts its output to the
{lane_id: 0 | 1} action format used by SumoEnvironment.

One agent is created per intersection (complex intersections are excluded because
the model was only trained on four-direction layouts).

How the model works:
    Input  : (image_size, image_size, 3) uint8 RGB image of the intersection.
    Output : 256 Q-values, one per discrete action (0 to 255).
    Action : argmax over Q-values gives an integer 0-255.
             That integer is decoded to 8 bits, one bit per control slot.

Bit layout (teacher's training order):
    bits[0] -> North leader    bits[1] -> North follower (ignored by us)
    bits[2] -> East leader     bits[3] -> East follower (ignored)
    bits[4] -> West leader     bits[5] -> West follower (ignored)
    bits[6] -> South leader    bits[7] -> South follower (ignored)

    We only use bits 0, 2, 4, 6 (the leader bits). The follower bits were
    used internally by the teacher's training environment but are not relevant
    here since we control followers separately via stop_non_leaders().

Safety filter (route-aware):
    After decoding the bits, _safety_filter() checks for conflicting simultaneous
    "go" decisions. It queries each vehicle's route via TraCI to determine the
    actual turn type (straight, left, right) and uses a conflict table to decide
    whether two directions can safely cross at the same time. This is less
    restrictive than blocking all perpendicular pairs and closer to the teacher's
    security_matrice.npy behavior.

    Priority order for conflict resolution: N > E > S > W.

Deadlock guard:
    If the model outputs all-zero (everyone stop), the deadlock guard immediately
    releases the lane whose leader has been waiting the longest. This prevents the
    simulation from freezing when the model gets stuck in a bad local decision.

Model sharing:
    For multi-intersection setups, all agents share the same Keras model instance
    loaded once from disk. Use the create_agents() class method to build them all
    at once and avoid loading the same weights multiple times.

Usage:
    agent = IntersectionAgent(model_path="./save_model", image_size=50)
    agents = IntersectionAgent.create_agents(count=4, model_path="./save_model")
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

# The four cardinal directions the model knows about.
_DIRECTIONS = ["N", "S", "E", "W"]

# Which bit index in the 8-bit decoded action corresponds to each direction's leader.
# Bits 0, 2, 4, 6 are the leader slots. Odd bits are follower slots (unused here).
_DIRECTION_BIT_INDEX: Dict[str, int] = {
    "N": 0,
    "E": 2,
    "W": 4,
    "S": 6,
}

# Parallel direction pairs that can ALWAYS safely cross at the same time.
# North + South and East + West never have crossing paths when going straight.
_ALWAYS_SAFE_PAIRS = {
    frozenset({"N", "S"}),
    frozenset({"E", "W"}),
}

# Priority order used when resolving conflicting "go" decisions.
# North is checked first, West last.
_PRIORITY_ORDER = ["N", "E", "S", "W"]

# Mapping from a direction to its opposite.
_OPPOSITE: Dict[str, str] = {"N": "S", "S": "N", "E": "W", "W": "E"}

# Turn classification table: (entry_direction, exit_direction) -> turn type.
# Based on standard right-hand traffic rules (driving on the right side of the road).
# Example: entering from N and exiting to S = straight through.
#          entering from N and exiting to W = right turn.
#          entering from N and exiting to E = left turn.
_TURN_TYPE: Dict[Tuple[str, str], str] = {
    ("N", "S"): "straight", ("N", "W"): "right",    ("N", "E"): "left",
    ("S", "N"): "straight", ("S", "E"): "right",    ("S", "W"): "left",
    ("E", "W"): "straight", ("E", "S"): "right",    ("E", "N"): "left",
    ("W", "E"): "straight", ("W", "N"): "right",    ("W", "S"): "left",
}

# Route-aware conflict table for perpendicular direction pairs.
# Key: (turn_type_of_A, turn_type_of_B). Value: True if they conflict.
# Two right turns never conflict because both vehicles merge without crossing.
# Left turns conflict with almost everything from perpendicular directions.
_PERPENDICULAR_CONFLICT: Dict[Tuple[str, str], bool] = {
    ("right",    "right"):    False,
    ("right",    "straight"): False,
    ("right",    "left"):     True,
    ("straight", "right"):    False,
    ("straight", "straight"): True,
    ("straight", "left"):     True,
    ("left",     "right"):    True,
    ("left",     "straight"): True,
    ("left",     "left"):     True,
}

# Regex to strip the trailing lane index (e.g. "_0", "_1") from a lane ID.
_LANE_INDEX_RE = re.compile(r"_\d+$")


def _strip_lane_index(lane_or_edge: str) -> str:
    """Remove the trailing lane index from a lane ID to get the parent edge ID.
    Example: "C_to_N_0" -> "C_to_N"
    """
    return _LANE_INDEX_RE.sub("", lane_or_edge)


def _lane_to_direction(lane_id: str) -> Optional[str]:
    """
    Infer the cardinal direction (N/S/E/W) of an incoming lane from its ID.

    Returns None for outgoing lanes, internal junction lanes, or formats we
    don't recognise. Handles single-intersection, linear-chain, and grid topologies.

    Single intersection example:  "N_to_C_0" -> origin is "N" -> returns "N"
    Multi-intersection example:   "N0_to_J0_0" -> origin is "N0" -> returns "N"
    Grid internal road example:   "J_0_0_to_J_0_1_0" -> infers direction from coords
    """
    if "_to_" not in lane_id:
        return None

    origin, dest_raw = lane_id.split("_to_", 1)
    dest_edge = _strip_lane_index(dest_raw)

    # Case 1: origin is a border node directly (e.g. "N", "S", "N0", "W_0_0").
    for d in _DIRECTIONS:
        if origin == d:
            return d
        if len(origin) > 1 and origin[0] == d and (origin[1].isdigit() or origin[1] == "_"):
            return d

    # Case 2: internal road between two junction nodes (e.g. J0 -> J1 in a chain).
    # We infer the direction from the relative positions encoded in the junction IDs.
    if not (origin.startswith("J") and dest_edge.startswith("J")):
        return None

    def _extract_nums(jid: str) -> List[int]:
        # Strip the "J" prefix and any leading underscores, then parse all numbers.
        inner = jid[1:].lstrip("_")
        return [int(s) for s in inner.replace("_", " ").split() if s.isdigit()]

    o_nums = _extract_nums(origin)
    d_nums = _extract_nums(dest_edge)

    # Linear chain (single index): J0 -> J1 means travelling east (from J0's perspective).
    if len(o_nums) == 1 and len(d_nums) == 1:
        if o_nums[0] < d_nums[0]: return "W"
        if o_nums[0] > d_nums[0]: return "E"

    # Grid (row, col indices): direction depends on whether row or col changes.
    elif len(o_nums) == 2 and len(d_nums) == 2:
        o_row, o_col = o_nums
        d_row, d_col = d_nums
        if o_col < d_col: return "W"
        if o_col > d_col: return "E"
        if o_row < d_row: return "S"
        if o_row > d_row: return "N"

    return None


def _get_vehicle_exit_direction(vid: str) -> Optional[str]:
    """
    Determine which direction a vehicle will exit the intersection from its route.

    Reads the last edge in the vehicle's route and extracts the border node direction
    from the edge name. Returns None if route info is unavailable.
    """
    if not TRACI_AVAILABLE:
        return None
    try:
        route = traci.vehicle.getRoute(vid)
        if not route:
            return None
        last_edge = route[-1]
        # The last edge always leads to a border node, e.g. "C_to_N" or "J0_to_S0".
        if "_to_" in last_edge:
            dest_node = last_edge.split("_to_")[-1]
            if dest_node and dest_node[0] in ("N", "S", "E", "W"):
                return dest_node[0]
    except Exception:
        pass
    return None


def _int_to_bits(action_int: int, n_bits: int = 8) -> List[int]:
    """
    Convert an integer (0-255) to a list of n_bits bits, MSB first.

    This mirrors the teacher's trad_action() function exactly.
    The MSB-first order means bit index 0 corresponds to the highest-value bit.
    Example: 6 = binary 00000110 -> [0, 0, 0, 0, 0, 1, 1, 0]
    """
    bits = []
    for _ in range(n_bits):
        bits.append(action_int % 2)
        action_int //= 2
    return list(reversed(bits))


class IntersectionAgent:
    """
    One DQN agent controlling a single (non-complex) intersection.

    Args:
        model_path: Path to the SavedModel directory on disk.
        image_size: Input image side length in pixels (must match training config).
        intersection_id: Index of this agent, used only for logging.
        shared_model: If provided, use this existing Keras model instead of loading
                      from disk. Used by create_agents() to share weights across agents.
    """

    def __init__(
        self,
        model_path:      str          = "./save_model",
        image_size:      int          = 50,
        intersection_id: int          = 0,
        shared_model                  = None,
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

    # ===================================================================== #
    #  Factory method for multi-agent setups (shared model)                #
    # ===================================================================== #

    @classmethod
    def create_agents(
        cls,
        count:      int,
        model_path: str = "./save_model",
        image_size: int = 50,
    ) -> List["IntersectionAgent"]:
        """
        Create 'count' agents that all share the same Keras model instance.

        Loading the model once and sharing it avoids reading the same weights
        from disk N times, which saves both time and memory on large networks.
        """
        print(f"[IntersectionAgent] Loading shared model from '{model_path}' ...")
        shared = keras.models.load_model(model_path)
        shared.summary()
        print(f"[IntersectionAgent] Creating {count} agent(s) with shared weights.")

        return [
            cls(
                model_path      = model_path,
                image_size      = image_size,
                intersection_id = i,
                shared_model    = shared,
            )
            for i in range(count)
        ]

    # ===================================================================== #
    #  Public API                                                          #
    # ===================================================================== #

    def act(self, obs: Dict[str, Any]) -> Dict[str, int]:
        """
        Choose a go/stop action for every incoming lane at this intersection.

        Returns {lane_id: 0 | 1} where 1=go and 0=stop.

        Processing pipeline:
            1. Run the CNN forward pass to get a greedy integer action (0-255).
            2. Decode that integer to 8 bits (one bit per direction slot).
            3. Map the leader bits to the actual lane IDs in this observation.
            4. Run the safety filter to remove conflicting go decisions.
            5. Run the deadlock guard if all lanes came out as stop.
        """
        image   = obs["image"]
        leaders = obs["leaders"]

        # Step 1: CNN inference -> integer action
        action_int = self._predict(image)
        bits       = _int_to_bits(action_int, n_bits=8)

        # Step 2: extract only the leader bits (indices 0, 2, 4, 6)
        direction_bits: Dict[str, int] = {
            d: bits[idx] for d, idx in _DIRECTION_BIT_INDEX.items()
        }

        # Step 3: map direction decisions to the actual lane IDs present in this obs
        action: Dict[str, int] = {}
        for lane_id, vehicle_id in leaders.items():
            if vehicle_id is None:
                continue
            direction = _lane_to_direction(lane_id)
            if direction is None:
                continue
            action[lane_id] = direction_bits[direction]

        # Step 4: safety filter removes conflicting simultaneous go decisions
        action = self._safety_filter(action, leaders)

        # Step 5: deadlock guard releases the longest-waiting vehicle if all are stopped
        if action and all(v == 0 for v in action.values()):
            action = self._deadlock_guard(action, leaders)

        return action

    # ===================================================================== #
    #  Route-aware safety filter                                           #
    # ===================================================================== #

    def _safety_filter(
        self,
        action:  Dict[str, int],
        leaders: Dict[str, Optional[str]],
    ) -> Dict[str, int]:
        """
        Remove conflicting go-decisions using route-aware conflict detection.

        For each pair of "go" directions:
            1. If they are parallel (N+S or E+W) they are always safe.
            2. For perpendicular pairs, we query each vehicle's route to determine
               its turn type (straight, left, right).
            3. We look up the (turn_A, turn_B) pair in the conflict table.
            4. If the pair conflicts, the lower-priority direction is stopped.

        Directions are processed in priority order (N, E, S, W) so the highest
        priority direction is always accepted first.
        """
        go_by_dir: Dict[str, List[Tuple[str, Optional[str]]]] = {}
        for lane_id, go_val in action.items():
            if go_val == 1:
                d = _lane_to_direction(lane_id)
                if d:
                    vid = leaders.get(lane_id)
                    go_by_dir.setdefault(d, []).append((lane_id, vid))

        if len(go_by_dir) <= 1:
            return action  # zero or one active direction, nothing to check

        accepted: Dict[str, str] = {}  # direction -> its turn type

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

        # Any direction not accepted is forced to stop.
        for d, lane_entries in go_by_dir.items():
            if d not in accepted:
                for lid, _ in lane_entries:
                    action[lid] = 0

        return action

    def _get_turn_type(
        self,
        entry_dir:    str,
        lane_entries: List[Tuple[str, Optional[str]]],
    ) -> str:
        """
        Determine the turn type for the leader vehicle in a given direction.

        Queries TraCI for the vehicle's route to find its exit direction, then
        looks up the (entry, exit) pair in the _TURN_TYPE table. Falls back to
        "straight" if route information is unavailable, since straight is the
        most likely to conflict and therefore the safest assumption.
        """
        for _, vid in lane_entries:
            if vid is None:
                continue
            exit_dir = _get_vehicle_exit_direction(vid)
            if exit_dir is not None and exit_dir != entry_dir:
                turn = _TURN_TYPE.get((entry_dir, exit_dir))
                if turn is not None:
                    return turn
        return "straight"

    @staticmethod
    def _pair_is_safe(
        dir_a: str, turn_a: str,
        dir_b: str, turn_b: str,
    ) -> bool:
        """
        Return True if two directions can safely cross the junction simultaneously.

        Parallel pairs (N+S, E+W) are always safe. For perpendicular pairs we check
        the turn-type conflict table. Returns False for any unknown combination
        to err on the side of safety.
        """
        pair = frozenset({dir_a, dir_b})

        if pair in _ALWAYS_SAFE_PAIRS:
            return True

        conflicts = _PERPENDICULAR_CONFLICT.get((turn_a, turn_b))
        if conflicts is not None:
            return not conflicts

        return False  # unknown combination, block to be safe
    
    # ===================================================================== #
    #  Deadlock guard                                                      #
    # ===================================================================== #
    def _deadlock_guard(
        self,
        action:  Dict[str, int],
        leaders: Dict[str, Optional[str]],
    ) -> Dict[str, int]:
        """
        Release the longest-waiting vehicle when the AI outputs all-stop.

        When every lane gets a stop decision the intersection is frozen and the
        state will never change on its own, so we intervene. We find the leader
        that has been waiting the longest and force its lane to go.

        Note: vehicles stopped via setStop() are sometimes reported with
        getAccumulatedWaitingTime() = 0, so we fall back to a minimum wait of 1.0
        to still pick a lane even in that case.
        """
        best_lane   = None
        best_wait   = -1.0
        any_stopped = False

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
                    wait = 1.0  # setStop vehicles may report 0, use 1.0 as a floor
            except Exception:
                wait = 1.0

            if wait > best_wait:
                best_wait = wait
                best_lane = lid

        if any_stopped and best_lane is not None:
            print(
                f"[DEADLOCK GUARD #{self.intersection_id}] "
                f"All stopped -> releasing {best_lane}."
            )
            action[best_lane] = 1

        return action

    # ===================================================================== #
    #  Internal                                                            #
    # ===================================================================== #

    def _predict(self, image: np.ndarray) -> int:
        """
        Run the CNN forward pass and return the greedy action integer (0-255).

        We convert the image to float32 before passing it to TensorFlow since
        the model expects float inputs even though the image is stored as uint8.
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        state_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        state_tensor = tf.expand_dims(state_tensor, 0)  # add batch dimension
        action_probs = self.model(state_tensor, training=False)
        return int(tf.argmax(action_probs[0]).numpy())

    def __repr__(self) -> str:
        return (
            f"IntersectionAgent(id={self.intersection_id}, "
            f"model='{self.model_path}', image_size={self.image_size})"
        )