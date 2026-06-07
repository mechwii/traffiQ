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

Safety filter
-------------
  After the model produces go/stop bits per direction, _safety_filter()
  removes conflicting simultaneous "go" decisions.  Two directions can
  safely cross at the same time only if they are PARALLEL (N+S or E+W).
  Perpendicular pairs (N+E, N+W, S+E, S+W) are NOT safe — the second
  one in priority order is set to STOP.

  Priority order: N > E > S > W  (matches teacher's convention).

  This replaces the teacher's security_matrice.npy lookup with a
  conservative rule-based check.  It is slightly more restrictive
  (the matrix also allows some perpendicular turns that don't actually
  cross) but is safe for all network types without needing the numpy
  file.

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
from typing import Dict, List, Optional, Any
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

# Direction pairs that can safely cross simultaneously (parallel traffic)
_SAFE_DIRECTION_PAIRS = {
    frozenset({"N", "S"}),
    frozenset({"E", "W"}),
}

# Priority order for resolving conflicts (higher priority kept first)
_PRIORITY_ORDER = ["N", "E", "S", "W"]

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

    # 1. Border node origin
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

    if len(o_nums) == 1 and len(d_nums) == 1:
        if o_nums[0] < d_nums[0]: return "W"
        if o_nums[0] > d_nums[0]: return "E"

    elif len(o_nums) == 2 and len(d_nums) == 2:
        o_row, o_col = o_nums
        d_row, d_col = d_nums
        if o_col < d_col: return "W"
        if o_col > d_col: return "E"
        if o_row < d_row: return "S"
        if o_row > d_row: return "N"

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
          4. Safety filter: remove conflicting perpendicular "go" pairs
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

        # 4. Safety filter: prevent conflicting perpendicular "go" actions
        action = self._safety_filter(action)

        # 5. Deadlock guard: if ALL are stopped and someone is waiting
        #    too long, release the longest-waiting leader
        if action and all(v == 0 for v in action.values()):
            action = self._deadlock_guard(action, leaders)

        return action

    # ------------------------------------------------------------------ #
    #  Safety filter                                                       #
    # ------------------------------------------------------------------ #

    def _safety_filter(self, action: Dict[str, int]) -> Dict[str, int]:
        """
        Remove conflicting go-actions for perpendicular directions.

        Only parallel direction pairs (N+S, E+W) are allowed to cross
        simultaneously.  When a perpendicular conflict is detected,
        the lower-priority direction is set to STOP.

        This is a conservative replacement for the teacher's
        security_matrice.npy — it may block some turns that would
        actually be safe, but it never allows unsafe crossings.
        """
        # Group "go" lanes by direction
        go_by_dir: Dict[str, List[str]] = {}
        for lane_id, go_val in action.items():
            if go_val == 1:
                d = _lane_to_direction(lane_id)
                if d:
                    go_by_dir.setdefault(d, []).append(lane_id)

        if len(go_by_dir) <= 1:
            return action   # 0 or 1 direction active -> no conflict

        # Greedily accept directions in priority order
        accepted: set = set()
        for d in _PRIORITY_ORDER:
            if d not in go_by_dir:
                continue
            is_safe = all(
                frozenset({d, a}) in _SAFE_DIRECTION_PAIRS
                for a in accepted
            )
            if is_safe:
                accepted.add(d)

        # Set rejected directions to STOP
        for d, lane_ids in go_by_dir.items():
            if d not in accepted:
                for lid in lane_ids:
                    action[lid] = 0

        return action

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
                # Use waiting time if available, else use 1.0 as
                # a tiebreaker so we still pick the "first" vehicle
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