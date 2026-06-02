# traffic_sim/agents/intersection_agent.py
"""
IntersectionAgent
=================
Wraps the pre-trained CNN (DQN) model and adapts its output to the
{lane_id: 0 | 1} action format used by SumoEnvironment.

One agent is created per intersection (complex intersections are excluded).

How the model works
------------------------------
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

    If the network has fewer than 4 incoming directions (e.g. T-junction with 3 arms), the extra bits are silently ignored.

_lane_to_direction -> lane naming ground-truth (from NetworkBuilder)
-------------------------------------------------------------------
  Single intersection (node "C"):
      "N_to_C_0"       origin="N"       -> N
      "W_to_C_1"       origin="W"       -> W
      "C_to_N_0"       outgoing          -> None

  Linear chain (nodes "J0", "J1"):
      "N0_to_J0_0"     origin="N0"      -> N
      "W0_to_J0_0"     origin="W0"      -> W
      "J0_to_J1_0"     origin="J0", dest_edge="J1"   J0<J1 -> W (coming from West)
      "J1_to_J0_0"     origin="J1", dest_edge="J0"   J1>J0 -> E (coming from East)

  Grid 2x2 (nodes "J_r_c"):
      "N_1_0_to_J_1_0_0"   origin="N_1_0"  -> N
      "J_0_0_to_J_0_1_0"   origin="J_0_0", dest_edge="J_0_1"
                            col 0 < col 1   -> W (vehicle comes from West)
      "J_0_1_to_J_0_0_0"   col 1 > col 0   -> E (comes from East)
      "J_0_0_to_J_1_0_0"   dest_edge="J_1_0", row 0 < row 1 -> S (from South)
      "J_1_0_to_J_0_0_0"   row 1 > row 0   -> N (from North)

  KEY FIX vs previous version:
      Lane IDs like "J0_to_J1_0" or "J_0_0_to_J_0_1_0" carry a trailing
      lane index ("_0") on the DESTINATION part after splitting on "_to_".
      The old code parsed the destination including that digit, getting
      3 numbers instead of 2 for grid lanes, causing the function to return
      None for every internal grid lane.
      Fix: strip the trailing "_<digit>" from the destination part before
      extracting junction coordinates.

Usage
-----
    from traffic_sim.agents.intersection_agent import IntersectionAgent

    # Single intersection
    agent = IntersectionAgent(model_path="./save_model", image_size=50)

    # In the RL loop:
    action = agent.act(obs)          # obs = env.step() / env.reset() output
    obs, reward, done, info = env.step(action)

    # Multiple intersections - one agent each, same shared model weights
    agents = [IntersectionAgent("./save_model", image_size=50, intersection_id=i)
              for i in range(num_intersections)]
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

# Cardinal directions the model knows about
_DIRECTIONS = ["N", "S", "E", "W"]

# Bit index in the 8-bit decoded action for each direction's leader decision
_DIRECTION_BIT_INDEX: Dict[str, int] = {
    "N": 0,
    "E": 2,
    "W": 4,
    "S": 6,
}

# Regex to strip the trailing lane index from a lane ID
# e.g. "J_0_1_0" -> "J_0_1",   "J1_0" -> "J1",   "J_0_0" -> "J_0_0" (no trailing single digit after _)
_LANE_INDEX_RE = re.compile(r"_\d+$")


def _strip_lane_index(lane_or_edge: str) -> str:
    """
    Strip the trailing lane index from a lane ID to recover the edge ID.

    SUMO lane IDs are formed as  <edge_id>_<lane_number>.
    Examples:
        "J_0_1_0"  ->  "J_0_1"   (grid junction edge, lane 0)
        "J1_0"     ->  "J1"      (chain junction edge, lane 0)
        "N0_to_J0" ->  "N0_to_J0"  (already an edge ID — no change)
    """
    return _LANE_INDEX_RE.sub("", lane_or_edge)


def _lane_to_direction(lane_id: str) -> Optional[str]:
    """
    Infer the cardinal direction (N/S/E/W) of an INCOMING lane.

    Returns None for:
      - outgoing lanes ("C_to_N_0", "J0_to_N0_0", ...)
      - internal lanes that don't map to a direction
      - unrecognised formats

    Handles all three network topologies produced by NetworkBuilder.
    """
    if "_to_" not in lane_id:
        return None

    # Split on the FIRST "_to_" only.
    # "J_0_0_to_J_0_1_0" -> origin="J_0_0", dest_raw="J_0_1_0"
    origin, dest_raw = lane_id.split("_to_", 1)

    # Remove the trailing lane index from the destination part.
    # This is the key fix: "J_0_1_0" -> "J_0_1", "J1_0" -> "J1"
    dest_edge = _strip_lane_index(dest_raw)

    # -- 1. Border node origin ------------------------------------------
    # Single: "N" -> N,  "S" -> S,  etc.
    # Chain:  "N0" -> N,  "W0" -> W,  etc.
    # Grid:   "N_1_0" -> N,  "S_0_2" -> S,  etc.
    for d in _DIRECTIONS:
        # Exact match (single intersection, no lane suffix)
        if origin == d:
            return d
        # Starts with direction letter followed by digit or underscore
        if len(origin) > 1 and origin[0] == d and (origin[1].isdigit() or origin[1] == "_"):
            return d

    # -- 2. Internal road between two junctions -------------------------
    # Both origin and dest_edge must start with "J"
    if not (origin.startswith("J") and dest_edge.startswith("J")):
        return None  # outgoing or unknown

    # Extract numeric coordinates from junction node names.
    # "J0"    -> [0]        (linear chain)
    # "J_0_1" -> [0, 1]    (grid, row=0 col=1)
    def _extract_nums(jid: str) -> List[int]:
        # Remove the leading "J" (and optional underscore) then split
        inner = jid[1:].lstrip("_")
        return [int(s) for s in inner.replace("_", " ").split() if s.isdigit()]

    o_nums = _extract_nums(origin)
    d_nums = _extract_nums(dest_edge)

    # Linear chain (J0, J1, ...)
    if len(o_nums) == 1 and len(d_nums) == 1:
        if o_nums[0] < d_nums[0]: return "W"   # moving right -> comes from West
        if o_nums[0] > d_nums[0]: return "E"   # moving left  -> comes from East

    # Grid (J_row_col)
    elif len(o_nums) == 2 and len(d_nums) == 2:
        o_row, o_col = o_nums
        d_row, d_col = d_nums
        if o_col < d_col: return "W"   # col increases -> from West
        if o_col > d_col: return "E"   # col decreases -> from East
        if o_row < d_row: return "S"   # row increases -> from South
        if o_row > d_row: return "N"   # row decreases -> from North

    return None


def _int_to_bits(action_int: int, n_bits: int = 8) -> List[int]:
    """
    Convert integer 0..255 to a list of n_bits bits, MSB first.
    Example: _int_to_bits(6, 8) -> [0, 0, 0, 0, 0, 1, 1, 0]
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
    model_path      : str   path to SavedModel directory (e.g. "./save_model")
    image_size      : int   input image side length in pixels (must match training)
    intersection_id : int   index of this intersection (for logging only)
    """

    def __init__(
        self,
        model_path: str = "./save_model",
        image_size: int = 50,
        intersection_id: int = 0,
    ):
        if not TF_AVAILABLE:
            raise EnvironmentError(
                "TensorFlow / Keras is required for IntersectionAgent.\n"
                "Install with:  pip install tensorflow"
            )

        self.image_size      = image_size
        self.intersection_id = intersection_id
        self.model_path      = model_path

        print(f"[IntersectionAgent #{intersection_id}] Loading model from '{model_path}' ...")
        self.model = keras.models.load_model(model_path)
        self.model.summary()
        print(f"[IntersectionAgent #{intersection_id}] Model loaded.")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Dict[str, int]:
        """
        Choose a go/stop action for every incoming lane at this intersection.

        Returns one dict entry per lane that has a vehicle:
            {lane_id: 0 | 1}    (1=go, 0=stop)

        All lanes of the same direction share the same model bit,
        but each lane gets its own independent key in the dict.

        Example with 3 lanes per direction (four_way):
            N_to_C_0 -> bits[0],  N_to_C_1 -> bits[0],  N_to_C_2 -> bits[0]
            E_to_C_0 -> bits[2],  E_to_C_1 -> bits[2],  E_to_C_2 -> bits[2]
            W_to_C_0 -> bits[4],  ...
            S_to_C_0 -> bits[6],  ...

        Deadlock guard: if ALL entries are 0 and vehicles are waiting,
        override to 1 so SUMO's right-of-way logic resolves the intersection.
        """
        image   = obs["image"]    # (H, W, 3) uint8
        leaders = obs["leaders"]  # {lane_id: vehicle_id | None}

        # CNN inference
        action_int = self._predict(image)
        bits       = _int_to_bits(action_int, n_bits=8)

        direction_bits: Dict[str, int] = {
            d: bits[idx] for d, idx in _DIRECTION_BIT_INDEX.items()
        }

        action: Dict[str, int] = {}
        for lane_id, vehicle_id in leaders.items():
            if vehicle_id is None:
                continue   # empty lane -> skip
            direction = _lane_to_direction(lane_id)
            if direction is None:
                continue   # outgoing or unrecognised -> skip
            action[lane_id] = direction_bits[direction]

        # Deadlock guard: all-zero -> override to all-go
        if action and all(v == 0 for v in action.values()):
            print("[DEADLOCK] We put all values to 1!")

            for lane_id in action:
                action[lane_id] = 1

        return action

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _predict(self, image: np.ndarray) -> int:
        """Run CNN forward pass, return greedy action integer 0-255."""
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        state_tensor     = tf.convert_to_tensor(image, dtype=tf.float32)
        state_tensor     = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        return int(tf.argmax(action_probs[0]).numpy())

    def __repr__(self) -> str:
        return (
            f"IntersectionAgent(id={self.intersection_id}, "
            f"model='{self.model_path}', image_size={self.image_size})"
        )