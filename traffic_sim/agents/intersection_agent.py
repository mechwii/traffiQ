# traffic_sim/agents/intersection_agent.py
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

Slot order (teacher + this agent)
-----------------------------------
  The env.py uses a fixed lane order:
      West, South, East, North   (4 leaders, indices 0-3)
  Since the model was trained on 4-lane intersections where each lane
  contributes one leader + one follower (8 bits total), we split the 8 bits as:
      bits[0], bits[2], bits[4], bits[6]  -> leaders  (West, South, East, North)
      bits[1], bits[3], bits[5], bits[7]  -> followers (ignored for now)

  If the network has fewer than 4 incoming directions (e.g. T-junction with 3
  arms), the extra bits are silently ignored.

Multi-intersection support
---------------------------
  Build one IntersectionAgent per intersection and feed it:
    - the intersection's cropped image  (or the full image for single-intersection)
    - the lanes belonging to that intersection only

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

from typing import Dict, List, Optional, Any
import numpy as np

# TensorFlow / Keras import 
try:
    import tensorflow as tf
    import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Lane direction order  (must match how NetworkBuilder names lanes)
# The teacher trained with West=0, South=1, East=2, North=3.
DIRECTION_ORDER: List[str] = ["W", "S", "E", "N"]

# The AI's brain only has 4 slots to receive information: "W", "S", "E", and "N".
# This dictionary tells the code how to map any complex lane names back into 
# those 4 simple slots so the AI doesn't crash when it receives data.
_EDGE_DIRECTION_MAP: Dict[str, str] = {
    "W": "W",   # "If a car is on a West lane, tell the AI it's West"
    "S": "S",  
    "E": "E",  
    "N": "N",   
    # --- FUTURE PROOFING EXAMPLE (If you generate complex intersections) --
    # "NW": "W",  # "If a car is on a Northwest lane, FORCE it into the AI's West slot"
    # "NE": "E",  # "If a car is on a Northeast lane, FORCE it into the AI's East slot"
}

def _lane_to_direction(lane_id: str) -> Optional[str]:
    """
    Infer the direction (W/S/E/N) from a lane ID.

    Handles the naming conventions produced by NetworkBuilder:
        "N_to_C_0"   -> "N"
        "W_to_C_1"   -> "W"
        "S_to_C"     -> "S"
    Returns None if the lane does not belong to a known incoming direction.
    """
    for direction in DIRECTION_ORDER:
        # Match lanes coming FROM a direction toward the center
        # EXPLANATION: WHY "_to_C"?
        # "C" stands for Center (the intersection).
        # The AI is acting as a traffic cop, so it only needs to look at cars 
        # driving TOWARD the center (e.g., "N_to_C"). 
        # It completely ignores cars driving away from the center (e.g., "C_to_N") 
        # because those cars have already crossed the intersection safely and 
        # no longer need a stop/go command.
        prefix = f"{direction}_to_C"
        if lane_id.startswith(prefix) or lane_id == prefix:
            return direction
    return None

def _int_to_bits(action_int: int, n_bits: int = 8) -> List[int]:
    """
    Convert an integer 0..255 to a list of n_bits bits, MSB first.

    n_bits will also allow to handle multiple lanes.

    Example:
        _int_to_bits(6, 8)  ->  [0, 0, 0, 0, 0, 1, 1, 0]
    """
    bits = []
    for _ in range(n_bits):
        # If the number is even, the remainder is 0. If odd, it's 1.
        # This gives us our binary digit!
        # Loop 1: 6 is even. (6 % 2 = 0). We append 0 to our list.
        bits.append(action_int % 2)

        # `// 2` divides the number by 2 and throws away any decimals.
        # This shifts our number down for the next loop.
        # Loop 1: 6 // 2 = 3. Next loop, we will do the math on the number 3.
        # Loop 2: 3 is odd (3 % 2 = 1). We append 1. Then 3 // 2 = 1.
        # Loop 3: 1 is odd (1 % 2 = 1). We append 1. Then 1 // 2 = 0
        action_int //= 2

    # By the end of 8 loops, our list looks like this: [0, 1, 1, 0, 0, 0, 0, 0]
    # The math above extracts the binary digits backwards (from right to left).
    # To match up with the correct "West, South, East, North" order the AI expects, 
    # we have to flip the list around so it reads from left to right.
    # [0, 1, 1, 0, 0, 0, 0, 0] becomes -> [0, 0, 0, 0, 0, 1, 1, 0]
    return list(reversed(bits))

class IntersectionAgent:
    """
    One DQN agent controlling a single (non-complex) intersection.

    Parameters
    ----------
    model_path : str
        Path to the SavedModel directory produced by the teacher's training
        script (e.g. "./save_model").
    image_size : int
        Side length of the input image in pixels.  Must match the value used
        during training (default 50).
    intersection_id : int
        Index of the intersection this agent controls.  Used only for logging.
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

        print(
            f"[IntersectionAgent #{intersection_id}] "
            f"Loading model from '{model_path}' …"
        )
        self.model = keras.models.load_model(model_path)
        self.model.summary()
        print(f"[IntersectionAgent #{intersection_id}] Model loaded.")

    ## ---------- PUBLIC API ----------

    def act(self, obs: Dict[str, Any]) -> Dict[str, int]:
        """
        Choose an action given the current observation.

        Parameters
        ----------
        obs : dict
            The observation dict returned by ``SumoEnvironment.step()`` /
            ``SumoEnvironment.reset()``.  Expected keys:
                ``"image"``   -> ndarray (image_size, image_size, 3) uint8
                ``"leaders"`` -> dict {lane_id: vehicle_id | None}

        Returns
        -------
        action : dict {lane_id: 0 | 1}
            1 = let the leader of this lane go.
            0 = hold the leader of this lane.
            Only lanes that belong to a known incoming direction
            (W / S / E / N) are included.
        """
        image   = obs["image"]        # (H, W, 3) uint8
        leaders = obs["leaders"]      # {lane_id: vehicle_id | None}

        # Run inference
        action_int = self._predict(image)

        # Decode integer action -> per-direction bits
        bits = _int_to_bits(action_int, n_bits=8)
        # bits layout from teacher:
        #   index  0 -> leader  West
        #   index  1 -> follower West  (ignored)
        #   index  2 -> leader  South
        #   index  3 -> follower South (ignored)
        #   index  4 -> leader  East
        #   index  5 -> follower East  (ignored)
        #   index  6 -> leader  North
        #   index  7 -> follower North (ignored)
        direction_bits: Dict[str, int] = {
            "W": bits[0],
            "S": bits[2],
            "E": bits[4],
            "N": bits[6],
        }

        # Build {lane_id: 0|1} for every leader lane we can classify
        action: Dict[str, int] = {}
        for lane_id, vehicle_id in leaders.items():
            if vehicle_id is None:
                continue   # empty lane - no action needed
            direction = _lane_to_direction(lane_id)
            if direction is None:
                continue   # outgoing lane or unknown - skip
            action[lane_id] = direction_bits.get(direction, 0)

        return action
    
    # ------ Internal 

    """
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
            action_binaire = env.trad_action(action)
            # Apply the sampled action in our environmentS
            print(f"Applying action: {action}")
            state_next, reward, done, average_waiting_time, cumulated_waiting_time, emission_of_co2, average_speed, evacuated_vehicle, nb_collision = \
                env.step(action_binaire, simulation_type, image_size, reward_type, coef, security=True)
    """

    def _predict(self, image: np.ndarray) -> int:
        """
        Run the CNN forward pass and return the greedy action integer.

        Mirrors the model.py inference code (sending by the teacher):
            state_tensor = tf.expand_dims(image, 0)
            action_probs = model(state_tensor, training=False)
            action       = tf.argmax(action_probs[0]).numpy()
        """
        # Ensure correct dtype and shape
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        state_tensor  = tf.convert_to_tensor(image, dtype=tf.float32)
        state_tensor  = tf.expand_dims(state_tensor, 0)          # (1, H, W, 3)
        action_probs  = self.model(state_tensor, training=False)  # (1, 256)
        action_int    = int(tf.argmax(action_probs[0]).numpy())   # 0..255
        return action_int

    def __repr__(self) -> str:
        return (
            f"IntersectionAgent(id={self.intersection_id}, "
            f"model='{self.model_path}', image_size={self.image_size})"
        )
