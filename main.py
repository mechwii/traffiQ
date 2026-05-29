# main.py
"""
SUMO Traffic Simulation - RL-ready demo
========================================

What this file does
--------------------
  Step 1 : Build a road network   (NetworkBuilder)
  Step 2 : Generate traffic demand (DemandGenerator)
  Step 3 : Run the RL loop        (SumoEnvironment)

One IntersectionAgent is created per intersection.
Each agent wraps the teacher's pre-trained CNN (DQN) and receives:
    - obs["image"]    ndarray (50, 50, 3) uint8   -> the full network image
    - obs["leaders"]  {lane_id: vehicle_id | None} -> filtered to its own lanes.

The agent returns  {lane_id: 0 | 1}  (1 = go, 0 = stop).

Reward
------
    +1 for every vehicle that completed its route this step.
    (traci.simulation.getArrivedNumber() -> teacher's reference formula)


Action order (per intersection, matching the teacher's training)
----------------------------------------------------------------
  West -> South -> East -> North
  (encoded as bits 0, 2, 4, 6 of the 8-bit integer action)

Supported scenarios
-------------------
  intersection_type : "four_way" | "t_junction"    (complex is excluded)
  num_intersections : 1 | 2 | 4 | 8
  num_lanes         : 1 | 2 | 3

Prerequisites
-------------
  1. SUMO installed   https://sumo.dlr.de/docs/Downloads.php
  2. SUMO_HOME set    export SUMO_HOME=/path/to/sumo
  3. Model saved at MODEL_PATH   (teacher's ./save_model directory)
  4. pip install numpy tensorflow keras
"""

""" TODO 
Re do the logic cause now we execute action each 5 seconds if it's okay
We can also take leader when arriving in the intersection
Also talk about deadlock

# Crop the 50x50 image per intersection to make the multiagent works
# Example of what the fix will look like later:
    for i, agent in enumerate(agents):
        
        # You would need a function that cuts the giant image into a 50x50 piece
        cropped_50x50_image = crop_image_for_intersection(obs["image"], intersection_index=i)
        
        local_obs = {
            "image":   cropped_50x50_image,   # Now the agent ONLY sees its own intersection!
            "leaders": local_leaders,
        }
        local_action = agent.act(local_obs)
"""

""" ---- DONE ---
# TODO un agent par intersection donc envoie une intersection a un agent
# et lui il va controler une intersection
# L'agent respecte un ordre 
#. West, Sud, Est, North 
# Une intersection par agent 
# Pas prendre en compte les Formes complex
# prendre les couleurs généré par le prof 
"""

import os
import sys
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_sim.network.network_builder   import NetworkBuilder
from traffic_sim.network.demand_generator  import DemandGenerator
from traffic_sim.sumo_env.sumo_environment import SumoEnvironment
from traffic_sim.agents.intersection_agent import IntersectionAgent

# ------------------------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------------------------

SCENARIO = {
    "intersection_type": "four_way", # "four_way" | "t_junction" | "complex"
    "num_intersections": 1,  # 1 | 2 | 4 | 8
    "num_lanes":         3, # 1 | 2 | 3 
}

DEMAND_LEVEL        = "low"   # "low" | "moderate" | "high" | "congested"
SIMULATION_DURATION = 300          # seconds
USE_GUI             = True
CONFIGS_DIR         = "configs"
PRINT_EVERY         = 50

# Path to the pre-trained SavedModel directory
MODEL_PATH = os.path.join(".", "traffic_sim", "save_model")

# Observation image size -> must match what the model was trained on
IMAGE_SIZE = 50  # the agent receives a (50 × 50 × 3) uint8 RGB image

DEST_COLORS: Optional[Dict] = None
# Example for the NetworkBuilder four_way_1int network:
# DEST_COLORS = {
#     "C_to_N": (1.0, 0.0, 0.0),   # red    -> north exit
#     "C_to_S": (0.0, 1.0, 0.0),   # green  -> south exit
#     "C_to_E": (0.0, 0.0, 1.0),   # blue   -> east  exit
#     "C_to_W": (1.0, 1.0, 0.0),   # yellow -> west  exit
# }

INTERSECTION_OUTGOING: Optional[set] = None
# Example:
# INTERSECTION_OUTGOING = {"C_to_N", "C_to_S", "C_to_E", "C_to_W"}

# ------------------------------------------------------------------------------
#  Helper: assign lanes to intersections
# ------------------------------------------------------------------------------

def assign_lanes_to_intersections(
    leaders: Dict[str, Any],
    num_intersections: int,
) -> List[Dict[str, Any]]:
    """
    Split the global leaders dict into one sub-dict per intersection.

    For a single-intersection network all lanes go to agent 0.
    For multi-intersection networks, lanes are partitioned by the
    intersection index encoded in the edge name:

        "N_to_C0_0"  -> intersection 0
        "W_to_C1_1"  -> intersection 1
        "S_to_C3"    -> intersection 3

    If no numeric suffix is found the lane goes to intersection 0 (default).

    Parameters
    ----------
    leaders : dict {lane_id: vehicle_id | None}
    num_intersections : int

    Returns
    -------
    list of dicts, one per intersection
    """
    if num_intersections == 1:
        return [leaders]

    buckets: List[Dict[str, Any]] = [{} for _ in range(num_intersections)]

    for lane_id, vehicle_id in leaders.items():
        # Try to extract the intersection index from the lane / edge name.
        # Convention: "X_to_C{idx}" or "X_to_C{idx}_{lane_num}"
        idx = 0
        try:
            # Find "C" in the lane id and read the digit that follows
            c_pos = lane_id.index("_to_C")
            after = lane_id[c_pos + len("_to_C"):]
            # after might be "0", "0_1", "0_lane_2", etc.
            digit_str = ""
            for ch in after:
                if ch.isdigit():
                    digit_str += ch
                else:
                    break
            if digit_str:
                idx = int(digit_str) % num_intersections
        except ValueError:
            pass  # no "_to_C" found -> default to 0

        if 0 <= idx < num_intersections:
            buckets[idx][lane_id] = vehicle_id

    return buckets

# ------------------------------------------------------------------------------
#  Multi-agent action combiner
# ------------------------------------------------------------------------------

def multi_agent_act(
    agents: List[IntersectionAgent],
    obs: Dict[str, Any],
    num_intersections: int,
) -> Dict[str, int]:
    """
    Run all agents and merge their actions into one global action dict.

    Each agent sees the same full-network image but only the leaders
    that belong to its intersection.

    Parameters
    ----------
    agents : list[IntersectionAgent]
    obs    : observation dict from SumoEnvironment
    num_intersections : int

    Returns
    -------
    action : dict {lane_id: 0 | 1}
    """

    # If num_intersections == 1, it just returns the whole dictionary in a list: [{all_lanes}]
    # If num_intersections == 4, it chops it up into 4 dictionaries: [{lanes_0}, {lanes_1}, ...]
    per_intersection_leaders = assign_lanes_to_intersections(
        obs["leaders"], num_intersections
    )

    combined_action: Dict[str, int] = {}

    # zip() pairs each agent with its specific bucket of lanes.
    # If there is 1 agent, this loop runs 1 time.
    # If there are 8 agents, this loop runs 8 times.
    for agent, local_leaders in zip(agents, per_intersection_leaders):
        local_obs = {
            "image":   obs["image"], # All agents look at the exact same 50x50 image
            "leaders": local_leaders, # BUT they only receive the lanes assigned to them!
            "state":   obs.get("state", {}),
        }

        # Ask the agent to make a Go/Stop decision based ONLY on its local lanes
        local_action = agent.act(local_obs)

        # Merge this agent's decision into the global master dictionary.
        # e.g., Agent 0 says "N_to_C0 goes", Agent 1 says "W_to_C1 stops" -> both go into the combined dict.
        combined_action.update(local_action)

    # Return the master dictionary containing instructions for every lane in the entire simulation
    return combined_action


# ------------------------------------------------------------------------------
#  Print helper
# ------------------------------------------------------------------------------

def print_step(step: int, reward: float, obs: Dict, info: Dict) -> None:
    stats = info["stats"]
    print(
        f"  step={step:>4} | "
        f"t={info['simulation_time']:>7.1f}s | "
        f"reward={reward:>5.1f} | "
        f"vehicles={stats['total_vehicles']:>3} | "
        f"speed={stats['mean_speed']:>5.2f} m/s | "
        f"wait={stats['mean_waiting_time']:>6.1f}s | "
        f"congestion={stats['congestion_index']:.2f}"
    )


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  SUMO Traffic Simulation - RL Interface Demo")
    print("=" * 65)

    num_intersections = SCENARIO["num_intersections"]
    intersection_type = SCENARIO["intersection_type"]


    # -- 1. Build the road network -----------------------------------------
    print("\n[1/5] Building road network...")
    builder  = NetworkBuilder(output_dir=CONFIGS_DIR)
    net_file = builder.build(
        intersection_type = intersection_type,
        num_intersections = num_intersections,
        num_lanes         = SCENARIO["num_lanes"],
    )

    # -- 2. Generate traffic demand ----------------------------------------
    print(f"\n[2/5] Generating '{DEMAND_LEVEL}' traffic demand...")
    demand_gen = DemandGenerator(net_file=net_file)
    route_file = demand_gen.generate(
        level    = DEMAND_LEVEL,
        duration = SIMULATION_DURATION,
    )

    # -- 3. Create the environment -----------------------------------------
    print(f"\n[3/5] Creating RL environment ({SIMULATION_DURATION}s episode)...")
    env = SumoEnvironment(
        net_file   = net_file,
        route_file = route_file,
        use_gui    = USE_GUI,
        step_length = 1.0,
        max_steps   = SIMULATION_DURATION,
        # -- observation ---------------
        image_size            = IMAGE_SIZE,
        dest_colors           = DEST_COLORS,
        intersection_outgoing = INTERSECTION_OUTGOING,
    )
    env.start()

    # -- 4. Create agents - one per intersection -----------------------
    print(f"\n[4/5] Loading {num_intersections} agent(s) from '{MODEL_PATH}'...")
    agents: List[IntersectionAgent] = [
        IntersectionAgent(
            model_path       = MODEL_PATH,
            image_size       = IMAGE_SIZE,
            intersection_id  = i,
        )
        for i in range(num_intersections)
    ]
    print(f"  {len(agents)} agent(s) ready.")

    # -- 5. RL loop --------------------------------------------------------
    print(f"\n[5/5] Running RL loop (print every {PRINT_EVERY} steps)...")
    print(
        f"\n  Each step the agent receives:\n"
        f"    obs['image']   - ndarray ({IMAGE_SIZE}×{IMAGE_SIZE}×3)  RGB pixel map\n"
        f"    obs['leaders'] - {{lane_id: vehicle_id | None}}\n"
        f"    (West->South->East->North order)\n"
        f"  And returns: {{lane_id: 0|1}}\n"
    )

    obs  = env.reset()
    done = False
    total_reward = 0.0

    print(f"  Initial observation - image shape: {obs['image'].shape}\n")

    # TODO Look for a better solution
    ACTION_REPEAT = 3  # The AI will take a decision each 3 secondes
    step_counter = 0   # Internal counter 
    current_action = {} # last AI decision

    while not done:
        """ OLD METHOD IN EACH SECOND
        # Each agent reads the image and its own intersection's leaders,
        # then returns {lane_id: 0|1}.
        action = multi_agent_act(agents, obs, num_intersections)
        """ 

        # We use the AI only each X seconds
        if step_counter % ACTION_REPEAT == 0:
            current_action = multi_agent_act(agents, obs, num_intersections)

        # -- Step the environment ------------------------------------------
        obs, reward, done, info = env.step(current_action)
        total_reward += reward
        step_counter += 1

        """ See a pic
        if info["step"] == 100:
            plt.imshow(obs["image"])
            plt.title(f"Generated vector at the step {info['step']}")
            plt.axis('off') 
            plt.show()
        """

        if info["step"] % PRINT_EVERY == 0 or done:
            print_step(info["step"], reward, obs, info)

    print(f"\n  Total episode reward (vehicles arrived): {total_reward:.0f}")
    print()
    env.statistics_collector.print_summary()

    try:
        df       = env.statistics_collector.to_dataframe()
        csv_path = os.path.join(
            os.path.dirname(route_file), f"results_{DEMAND_LEVEL}.csv"
        )
        df.to_csv(csv_path, index=False)
        print(f"  Results saved to: {csv_path}")
    except ImportError:
        print("  (Install pandas to export results to CSV)")

    env.close()
    print("\nSimulation complete.")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------
#  Placeholder agent
#  -----------------
#  This is where your AI model goes.
#
#  Input
#  -----
#  obs["image"]   ndarray (50, 50, 3) uint8
#                 The RGB pixel map.  Each pixel = one vehicle.
#                 Intensity = speed.  Colour = destination direction.
#                 White = already past the intersection.
#
#  obs["leaders"] dict {lane_id: vehicle_id | None}
#                 Which vehicle is the leader on each lane right now.
#
#  Output
#  ------
#  action : dict {lane_id: 0 | 1}
#                 1 = allow the leader to go  (speed restored to SUMO default)
#                 0 = stop the leader          (speed = 0)
#
#  Placeholder policy: let every leader go unconditionally (all 1s).
# ------------------------------------------------------------------------------

# def placeholder_agent(obs: Dict[str, Any]) -> Dict[str, int]:
#     """
#     Rule-based placeholder -> lets every leader pass.

#     Replace the body of this function with your real model, e.g.:

#         image  = obs["image"]                    # shape (50, 50, 3)
#         tensor = preprocess(image)               # your preprocessing
#         action = my_model.predict(tensor)        # {lane_id: 0|1}
#         return action
#     """
#     action = {}
#     for lane_id, vehicle_id in obs["leaders"].items():
#         if vehicle_id is not None:
#             action[lane_id] = 1   # go
#     return action
