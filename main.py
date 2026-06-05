# main.py
"""
SUMO Traffic Simulation - DQN Agent
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

Three problems solved vs the previous version
----------------------------------------------
 
  FIX 1 : Smart agent call timing
      The AI is no longer called on every simulation step.
      Instead, the agent is called only when something meaningful changes
      at its intersection: a new leader appears on a lane that was empty,
      OR a previously allowed vehicle has cleared the intersection.
      The current decision is held until that condition triggers again.
 
  FIX 2 : Variable action dict size (multi-lane)
      With num_lanes=3, each direction produces 3 lanes:
          E_to_C_0, E_to_C_1, E_to_C_2
      The old code only output 4 bits, so some lanes were missing.
      Now ALL lanes of the same direction share the same decision bit
      (handled inside IntersectionAgent.act()).
 
  FIX 3 : Per-intersection cropped image
      For multi-intersection networks, each agent now receives a 50x50
      image cropped to ITS OWN intersection's bounding box instead of
      the full network image.  Built by ObservationBuilder.build_image_for_intersection().

Reward
------
    +1 for every vehicle that completed its route this step.
    (traci.simulation.getArrivedNumber() -> teacher's reference formula)


Action order (per intersection, matching the teacher's training)
----------------------------------------------------------------
  North -> East -> West -> South
  (mapped to bits 0, 2, 4, 6 of the 8-bit integer action)

Supported scenarios
-------------------
  intersection_type : "four_way" | "t_junction"    (complex is excluded)
  num_intersections : 1 | 2 | 4 | 8
  num_lanes         : 1 | 2 | 3

  Junction ID conventions (NetworkBuilder)
-----------------------------------------
  single intersection      : "C"
  linear chain (count=2)   : "J0", "J1"
  grid 2×2  (count=4)      : "J_0_0", "J_0_1", "J_1_0", "J_1_1"
  grid 2×4  (count=8)      : "J_0_0" ... "J_1_3"
 
Lane naming conventions
-----------------------
  single:   "N_to_C_0", "E_to_C_1", ...
  chain:    "N0_to_J0_0", "S1_to_J1_0", "W0_to_J0_0", ...
  grid:     "N_0_0_to_J_0_0_0", "W_0_0_to_J_0_0_0", .

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

"""
TODO also do the crop of different intersections cause now we are sending 1 big pic
We have to send X intersections pics
"""


"""TODO fix deadlock issues
Start redaction of the report, soutenance -> 16 june 

Feu de signalisation modificaiton au niveau du fichier net, rajouter que 
l'intersection est généré avec le temps de signalisation

Ajouter le premier arrivé, premier servi (modificaiton de parameetre dans le net pour respecter ça )
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
from traffic_sim.agents.agent_call_manager import AgentCallManager

# ------------------------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------------------------

SCENARIO = {
    "intersection_type": "four_way", # "four_way" | "t_junction" | "complex"
    "num_intersections": 1,  # 1 | 2 | 4 | 8
    "num_lanes":         3, # 1 | 2 | 3 
}

DEMAND_LEVEL= "low"   # "low" | "moderate" | "high" | "congested"
SIMULATION_DURATION = 300          # seconds
USE_GUI     = True
CONFIGS_DIR = "configs"
PRINT_EVERY = 50

# Path to the pre-trained SavedModel directory
MODEL_PATH = os.path.join(".", "traffic_sim", "save_model")

# Observation image size -> must match what the model was trained on
IMAGE_SIZE = 50  # the agent receives a (50 × 50 × 3) uint8 RGB image
SHOW_CROPPED_IMAGE = True

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
#  Helper: Junction ID resolver
# ------------------------------------------------------------------------------
def get_junction_ids(num_intersections: int) -> List[str]:
    """
    Return the ordered list of junction node IDs for a given intersection count.
 
    Matches the naming produced by NetworkBuilder exactly:
        1  -> ["C"]
        2  -> ["J0", "J1"]
        4  -> ["J_0_0", "J_0_1", "J_1_0", "J_1_1"]
        8  -> ["J_0_0", "J_0_1", "J_0_2", "J_0_3",
                "J_1_0", "J_1_1", "J_1_2", "J_1_3"]
    """
    if num_intersections == 1:
        return ["C"]
    if num_intersections == 2:
        return ["J0", "J1"]
    if num_intersections == 4:
        return [f"J_{r}_{c}" for r in range(2) for c in range(2)]
    if num_intersections == 8:
        return [f"J_{r}_{c}" for r in range(2) for c in range(4)]
    raise ValueError(f"Unsupported num_intersections: {num_intersections}")


# ------------------------------------------------------------------------------
#  Helper: assign lanes to intersections
# ------------------------------------------------------------------------------

"""
def assign_lanes_to_intersections(
    leaders: Dict[str, Any],
    num_intersections: int,
) -> List[Dict[str, Any]]:
    
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
"""

def assign_lanes_to_intersections(
    leaders: Dict[str, Any],
    num_intersections: int,
    junction_ids: List[str],
) -> List[Dict[str, Any]]:
    """
    Split the global leaders dict into one sub-dict per intersection.
 
    Matching strategy — each lane ID is checked against every junction ID:
        Single "C":     "N_to_C_0"     contains "C"     -> intersection 0
        Chain  "J0":    "N0_to_J0_0"   contains "J0"    -> intersection 0
               "J1":    "S1_to_J1_0"   contains "J1"    -> intersection 1
        Grid "J_0_1":   "N_0_1_to_J_0_1_0" contains "J_0_1" -> that intersection
 
    A lane is assigned to the FIRST junction whose ID appears in the lane string.
    Lanes that don't match any junction go to intersection 0 (safety fallback).
 
    Parameters
    ----------
    leaders : dict {lane_id: vehicle_id | None}
    num_intersections : int
    junction_ids : list[str]  ordered junction IDs (from get_junction_ids)
 
    Returns
    -------
    list of dicts, one per intersection
    """
    if num_intersections == 1:
        return [dict(leaders)]
 
    buckets: List[Dict[str, Any]] = [{} for _ in range(num_intersections)]
 
    for lane_id, vehicle_id in leaders.items():
        assigned = False
        for idx, jid in enumerate(junction_ids):
            # Check if this junction ID appears in the lane name
            # Use "_to_{jid}" to avoid "J0" matching inside "J0_to_J1"
            if f"_to_{jid}" in lane_id or f"_to_{jid}_" in lane_id:
                buckets[idx][lane_id] = vehicle_id
                assigned = True
                break
        if not assigned:
            buckets[0][lane_id] = vehicle_id  # safety fallback
 
    return buckets


# ------------------------------------------------------------------------------
# Per-intersection cropped image builder
# ------------------------------------------------------------------------------

def build_per_intersection_obs(
    obs:               Dict[str, Any],
    env:               SumoEnvironment,
    per_int_leaders:   List[Dict[str, Any]],
    junction_ids:      List[str],
    num_intersections: int,
    image_size:        int,
) -> List[Dict[str, Any]]:
    """
    Build one observation dict per intersection.
 
    Single intersection -> full image (no crop needed).
    Multiple intersections -> each agent gets a 50x50 image cropped to
    its junction's bounding box (junction centre ± _CROP_RADIUS_M metres).
 
    Parameters
    ----------
    obs               : global observation from env.step() / env.reset()
    env               : SumoEnvironment (access to observation_builder)
    per_int_leaders   : list of per-intersection leader dicts
    junction_ids      : ordered list of junction IDs (from get_junction_ids)
    num_intersections : int
    image_size        : int
 
    Returns
    -------
    list of obs dicts, one per intersection
    """
    if num_intersections == 1:
        return [{
            "image":   obs["image"],
            "leaders": per_int_leaders[0],
            "state":   obs.get("state", {}),
        }]
 
    result = []
    for i, (local_leaders, jid) in enumerate(zip(per_int_leaders, junction_ids)):
        # Crop the image to this junction's bbox
        cropped_image = env.observation_builder.build_image_for_intersection(
            intersection_id = i,
            junction_id     = jid,
            n               = image_size,
        )
        result.append({
            "image":   cropped_image,
            "leaders": local_leaders,
            "state":   obs.get("state", {}),
        })
 
    return result

# ------------------------------------------------------------------------------
#  Multi-agent action combiner
# ------------------------------------------------------------------------------

def multi_agent_act(
    agents:             List[IntersectionAgent],
    call_managers:      List[AgentCallManager],
    per_int_obs:        List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Run each agent if its AgentCallManager says a new decision is needed,
    otherwise replay the agent's current held action.
 
    Parameters
    ----------
    agents         : one IntersectionAgent per intersection
    call_managers  : one AgentCallManager per intersection
    per_int_obs    : one observation dict per intersection (cropped image + local leaders)
 
    Returns
    -------
    combined_action : dict {lane_id: 0 | 1}
    """
    combined_action: Dict[str, int] = {}

    # zip() pairs each agent with its specific bucket of lanes.
    # If there is 1 agent, this loop runs 1 time.
    # If there are 8 agents, this loop runs 8 times.
    for agent, manager, local_obs in zip(agents, call_managers, per_int_obs):
        local_leaders = local_obs["leaders"]
 
        if manager.needs_new_decision(local_leaders):
            # New vehicles at the intersection -> ask the agent
            local_action = agent.act(local_obs)
            manager.update(local_leaders, local_action)
        else:
            # Nothing changed -> hold the previous decision
            local_action = manager.current_action()
 
        combined_action.update(local_action)
 
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
# Visualization helper
# ------------------------------------------------------------------------------
def visualize_observations(step: int, global_obs: Dict[str, Any], per_int_obs: List[Dict[str, Any]]) -> None:
    """
    Displays the global image and the cropped images for each intersection.
    """
    import numpy as np  # Make sure NumPy is imported at the top of your file
    num_intersections = len(per_int_obs)

    # Create a figure: 1 panel for the global image + 1 panel per intersection
    fig, axes = plt.subplots(1, num_intersections + 1, figsize=(4 * (num_intersections + 1), 4))

    # Safety check in case matplotlib returns a single object instead of an array
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # 1. Global image (full network view)
    axes[0].imshow(global_obs["image"])
    axes[0].set_title(f"Global Network (Step {step})")
    axes[0].axis('off')

    # 2. Local images (the individual view of each agent)
    for i, local_obs in enumerate(per_int_obs):
        ax = axes[i + 1]
        ax.imshow(local_obs["image"])
        ax.set_title(f"Agent {i} Crop (50x50)")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  SUMO Traffic Simulation - DQN Agent")
    print("=" * 65)

    num_intersections = SCENARIO["num_intersections"]
    intersection_type = SCENARIO["intersection_type"]

    junction_ids = get_junction_ids(num_intersections)


    # -- 1. Build the road network -----------------------------------------
    print("\n[1/5] Building road network...")
    builder  = NetworkBuilder(output_dir=CONFIGS_DIR)
    net_file = builder.build(
        intersection_type = intersection_type,
        num_intersections = num_intersections,
        num_lanes = SCENARIO["num_lanes"],
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
        net_file      = net_file,
        route_file    = route_file,
        use_gui       = USE_GUI,
        step_length   = 1.0,
        max_steps     = SIMULATION_DURATION,
        image_size    = IMAGE_SIZE,
        dest_colors   = DEST_COLORS,
        intersection_outgoing = INTERSECTION_OUTGOING,
    )
    env.start()

    # -- 4. Create agents and call managers : one per intersection -----------------------
    print(f"\n[4/5] Loading {num_intersections} agent(s) from '{MODEL_PATH}'...")
    agents: List[IntersectionAgent] = [
        IntersectionAgent(
            model_path      = MODEL_PATH,
            image_size      = IMAGE_SIZE,
            intersection_id = i,
        )
        for i in range(num_intersections)
    ]
    call_managers: List[AgentCallManager] = [
        AgentCallManager(intersection_id=i)
        for i in range(num_intersections)
    ]
    print(f"  {len(agents)} agent(s) ready.")
    print(f"  Junction IDs: {junction_ids}")

    # -- 5. RL loop --------------------------------------------------------
    print(f"\n[5/5] Running RL loop (print every {PRINT_EVERY} steps)...")
 
    obs = env.reset()

    # Clear the intersection bbox cache so it's recomputed for this episode
    env.observation_builder.reset_bbox_cache()
    for mgr in call_managers:
        mgr.reset()
 
    done = False
    total_reward = 0.0
 
    print(f"  Initial observation — image shape: {obs['image'].shape}\n")


    while not done:
        # Split leaders per intersection using junction ID matching
        per_int_leaders = assign_lanes_to_intersections(
            obs["leaders"], num_intersections, junction_ids
        )
 
        # Build one cropped image per intersection
        per_int_obs = build_per_intersection_obs(
            obs               = obs,
            env               = env,
            per_int_leaders   = per_int_leaders,
            junction_ids      = junction_ids,
            num_intersections = num_intersections,
            image_size        = IMAGE_SIZE,
        )
 
        # Smart agent call + full per-lane action dict
        action = multi_agent_act(agents, call_managers, per_int_obs)
 
        # Step the environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
 
        # Visualize at step 100 if enabled
        if info["step"] == 100 and SHOW_CROPPED_IMAGE:
            visualize_observations(info["step"], obs, per_int_obs)
 
        if info["step"] % PRINT_EVERY == 0 or done:
            print_step(info["step"], reward, obs, info)
 
    print(f"\n  Total episode reward (vehicles arrived): {total_reward:.0f}")
    print()
    env.statistics_collector.print_summary()
 
    try:
        df = env.statistics_collector.to_dataframe()
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
