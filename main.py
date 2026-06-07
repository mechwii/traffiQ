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

Key design decisions
--------------------

  1. Smart agent call timing (AgentCallManager)
      The AI is called only when something meaningful changes:
      a new leader appears on a lane that was empty, OR a previously
      allowed vehicle has cleared the intersection.  The current
      decision is held until that condition triggers again.

  2. Multi-step step()
      env.step(action) advances the simulation MULTIPLE sub-steps
      until the "go" leaders have crossed the intersection.  This
      matches the teacher's reference and prevents stop/go oscillation.

  3. Per-intersection cropped image
      For multi-intersection networks, each agent receives a 50x50
      image cropped to its own intersection's bounding box.

  4. Shared model weights
      All agents share a single Keras model instance loaded once from
      disk.  This saves memory and load time.

  5. Safety filter
      The agent's _safety_filter() prevents conflicting perpendicular
      "go" decisions (e.g. N and E simultaneously).  Only parallel
      direction pairs (N+S or E+W) are allowed.

Reward
------
    +1 for every vehicle that completed its route (summed across all
    sub-steps within one agent step).

Action order (per intersection, matching the teacher's training)
----------------------------------------------------------------
  North -> East -> West -> South
  (mapped to bits 0, 2, 4, 6 of the 8-bit integer action)

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
    "intersection_type": "four_way",  # "four_way" | "t_junction" | "complex"
    "num_intersections": 1,           # 1 | 2 | 4 | 8
    "num_lanes":         1,           # 1 | 2 | 3
}

DEMAND_LEVEL        = "congested"     # "low" | "moderate" | "high" | "congested"
SIMULATION_DURATION = 300       # seconds
USE_GUI             = True
CONFIGS_DIR         = "configs"
PRINT_EVERY         = 50

# Path to the pre-trained SavedModel directory
MODEL_PATH = os.path.join(".", "traffic_sim", "save_model")

# Observation image size — must match what the model was trained on
IMAGE_SIZE = 50
SHOW_CROPPED_IMAGE = True

DEST_COLORS: Optional[Dict] = None
INTERSECTION_OUTGOING: Optional[set] = None


# ------------------------------------------------------------------------------
#  Helper: Junction ID resolver
# ------------------------------------------------------------------------------

def get_junction_ids(num_intersections: int) -> List[str]:
    """
    Return the ordered list of junction node IDs for a given count.

    Matches NetworkBuilder naming exactly:
        1  -> ["C"]
        2  -> ["J0", "J1"]
        4  -> ["J_0_0", "J_0_1", "J_1_0", "J_1_1"]
        8  -> ["J_0_0" ... "J_1_3"]
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

def assign_lanes_to_intersections(
    leaders: Dict[str, Any],
    num_intersections: int,
    junction_ids: List[str],
) -> List[Dict[str, Any]]:
    """
    Split the global leaders dict into one sub-dict per intersection.

    A lane is INCOMING to a junction when the junction ID appears in
    the DESTINATION part (after "_to_") of the lane name.

    Outgoing lanes (destination is a border node N/S/E/W) are SKIPPED
    rather than falling back to bucket 0.
    """
    if num_intersections == 1:
        return [dict(leaders)]

    buckets: List[Dict[str, Any]] = [{} for _ in range(num_intersections)]

    for lane_id, vehicle_id in leaders.items():
        if "_to_" not in lane_id:
            continue

        dest_part = lane_id.split("_to_", 1)[1]

        assigned = False
        for idx, jid in enumerate(junction_ids):
            if dest_part == jid or dest_part.startswith(jid + "_"):
                buckets[idx][lane_id] = vehicle_id
                assigned = True
                break

        # FIX: outgoing lanes are simply skipped — no fallback to bucket 0
        if not assigned:
            pass

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
    Multiple intersections -> each agent gets a cropped image centred
    on its junction.
    """
    if num_intersections == 1:
        return [{
            "image":   obs["image"],
            "leaders": per_int_leaders[0],
            "state":   obs.get("state", {}),
        }]

    result = []
    for i, (local_leaders, jid) in enumerate(zip(per_int_leaders, junction_ids)):
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
    agents:        List[IntersectionAgent],
    call_managers: List[AgentCallManager],
    per_int_obs:   List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Run each agent if its AgentCallManager says a new decision is needed,
    otherwise replay the agent's current held action.
    """
    combined_action: Dict[str, int] = {}

    for agent, manager, local_obs in zip(agents, call_managers, per_int_obs):
        local_leaders = local_obs["leaders"]

        if manager.needs_new_decision(local_leaders):
            local_action = agent.act(local_obs)
            manager.update(local_leaders, local_action)
        else:
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

def visualize_observations(
    step: int,
    global_obs: Dict[str, Any],
    per_int_obs: List[Dict[str, Any]],
) -> None:
    """Display the global image and the cropped images for each intersection."""
    import numpy as np
    num_intersections = len(per_int_obs)

    fig, axes = plt.subplots(
        1, num_intersections + 1,
        figsize=(4 * (num_intersections + 1), 4),
    )

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    axes[0].imshow(global_obs["image"])
    axes[0].set_title(f"Global Network (Step {step})")
    axes[0].axis('off')

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

    # Guard: complex intersections are not supported by the 4-direction model
    if intersection_type == "complex":
        print(
            "\n  WARNING: 'complex' intersection type has more than 4 arms.\n"
            "  The pre-trained DQN model only supports 4 directions (N/E/W/S).\n"
            "  Falling back to 'four_way'.\n"
        )
        intersection_type = "four_way"

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

    # -- 4. Create agents (shared model) and call managers -----------------
    print(f"\n[4/5] Loading {num_intersections} agent(s) from '{MODEL_PATH}'...")
    agents: List[IntersectionAgent] = IntersectionAgent.create_agents(
        count      = num_intersections,
        model_path = MODEL_PATH,
        image_size = IMAGE_SIZE,
    )
    call_managers: List[AgentCallManager] = [
        AgentCallManager(intersection_id=i)
        for i in range(num_intersections)
    ]
    print(f"  {len(agents)} agent(s) ready.")
    print(f"  Junction IDs: {junction_ids}")

    # -- 5. RL loop --------------------------------------------------------
    print(f"\n[5/5] Running RL loop (print every {PRINT_EVERY} steps)...")

    obs = env.reset()

    env.observation_builder.reset_bbox_cache()
    for mgr in call_managers:
        mgr.reset()

    done = False
    total_reward = 0.0
    agent_steps  = 0

    print(f"  Initial observation — image shape: {obs['image'].shape}\n")

    while not done:
        agent_steps += 1

        # Split leaders per intersection
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

        # Step the environment (multi-step internally)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Visualize at step 100 if enabled
        if info["step"] == 100 and SHOW_CROPPED_IMAGE:
            visualize_observations(info["step"], obs, per_int_obs)

        if info["step"] % PRINT_EVERY == 0 or done:
            print_step(info["step"], reward, obs, info)

    print(f"\n  Total episode reward (vehicles arrived): {total_reward:.0f}")
    print(f"  Agent steps: {agent_steps}")
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