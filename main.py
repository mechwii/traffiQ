# main.py
"""
SUMO Traffic Simulation - DQN Agent

This is the entry point of the project. It wires together all the components
and runs a full episode of the RL simulation.

What happens when you run this file:
    Step 1: Build a road network file (.net.xml) via NetworkBuilder.
    Step 2: Generate a traffic demand file (.rou.xml) via DemandGenerator.
    Step 3: Run the RL control loop using SumoEnvironment.

One IntersectionAgent is created per intersection. Each agent wraps the
teacher's pre-trained CNN (DQN) and receives:
    obs["image"]   -> (50, 50, 3) uint8 RGB image of the network
    obs["leaders"] -> {lane_id: vehicle_id | None} filtered to its own lanes

The agent returns {lane_id: 0 | 1} (1=go, 0=stop).

Key design decisions:

    Smart agent call timing (AgentCallManager):
        The AI is only re-called when something meaningful has changed at an
        intersection: a new leader appeared on an empty lane, or a previously
        allowed vehicle just cleared the junction. Between those events the last
        decision is simply replayed. This avoids stop/go oscillation.

    Single-step step():
        env.step() advances the simulation by exactly one simulationStep().
        The teacher's reference uses multi-step (loop until leaders cross),
        which causes stuttering on multi-intersection networks because one
        intersection blocks all others. Single-step avoids that.

    Per-intersection cropped image:
        For multi-intersection networks, each agent receives a 50x50 image
        cropped around its own junction rather than the full network image.

    Shared model weights:
        All agents share a single Keras model loaded once from disk. Loading it
        N times for N intersections would waste memory and startup time.

    Safety filter:
        IntersectionAgent._safety_filter() prevents conflicting perpendicular
        "go" decisions (e.g. North and East crossing at the same time when both
        go straight). Only combinations proven safe by the route-aware conflict
        table are allowed through.

Action bit order (matching the teacher's training convention):
    North -> East -> West -> South
    (mapped to bits 0, 2, 4, 6 of the 8-bit integer action)

Supported scenarios:
    intersection_type : "four_way" | "t_junction"  (complex is not supported by the model)
    num_intersections : 1 | 2 | 4 | 8
    num_lanes         : 1 | 2 | 3

Prerequisites:
    1. SUMO installed: https://sumo.dlr.de/docs/Downloads.php
    2. SUMO_HOME set:  export SUMO_HOME=/path/to/sumo
    3. Pre-trained model at MODEL_PATH (teacher's ./save_model directory)
    4. pip install numpy tensorflow keras
"""

import os
import sys
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_sim.network.network_builder   import NetworkBuilder
from traffic_sim.network.demand_generator  import DemandGenerator
from traffic_sim.env.sumo_environment      import SumoEnvironment
from traffic_sim.agents.intersection_agent import IntersectionAgent
from traffic_sim.agents.agent_call_manager import AgentCallManager
from plot_result import plot_metrics


# Scenario configuration. Change these to run a different network setup.
SCENARIO = {
    "intersection_type": "four_way",   # "four_way" | "t_junction" | "complex"
    "num_intersections": 1,            # 1 | 2 | 4 | 8
    "num_lanes":         1,            # 1 | 2 | 3
}

DEMAND_LEVEL        = "congested"  # "low" | "moderate" | "high" | "congested"
SIMULATION_DURATION = 300          # total episode duration in seconds
USE_GUI             = True
CONFIGS_DIR         = "configs"
PRINT_EVERY         = 50           # print a status line every N steps

# Path to the teacher's pre-trained SavedModel directory.
MODEL_PATH = os.path.join(".", "traffic_sim", "models", "save_model")

# Image side length in pixels. Must match what the model was trained on (50).
IMAGE_SIZE = 50

# If True, show a matplotlib window with the per-intersection cropped images at step 100.
SHOW_CROPPED_IMAGE = True

# Optional: provide custom color tables for the observation image.
# None means we use the built-in defaults from ObservationBuilder.
DEST_COLORS:           Optional[Dict] = None
INTERSECTION_OUTGOING: Optional[set]  = None

REWARD_TYPE = "combined"  # "arrived" | "waiting" | "congestion" | "combined"

# =====================================================================
#  Helper: Junction ID resolver
# =====================================================================

def get_junction_ids(num_intersections: int) -> List[str]:
    """
    Return the ordered list of junction node IDs for a given intersection count.

    These must match the naming convention used by NetworkBuilder exactly:
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


# =====================================================================
#  Helper: assign lanes to intersections
# =====================================================================

def assign_lanes_to_intersections(
    leaders:           Dict[str, Any],
    num_intersections: int,
    junction_ids:      List[str],
) -> List[Dict[str, Any]]:
    """
    Split the global leaders dict into one sub-dict per intersection.

    A lane belongs to a junction when the junction ID appears in the destination
    part (after "_to_") of the lane name. Outgoing lanes (whose destination is a
    border node N/S/E/W) are simply skipped rather than being assigned to bucket 0,
    which would pollute that intersection's decision with irrelevant lanes.

    Args:
        leaders: The full {lane_id: vehicle_id | None} dict from the observation.
        num_intersections: Total number of intersections in the network.
        junction_ids: Ordered list of junction node IDs from get_junction_ids().

    Returns:
        A list of per-intersection leader dicts, one per agent.
    """
    if num_intersections == 1:
        return [dict(leaders)]

    buckets: List[Dict[str, Any]] = [{} for _ in range(num_intersections)]

    for lane_id, vehicle_id in leaders.items():
        if "_to_" not in lane_id:
            continue

        dest_part = lane_id.split("_to_", 1)[1]

        for idx, jid in enumerate(junction_ids):
            if dest_part == jid or dest_part.startswith(jid + "_"):
                buckets[idx][lane_id] = vehicle_id
                break
        # Outgoing lanes (destination is a border node) don't match any junction
        # and are intentionally left out of all buckets.

    return buckets


# =====================================================================
# Per-intersection cropped image builder
# =====================================================================

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

    For a single intersection we just return the full image as-is.
    For multi-intersection networks we crop a 50x50 window around each junction
    so each agent only sees the area it controls.

    Args:
        obs: The raw observation returned by env.step() or env.reset().
        env: The running environment (needed to access ObservationBuilder).
        per_int_leaders: List of per-intersection leader dicts.
        junction_ids: Ordered junction IDs from get_junction_ids().
        num_intersections: Total number of intersections.
        image_size: Output image side length in pixels.

    Returns:
        List of observation dicts, one per agent.
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


# =====================================================================
#  Multi-agent action combiner
# =====================================================================

def multi_agent_act(
    agents:        List[IntersectionAgent],
    call_managers: List[AgentCallManager],
    per_int_obs:   List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Collect a go/stop decision from every agent and merge them into one dict.

    Each agent is only called if its AgentCallManager says the state has changed
    enough to warrant a new decision. Otherwise the previous decision is replayed.

    Returns a single combined {lane_id: 0|1} dict covering all intersections.
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


# =====================================================================
#  Print helper
# =====================================================================

def print_step(step: int, reward: float, obs: Dict, info: Dict) -> None:
    """Print a one-line status summary for the current step."""
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


# =====================================================================
# Visualization helper
# =====================================================================

def visualize_observations(
    step:        int,
    global_obs:  Dict[str, Any],
    per_int_obs: List[Dict[str, Any]],
) -> None:
    """
    Display the global network image alongside each agent's cropped image.

    Only called once at step 100 when SHOW_CROPPED_IMAGE is True.
    Useful for verifying that the crop bounding boxes are centered correctly
    and that the color encoding looks right.
    """
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
    axes[0].axis("off")

    for i, local_obs in enumerate(per_int_obs):
        axes[i + 1].imshow(local_obs["image"])
        axes[i + 1].set_title(f"Agent {i} Crop (50x50)")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


# =====================================================================
#  Main
# =====================================================================

def main():
    print("=" * 65)
    print("  SUMO Traffic Simulation - DQN Agent")
    print("=" * 65)

    num_intersections = SCENARIO["num_intersections"]
    intersection_type = SCENARIO["intersection_type"]

    # The pre-trained model only supports 4-direction layouts.
    # Fall back to four_way if complex is requested.
    if intersection_type == "complex":
        print(
            "\n  WARNING: 'complex' intersection type has more than 4 arms.\n"
            "  The pre-trained DQN model only supports 4 directions (N/E/W/S).\n"
            "  Falling back to 'four_way'.\n"
        )
        intersection_type = "four_way"

    junction_ids = get_junction_ids(num_intersections)

    # Step 1: Build the road network file (.net.xml)
    print("\n[1/5] Building road network...")
    builder  = NetworkBuilder(output_dir=CONFIGS_DIR)
    net_file = builder.build(
        intersection_type = intersection_type,
        num_intersections = num_intersections,
        num_lanes         = SCENARIO["num_lanes"],
    )

    # Step 2: Generate traffic demand file (.rou.xml)
    print(f"\n[2/5] Generating '{DEMAND_LEVEL}' traffic demand...")
    demand_gen = DemandGenerator(net_file=net_file)
    route_file = demand_gen.generate(
        level    = DEMAND_LEVEL,
        duration = SIMULATION_DURATION,
    )

    # Step 3: Create RL environment
    print(f"\n[3/5] Creating RL environment ({SIMULATION_DURATION}s episode)...")
    env = SumoEnvironment(
        net_file              = net_file,
        route_file            = route_file,
        use_gui               = USE_GUI,
        step_length           = 1.0,
        max_steps             = SIMULATION_DURATION,
        image_size            = IMAGE_SIZE,
        dest_colors           = DEST_COLORS,
        intersection_outgoing = INTERSECTION_OUTGOING,
        reward_type           = REWARD_TYPE,
    )
    env.start()

    # Step 4: Create agents and their call managers
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

    # Step 5: Run the RL loop
    print(f"\n[5/5] Running RL loop (print every {PRINT_EVERY} steps)...")

    obs = env.reset()

    # Clear the bbox cache so crops are recomputed for the new episode.
    env.observation_builder.reset_bbox_cache()
    for mgr in call_managers:
        mgr.reset()

    done         = False
    total_reward = 0.0
    agent_steps  = 0

    print(f"  Initial observation -> image shape: {obs['image'].shape}\n")

    while not done:
        agent_steps += 1

        # Split the global leaders dict into one sub-dict per intersection for the agents.
        per_int_leaders = assign_lanes_to_intersections(
            obs["leaders"], num_intersections, junction_ids
        )

        # Build one observation dict per intersection, cropping the image around each junction.
        per_int_obs = build_per_intersection_obs(
            obs               = obs,
            env               = env,
            per_int_leaders   = per_int_leaders,
            junction_ids      = junction_ids,
            num_intersections = num_intersections,
            image_size        = IMAGE_SIZE,
        )

        # Collect a go/stop decision from every agent and merge them into one dict.
        action = multi_agent_act(agents, call_managers, per_int_obs)

        # Take a step in the environment with the combined action and get the new observation and reward.
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Show a visualization window once at step 100 for debugging.
        if info["step"] == 100 and SHOW_CROPPED_IMAGE:
            visualize_observations(info["step"], obs, per_int_obs)

        if info["step"] % PRINT_EVERY == 0 or done:
            print_step(info["step"], reward, obs, info)

    print(f"\n  Total episode reward ({REWARD_TYPE}): {total_reward:.0f}")
    print(f"  Agent steps: {agent_steps}")
    print()
    env.statistics_collector.print_summary()

    try:
        df       = env.statistics_collector.to_dataframe()
        csv_path = os.path.join(
            os.path.dirname(route_file), f"results_{DEMAND_LEVEL}.csv"
        )
        df.to_csv(csv_path, index=False)
        plot_metrics(csv_path)
        print(f"  Results saved to: {csv_path}")
    except ImportError:
        print("  (Install pandas to export results to CSV)")

    env.close()
    print("\nSimulation complete.")


if __name__ == "__main__":
    main()