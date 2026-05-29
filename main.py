# main.py
"""
SUMO Traffic Simulation — RL-ready demo
========================================

What this file shows
--------------------
  Step 1 : Build a road network   (NetworkBuilder)
  Step 2 : Generate traffic demand (DemandGenerator)
  Step 3 : Run the RL loop        (SumoEnvironment)

           obs = env.reset()           -> {"image": ndarray(50,50,3), ...}
           obs, reward, done, info = env.step(action)

  The "agent" used here is a rule-based placeholder that lets every leader
  pass.  Replace ``placeholder_agent()`` with your trained model.

Action format
-------------
    {lane_id: 0 | 1}
        1 -> the leader of that lane is allowed to go
        0 -> the leader of that lane is stopped

Reward
------
    +1 for every vehicle that completed its route this step.
    (traci.simulation.getArrivedNumber() — teacher's reference formula)

Prerequisites
-------------
    1. SUMO installed   https://sumo.dlr.de/docs/Downloads.php
    2. SUMO_HOME set    export SUMO_HOME=/path/to/sumo
    3. Dependencies     pip install numpy
"""

import os
import sys
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_sim.network.network_builder   import NetworkBuilder
from traffic_sim.network.demand_generator  import DemandGenerator
from traffic_sim.sumo_env.sumo_environment import SumoEnvironment

# ------------------------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------------------------

SCENARIO = {
    "intersection_type": "four_way", # "four_way" | "t_junction" | "complex"
    "num_intersections": 1,  # 1 | 2 | 4 | 8
    "num_lanes":         3, # 1 | 2 | 3 
}

DEMAND_LEVEL        = "moderate"   # "low" | "moderate" | "high" | "congested"
SIMULATION_DURATION = 300          # seconds
USE_GUI             = True
CONFIGS_DIR         = "configs"
PRINT_EVERY         = 50

# -- Observation ---------------------------------------------------------------
IMAGE_SIZE = 50   # the agent receives a (50 × 50 × 3) uint8 RGB image

# -- Adapt these to YOUR network's exit edge names -----------------------------
#    (leave as None to use the teacher's default tables)
DEST_COLORS: Optional[Dict] = None
# Example for the NetworkBuilder four_way_1int network:
# DEST_COLORS = {
#     "C_to_N": (1.0, 0.0, 0.0),   # red    -> north exit
#     "C_to_S": (0.0, 1.0, 0.0),   # green  -> south exit
#     "C_to_E": (0.0, 0.0, 1.0),   # blue   -> east  exit
#     "C_to_W": (1.0, 1.0, 0.0),   # yellow -> west  exit
# }

# TODO un agent par intersection donc envoie une intersection a un agent
# et lui il va controler une intersection
# L'agent respecte un ordre 
#. West, Sud, Est, North 
# Une intersection par agent 
# Pas prendre en compte les Formes complex
# prendre les couleurs généré par le prof 

INTERSECTION_OUTGOING: Optional[set] = None
# Example:
# INTERSECTION_OUTGOING = {"C_to_N", "C_to_S", "C_to_E", "C_to_W"}


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

def placeholder_agent(obs: Dict[str, Any]) -> Dict[str, int]:
    """
    Rule-based placeholder — lets every leader pass.

    Replace the body of this function with your real model, e.g.:

        image  = obs["image"]                    # shape (50, 50, 3)
        tensor = preprocess(image)               # your preprocessing
        action = my_model.predict(tensor)        # {lane_id: 0|1}
        return action
    """
    action = {}
    for lane_id, vehicle_id in obs["leaders"].items():
        if vehicle_id is not None:
            action[lane_id] = 1   # go
    return action


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
    print("  SUMO Traffic Simulation — RL Interface Demo")
    print("=" * 65)

    # -- 1. Build the road network -----------------------------------------
    print("\n[1/4] Building road network...")
    builder  = NetworkBuilder(output_dir=CONFIGS_DIR)
    net_file = builder.build(
        intersection_type = SCENARIO["intersection_type"],
        num_intersections = SCENARIO["num_intersections"],
        num_lanes         = SCENARIO["num_lanes"],
    )

    # -- 2. Generate traffic demand ----------------------------------------
    print(f"\n[2/4] Generating '{DEMAND_LEVEL}' traffic demand...")
    demand_gen = DemandGenerator(net_file=net_file)
    route_file = demand_gen.generate(
        level    = DEMAND_LEVEL,
        duration = SIMULATION_DURATION,
    )

    # -- 3. Create the environment -----------------------------------------
    print(f"\n[3/4] Creating RL environment ({SIMULATION_DURATION}s episode)...")
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

    # -- 4. RL loop --------------------------------------------------------
    print(f"\n[4/4] Running RL loop (print every {PRINT_EVERY} steps)...")
    print(
        f"\n  Each step the agent receives:\n"
        f"    obs['image']   — ndarray ({IMAGE_SIZE}×{IMAGE_SIZE}×3)  RGB pixel map\n"
        f"    obs['leaders'] — {{lane_id: vehicle_id | None}}\n"
        f"  And returns: {{lane_id: 0|1}}\n"
    )

    obs  = env.reset()
    done = False
    total_reward = 0.0

    print(f"  Initial observation — image shape: {obs['image'].shape}\n")

    while not done:
        # -- Agent chooses an action based on the image --------------------
        # obs["image"]   -> ndarray (50, 50, 3) — the only input to your model
        # obs["leaders"] -> which vehicle is the leader per lane (for action keys)
        action = placeholder_agent(obs)

        # -- Step the environment ------------------------------------------
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if info["step"] == 100:
            plt.imshow(obs["image"])
            plt.title(f"Generated vector at the step {info['step']}")
            plt.axis('off') # On cache les axes (0 à 50)
            plt.show()

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