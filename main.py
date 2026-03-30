# main.py
"""
This script demonstrates the full pipeline:

  Step 1 : Build a road network with NetworkBuilder
  Step 2 : Generate traffic demand with DemandGenerator
  Step 3 : Run the simulation with SumoEnvironment
  Step 4 : Print statistics at each step
  Step 5 : Print the episode summary

Run this file directly:
    python main.py

Before running, make sure:
  1. SUMO is installed:  https://sumo.dlr.de/docs/Downloads.php
  2. SUMO_HOME environment variable is set:
       Linux/Mac: export SUMO_HOME=/path/to/sumo
       Windows:   set SUMO_HOME=C:\\path\\to\\sumo
  3. Dependencies are installed:
       pip install -r requirements.txt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_sim.network.network_builder import NetworkBuilder
from traffic_sim.network.demand_generator import DemandGenerator
from traffic_sim.sumo_env.sumo_environment import SumoEnvironment



SCENARIO = {
    "intersection_type": "t_junction",   # "four_way" | "t_junction" | "complex"
    "num_intersections": 4,            # 1 | 2 | 4 | 8
    "num_lanes":         3,            # 1 | 2 | 3
}

DEMAND_LEVEL  = "moderate"   # "low" | "moderate" | "high" | "congested"
SIMULATION_DURATION = 1000   
USE_GUI       = True         #  True for opening the visual window
CONFIGS_DIR   = "configs"
PRINT_EVERY   = 50           # print stats every N steps


def print_state(step: int, state: dict, stats: dict) -> None:
    """Print a one-line summary of the current simulation state."""
    print(
        f"  step={step:>4} | "
        f"t={state['simulation_time']:>7.1f}s | "
        f"vehicles={stats['total_vehicles']:>3} | "
        f"mean_speed={stats['mean_speed']:>5.2f} m/s | "
        f"waiting={stats['mean_waiting_time']:>6.1f}s | "
        f"throughput={stats['throughput']:>4} | "
        f"congestion={stats['congestion_index']:.2f}"
    )

def main():
    print("=" * 60)
    print("  SUMO Traffic Simulation — Demo")
    print("=" * 60)

    # 1. Here we build the road network
    print("\n[1/4] Building road network...")

    builder = NetworkBuilder(output_dir=CONFIGS_DIR)
    net_file = builder.build(
        intersection_type=SCENARIO["intersection_type"],
        num_intersections=SCENARIO["num_intersections"],
        num_lanes=SCENARIO["num_lanes"],
    )

    # 2. Generate traffic demand
    print(f"\n[2/4] Generating '{DEMAND_LEVEL}' traffic demand...")

    demand_gen = DemandGenerator(net_file=net_file)
    route_file = demand_gen.generate(
        level=DEMAND_LEVEL,
        duration=SIMULATION_DURATION,
    )

    # 3. Initialize and run the simulation
    print(f"\n[3/4] Starting simulation ({SIMULATION_DURATION}s episode)...")

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=USE_GUI,
        step_length=1.0,
        max_steps=SIMULATION_DURATION,
    )

    # start() validates the environment (checks SUMO is installed etc.)
    env.start()

    # reset() launches SUMO, connects TraCI, returns the initial state
    state = env.reset()
    print(f"\n  Initial state: {len(state['vehicle_ids'])} vehicles in network")


    # Main control loop
    print(f"\n[4/4] Running simulation loop (print every {PRINT_EVERY} steps)...\n")

    step = 0
    done = False

    while not done:
        # Retrieve leaders
        # Get the frontmost vehicle per lane
        leaders = env.get_leaders()

        # Compose an action 
        # Example policy: give leaders their full speed, stop everyone else.
        # This demonstrates set_action() and set_loaded_veh().
        #
        # For a pure no-control baseline, pass action=None to step():
        #   state, info = env.step(action=None)
        #
        env.action_handler.set_loaded_veh(leaders)   # stop non-leaders
        env.action_handler.set_speedMode(mode=22)    # give code speed control

        #  Step the simulation 
        state, info = env.step(action=None)   # action already applied above
        done = info["done"]
        step = info["step"]

        #  Collect and display stats 
        if step % PRINT_EVERY == 0 or done:
            stats = env.statistics()
            print_state(step, state, stats)

    # 5. Episode summary
    print()
    env.statistics_collector.print_summary()

    # Optional : export to CSV for analysis
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