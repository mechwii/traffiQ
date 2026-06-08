# traffic_sim/env/statistics.py
"""
StatisticsCollector

Collects and computes traffic simulation metrics from the running SUMO simulation via TraCI.
This class acts as the central metrics hub, implementing the statistics() function 
from the project spec, while also handling per-episode data logging and DataFrame exports.

Metrics collected:
  - mean_speed: Average speed of all active vehicles (m/s).
  - mean_waiting_time: Average accumulated waiting time per vehicle (s).
  - throughput: Number of vehicles that have completed their trip.
  - total_vehicles: Vehicles currently active in the network.
  - mean_travel_time: Average travel time of completed trips (s).
  - step: Current simulation step.

TraCI calls used:
  traci.vehicle.getIDList()                    -> Get active vehicles
  traci.vehicle.getSpeed(vid)                  -> Get current speed
  traci.vehicle.getAccumulatedWaitingTime(vid) -> Get waiting time
  traci.simulation.getTime()                   -> Get simulation clock
  traci.simulation.getArrivedNumber()          -> Vehicles that finished this step
  traci.simulation.getDepartedNumber()         -> Vehicles that entered this step
  traci.simulation.getCollisions()             -> List of collisions this step
"""

from typing import Any, Dict, List, Optional
import time

# We wrap the imports in try/except blocks so the module can still be 
# imported and inspected even if the environment lacks SUMO or pandas.
try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Standard SUMO convention: any vehicle moving slower than 0.1 m/s 
# is considered stationary/waiting in a queue.
WAITING_SPEED_THRESHOLD = 0.1


class StatisticsCollector:
    """
    Collects and tracks simulation metrics over a single episode.

    Instantiated by SumoEnvironment right after TraCI connects. 
    The statistics() method should be called at each step to build the episode log.
    """

    def __init__(self):
        # Cumulative counters (these get reset at the start of each episode)
        self._throughput: int = 0
        self._total_departed: int = 0
        self._total_collisions: int = 0

        # Stores the metric dictionary for every single step
        self._history: List[Dict[str, Any]] = []

        # Tracking dictionaries for travel time calculations
        # _departure_times maps vehicle_id to their exact spawn time
        self._departure_times: Dict[str, float] = {}
        self._travel_times: List[float] = [] 

        # We also track wall-clock time to monitor simulation performance
        self._episode_start: float = time.time()

    def statistics(self) -> Dict[str, Any]:
        """
        Pulls the latest data from TraCI, computes derived metrics, and logs them.

        Returns:
            dict: A snapshot of all current metrics (simulation_time, step, 
                  total_vehicles, mean_speed, mean_waiting_time, throughput, 
                  mean_travel_time, vehicles_per_minute, congestion_index).
        """
        sim_time = traci.simulation.getTime()

        # Fetch active vehicles
        vehicle_ids = list(traci.vehicle.getIDList())
        n_vehicles  = len(vehicle_ids)

        # Calculate mean speed across the entire network
        speeds = [traci.vehicle.getSpeed(vid) for vid in vehicle_ids]
        mean_speed = sum(speeds) / n_vehicles if n_vehicles else 0.0

        # Calculate mean accumulated waiting time
        wait_times = [
            traci.vehicle.getAccumulatedWaitingTime(vid)
            for vid in vehicle_ids
        ]
        mean_waiting_time = sum(wait_times) / n_vehicles if n_vehicles else 0.0

        # Update cumulative counters for this step
        arrived_this_step  = traci.simulation.getArrivedNumber()
        departed_this_step = traci.simulation.getDepartedNumber()
        self._throughput       += arrived_this_step
        self._total_departed   += departed_this_step

        # We use getCollisions() instead of getCollidingVehiclesNumber() because 
        # it gives a more reliable count depending on the SUMO version/config.
        collisions_this_step = len(traci.simulation.getCollisions())
        self._total_collisions += collisions_this_step

        # Track when new vehicles enter the network
        for vid in vehicle_ids:
            if vid not in self._departure_times:
                self._departure_times[vid] = sim_time

        # Calculate travel time for vehicles that arrived during this exact step.
        # We use .pop(vid) to remove them from our tracking dict, which calculates 
        # the final travel time AND prevents memory leaks over long simulations.
        for vid in traci.simulation.getArrivedIDList():
            if vid in self._departure_times:
                travel_time = sim_time - self._departure_times.pop(vid)
                if travel_time > 0:
                    self._travel_times.append(travel_time)

        # Average travel time of all completed trips so far
        mean_travel_time = (
            sum(self._travel_times) / len(self._travel_times)
            if self._travel_times
            else 0.0
        )

        # Compute derived system metrics
        elapsed_minutes = (sim_time / 60.0) if sim_time > 0 else 1.0
        vehicles_per_minute = self._throughput / elapsed_minutes

        # Congestion index: ratio of stopped vehicles vs total active vehicles
        waiting_count = sum(1 for s in speeds if s < WAITING_SPEED_THRESHOLD)
        congestion_index = waiting_count / n_vehicles if n_vehicles else 0.0

        # Package the results
        step = len(self._history) + 1
        result = {
            "simulation_time":     sim_time,
            "step":                step,
            "total_vehicles":      n_vehicles,
            "mean_speed":          round(mean_speed, 3),
            "mean_waiting_time":   round(mean_waiting_time, 3),
            "throughput":          self._throughput,
            "mean_travel_time":    round(mean_travel_time, 3),
            "vehicles_per_minute": round(vehicles_per_minute, 2),
            "congestion_index":    round(congestion_index, 3),
            "total_collisions":    self._total_collisions,
        }

        # Save to internal history log
        self._history.append(result)

        return result

    def episode_summary(self) -> Dict[str, Any]:
        """
        Aggregates the per-step history into a single episode-level summary.
        Should be called after the environment signals done=True.

        Returns:
            dict: High-level metrics for the entire run (averages, peaks, totals).
        """
        if not self._history:
            return {"error": "No data collected - call statistics() first."}

        all_speeds     = [h["mean_speed"] for h in self._history]
        all_waits      = [h["mean_waiting_time"] for h in self._history]
        all_congestion = [h["congestion_index"] for h in self._history]

        elapsed_wall = time.time() - self._episode_start

        return {
            "total_steps":             len(self._history),
            "total_throughput":        self._throughput,
            "total_departed":          self._total_departed,
            "episode_mean_speed":      round(sum(all_speeds) / len(all_speeds), 3),
            "episode_mean_wait":       round(sum(all_waits) / len(all_waits), 3),
            "episode_mean_congestion": round(sum(all_congestion) / len(all_congestion), 3),
            "peak_congestion":         round(max(all_congestion), 3),
            "mean_travel_time":        self._history[-1]["mean_travel_time"],
            "wall_time_seconds":       round(elapsed_wall, 2),
            "total_collisions":        self._total_collisions,
        }

    def to_dataframe(self):
        """
        Converts the accumulated history into a pandas DataFrame.
        This makes it trivial to plot results or dump to CSV later.
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for to_dataframe().\n"
                "Install it with: pip install pandas"
            )
        return pd.DataFrame(self._history)

    def print_summary(self) -> None:
        """Helper to print a clean terminal summary of the episode."""
        summary = self.episode_summary()
        print("\n" + "=" * 50)
        print("  EPISODE SUMMARY")
        print("=" * 50)
        for key, value in summary.items():
            label = key.replace("_", " ").title()
            print(f"  {label:<30} {value}")
        print("=" * 50 + "\n")

    def reset(self) -> None:
        """
        Clears all metrics and restarts the wall-clock timer.
        Called automatically by SumoEnvironment.reset() at the start of a new episode.
        """
        self._throughput        = 0
        self._total_departed    = 0
        self._total_collisions  = 0
        self._history           = []
        self._departure_times   = {}
        self._travel_times      = []
        self._episode_start     = time.time()

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Returns the full list of step-by-step metric dictionaries."""
        return list(self._history)

    @property
    def throughput(self) -> int:
        """Returns the total number of vehicles that completed their routes."""
        return self._throughput