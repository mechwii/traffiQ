# traffic_sim/sumo_env/statistics.py
"""
StatisticsCollector

Collects and computes traffic simulation metrics from the running SUMO simulation via TraCI.

Implements the statistics() function from the project specification, plus additional helpers for recording per-episode data and exporting results.

Metrics collected
-----------------
  - mean_speed        => average speed of all active vehicles (m/s)
  - mean_waiting_time => average accumulated waiting time per vehicle (s)
  - throughput        => number of vehicles that have completed their trip
  - total_vehicles    => vehicles currently in the network
  - mean_travel_time  => average travel time of completed trips (s)
  - step              => current simulation step

Episode history :
    The collector keeps a per-step log that can be exported to a pandas DataFrame for analysis and plotting.

TraCI calls used :
    traci.vehicle.getIDList()               - active vehicles
    traci.vehicle.getSpeed(vid)             - current speed
    traci.vehicle.getAccumulatedWaitingTime(vid) - waiting time
    traci.simulation.getTime()             - simulation clock
    traci.simulation.getArrivedNumber()    - vehicles that finished this step
    traci.simulation.getDepartedNumber()   - vehicles that entered this step
"""

from typing import Any, Dict, List, Optional
import time

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

# A vehicle is considered "waiting" when its speed falls below this threshold.
# 0.1 m/s is the standard SUMO convention for a stationary vehicle.
WAITING_SPEED_THRESHOLD = 0.1


class StatisticsCollector:
    """
    Collects and tracks simulation metrics over an episode.

    Instantiated by SumoEnvironment after TraCI connects.
    Call statistics() at each step (or less frequently) to update the log.
    """

    def __init__(self):
        # Cumulative counters reset at each episode (reset())
        self._throughput: int = 0          # total vehicles that arrived
        self._total_departed: int = 0      # total vehicles that departed
        self._total_collisions: int = 0

        # Per-step history (list of stat dicts)
        self._history: List[Dict[str, Any]] = []

        # Track travel times of completed vehicles
        # {vehicle_id: departure_time}
        self._departure_times: Dict[str, float] = {}
        self._travel_times: List[float] = []    # completed trips

        # Wall-clock time when the episode started
        self._episode_start: float = time.time()


    def statistics(self) -> Dict[str, Any]:
        """
        Collect and return current simulation statistics.

        Call this at every step or at any desired frequency.

        Returns :
            dict with keys:

            simulation_time : float
                Current simulation clock in seconds.

            step : int
                Number of times statistics() has been called this episode.

            total_vehicles : int
                Vehicles currently active in the network.

            mean_speed : float
                Mean speed of all active vehicles in m/s.
                Returns 0.0 if no vehicles are present.

            mean_waiting_time : float
                Mean accumulated waiting time per vehicle in seconds.
                A vehicle is waiting when its speed < 0.1 m/s.

            throughput : int
                Cumulative count of vehicles that have completed their route
                since the last reset().

            mean_travel_time : float
                Mean travel time (departure → arrival) of completed trips.
                Returns 0.0 if no trips have been completed yet.

            vehicles_per_minute : float
                Throughput expressed as vehicles per minute.

            congestion_index : float
                Ratio of waiting vehicles to total vehicles [0.0 - 1.0].
                0.0 = free flow, 1.0 = everyone is stopped.
        """
        sim_time = traci.simulation.getTime()

        # Active vehicles
        vehicle_ids = list(traci.vehicle.getIDList())
        n_vehicles  = len(vehicle_ids)

        # Speed
        speeds = [traci.vehicle.getSpeed(vid) for vid in vehicle_ids]
        mean_speed = sum(speeds) / n_vehicles if n_vehicles else 0.0

        # Waiting time
        wait_times = [
            traci.vehicle.getAccumulatedWaitingTime(vid)
            for vid in vehicle_ids
        ]
        mean_waiting_time = sum(wait_times) / n_vehicles if n_vehicles else 0.0

        # Throughput: count newly arrived vehicles this step
        arrived_this_step  = traci.simulation.getArrivedNumber()
        departed_this_step = traci.simulation.getDepartedNumber()
        self._throughput       += arrived_this_step
        self._total_departed   += departed_this_step

        # Collision count (if enabled in SUMO config)
        # collisions_this_step = traci.simulation.getCollidingVehiclesNumber()
        collisions_this_step = len(traci.simulation.getCollisions())
        self._total_collisions += collisions_this_step

        

        # Track departure times: record sim_time when each vehicle first appears.
        for vid in vehicle_ids:
            if vid not in self._departure_times:
                self._departure_times[vid] = sim_time

        # Compute travel times for vehicles that completed their trip this step.
        # getArrivedIDList() returns every vehicle that reached its destination
        # during this simulation step - this is the correct TraCI call (was
        # previously missing, which caused mean_travel_time to always be 0.0).
        for vid in traci.simulation.getArrivedIDList():
            if vid in self._departure_times:
                travel_time = sim_time - self._departure_times.pop(vid)
                if travel_time > 0:
                    self._travel_times.append(travel_time)

        # Mean travel time
        mean_travel_time = (
            sum(self._travel_times) / len(self._travel_times)
            if self._travel_times
            else 0.0
        )

        # Derived metrics
        elapsed_minutes    = (sim_time / 60.0) if sim_time > 0 else 1.0
        vehicles_per_minute = self._throughput / elapsed_minutes

        waiting_count     = sum(1 for s in speeds if s < WAITING_SPEED_THRESHOLD)
        congestion_index  = waiting_count / n_vehicles if n_vehicles else 0.0

        # Build result dict
        step = len(self._history) + 1
        result = {
            "simulation_time":    sim_time,
            "step":               step,
            "total_vehicles":     n_vehicles,
            "mean_speed":         round(mean_speed,        3),
            "mean_waiting_time":  round(mean_waiting_time, 3),
            "throughput":         self._throughput,
            "mean_travel_time":   round(mean_travel_time,  3),
            "vehicles_per_minute": round(vehicles_per_minute, 2),
            "congestion_index":   round(congestion_index,  3),
            "total_collisions":   self._total_collisions,
        }

        # Append to history
        self._history.append(result)

        # arrived_ids = traci.simulation.getArrivedIDList()
        # if len(arrived_ids) > 0:
        #    print(f"[DEBUG] t={sim_time}s -> Theses cars have arrived : {arrived_ids}")

        return result

    def episode_summary(self) -> Dict[str, Any]:
        """
        Return aggregate statistics for the completed episode.

        Best called after the simulation has ended (done=True).

        Returns :
            dict with overall episode-level metrics.
        """
        if not self._history:
            return {"error": "No data collected - call statistics() first."}

        all_speeds  = [h["mean_speed"]       for h in self._history]
        all_waits   = [h["mean_waiting_time"] for h in self._history]
        all_congestion = [h["congestion_index"] for h in self._history]

        elapsed_wall = time.time() - self._episode_start

        return {
            "total_steps":           len(self._history),
            "total_throughput":      self._throughput,
            "total_departed":        self._total_departed,
            "episode_mean_speed":    round(sum(all_speeds)     / len(all_speeds), 3),
            "episode_mean_wait":     round(sum(all_waits)      / len(all_waits),  3),
            "episode_mean_congestion": round(sum(all_congestion) / len(all_congestion), 3),
            "peak_congestion":       round(max(all_congestion), 3),
            "mean_travel_time":      self._history[-1]["mean_travel_time"],
            "wall_time_seconds":     round(elapsed_wall, 2),
            "total_collisions":   self._total_collisions,
        }

    # ---- Exportation ----


    def to_dataframe(self):
        """
        Export the per-step history as a pandas DataFrame.

        Returns :
            pandas.DataFrame
                One row per statistics() call, columns = metric names.

        Raises :
            ImportError
                If pandas is not installed.
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for to_dataframe().\n"
                "Install it with:  pip install pandas"
            )
        import pandas as pd
        return pd.DataFrame(self._history)

    def print_summary(self) -> None:
        """Print a formatted episode summary to stdout."""
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
        Reset all counters.  Called automatically by SumoEnvironment.reset().
        """
        self._throughput        = 0
        self._total_departed    = 0
        self._history           = []
        self._departure_times   = {}
        self._travel_times      = []
        self._episode_start     = time.time()
        self._total_collisions = 0

    # ---- Properties ----

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Full per-step statistics history for this episode."""
        return list(self._history)

    @property
    def throughput(self) -> int:
        """Total vehicles that have completed their trip this episode."""
        return self._throughput