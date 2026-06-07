# traffic_sim/env/reward.py
"""
RewardCalculator
================
Computes the scalar reward signal returned to the AI agent at every step.

Three reward types are supported (configurable at environment creation):

  "arrived"
      +1 for every vehicle that completed its route this step.
      Simple, sparse, directly optimises throughput.
      → Mirrors the teacher's reference implementation.

  "waiting"
      - (mean accumulated waiting time across all active vehicles).
      Dense signal; encourages the agent to reduce queue lengths.

  "congestion"
      - (fraction of vehicles that are stopped).
      Dense; ranges from 0 (free flow) to -1 (full gridlock).

  "combined"   ← DEFAULT
      arrived_weight  * n_arrived
    - waiting_weight  * mean_waiting_time
    - collision_coef  * n_collisions

      Balances throughput, queue length, and safety.
      All weights are configurable.

Usage
-----
    from traffic_sim.env.reward import RewardCalculator

    calc = RewardCalculator(reward_type="combined",
                            arrived_weight=1.0,
                            waiting_weight=0.01,
                            collision_coef=5.0)

    reward = calc.compute()   # call once per step while TraCI is active

TraCI calls used
----------------
    traci.vehicle.getIDList()
    traci.vehicle.getSpeed(vid)
    traci.vehicle.getAccumulatedWaitingTime(vid)
    traci.simulation.getArrivedNumber()
    traci.simulation.getCollidingVehiclesNumber()
"""

from __future__ import annotations

from typing import Optional

WAITING_THRESHOLD = 0.1   # m/s -> vehicle is considered "stopped" below this

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False


# Supported reward type identifiers
REWARD_TYPES = ("arrived", "waiting", "congestion", "combined")


class RewardCalculator:
    """
    Computes a scalar reward for one simulation step.

    Parameters
    ----------
    reward_type : str
        One of ``"arrived"``, ``"waiting"``, ``"congestion"``, ``"combined"``.
        Default is ``"combined"``.
    arrived_weight : float
        Weight on the arrived-vehicles term  (combined mode only).
    waiting_weight : float
        Weight on the mean-waiting-time penalty  (combined mode only).
    collision_coef : float
        Penalty multiplied by the number of collisions  (combined mode only).
    """

    def __init__(
        self,
        reward_type:     str   = "combined",
        arrived_weight:  float = 1.0,
        waiting_weight:  float = 0.01,
        collision_coef:  float = 5.0,
    ):
        if reward_type not in REWARD_TYPES:
            raise ValueError(
                f"Unknown reward_type '{reward_type}'. "
                f"Choose from: {REWARD_TYPES}"
            )
        self.reward_type    = reward_type
        self.arrived_weight = arrived_weight
        self.waiting_weight = waiting_weight
        self.collision_coef = collision_coef


    def compute(self) -> float:
        """
        Compute and return the reward for the current simulation step.

        Must be called **after** ``traci.simulationStep()`` so that the
        arrived / collision counters have been updated for this step.

        Returns
        -------
        float
            Scalar reward value.
        """
        if self.reward_type == "arrived":
            return self._reward_arrived()
        elif self.reward_type == "waiting":
            return self._reward_waiting()
        elif self.reward_type == "congestion":
            return self._reward_congestion()
        else:  # "combined"
            return self._reward_combined()


    def _reward_arrived(self) -> float:
        """
        +1 per vehicle that finished its route this step.
        Mirrors the teacher's reference: reward += traci.simulation.getArrivedNumber()
        """
        return float(traci.simulation.getArrivedNumber())

    def _reward_waiting(self) -> float:
        """
        Negative mean accumulated waiting time.
        Encourages the agent to reduce queues.
        Returns 0.0 if no vehicles are present.
        """
        vehicle_ids = list(traci.vehicle.getIDList())
        if not vehicle_ids:
            return 0.0
        total_wait = sum(
            traci.vehicle.getAccumulatedWaitingTime(vid)
            for vid in vehicle_ids
        )
        return -(total_wait / len(vehicle_ids))

    def _reward_congestion(self) -> float:
        """
        Negative congestion index: −(stopped_vehicles / total_vehicles).
        Returns 0.0 when the network is empty.
        """
        vehicle_ids = list(traci.vehicle.getIDList())
        if not vehicle_ids:
            return 0.0
        n_stopped = sum(
            1 for vid in vehicle_ids
            if traci.vehicle.getSpeed(vid) < WAITING_THRESHOLD
        )
        return -(n_stopped / len(vehicle_ids))

    def _reward_combined(self) -> float:
        """
        Throughput reward - waiting penalty - collision penalty.

        combined = arrived_weight  * n_arrived
                 - waiting_weight  * mean_waiting_time
                 - collision_coef  * n_collisions
        """
        n_arrived    = traci.simulation.getArrivedNumber()
        n_collisions = traci.simulation.getCollidingVehiclesNumber()

        vehicle_ids  = list(traci.vehicle.getIDList())
        if vehicle_ids:
            mean_wait = sum(
                traci.vehicle.getAccumulatedWaitingTime(vid)
                for vid in vehicle_ids
            ) / len(vehicle_ids)
        else:
            mean_wait = 0.0

        return (
              self.arrived_weight * n_arrived
            - self.waiting_weight * mean_wait
            - self.collision_coef * n_collisions
        )
    
    # ------- Convenience --------

    def __repr__(self) -> str:
        return (
            f"RewardCalculator(type={self.reward_type!r}, "
            f"arrived_w={self.arrived_weight}, "
            f"waiting_w={self.waiting_weight}, "
            f"collision_coef={self.collision_coef})"
        )