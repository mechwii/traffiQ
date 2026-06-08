# traffic_sim/env/reward.py
"""
RewardCalculator
================
Computes the scalar reward signal returned to the AI agent at every step.

We support four reward strategies here. I kept the teacher's reference 
("arrived"), but added a few others ("waiting", "congestion", "combined") 
to give us more flexibility during training depending on what we want to optimize.

Reward types:
  "arrived"
      +1 for every vehicle that completed its route this step.
      Simple, sparse, and directly optimizes throughput. 
      This is the exact logic from the teacher's reference.

  "waiting"
      Negative mean accumulated waiting time across all active vehicles.
      Dense signal that forces the agent to actively reduce queue lengths.

  "congestion"
      Negative fraction of vehicles that are currently stopped.
      Dense signal ranging from 0 (free flow) to -1 (full gridlock).

  "combined" (Default)
      Balances throughput, queue length, and safety.
      Calculation: 
      (arrived_weight * n_arrived) - (waiting_weight * mean_waiting) - (collision_coef * n_collisions)

Usage:
    from traffic_sim.env.reward import RewardCalculator

    calc = RewardCalculator(
        reward_type="combined",
        arrived_weight=1.0,
        waiting_weight=0.01,
        collision_coef=5.0
    )
    reward = calc.compute()  # Call once per step while TraCI is active

TraCI calls used:
    traci.vehicle.getIDList()
    traci.vehicle.getSpeed(vid)
    traci.vehicle.getAccumulatedWaitingTime(vid)
    traci.simulation.getArrivedNumber()
    traci.simulation.getCollidingVehiclesNumber()
"""

from __future__ import annotations

from typing import Optional

# Any vehicle moving slower than this (m/s) is considered "stopped"
WAITING_THRESHOLD = 0.1

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

    Args:
        reward_type (str): The strategy to use ("arrived", "waiting", "congestion", "combined").
                           Defaults to "combined".
        arrived_weight (float): Multiplier for vehicles that finished their route (combined mode only).
        waiting_weight (float): Penalty multiplier for the mean waiting time (combined mode only).
        collision_coef (float): Huge penalty multiplier for crashes (combined mode only).
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
        Computes and returns the reward for the current simulation step.

        CRITICAL: This must be called AFTER traci.simulationStep() so that SUMO 
        has already updated the internal counters for arrivals and collisions.

        Returns:
            float: The computed scalar reward value.
        """
        if self.reward_type == "arrived":
            return self._reward_arrived()
        elif self.reward_type == "waiting":
            return self._reward_waiting()
        elif self.reward_type == "congestion":
            return self._reward_congestion()
        else:
            return self._reward_combined()

    def _reward_arrived(self) -> float:
        """
        Grants +1 for every vehicle that left the network this step.
        Mirrors the teacher's original logic exactly.
        """
        return float(traci.simulation.getArrivedNumber())

    def _reward_waiting(self) -> float:
        """
        Calculates the negative mean accumulated waiting time.
        Used to punish the agent for making cars wait too long at red lights.
        Returns 0.0 if the network is currently empty.
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
        Calculates a negative congestion index: -(stopped_vehicles / total_vehicles).
        Returns 0.0 if the network is empty.
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
        Our custom blended reward function.
        Balances throughput (good), waiting times (bad), and collisions (very bad).
        """
        n_arrived    = traci.simulation.getArrivedNumber()
        n_collisions = traci.simulation.getCollidingVehiclesNumber()

        vehicle_ids = list(traci.vehicle.getIDList())
        if vehicle_ids:
            mean_wait = sum(
                traci.vehicle.getAccumulatedWaitingTime(vid)
                for vid in vehicle_ids
            ) / len(vehicle_ids)
        else:
            mean_wait = 0.0

        return (
              (self.arrived_weight * n_arrived)
            - (self.waiting_weight * mean_wait)
            - (self.collision_coef * n_collisions)
        )

    def __repr__(self) -> str:
        """Quick debug representation of the current calculator config."""
        return (
            f"RewardCalculator(type={self.reward_type!r}, "
            f"arrived_w={self.arrived_weight}, "
            f"waiting_w={self.waiting_weight}, "
            f"collision_coef={self.collision_coef})"
        )