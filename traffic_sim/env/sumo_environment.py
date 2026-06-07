# traffic_sim/env/environment.py
"""
SumoEnvironment

    The central class of the simulation package.  It wraps the SUMO/TraCI
    interface and exposes the control loop defined in the project specification:

    env = SumoEnvironment(...)
    env.start()
    obs = env.reset()
    while not done:
        obs, reward, done, info = env.step(action)
    env.close()

--------------------------------------------------------------------------
HOW step() WORKS  (single-step, compatible with multi-intersection)
--------------------------------------------------------------------------
  Each call to step():
    1. Retrieve current leaders (single query, reused everywhere).
    2. Set speed mode 55 for newly loaded vehicles (disables junction ROW
       so the AI agent has full control over who crosses).
    3. Stop all non-leader vehicles on incoming edges (follower control).
    4. Apply go/stop to leaders via traci.vehicle.setStop().
    5. Advance the simulation by ONE step (traci.simulationStep()).
    6. Return (observation, reward, done, info).

  The AgentCallManager in main.py decides whether to re-call the agent
  or replay the previous action.  This avoids the "stuttering" problem
  of multi-step step() in multi-intersection networks, where waiting
  for leaders to cross at one intersection blocks decisions at others.

--------------------------------------------------------------------------
Follower control  (matching the teacher's set_loaded_vehicle)
--------------------------------------------------------------------------
  Non-leader vehicles are stopped 10 m before the end of their edge
  using traci.vehicle.setStop().  This prevents followers from pushing
  into the intersection uncontrolled.

  ALL follower-control and action-application logic is delegated to
  ActionHandler — SumoEnvironment does NOT duplicate it.

--------------------------------------------------------------------------
Leader actions  (matching the teacher's set_leaders_actions2)
--------------------------------------------------------------------------
  action=0 -> traci.vehicle.setStop(vid, edge, pos)    = stop before junction
  action=1 -> traci.vehicle.setStop(vid, edge, pos, duration=0) = release
              traci.vehicle.setSpeed(vid, -1)           = return to SUMO

--------------------------------------------------------------------------
Speed mode  (matching the teacher's set_speed_mode)
--------------------------------------------------------------------------
  Mode 55 = binary 110111 = disable right-of-way check at junctions
  (bit 3).  Applied to all newly loaded vehicles each step so the AI
  agent controls who crosses, not SUMO's default priority rules.

--------------------------------------------------------------------------
Reward  (configurable via RewardCalculator)
--------------------------------------------------------------------------
  The reward is computed by RewardCalculator, which supports four modes:
    "arrived"    -> +1 per vehicle completing its route
    "waiting"    -> negative mean waiting time
    "congestion" -> negative congestion index
    "combined"   -> weighted sum of arrived, waiting, and collisions
"""

import os
import sys
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

# Safe import of traci
try:
    import traci
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    class _TraciStub:
        def __getattr__(self, name):
            raise EnvironmentError(
                "TraCI (SUMO's Python API) is not installed.\n"
                "Install SUMO and make sure the SUMO_HOME environment "
                "variable is set, then add $SUMO_HOME/tools to PYTHONPATH.\n"
                "Download SUMO: https://sumo.dlr.de/docs/Downloads.php"
            )
    traci = _TraciStub()

from .state import StateExtractor
from .actions import ActionHandler
from .statistics import StatisticsCollector
from .observation import ObservationBuilder
from .reward import RewardCalculator


class SumoEnvironment:
    """
    Main interface to the SUMO simulation.

    Lifecycle
        1. start()  => validates files, configures SUMO command.
        2. reset()  => (re)launches SUMO, connects TraCI, returns first obs.
        3. step()   => applies action, advances 1 step, returns new obs.
        4. close()  => terminates SUMO and disconnects TraCI.

    Parameters
    ----------
    net_file : str
        Path to the compiled .net.xml file.
    route_file : str
        Path to the .rou.xml traffic demand file.
    use_gui : bool
        Launch sumo-gui (visual) instead of headless sumo.
    step_length : float
        Duration of each simulation step in seconds.
    max_steps : int
        Maximum simulation time in seconds before done=True.
    sumo_home : str | None
        SUMO installation directory.  Falls back to $SUMO_HOME.
    port : int
        TraCI connection port.
    seed : int
        Random seed for reproducible vehicle insertion.
    image_size : int
        Side length in pixels of the RGB observation image.
    dest_colors : dict | None
        {edge_id: (R, G, B)} colour table for the observation image.
    intersection_outgoing : set | None
        Edge IDs drawn white in the observation image.
    bbox : tuple | None
        World-space bounding box: ((xmin, ymin), (xmax, ymax)).
    reward_type : str
        Reward calculation mode: "arrived", "waiting", "congestion",
        or "combined".  Default is "arrived".
    arrived_weight : float
        Weight on the arrived-vehicles term (combined mode only).
    waiting_weight : float
        Weight on the mean-waiting-time penalty (combined mode only).
    collision_coef : float
        Penalty multiplied by the number of collisions (combined mode only).
    """

    def __init__(
        self,
        net_file: str,
        route_file: str,
        use_gui: bool = False,
        step_length: float = 1.0,
        max_steps: int = 3600,
        sumo_home: str = None,
        port: int = 8813,
        seed: int = 42,
        image_size:             int             = 50,
        dest_colors:            Optional[Dict]  = None,
        intersection_outgoing:  Optional[set]   = None,
        bbox:                   Optional[Any]   = None,
        reward_type:            str             = "arrived",
        arrived_weight:         float           = 1.0,
        waiting_weight:         float           = 0.01,
        collision_coef:         float           = 5.0,
    ):
        if not os.path.isfile(net_file):
            raise FileNotFoundError(
                f"Network file not found: '{net_file}'\n"
                "Run NetworkBuilder.build() first."
            )
        if not os.path.isfile(route_file):
            raise FileNotFoundError(
                f"Route file not found: '{route_file}'\n"
                "Run DemandGenerator.generate() first."
            )

        self.net_file    = net_file
        self.route_file  = route_file
        self.use_gui     = use_gui
        self.step_length = step_length
        self.max_steps   = max_steps
        self.port        = port
        self.seed        = seed

        self._image_size            = image_size
        self._dest_colors           = dest_colors
        self._intersection_outgoing = intersection_outgoing
        self._bbox                  = bbox

        # Reward configuration
        self._reward_type    = reward_type
        self._arrived_weight = arrived_weight
        self._waiting_weight = waiting_weight
        self._collision_coef = collision_coef

        # Resolve SUMO binary
        sumo_home = sumo_home or os.environ.get("SUMO_HOME", "")
        if sumo_home:
            bin_name   = "sumo-gui" if use_gui else "sumo"
            self._sumo_bin = os.path.join(sumo_home, "bin", bin_name)
            tools_dir = os.path.join(sumo_home, "tools")
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
        else:
            self._sumo_bin = "sumo-gui" if use_gui else "sumo"

        # Internal state
        self._simulation_running: bool = False
        self._current_step: int = 0
        self._sumo_process: Optional[subprocess.Popen] = None

        # Sub-components (initialised after TraCI connects in reset())
        self.state_extractor:      Optional[StateExtractor]      = None
        self.action_handler:       Optional[ActionHandler]        = None
        self.statistics_collector: Optional[StatisticsCollector]  = None
        self.observation_builder:  Optional[ObservationBuilder]   = None
        self.reward_calculator:    Optional[RewardCalculator]     = None

        self._started = False

        print(
            f"[SumoEnvironment] Initialised\n"
            f"  net_file    : {self.net_file}\n"
            f"  route_file  : {self.route_file}\n"
            f"  gui         : {self.use_gui}\n"
            f"  step_length : {self.step_length}s\n"
            f"  max_steps   : {self.max_steps}\n"
            f"  port        : {self.port}\n"
            f"  image_size  : {self._image_size}px\n"
            f"  reward_type : {self._reward_type}\n"
        )

    # ================================================================== #
    #  Lifecycle methods                                                   #
    # ================================================================== #

    def start(self) -> None:
        """
        Validate the environment and prepare the SUMO command.

        Does NOT launch SUMO — that happens in reset().
        """
        if not TRACI_AVAILABLE:
            raise EnvironmentError(
                "TraCI is not available.  Make sure SUMO is installed and "
                "$SUMO_HOME/tools is in your PYTHONPATH."
            )
        if not self._sumo_bin_exists():
            raise EnvironmentError(
                f"SUMO binary not found: '{self._sumo_bin}'\n"
                "Set the SUMO_HOME environment variable or add the SUMO "
                "bin/ directory to your system PATH."
            )
        self._started = True
        print("[SumoEnvironment] start() complete -> ready to reset().")

    def reset(self) -> Dict[str, Any]:
        """
        Launch (or relaunch) SUMO, connect via TraCI, and return the
        initial observation.

        Returns
        -------
        observation : dict
            "image"   -> ndarray (n, n, 3) uint8
            "state"   -> dict
            "leaders" -> dict {lane_id: vid|None}
        """
        if not self._started:
            raise RuntimeError("Call start() before reset().")

        if self._simulation_running:
            self._close_traci()

        cfg_path = self._write_sumocfg()
        sumo_cmd = self._build_sumo_command(cfg_path)
        print(f"[SumoEnvironment] Launching SUMO: {' '.join(sumo_cmd)}")

        traci.start(sumo_cmd, port=self.port)
        self._simulation_running = True
        self._current_step = 0

        # Initialise sub-components
        self.state_extractor      = StateExtractor()
        self.action_handler       = ActionHandler(step_length=self.step_length)
        self.statistics_collector = StatisticsCollector()
        self.observation_builder  = ObservationBuilder(
            dest_colors           = self._dest_colors,
            intersection_outgoing = self._intersection_outgoing,
            bbox                  = self._bbox,
        )
        self.reward_calculator    = RewardCalculator(
            reward_type    = self._reward_type,
            arrived_weight = self._arrived_weight,
            waiting_weight = self._waiting_weight,
            collision_coef = self._collision_coef,
        )

        obs = self._build_observation()
        print(
            f"[SumoEnvironment] reset() complete -> "
            f"image shape {obs['image'].shape}, "
            f"{len(obs['leaders'])} lanes tracked."
        )
        return obs

    # ================================================================== #
    #  step()  — SINGLE-STEP, multi-intersection friendly                  #
    # ================================================================== #

    def step(
        self,
        action: Any = None,
        simulation_time: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Apply an action, advance the simulation by ONE step, and return
        the new observation.

        Single-step design rationale
        ----------------------------
        In multi-intersection networks, a multi-step step() that blocks
        until leaders cross causes "stuttering": intersection A waits for
        its leader to cross while intersection B urgently needs a new
        decision.  Single-step avoids this; the AgentCallManager in
        main.py handles re-triggering only when meaningful changes occur.

        The critical fixes from the teacher's code are applied WITHIN
        each single step, ALL delegated to ActionHandler:
          - Follower control  (non-leaders stopped before intersection)
          - setStop / release (leaders controlled via setStop, not setSpeed)
          - Speed mode 55     (disable SUMO junction right-of-way)

        Parameters
        ----------
        action : dict | None
            {lane_id: 0 | 1}  where 1=go, 0=stop.
            Pass None for free-running SUMO (no external control).
        simulation_time : float | None
            Override for max simulation time (seconds).  If provided,
            overrides the max_steps set at construction.  Matches the
            spec signature: step(action, simulation_time).

        Returns
        -------
        observation : dict
        reward      : float
        done        : bool
        info        : dict
        """
        self._check_running()

        # Override max_steps if simulation_time is provided
        effective_max_steps = (
            simulation_time if simulation_time is not None
            else self.max_steps
        )

        # ---- 1. Refresh ActionHandler cache ----
        self.action_handler.refresh_cache()

        # ---- 2. Set speed mode for newly loaded vehicles ----
        self.action_handler.set_speed_modes_loaded()

        # ---- 3. Apply action ----
        incoming_leaders = None

        if action is not None and isinstance(action, dict) and self._looks_binary(action):
            # Get current leaders (single query, reused by stop + action)
            all_leaders      = self.state_extractor.get_leaders()
            incoming_leaders = self._filter_incoming_leaders(all_leaders)

            # Stop all non-leader vehicles on incoming edges
            self.action_handler.stop_non_leaders(incoming_leaders)

            # Apply go/stop to leaders via setStop
            self.action_handler.apply_binary_action(action, incoming_leaders)

        elif action is not None:
            # Legacy path: speed dict or list of action dicts
            self.action_handler.set_action(action)

        # ---- 4. Advance simulation by ONE step ----
        traci.simulationStep()
        self._current_step += 1

        # Set speed mode for vehicles that just loaded during this step
        self.action_handler.set_speed_modes_loaded()

        # Stop vehicles that just departed and are not leaders
        if incoming_leaders is not None:
            self.action_handler.stop_departed_non_leaders(incoming_leaders)

        # ---- 5. Reward: computed by RewardCalculator ----
        reward = self.reward_calculator.compute()

        # ---- 6. Build observation ----
        obs  = self._build_observation()
        done = self._is_done(effective_max_steps)

        # ---- 7. Collect statistics ----
        stats = self.statistics_collector.statistics()

        info = {
            "step":            self._current_step,
            "simulation_time": traci.simulation.getTime(),
            "done":            done,
            "stats":           stats,
        }

        if done:
            print(
                f"[SumoEnvironment] Episode finished at step "
                f"{self._current_step} (t={info['simulation_time']:.1f}s)."
            )

        return obs, reward, done, info

    def close(self) -> None:
        """Cleanly close the TraCI connection and terminate SUMO."""
        if self._simulation_running:
            self._close_traci()
        print("[SumoEnvironment] Environment closed.")

    # ================================================================== #
    #  Spec functions: state, statistics, leaders                          #
    # ================================================================== #

    def create_state(self) -> Dict[str, Any]:
        """Retrieve the current environment state."""
        self._check_running()
        return self.state_extractor.create_state()

    def statistics(self) -> Dict[str, Any]:
        """Collect simulation-wide statistics."""
        self._check_running()
        return self.statistics_collector.statistics()

    def get_leaders(self) -> Dict[str, Optional[str]]:
        """Retrieve the leader vehicle for each lane."""
        self._check_running()
        return self.state_extractor.get_leaders()

    # ================================================================== #
    #  Context manager support                                             #
    # ================================================================== #

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ================================================================== #
    #  Properties                                                          #
    # ================================================================== #

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def simulation_time(self) -> float:
        if self._simulation_running:
            return traci.simulation.getTime()
        return 0.0

    @property
    def is_running(self) -> bool:
        return self._simulation_running

    # ================================================================== #
    #  Leader / edge helpers                                               #
    # ================================================================== #

    def _filter_incoming_leaders(
        self,
        all_leaders: Dict[str, Optional[str]],
    ) -> Dict[str, Optional[str]]:
        """
        Keep only lanes that are INCOMING to an intersection.

        Since get_leaders() now filters out outgoing lanes at the source,
        this method only needs to handle edge cases (internal lanes that
        might slip through).
        """
        incoming: Dict[str, Optional[str]] = {}
        for lane_id, vid in all_leaders.items():
            if lane_id.startswith(":"):
                continue
            incoming[lane_id] = vid
        return incoming

    @staticmethod
    def _lane_to_edge(lane_id: str) -> str:
        """Strip the trailing '_<digit>' lane index from a lane ID."""
        if "_" in lane_id:
            parts = lane_id.rsplit("_", 1)
            if parts[1].isdigit():
                return parts[0]
        return lane_id

    @staticmethod
    def _is_outgoing_edge(edge_id: str) -> bool:
        """
        Return True if edge_id goes toward a border node.

        Border nodes start with N/S/E/W.  Outgoing edges have one of
        these as their destination:
            "C_to_N"          -> dest "N"       -> outgoing
            "J0_to_N0"        -> dest "N0"      -> outgoing
            "J_0_0_to_S_0_0"  -> dest "S_0_0"   -> outgoing
            "J0_to_J1"        -> dest "J1"      -> NOT outgoing
        """
        if "_to_" not in edge_id:
            return False
        dest = edge_id.split("_to_")[-1]
        return bool(dest) and dest[0] in ("N", "S", "E", "W")

    # ================================================================== #
    #  Observation builder                                                 #
    # ================================================================== #

    def _build_observation(self) -> Dict[str, Any]:
        """Build the observation dict returned by reset() and step()."""
        raw_state = self.state_extractor.create_state()
        image     = self.observation_builder.build_image(n=self._image_size)
        return {
            "image":   image,
            "state":   raw_state,
            "leaders": raw_state["leaders"],
        }

    # ================================================================== #
    #  Action format detection                                             #
    # ================================================================== #

    @staticmethod
    def _looks_binary(action: dict) -> bool:
        """Return True if every value in the action dict is 0 or 1."""
        return all(v in (0, 1, 0.0, 1.0) for v in action.values())

    # ================================================================== #
    #  Episode termination                                                 #
    # ================================================================== #

    def _is_done(self, max_steps: Optional[float] = None) -> bool:
        """
        Return True when the episode should end.
          - simulation time >= max_steps
          - no more vehicles expected AND simulation time > 60s
        """
        effective_max = max_steps if max_steps is not None else self.max_steps
        sim_time = traci.simulation.getTime()
        if sim_time >= effective_max:
            return True
        if sim_time > 60 and traci.simulation.getMinExpectedNumber() == 0:
            return True
        return False

    # ================================================================== #
    #  SUMO process management                                             #
    # ================================================================== #

    def _close_traci(self) -> None:
        try:
            traci.close()
        except Exception:
            pass
        self._simulation_running = False
        print("[SumoEnvironment] TraCI connection closed.")

    def _check_running(self) -> None:
        if not self._simulation_running:
            raise RuntimeError("No simulation running.  Call reset() first.")

    def _sumo_bin_exists(self) -> bool:
        if os.path.isabs(self._sumo_bin):
            return os.path.isfile(self._sumo_bin)
        import shutil
        return shutil.which(self._sumo_bin) is not None

    def _write_sumocfg(self) -> str:
        """
        Write a .sumocfg configuration file next to the net file.

        Key settings:
          - time-to-teleport = 300 (removes stuck vehicles after 5 min;
            prevents permanent deadlocks)
          - collision.action = warn
          - collision.check-junctions = true
        """
        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        config = ET.Element("configuration")
        config.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        config.set(
            "xsi:noNamespaceSchemaLocation",
            "http://sumo.dlr.de/xsd/sumoConfiguration.xsd",
        )

        inp = ET.SubElement(config, "input")
        ET.SubElement(inp, "net-file",
                      attrib={"value": os.path.basename(self.net_file)})
        ET.SubElement(inp, "route-files",
                      attrib={"value": os.path.basename(self.route_file)})

        tim = ET.SubElement(config, "time")
        ET.SubElement(tim, "step-length",
                      attrib={"value": str(self.step_length)})

        rep = ET.SubElement(config, "report")
        ET.SubElement(rep, "no-warnings",  attrib={"value": "true"})
        ET.SubElement(rep, "no-step-log",  attrib={"value": "true"})

        proc = ET.SubElement(config, "processing")
        ET.SubElement(proc, "time-to-teleport",
                      attrib={"value": "300"})
        ET.SubElement(proc, "waiting-time-memory",
                      attrib={"value": "10000"})
        ET.SubElement(proc, "collision.action",
                      attrib={"value": "warn"})
        ET.SubElement(proc, "collision.check-junctions",
                      attrib={"value": "true"})

        raw      = ET.tostring(config, encoding="unicode")
        pretty   = minidom.parseString(raw).toprettyxml(indent="    ")
        base = (self.net_file[:-len(".net.xml")]
                if self.net_file.endswith(".net.xml")
                else os.path.splitext(self.net_file)[0])
        cfg_path = base + ".sumocfg"
        with open(cfg_path, "w", encoding="utf-8") as fh:
            fh.write(pretty)

        print(f"[SumoEnvironment] sumocfg written: {cfg_path}")
        return cfg_path

    def _build_sumo_command(self, cfg_path: str) -> List[str]:
        cmd = [
            self._sumo_bin,
            "-c", cfg_path,
            "--seed", str(self.seed),
        ]
        if self.use_gui:
            cmd += ["--start", "--quit-on-end"]
        return cmd