# traffic_sim/env/sumo_environment.py
"""
SumoEnvironment

The central class of the simulation package. It owns the SUMO process,
manages the TraCI connection, and exposes the standard RL control loop:

    env = SumoEnvironment(net_file=..., route_file=...)
    env.start()
    obs = env.reset()
    while not done:
        obs, reward, done, info = env.step(action)
    env.close()

Why single-step instead of multi-step:
    The teacher's reference uses a multi-step step() that loops until the
    "go" leaders have physically crossed the intersection. That works fine
    for a single intersection, but breaks down for multi-intersection networks:
    if step() blocks waiting for intersection A's leader to cross, intersection B
    gets no decisions during that time, causing visible "stuttering" in traffic flow.
    Our step() advances exactly one simulationStep() per call. The AgentCallManager
    in main.py handles re-triggering each agent only when something meaningful
    has actually changed at its intersection.

What happens inside each step() call:
    1. Refresh the vehicle ID cache so ActionHandler can do fast lookups.
    2. Set speed mode 55 on vehicles that just loaded (disables junction ROW).
    3. Stop all non-leader vehicles on incoming edges (follower control).
    4. Apply the AI's go/stop decision to leaders via setStop.
    5. Call traci.simulationStep() to advance time by one step_length.
    6. Set speed mode 55 again for vehicles that just departed mid-step.
    7. Stop any newly departed non-leaders.
    8. Compute reward (must be after the step so arrival counters are updated).
    9. Build observation, collect statistics, return.

Delegation:
    All TraCI vehicle commands go to ActionHandler.
    State extraction goes to StateExtractor.
    Image rendering goes to ObservationBuilder.
    Reward goes to RewardCalculator.
    Metrics go to StatisticsCollector.
    SumoEnvironment itself only orchestrates the order of calls.
"""

import os
import sys
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import traci
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    # Stub that raises a clear error message if any TraCI call is attempted without SUMO.
    class _TraciStub:
        def __getattr__(self, name):
            raise EnvironmentError(
                "TraCI (SUMO's Python API) is not installed.\n"
                "Install SUMO and make sure the SUMO_HOME environment "
                "variable is set, then add $SUMO_HOME/tools to PYTHONPATH.\n"
                "Download SUMO: https://sumo.dlr.de/docs/Downloads.php"
            )
    traci = _TraciStub()

from .state       import StateExtractor
from .actions     import ActionHandler
from .statistics  import StatisticsCollector
from .observation import ObservationBuilder
from .reward      import RewardCalculator


class SumoEnvironment:
    """
    Main interface to the SUMO simulation.

    Lifecycle:
        1. start()  -> validate files and resolve the SUMO binary. Does NOT launch SUMO.
        2. reset()  -> (re)launch SUMO, connect via TraCI, return the first observation.
        3. step()   -> apply action, advance one step, return obs/reward/done/info.
        4. close()  -> disconnect TraCI and shut down SUMO.

    Also works as a context manager:
        with SumoEnvironment(...) as env:
            obs = env.reset()
            ...

    Args:
        net_file: Path to the compiled .net.xml road network file.
        route_file: Path to the .rou.xml traffic demand file.
        use_gui: If True, launch sumo-gui for visual inspection. False = headless.
        step_length: Simulation step duration in seconds.
        max_steps: Episode ends when simulation time reaches this value (seconds).
        sumo_home: Path to the SUMO installation directory. Falls back to $SUMO_HOME.
        port: TCP port for the TraCI connection. Change if the default is already in use.
        seed: Random seed passed to SUMO for reproducible vehicle insertion order.
        image_size: Side length in pixels of the RGB observation image.
        dest_colors: {edge_id: (R, G, B)} color table for the observation image.
                     Defaults to the built-in single-intersection table.
        intersection_outgoing: Edge IDs drawn white in the observation (cleared vehicles).
        bbox: Fixed world bounding box ((xmin,ymin),(xmax,ymax)) for image rendering.
              If None, the full network boundary is used.
        reward_type: Which reward strategy to use. See RewardCalculator for options.
        arrived_weight: Weight on the throughput term (combined reward only).
        waiting_weight: Weight on the waiting-time penalty (combined reward only).
        collision_coef: Penalty per colliding vehicle (combined reward only).
    """

    def __init__(
        self,
        net_file:               str,
        route_file:             str,
        use_gui:                bool            = False,
        step_length:            float           = 1.0,
        max_steps:              int             = 3600,
        sumo_home:              str             = None,
        port:                   int             = 8813,
        seed:                   int             = 42,
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

        # Observation config (passed through to ObservationBuilder at reset time).
        self._image_size            = image_size
        self._dest_colors           = dest_colors
        self._intersection_outgoing = intersection_outgoing
        self._bbox                  = bbox

        # Reward config (passed through to RewardCalculator at reset time).
        self._reward_type    = reward_type
        self._arrived_weight = arrived_weight
        self._waiting_weight = waiting_weight
        self._collision_coef = collision_coef

        # Resolve which SUMO binary to launch. If SUMO_HOME is set we build the
        # full path; otherwise we assume the binary is on the system PATH.
        sumo_home = sumo_home or os.environ.get("SUMO_HOME", "")
        if sumo_home:
            bin_name       = "sumo-gui" if use_gui else "sumo"
            self._sumo_bin = os.path.join(sumo_home, "bin", bin_name)
            tools_dir      = os.path.join(sumo_home, "tools")
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
        else:
            self._sumo_bin = "sumo-gui" if use_gui else "sumo"

        # Runtime state (passed through to Sub-components at reset time).
        self._simulation_running: bool                        = False
        self._current_step:       int                         = 0
        self._sumo_process:       Optional[subprocess.Popen] = None

        # Sub-components are created fresh inside reset() after TraCI connects,
        # because some of them (e.g. ObservationBuilder) query the live simulation.
        self.state_extractor:      Optional[StateExtractor]      = None
        self.action_handler:       Optional[ActionHandler]       = None
        self.statistics_collector: Optional[StatisticsCollector] = None
        self.observation_builder:  Optional[ObservationBuilder]  = None
        self.reward_calculator:    Optional[RewardCalculator]    = None

        # Guards against calling reset() before start().
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
        Validate the environment setup before launching the first episode.

        Checks that TraCI is importable and that the SUMO binary exists on disk
        or on the system PATH. Does NOT launch SUMO, that happens in reset() so
        the same environment object can be reused across multiple episodes.
        """
        if not TRACI_AVAILABLE:
            raise EnvironmentError(
                "TraCI is not available. Make sure SUMO is installed and "
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
        Launch (or relaunch) SUMO, connect via TraCI, and return the first observation.

        If a simulation is already running it is closed first, so reset() can be
        called repeatedly without restarting the Python process between episodes.

        Returns:
            observation dict with keys:
                "image"   -> ndarray (image_size, image_size, 3) uint8
                "state"   -> full state dict from StateExtractor
                "leaders" -> {lane_id: vehicle_id | None}
        """
        if not self._started:
            raise RuntimeError("Call start() before reset().")

        if self._simulation_running:
            self._close_traci()

        # Write a fresh config file pointing to the current net/route files,
        # then build and launch the SUMO process.
        cfg_path = self._write_sumocfg()
        sumo_cmd = self._build_sumo_command(cfg_path)
        print(f"[SumoEnvironment] Launching SUMO: {' '.join(sumo_cmd)}")

        traci.start(sumo_cmd, port=self.port)
        self._simulation_running = True
        self._current_step       = 0

        # Create fresh sub-components for the new episode.
        # They hold per-episode state, so they must be recreated (not just reset).
        self.state_extractor      = StateExtractor()
        self.action_handler       = ActionHandler(step_length=self.step_length)
        self.statistics_collector = StatisticsCollector()
        self.observation_builder  = ObservationBuilder(
            dest_colors           = self._dest_colors,
            intersection_outgoing = self._intersection_outgoing,
            bbox                  = self._bbox,
        )
        self.reward_calculator = RewardCalculator(
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
    #  step() -> SINGLE-STEP, multi-intersection                  #
    # ================================================================== #

    def step(
        self,
        action:          Any             = None,
        simulation_time: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Apply an action, advance the simulation by one step, and return the result.

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
        Args:
            action: {lane_id: 0 | 1} where 1=go, 0=stop.
                    Pass None to let SUMO run freely without external control.
            simulation_time: If provided, overrides max_steps as the episode time limit.
                             Matches the spec signature: step(action, simulation_time).

        Returns:
            observation (dict): "image", "state", "leaders"
            reward (float): scalar from RewardCalculator
            done (bool): True when the episode should end
            info (dict): "step", "simulation_time", "done", "stats"
        """
        self._check_running()

        # Allow callers to override the episode length without rebuilding the env.
        effective_max_steps = (
            simulation_time if simulation_time is not None else self.max_steps
        )

        # Step 1: refresh the vehicle cache so ActionHandler lookups are fast.
        self.action_handler.refresh_cache()

        # Step 2: apply speed mode 55 to vehicles already loaded before this step.
        self.action_handler.set_speed_modes_loaded()

        # Step 3: apply the action.
        incoming_leaders = None

        if action is not None and isinstance(action, dict) and self._looks_binary(action):
            # Standard DQN pipeline: binary {lane_id: 0|1} action dict.
            # We query leaders once and reuse the snapshot for both stop_non_leaders
            # and apply_binary_action to keep the state consistent.
            all_leaders      = self.state_extractor.get_leaders()
            incoming_leaders = self._filter_incoming_leaders(all_leaders)

            self.action_handler.stop_non_leaders(incoming_leaders)
            self.action_handler.apply_binary_action(action, incoming_leaders)

        elif action is not None:
            # Legacy path: raw speed dict or list-of-command-dicts.
            self.action_handler.set_action(action)

        # Step 4: advance the simulation by exactly one step_length second.
        traci.simulationStep()
        self._current_step += 1

        # Step 5 & 6: handle vehicles that loaded or departed during the step.
        # They weren't in the pre-step leader snapshot, so we process them here.
        self.action_handler.set_speed_modes_loaded()
        if incoming_leaders is not None:
            self.action_handler.stop_departed_non_leaders(incoming_leaders)

        # Step 7: compute reward AFTER the step so arrival counters are updated.
        reward = self.reward_calculator.compute()

        # Step 8: build the new observation.
        obs  = self._build_observation()
        done = self._is_done(effective_max_steps)

        # Step 9: collect statistics and bundle the info dict.
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
        """Disconnect TraCI and terminate the SUMO process."""
        if self._simulation_running:
            self._close_traci()
        print("[SumoEnvironment] Environment closed.")

    # ================================================================== #
    #  Spec functions: state, statistics, leaders                          #
    # ================================================================== #

    def create_state(self) -> Dict[str, Any]:
        """Return the current full state snapshot (delegates to StateExtractor)."""
        self._check_running()
        return self.state_extractor.create_state()

    def statistics(self) -> Dict[str, Any]:
        """Return current step statistics (delegates to StatisticsCollector)."""
        self._check_running()
        return self.statistics_collector.statistics()

    def get_leaders(self) -> Dict[str, Optional[str]]:
        """Return the leader vehicle per lane (delegates to StateExtractor)."""
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
        return False  # do not suppress exceptions

    # ================================================================== #
    #  Properties                                                          #
    # ================================================================== #
    @property
    def current_step(self) -> int:
        """Number of simulation steps completed in the current episode."""
        return self._current_step

    @property
    def simulation_time(self) -> float:
        """Current simulation clock in seconds (0.0 if not running)."""
        return traci.simulation.getTime() if self._simulation_running else 0.0

    @property
    def is_running(self) -> bool:
        """True while a TraCI connection is active."""
        return self._simulation_running

    # ================================================================== #
    #  Leader / edge helpers                                               #
    # ================================================================== #

    def _filter_incoming_leaders(
        self,
        all_leaders: Dict[str, Optional[str]],
    ) -> Dict[str, Optional[str]]:
        """
        Drop any internal SUMO lanes (starting with ':') from the leaders dict.

        StateExtractor.get_leaders() already filters most of them, but this is a
        safety net for edge cases where a vehicle briefly appears on an internal lane.
        """
        return {
            lane_id: vid
            for lane_id, vid in all_leaders.items()
            if not lane_id.startswith(":")
        }

    @staticmethod
    def _lane_to_edge(lane_id: str) -> str:
        """
        Strip the trailing '_<index>' suffix from a lane ID to get the edge ID.
        Example: "C_to_N_0" -> "C_to_N"
        """
        if "_" in lane_id:
            parts = lane_id.rsplit("_", 1)
            if parts[1].isdigit():
                return parts[0]
        return lane_id

    @staticmethod
    def _is_outgoing_edge(edge_id: str) -> bool:
        """
        Return True if the edge leads toward a border (exit) node.

        Border nodes are named with a directional prefix (N, S, E, W).
            "C_to_N"         -> outgoing (exits northward)
            "J0_to_N0"       -> outgoing
            "J_0_0_to_S_0_0" -> outgoing
            "J0_to_J1"       -> NOT outgoing (internal junction-to-junction road)
        """
        if "_to_" not in edge_id:
            return False
        dest = edge_id.split("_to_")[-1]
        return bool(dest) and dest[0] in ("N", "S", "E", "W")

    # ================================================================== #
    #  Observation builder                                                 #
    # ================================================================== #

    def _build_observation(self) -> Dict[str, Any]:
        """
        Assemble the observation dict returned by reset() and step().

        We call create_state() once and reuse the result for both the leaders
        field and the raw state field to avoid an unnecessary second TraCI round-trip.
        """
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
        """
        Return True if every value in the action dict is 0 or 1.

        Used to distinguish the DQN binary action format from the legacy
        speed-dict format that ActionHandler.set_action() also accepts.
        """
        return all(v in (0, 1, 0.0, 1.0) for v in action.values())

    # ================================================================== #
    #  Episode termination                                                 #
    # ================================================================== #

    def _is_done(self, max_steps: Optional[float] = None) -> bool:
        """
        Return True when the episode should end.

        Two conditions trigger done:
            1. Simulation time has reached the configured limit.
            2. No vehicles are expected anymore AND at least 60 seconds have passed.
               This catches scenarios where all demand is exhausted early so we don't
               keep running an empty network.
        """
        effective_max = max_steps if max_steps is not None else self.max_steps
        sim_time      = traci.simulation.getTime()

        if sim_time >= effective_max:
            return True
        if sim_time > 60 and traci.simulation.getMinExpectedNumber() == 0:
            return True
        return False

    # ================================================================== #
    #  SUMO process management                                             #
    # ================================================================== #

    def _close_traci(self) -> None:
        """Attempt to close the TraCI connection gracefully."""
        try:
            traci.close()
        except Exception:
            pass  # already closed or never properly opened
        self._simulation_running = False
        print("[SumoEnvironment] TraCI connection closed.")

    def _check_running(self) -> None:
        """Raise an error if step() or statistics() are called without an active simulation."""
        if not self._simulation_running:
            raise RuntimeError("No simulation running. Call reset() first.")

    def _sumo_bin_exists(self) -> bool:
        """Check whether the resolved SUMO binary is accessible on disk or PATH."""
        if os.path.isabs(self._sumo_bin):
            return os.path.isfile(self._sumo_bin)
        import shutil
        return shutil.which(self._sumo_bin) is not None

    def _write_sumocfg(self) -> str:
        """
        Write a SUMO configuration (.sumocfg) file next to the net file.

        Key settings written here:
            time-to-teleport = 300     -> removes stuck vehicles after 5 minutes
                                          to prevent permanent deadlocks
            collision.action = warn    -> log collisions without stopping the simulation
            collision.check-junctions  -> detect collisions inside junction boxes too
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
        ET.SubElement(inp, "net-file",    attrib={"value": os.path.basename(self.net_file)})
        ET.SubElement(inp, "route-files", attrib={"value": os.path.basename(self.route_file)})

        tim = ET.SubElement(config, "time")
        ET.SubElement(tim, "step-length", attrib={"value": str(self.step_length)})

        rep = ET.SubElement(config, "report")
        ET.SubElement(rep, "no-warnings", attrib={"value": "true"})
        ET.SubElement(rep, "no-step-log", attrib={"value": "true"})

        proc = ET.SubElement(config, "processing")
        ET.SubElement(proc, "time-to-teleport",          attrib={"value": "300"})
        ET.SubElement(proc, "waiting-time-memory",       attrib={"value": "10000"})
        ET.SubElement(proc, "collision.action",          attrib={"value": "warn"})
        ET.SubElement(proc, "collision.check-junctions", attrib={"value": "true"})

        raw    = ET.tostring(config, encoding="unicode")
        pretty = minidom.parseString(raw).toprettyxml(indent="    ")

        base     = (self.net_file[:-len(".net.xml")]
                    if self.net_file.endswith(".net.xml")
                    else os.path.splitext(self.net_file)[0])
        cfg_path = base + ".sumocfg"

        with open(cfg_path, "w", encoding="utf-8") as fh:
            fh.write(pretty)

        print(f"[SumoEnvironment] sumocfg written: {cfg_path}")
        return cfg_path

    def _build_sumo_command(self, cfg_path: str) -> List[str]:
        """
        Build the command-line arguments for launching SUMO.

        --start and --quit-on-end are added in GUI mode so the window opens
        automatically and closes when the episode ends without manual input.
        """
        cmd = [
            self._sumo_bin,
            "-c", cfg_path,
            "--seed", str(self.seed),
        ]
        if self.use_gui:
            cmd += ["--start", "--quit-on-end"]
        return cmd