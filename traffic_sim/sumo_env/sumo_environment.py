# traffic_sim/sumo_env/sumo_environment.py
"""
SumoEnvironment

    The central class of the simulation package.  It wraps the SUMO/TraCI interface and exposes the control loop defined in the project specification:

    env = SumoEnvironment(...)
    env.start()
    obs = env.reset()
    while not done:
        obs, info = env.step(action, simulation_time)
    env.close()

TraCI (Traffic Control Interface) :

TraCI is SUMO's Python API.  It lets an external process (our Python code) connect to a running SUMO instance over a socket and query / control the
simulation at every time step.

Key TraCI concepts used here:
  - traci.start()       => launch SUMO and open the connection
  - traci.simulationStep() => advance simulation by one step
  - traci.close()       => shut down the connection
  - traci.vehicle.*     => vehicle state and control commands
  - traci.edge.*        => edge-level statistics

Parameters :
    net_file : str
        Path to the compiled .net.xml file.
    route_file : str
        Path to the .rou.xml traffic demand file.
    use_gui : bool
        If True, launches sumo-gui (visual mode).  If False, uses sumo
        (headless, faster for experiments).
    step_length : float
        Duration of each simulation step in seconds (default 1.0).
    max_steps : int
        Maximum number of steps before the episode ends (default 3600 = 1 hour).
    sumo_home : str | None
        Path to SUMO installation.  Falls back to $SUMO_HOME env variable.
    port : int
        TraCI connection port (default 8813).  Change if running multiple
        instances in parallel.
    seed : int
        Random seed passed to SUMO for reproducible vehicle insertion.
"""

import os
import sys
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

# Cleaning import in cas of not installed SUMO
try:
    import traci
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    # Define a stub so the rest of the module can be imported without SUMO
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


class SumoEnvironment:
    """
    Main interface to the SUMO simulation.

    Lifecycle
        1. start()  =>> one-time setup: validates files, configures SUMO command.
        2. reset()  =>> (re)launches SUMO, connects TraCI, returns first state.
        3. step()   =>> applies an action, advances simulation, returns new state.
        4. close()  =>> terminates the SUMO process and disconnects TraCI.

    The environment can be reset() multiple times without calling start()
    again (useful for running multiple episodes in a training loop).
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

        # Resolve SUMO binary
        sumo_home = sumo_home or os.environ.get("SUMO_HOME", "")
        if sumo_home:
            bin_name   = "sumo-gui" if use_gui else "sumo"
            self._sumo_bin = os.path.join(sumo_home, "bin", bin_name)

            # Also add SUMO tools to sys.path so traci can be imported
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
        self.state_extractor:     Optional[StateExtractor]     = None
        self.action_handler:      Optional[ActionHandler]       = None
        self.statistics_collector: Optional[StatisticsCollector] = None

        # Flag: start() has been called
        self._started = False

        print(
            f"[SumoEnvironment] Initialised\n"
            f"  net_file   : {self.net_file}\n"
            f"  route_file : {self.route_file}\n"
            f"  gui        : {self.use_gui}\n"
            f"  step_length: {self.step_length}s\n"
            f"  max_steps  : {self.max_steps}\n"
            f"  port       : {self.port}\n"
        )

    def start(self) -> None:
        """
        Validate the environment and prepare the SUMO command.

        This method does NOT launch SUMO yet -> that happens in reset().
        Call start() once before the first reset().

        Raises:
            EnvironmentError
                If the SUMO binary cannot be found.
        """
        if not TRACI_AVAILABLE:
            raise EnvironmentError(
                "TraCI is not available.  Make sure SUMO is installed and "
                "$SUMO_HOME/tools is in your PYTHONPATH."
            )

        # Check the SUMO binary exists
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
        initial state observation.

        Returns :
            dict
                The initial state of the simulation (see StateExtractor).

        Raises :
            RuntimeError
                If start() has not been called first.
        """
        if not self._started:
            raise RuntimeError(
                "Call start() before reset()."
            )

        # If a simulation is already running, close it cleanly
        if self._simulation_running:
            self._close_traci()

        # Build the SUMO command (writes .sumocfg first for realistic junction behaviour)
        cfg_path = self._write_sumocfg()
        sumo_cmd = self._build_sumo_command(cfg_path)
        print(f"[SumoEnvironment] Launching SUMO: {' '.join(sumo_cmd)}")

        # Start SUMO and connect TraCI
        traci.start(sumo_cmd, port=self.port)
        self._simulation_running = True
        self._current_step = 0

        # Initialise sub-components now that TraCI is connected
        self.state_extractor      = StateExtractor()
        self.action_handler       = ActionHandler(step_length=self.step_length)
        self.statistics_collector = StatisticsCollector()

        # Return first observation
        initial_state = self.create_state()
        print("[SumoEnvironment] reset() complete => simulation started.")
        return initial_state


    def step(
        self,
        action: Any = None,
        simulation_time: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Apply an action and advance the simulation by one time step.

        Parameters :

            action : Any
                Control action to apply to vehicles.  Can be:
                - None       => no control, vehicles follow SUMO defaults
                - dict       => {vehicle_id: speed} mapping (speed control)
                - list       => list of actions to apply via ActionHandler

            simulation_time : float | None
                If provided, advance the simulation to this absolute time (s),
                applying the action at every sub-step along the way.
                If None, advance by exactly one step_length.

        Returns :
            state : dict
                New simulation state after the step.
            info : dict
                Diagnostic information (step number, simulation time, done flag).

        Raises :
            RuntimeError
                If the simulation is not running (call reset() first).
        """
        if not self._simulation_running:
            raise RuntimeError(
                "Simulation is not running.  Call reset() first."
            )

        # 1. Advance simulation (action applied inside the loop / before the step)
        if simulation_time is not None:
            # Advance to a specific absolute time, applying the action at
            # every sub-step so control is consistent across the whole interval.
            while traci.simulation.getTime() < simulation_time:
                if action is not None:
                    self.action_handler.set_action(action)
                traci.simulationStep()
                self._current_step += 1
        else:
            # Single step: apply action once, then advance.
            if action is not None:
                self.action_handler.set_action(action)
            traci.simulationStep()
            self._current_step += 1

        # 2. Collect new state
        state = self.create_state()

        # 3. Check termination
        done = self._is_done()

        info = {
            "step":            self._current_step,
            "simulation_time": traci.simulation.getTime(),
            "done":            done,
        }

        if done:
            print(
                f"[SumoEnvironment] Episode finished at step "
                f"{self._current_step} "
                f"(t={info['simulation_time']:.1f}s)."
            )

        return state, info

    def close(self) -> None:
        """
        Cleanly close the TraCI connection and terminate SUMO.
        Call this when you are completely done with the environment.
        """
        if self._simulation_running:
            self._close_traci()
        print("[SumoEnvironment] Environment closed.")

    #  ---- Spec functions: state, statistics, leaders ----

    def create_state(self) -> Dict[str, Any]:
        """
        Retrieve the current environment state.

        Returns a dictionary with:
          - vehicle_ids      => list of all vehicles currently in the network
          - vehicle_speeds   => {vehicle_id: speed_m_s}
          - vehicle_positions=> {vehicle_id: (x, y)}
          - vehicle_edges    => {vehicle_id: edge_id}
          - edge_occupancies => {edge_id: occupancy_%}
          - edge_speeds      => {edge_id: mean_speed_m_s}
          - waiting_times    => {vehicle_id: accumulated_waiting_time_s}
          - simulation_time  => current simulation clock (s)
          - leaders          => output of get_leaders()
        """
        self._check_running()
        return self.state_extractor.create_state()

    def statistics(self) -> Dict[str, Any]:
        """
        Collect simulation-wide statistics.

        Returns :
            dict with keys:
                - mean_speed        => average speed of all vehicles (m/s)
                - mean_waiting_time => average waiting time per vehicle (s)
                - throughput        => vehicles that have completed their trip
                - total_vehicles    => vehicles currently in network
                - mean_travel_time  => average travel time of completed trips (s)
                - step             => current step number
        """
        self._check_running()
        return self.statistics_collector.statistics()

    def get_leaders(self) -> Dict[str, Optional[str]]:
        """
        Retrieve the leader vehicle for each lane on each edge.

        A "leader" vehicle is the frontmost vehicle on a given lane =>
        the one that determines the speed of vehicles behind it.

        Returns :
  
            dict : {lane_id: vehicle_id | None}
                Maps each lane to its current leader, or None if the lane
                is empty.
        """
        self._check_running()
        return self.state_extractor.get_leaders()

    # ---- Context manager support  (with SumoEnvironment(...) as env:) ----

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False   # do not suppress exceptions


    # ---- Properties ----

    @property
    def current_step(self) -> int:
        """Number of simulation steps taken since last reset."""
        return self._current_step

    @property
    def simulation_time(self) -> float:
        """Current simulation clock in seconds (0.0 if not running)."""
        if self._simulation_running:
            return traci.simulation.getTime()
        return 0.0

    @property
    def is_running(self) -> bool:
        """True if SUMO is currently connected and running."""
        return self._simulation_running

    # ---- Internal Helpers ----

    def _write_sumocfg(self) -> str:
        """
        Write a .sumocfg configuration file next to the net file and return
        its path.  Using a .sumocfg is the standard SUMO workflow and allows
        netconvert / SUMO to apply correct turn-movement rules at
        unsignalized intersections (right-of-way type = "right_before_left"
        by default, which is the most realistic setting for urban intersections).
        The file is (re)written on every reset() so it always reflects the
        current net_file / route_file paths.
        """
        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        config = ET.Element("configuration")
        config.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        config.set(
            "xsi:noNamespaceSchemaLocation",
            "http://sumo.dlr.de/xsd/sumoConfiguration.xsd",
        )

        # <input> block
        inp = ET.SubElement(config, "input")
        ET.SubElement(inp, "net-file",    attrib={"value": os.path.basename(self.net_file)})
        ET.SubElement(inp, "route-files", attrib={"value": os.path.basename(self.route_file)})

        # <time> block
        tim = ET.SubElement(config, "time")
        ET.SubElement(tim, "step-length", attrib={"value": str(self.step_length)})

        # <report> block — suppress noisy console output
        rep = ET.SubElement(config, "report")
        ET.SubElement(rep, "no-warnings",  attrib={"value": "true"})
        ET.SubElement(rep, "no-step-log",  attrib={"value": "true"})

        # <processing> block — realistic unsignalized intersection behaviour
        proc = ET.SubElement(config, "processing")
        ET.SubElement(proc, "time-to-teleport",      attrib={"value": "-1"})
        ET.SubElement(proc, "waiting-time-memory",   attrib={"value": "10000"})
        ET.SubElement(proc, "collision.action",      attrib={"value": "warn"})

        # Pretty-print and write
        raw      = ET.tostring(config, encoding="unicode")
        pretty   = minidom.parseString(raw).toprettyxml(indent="    ")
        base = (self.net_file[:-len(".net.xml")] if self.net_file.endswith(".net.xml") else os.path.splitext(self.net_file)[0])
        cfg_path = base + ".sumocfg"
        with open(cfg_path, "w", encoding="utf-8") as fh:
            fh.write(pretty)

        print(f"[SumoEnvironment] sumocfg written: {cfg_path}")
        return cfg_path

    def _build_sumo_command(self, cfg_path: str) -> List[str]:
        """Assemble the command line arguments for launching SUMO."""
        cmd = [
            self._sumo_bin,
            "-c", cfg_path,           # use the generated .sumocfg
            "--seed", str(self.seed),
        ]
        # GUI-specific options
        if self.use_gui:
            cmd += ["--start", "--quit-on-end"]
        return cmd

    def _is_done(self) -> bool:
        """
        Return True when the episode should end.
        Conditions:
          - max_steps reached
          - no more vehicles in the network AND simulation time > 60s
            (allow some warm-up time before declaring done)
        """
        if self._current_step >= self.max_steps:
            return True
        sim_time = traci.simulation.getTime()
        if sim_time > 60 and traci.simulation.getMinExpectedNumber() == 0:
            return True
        return False

    def _close_traci(self) -> None:
        """Disconnect TraCI and set the running flag to False."""
        try:
            traci.close()
        except Exception:
            pass   # ignore errors during cleanup
        self._simulation_running = False
        print("[SumoEnvironment] TraCI connection closed.")

    def _check_running(self) -> None:
        """Raise RuntimeError if the simulation is not active."""
        if not self._simulation_running:
            raise RuntimeError(
                "No simulation running.  Call reset() first."
            )

    def _sumo_bin_exists(self) -> bool:
        """Check whether the SUMO binary is reachable."""
        # If it is an absolute path, check directly
        if os.path.isabs(self._sumo_bin):
            return os.path.isfile(self._sumo_bin)
        # Otherwise, check if it is on the system PATH
        import shutil
        return shutil.which(self._sumo_bin) is not None