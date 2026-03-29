# sumo/network/demand_generator.py
"""
DemandGenerator

Generates compatible traffic demand files (.rou.xml) for a given road network scenario.

The system must be able to generate different traffic conditions. We need to define the
low, moderate, high, and congested traffic flows  that will populate the roads you just built.

SUMO needs to know :
  - Which vehicle types exist (car, truck, …)
  - Which routes vehicles can follow (a sequence of edges)
  - How many vehicles are injected per time unit (traffic flow)

Traffic demand levels (from the specification):
  ─────────────────────────────────────────────────────
  Level      | vehicles / hour per lane | Description
  ─────────────────────────────────────────────────────
  low        |  ~100                    | Free-flow, no queuing
  moderate   |  ~300                    | Occasional queuing
  high       |  ~600                    | Persistent queuing
  congested  |  ~900                    | Near-capacity, heavy delay
  ─────────────────────────────────────────────────────

How it works :
  The generator reads the edge list from the compiled .net.xml file using sumolib, then automatically builds all valid through-routes (routes that
  enter from a border node, cross the intersection(s), and exit at another border node).  One flow element is written for each route.

Output file:
  <scenario_dir>/traffic_<level>.rou.xml

Usage :

    from traffic_sim.network.demand_generator import DemandGenerator

    gen = DemandGenerator(net_file="configs/four_way_1int_2lanes/network.net.xml")

    # Generate moderate traffic for a 3600-second simulation
    gen.generate(level="moderate", duration=3600)

    # Generate all four levels
    gen.generate_all(duration=3600)
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Tuple, Dict

#  vehicles per hour injected on EACH entry edge for each demand level
DEMAND_LEVELS: Dict[str, int] = {
    "low":       100,
    "moderate":  300,
    "high":      600,
    "congested": 900,
}

# Default vehicle type parameters (passenger car)
DEFAULT_VTYPE = {
    "id":           "passenger",
    "accel":        "2.6",      # m/s²
    "decel":        "4.5",      # m/s²
    "sigma":        "0.5",      # driver imperfection [0,1]
    "length":       "5.0",      # metres
    "minGap":       "2.5",      # metres
    "maxSpeed":     "13.89",    # m/s  (50 km/h)
    "guiShape":     "passenger",
    "color":        "0.8,0.8,0.0",  # yellowish
}

# ============ HELPER UTILITIES ============

def _pretty_xml(element: ET.Element) -> str:
    """Return a nicely indented XML string from an ElementTree element."""
    raw = ET.tostring(element, encoding="unicode")
    reparsed = minidom.parseString(raw)
    return reparsed.toprettyxml(indent="    ")


def _write_xml(path: str, element: ET.Element) -> None:
    """Write a pretty-printed XML element to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_pretty_xml(element))

# ---- Route Discovery ----

def _parse_net_edges(net_file: str) -> Tuple[List[str], List[str]]:
    """
    Parse a .net.xml file WITHOUT importing sumolib (so the file can be
    generated and inspected even when SUMO is not installed yet).

    Returns :
      entry_edges : list[str]
          Edges whose 'from' node is a dead_end (border) node — these are
          the edges on which vehicles enter the network.
      internal_edges : list[str]
          All other non-internal edges (used to build through-routes).

    Note: SUMO internal edges start with ':' => we always skip those.
    """
    tree = ET.parse(net_file)
    root = tree.getroot()

    # Build a set of dead-end node ids
    dead_ends = set()
    for node_el in root.iter("junction"):
        if node_el.get("type") == "dead_end":
            dead_ends.add(node_el.get("id"))

    # Separate entry edges from internal / exit edges
    entry_edges: List[str] = []
    all_edges: List[str] = []

    for edge_el in root.iter("edge"):
        eid = edge_el.get("id", "")

        if eid.startswith(":"): # SUMO internal connector edge => skip
            continue
        
        from_node = edge_el.get("from", "")
        to_node   = edge_el.get("to",   "")

        all_edges.append(eid)

        # Entry edge: starts at a dead-end border node -> going INTO network
        if from_node in dead_ends and to_node not in dead_ends:
            entry_edges.append(eid)

    return entry_edges, all_edges

def _build_routes(entry_edges: List[str], all_edges: List[str]) -> List[Tuple[str, str]]:
    """
    Build simple two-edge through-routes: entry_edge -> exit_edge.

    Strategy: for each entry edge "X_to_Y", the complementary exit edge
    is "Y_to_Z" for each Z != X.  This covers straight-through and turning
    movements without path-finding.

    For multi-intersection networks a full graph search would be needed;
    here we use a heuristic:  pair every entry edge with every other entry
    edge whose direction is not exactly opposite (avoids U-turns).

    Returns:
      list of (route_id, edge_sequence_string) tuples
    """
    routes: List[Tuple[str, str]] = []

    # Build a fast look-up: edge_id -> (from_node, to_node)
    # We reconstruct this from edge naming convention "X_to_Y"
    def nodes_from_id(eid: str):
        parts = eid.split("_to_")
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, None
    
    # Group entry edges by their destination junction
    by_dest: Dict[str, List[str]] = {}
    for eid in entry_edges:
        _, dest = nodes_from_id(eid)
        if dest:
            by_dest.setdefault(dest, []).append(eid)

    # For each junction, pair every inbound entry edge with every outbound
    # entry edge that leads away from that junction
    # (outbound edges start AT the junction, their 'from' = junction)
    outbound: Dict[str, List[str]] = {}
    for eid in all_edges:
        frm, _ = nodes_from_id(eid)
        if frm:
            outbound.setdefault(frm, []).append(eid)

    route_idx = 0
    seen = set()

    for junction, inbounds in by_dest.items():
        exits = outbound.get(junction, [])
        for in_edge in inbounds:
            in_from, _ = nodes_from_id(in_edge)
            for out_edge in exits:
                _, out_to = nodes_from_id(out_edge)
                # Skip U-turns (returning to the same border)
                if out_to == in_from:
                    continue
                # Skip duplicate pairs
                key = (in_edge, out_edge)
                if key in seen:
                    continue
                seen.add(key)
                route_id = f"route_{route_idx}"
                routes.append((route_id, f"{in_edge} {out_edge}"))
                route_idx += 1

    return routes

# ============ MAIN CLASS ============
class DemandGenerator:
    """
    Generates SUMO traffic demand (.rou.xml) files for a compiled network.

    Parameters:
      net_file : str
          Path to the compiled .net.xml file produced by NetworkBuilder.
      output_dir : str | None
          Directory where .rou.xml files will be written.
          Defaults to the same directory as net_file.

    Example :
        gen = DemandGenerator("configs/four_way_1int_2lanes/network.net.xml")
        gen.generate("moderate", duration=3600)
        gen.generate_all(duration=3600)
    """

    def __init__(self, net_file: str, output_dir: str = None):
        if not os.path.isfile(net_file):
            raise FileNotFoundError(
                f"Network file not found: '{net_file}'\n"
                "Run NetworkBuilder.build() first to generate it."
            )
        self.net_file = net_file
        self.output_dir = output_dir or os.path.dirname(net_file)

        # Parse the network once and cache routes
        print(f"[DemandGenerator] Parsing network: {net_file}")
        self._entry_edges, self._all_edges = _parse_net_edges(net_file)
        self._routes = _build_routes(self._entry_edges, self._all_edges)

        print(
            f"[DemandGenerator] CONSTRUCTOR : Found {len(self._entry_edges)} entry edges, "
            f"{len(self._routes)} through-routes"
        )

    # ---- Callable Functions ----

    def generate(
        self,
        level: str = "moderate",
        duration: int = 3600,
        begin: int = 0,
    ) -> str:
        """
        Write a .rou.xml file for a given demand level.

        Parameters :
          level : str
              One of "low", "moderate", "high", "congested".
          duration : int
              Simulation duration in seconds.  Vehicles are injected from
              *begin* to *begin + duration*.
          begin : int
              Simulation start time in seconds (default 0).

        Returns :
          str
              Path to the generated .rou.xml file.
        """
        if level not in DEMAND_LEVELS:
            raise ValueError(f"Unknown demand level: '{level}'. "
                  f"Choose from: {list(DEMAND_LEVELS.keys())}"
            )
        
        if not self.routes:
             raise RuntimeError(
                "No valid through-routes could be built from this network. "
                "Check that the .net.xml contains proper dead_end border nodes."
            )          

        vph = DEMAND_LEVELS[level] # vehicles per hour per entry edge
        end = begin + duration 

        root = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set(
            "xsi:noNamespaceSchemaLocation",
            "http://sumo.dlr.de/xsd/routes_file.xsd",
        )

        # Vehicle type definition
        vtype_el = ET.SubElement(root, "vType")
        for attr, val in DEFAULT_VTYPE.items():
            vtype_el.set(attr, val)

        # Route definitions 
        for route_id, edges in self._routes:
            route_el = ET.SubElement(root, "route")
            route_el.set("id", route_id)
            route_el.set("edges", edges)

        # Flow definitions
        # Each flow uses one route and injects vehicles at a constant rate.
        # SUMO converts vehsPerHour to per-second headways automatically.
        for i, (route_id, _) in enumerate(self._routes):
            flow_el = ET.SubElement(root, "flow")
            flow_el.set("id",          f"flow_{level}_{i}")
            flow_el.set("type",        DEFAULT_VTYPE["id"])
            flow_el.set("route",       route_id)
            flow_el.set("begin",       str(begin))
            flow_el.set("end",         str(end))
            flow_el.set("vehsPerHour", str(vph))
            flow_el.set("departSpeed", "max")   # enter at road max speed
            flow_el.set("departLane",  "best")  # SUMO chooses optimal lane

        # Write file
        out_path = os.path.join(self.output_dir, f"traffic_{level}.rou.xml")
        _write_xml(out_path, root)

        print(
            f"[DemandGenerator] SUCCESS : {level.capitalize()} demand file written: "
            f"{out_path}  ({len(self._routes)} flows, {vph} veh/h each)"
        )
        return out_path
    
    def generate_all(self, duration: int = 3600, begin: int = 0) -> List[str]:
        """
        Generate .rou.xml files for all four demand levels.

        Returns :

        list[str]
            Paths to all generated .rou.xml files.
        """
        paths = []
        for level in DEMAND_LEVELS:
            path = self.generate(level=level, duration=duration, begin=begin)
            paths.append(path)
        print(f"\n[DemandGenerator] SUCCESS : All demand levels generated.")
        return paths


    # ---- Accessors (for the simulation environment) ----
    @property
    def entry_edges(self) -> List[str]:
        """List of entry edge IDs (vehicles enter the network here)."""
        return list(self._entry_edges)

    @property
    def routes(self) -> List[Tuple[str, str]]:
        """List of (route_id, edge_sequence) tuples."""
        return list(self._routes)

    @staticmethod
    def demand_levels() -> Dict[str, int]:
        """Return the demand level → vehicles/hour mapping."""
        return dict(DEMAND_LEVELS)