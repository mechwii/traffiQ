# traffic_sim/network/demand_generator.py
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
    Parse a COMPILED .net.xml file (output of netconvert).

    The compiled format differs from the input .nod.xml / .edg.xml files:
      - The root tag is <net>, not <nodes> or <edges>.
      - Junctions are <junction id="..." type="..." ...> inside <net>.
      - Road edges are <edge id="..." from="..." to="..."> with NO
        'function' attribute (or function="normal").
      - Internal SUMO connector edges have function="internal" AND their
        id starts with ':' => we always skip both.
 
    Dead-end border nodes have type="dead_end" in the compiled file,
    exactly as we set in the .nod.xml - netconvert preserves this.

    Returns :
      entry_edges : list[str]
          Edges whose 'from' node is a dead_end (border) node - these are
          the edges on which vehicles enter the network.
      all_edges : list[str]
        All non-internal road edges (entry + internal road segments).

    Note: SUMO internal edges start with ':' => we always skip those.
    """
    tree = ET.parse(net_file)
    root = tree.getroot()

    # 1. Collect dead-end (border) junction IDs
    dead_ends: set = set()
    for junction in root.iter("junction"):
        jtype = junction.get("type", "")
        if jtype == "dead_end":
            dead_ends.add(junction.get("id"))

    print(f"  [DemandGenerator] Dead-end junctions found: {sorted(dead_ends)}")

    # 2. Collect all road edges (skip internal connector edges)
    entry_edges: List[str] = []
    all_edges:   List[str] = []

    for edge_el in root.iter("edge"):
        eid      = edge_el.get("id", "")
        function = edge_el.get("function", "")
 
        # Skip SUMO internal connector edges (id starts with ':' OR function == "internal")
        if eid.startswith(":") or function == "internal":
            continue
 
        from_node = edge_el.get("from", "")
        to_node   = edge_el.get("to",   "")
 
        all_edges.append(eid)
 
        # Entry edge: originates at a border dead-end node, points inward
        if from_node in dead_ends and to_node not in dead_ends:
            entry_edges.append(eid)
 
    # 3. Fallback: if dead_end detection failed (e.g. netconvert changed
    #    the type label), infer entry edges from naming convention.
    #    Our NetworkBuilder names border nodes N, S, E, W, N0, S0, etc.
    #    Entry edges therefore have IDs like "N_to_C", "W0_to_J0", etc.
    if not entry_edges and all_edges:
        print(
            "  [DemandGenerator] WARNING: No dead-end junctions detected. "
            "Falling back to name-based entry-edge detection."
        )
        # Border node prefixes used by NetworkBuilder
        BORDER_PREFIXES = ("N", "S", "E", "W")
 
        for edge_el in root.iter("edge"):
            eid      = edge_el.get("id", "")
            function = edge_el.get("function", "")
            if eid.startswith(":") or function == "internal":
                continue
 
            from_node = edge_el.get("from", "")
            # from_node is a border node if it starts with N/S/E/W
            # AND the edge id follows the "X_to_Y" naming convention
            if (
                from_node
                and any(from_node.startswith(p) for p in BORDER_PREFIXES)
                and "_to_" in eid
            ):
                entry_edges.append(eid)
 
    return entry_edges, all_edges


def _build_routes(entry_edges: List[str], all_edges: List[str]) -> List[Tuple[str, str]]:
    """
    Build all valid through-routes using BFS on the edge graph.

    This replaces the old two-edge heuristic that failed for multi-intersection
    networks (4 / 8 intersections) because vehicles need to traverse several
    edges to cross the full grid.

    Strategy:
      1. Build an adjacency map:  from_node -> [edge_id, ...]
      2. For every entry edge, run BFS until we reach another border node
         (a dead_end that is NOT the one we started from).
      3. Every such complete path becomes a route.

    Dead_end border nodes are inferred from the entry_edges list:
    an entry edge "X_to_Y" starts at border node X.

    Returns:
        list of (route_id, edge_sequence_string) tuples
    """

    # ---- helpers ----
    def nodes_from_id(eid: str):
        """Extract (from_node, to_node) from edge id 'X_to_Y'."""
        parts = eid.split("_to_")
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, None

    # ---- build adjacency: to_node -> list[(edge_id, to_node)] ----
    # We index by the node a vehicle ARRIVES AT so we can extend paths.
    adjacency: Dict[str, List[Tuple[str, str]]] = {}
    for eid in all_edges:
        frm, to = nodes_from_id(eid)
        if frm and to:
            adjacency.setdefault(frm, []).append((eid, to))

    # ---- collect border nodes (dead_ends) ----
    # Every entry edge originates at a border node.
    border_nodes: set = set()
    for eid in entry_edges:
        frm, _ = nodes_from_id(eid)
        if frm:
            border_nodes.add(frm)

    # ---- BFS from each entry edge ----
    routes: List[Tuple[str, str]] = []
    route_idx = 0
    seen_paths: set = set()  # avoid duplicates

    for start_edge in entry_edges:
        start_from, start_to = nodes_from_id(start_edge)
        if not start_to:
            continue

        # BFS state: (current_node, path_of_edges_so_far)
        # We limit depth to avoid infinite loops in cyclic grids.
        MAX_DEPTH = 20
        queue: List[Tuple[str, List[str]]] = [(start_to, [start_edge])]

        while queue:
            current_node, path = queue.pop(0)

            if len(path) > MAX_DEPTH:
                continue

            # If we reached a border node that is NOT our origin => valid route
            if current_node in border_nodes and current_node != start_from:
                key = tuple(path)
                if key not in seen_paths:
                    seen_paths.add(key)
                    route_id = f"route_{route_idx}"
                    routes.append((route_id, " ".join(path)))
                    route_idx += 1
                continue  # don't extend further past a border exit

            # Extend path through all outgoing edges from current_node
            for next_edge, next_node in adjacency.get(current_node, []):
                # Avoid revisiting nodes already in this path (no loops)
                nodes_visited = {nodes_from_id(e)[0] for e in path}
                nodes_visited.add(start_from)
                if next_node not in nodes_visited:
                    queue.append((next_node, path + [next_edge]))

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
        
        if not self._routes:
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