# traffic_sim/network/demand_generator.py
"""
DemandGenerator

Generates compatible traffic demand files (.rou.xml) for a given road network
scenario.

Traffic demand levels (from the specification):
  Level      | vehicles / hour per ENTRY EDGE | Description
  low        |  ~100                          | Free-flow, no queuing
  moderate   |  ~300                          | Occasional queuing
  high       |  ~600                          | Persistent queuing
  congested  |  ~900                          | Near-capacity, heavy delay

IMPORTANT — flow rate is per ENTRY EDGE, not per route.
  A single entry edge (e.g. "N_to_C") fans out into multiple routes
  (N->S, N->E, N->W).  The vph value is SPLIT across those routes so
  the TOTAL injection from each entry edge matches the intended level.

  Example: four-way intersection, "low" = 100 veh/h per entry edge.
    Entry "N_to_C" has 3 routes (to S, E, W).
    Each route gets 100/3 ≈ 33 veh/h.
    Total from N_to_C = 33*3 = ~100 veh/h.  Correct.
    Total for all 4 entries = ~400 veh/h.

How it works:
  The generator reads the edge list from the compiled .net.xml file,
  builds all valid through-routes via BFS, counts how many routes share
  each entry edge, and divides the per-entry vph accordingly.

Usage:
    from traffic_sim.network.demand_generator import DemandGenerator

    gen = DemandGenerator(net_file="configs/four_way_1int_2lanes/network.net.xml")
    gen.generate(level="moderate", duration=3600)
    gen.generate_all(duration=3600)
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Tuple, Dict

#  vehicles per hour per ENTRY EDGE for each demand level
DEMAND_LEVELS: Dict[str, int] = {
    "low":       100,
    "moderate":  300,
    "high":      600,
    "congested": 900,
}

# Default vehicle type parameters (passenger car)
DEFAULT_VTYPE = {
    "id":           "passenger",
    "accel":        "2.6",
    "decel":        "4.5",
    "sigma":        "0.5",
    "length":       "5.0",
    "minGap":       "2.5",
    "maxSpeed":     "13.89",
    "guiShape":     "passenger",
    "color":        "0.8,0.8,0.0",
}

# ============ HELPER UTILITIES ============

def _pretty_xml(element: ET.Element) -> str:
    raw = ET.tostring(element, encoding="unicode")
    reparsed = minidom.parseString(raw)
    return reparsed.toprettyxml(indent="    ")


def _write_xml(path: str, element: ET.Element) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_pretty_xml(element))


# ---- Route Discovery ----

def _parse_net_edges(net_file: str) -> Tuple[List[str], List[str]]:
    """
    Parse a compiled .net.xml file (output of netconvert).

    Returns:
      entry_edges : list[str]
          Edges whose 'from' node is a dead_end border node.
      all_edges : list[str]
          All non-internal road edges.
    """
    tree = ET.parse(net_file)
    root = tree.getroot()

    dead_ends: set = set()
    for junction in root.iter("junction"):
        jtype = junction.get("type", "")
        if jtype == "dead_end":
            dead_ends.add(junction.get("id"))

    print(f"  [DemandGenerator] Dead-end junctions found: {sorted(dead_ends)}")

    entry_edges: List[str] = []
    all_edges:   List[str] = []

    for edge_el in root.iter("edge"):
        eid      = edge_el.get("id", "")
        function = edge_el.get("function", "")

        if eid.startswith(":") or function == "internal":
            continue

        from_node = edge_el.get("from", "")
        to_node   = edge_el.get("to",   "")

        all_edges.append(eid)

        if from_node in dead_ends and to_node not in dead_ends:
            entry_edges.append(eid)

    # Fallback: name-based detection
    if not entry_edges and all_edges:
        print(
            "  [DemandGenerator] WARNING: No dead-end junctions detected. "
            "Falling back to name-based entry-edge detection."
        )
        BORDER_PREFIXES = ("N", "S", "E", "W")

        for edge_el in root.iter("edge"):
            eid      = edge_el.get("id", "")
            function = edge_el.get("function", "")
            if eid.startswith(":") or function == "internal":
                continue
            from_node = edge_el.get("from", "")
            if (
                from_node
                and any(from_node.startswith(p) for p in BORDER_PREFIXES)
                and "_to_" in eid
            ):
                entry_edges.append(eid)

    return entry_edges, all_edges


def _build_routes(
    entry_edges: List[str],
    all_edges: List[str],
) -> List[Tuple[str, str]]:
    """
    Build all valid through-routes using BFS on the edge graph.

    A through-route enters at a border node, crosses the network,
    and exits at a different border node.
    """

    def nodes_from_id(eid: str):
        parts = eid.split("_to_")
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, None

    # Build adjacency: from_node -> [(edge_id, to_node)]
    adjacency: Dict[str, List[Tuple[str, str]]] = {}
    for eid in all_edges:
        frm, to = nodes_from_id(eid)
        if frm and to:
            adjacency.setdefault(frm, []).append((eid, to))

    # Collect border nodes
    border_nodes: set = set()
    for eid in entry_edges:
        frm, _ = nodes_from_id(eid)
        if frm:
            border_nodes.add(frm)

    # BFS from each entry edge
    routes: List[Tuple[str, str]] = []
    route_idx = 0
    seen_paths: set = set()
    MAX_DEPTH = 20

    for start_edge in entry_edges:
        start_from, start_to = nodes_from_id(start_edge)
        if not start_to:
            continue

        queue: List[Tuple[str, List[str]]] = [(start_to, [start_edge])]

        while queue:
            current_node, path = queue.pop(0)

            if len(path) > MAX_DEPTH:
                continue

            if current_node in border_nodes and current_node != start_from:
                key = tuple(path)
                if key not in seen_paths:
                    seen_paths.add(key)
                    route_id = f"route_{route_idx}"
                    routes.append((route_id, " ".join(path)))
                    route_idx += 1
                continue

            for next_edge, next_node in adjacency.get(current_node, []):
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
    """

    def __init__(self, net_file: str, output_dir: str = None):
        if not os.path.isfile(net_file):
            raise FileNotFoundError(
                f"Network file not found: '{net_file}'\n"
                "Run NetworkBuilder.build() first to generate it."
            )
        self.net_file = net_file
        self.output_dir = output_dir or os.path.dirname(net_file)

        print(f"[DemandGenerator] Parsing network: {net_file}")
        self._entry_edges, self._all_edges = _parse_net_edges(net_file)
        self._routes = _build_routes(self._entry_edges, self._all_edges)

        # Pre-compute how many routes share each entry edge so we can
        # split the per-entry vph correctly.
        self._routes_per_entry: Dict[str, int] = {}
        for _, edges_str in self._routes:
            first_edge = edges_str.split()[0]
            self._routes_per_entry[first_edge] = (
                self._routes_per_entry.get(first_edge, 0) + 1
            )

        print(
            f"[DemandGenerator] Found {len(self._entry_edges)} entry edges, "
            f"{len(self._routes)} through-routes"
        )

    def generate(
        self,
        level: str = "moderate",
        duration: int = 3600,
        begin: int = 0,
    ) -> str:
        """
        Write a .rou.xml file for a given demand level.

        The per-entry-edge vph is SPLIT across the routes that share
        that entry edge.  This ensures the total injection from each
        entry edge matches the intended demand level.

        Parameters:
          level    : "low" | "moderate" | "high" | "congested"
          duration : simulation duration in seconds
          begin    : simulation start time in seconds

        Returns:
          str : path to the generated .rou.xml file.
        """
        if level not in DEMAND_LEVELS:
            raise ValueError(
                f"Unknown demand level: '{level}'. "
                f"Choose from: {list(DEMAND_LEVELS.keys())}"
            )
        if not self._routes:
            raise RuntimeError(
                "No valid through-routes could be built from this network. "
                "Check that the .net.xml contains proper dead_end border nodes."
            )

        vph_per_entry = DEMAND_LEVELS[level]
        end = begin + duration

        root = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set(
            "xsi:noNamespaceSchemaLocation",
            "http://sumo.dlr.de/xsd/routes_file.xsd",
        )

        # Vehicle type
        vtype_el = ET.SubElement(root, "vType")
        for attr, val in DEFAULT_VTYPE.items():
            vtype_el.set(attr, val)

        # Route definitions
        for route_id, edges in self._routes:
            route_el = ET.SubElement(root, "route")
            route_el.set("id", route_id)
            route_el.set("edges", edges)

        # Flow definitions
        # FIX: vph is divided by the number of routes sharing the same
        # entry edge so the TOTAL flow from each entry = vph_per_entry.
        for i, (route_id, edges_str) in enumerate(self._routes):
            first_edge  = edges_str.split()[0]
            n_sharing   = self._routes_per_entry.get(first_edge, 1)
            route_vph   = max(1, int(round(vph_per_entry / n_sharing)))

            flow_el = ET.SubElement(root, "flow")
            flow_el.set("id",          f"flow_{level}_{i}")
            flow_el.set("type",        DEFAULT_VTYPE["id"])
            flow_el.set("route",       route_id)
            flow_el.set("begin",       str(begin))
            flow_el.set("end",         str(end))
            flow_el.set("vehsPerHour", str(route_vph))
            flow_el.set("departSpeed", "max")
            flow_el.set("departLane",  "best")

        # Write file
        out_path = os.path.join(self.output_dir, f"traffic_{level}.rou.xml")
        _write_xml(out_path, root)

        print(
            f"[DemandGenerator] {level.capitalize()} demand written: {out_path}\n"
            f"  {len(self._routes)} flows, ~{vph_per_entry} veh/h per entry edge"
        )
        return out_path

    def generate_all(self, duration: int = 3600, begin: int = 0) -> List[str]:
        """Generate .rou.xml files for all four demand levels."""
        paths = []
        for level in DEMAND_LEVELS:
            path = self.generate(level=level, duration=duration, begin=begin)
            paths.append(path)
        print(f"\n[DemandGenerator] All demand levels generated.")
        return paths

    @property
    def entry_edges(self) -> List[str]:
        return list(self._entry_edges)

    @property
    def routes(self) -> List[Tuple[str, str]]:
        return list(self._routes)

    @staticmethod
    def demand_levels() -> Dict[str, int]:
        return dict(DEMAND_LEVELS)