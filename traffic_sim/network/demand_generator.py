# traffic_sim/network/demand_generator.py
"""
DemandGenerator

Generates traffic demand files (.rou.xml) compatible with our SUMO network scenarios.

Traffic demand levels defined by the project spec:
  - low: ~100 veh/h per ENTRY EDGE (Free-flow, no queuing)
  - moderate: ~300 veh/h per ENTRY EDGE (Occasional queuing)
  - high: ~600 veh/h per ENTRY EDGE (Persistent queuing)
  - congested: ~900 veh/h per ENTRY EDGE (Near-capacity)

CRITICAL DETAIL regarding flow rates:
The target volume (vph) is per ENTRY EDGE, not per route. 
Since a single entry edge (like "N_to_C") usually fans out into multiple routes 
(e.g., straight, left, right), we have to divide the entry's total target flow 
by the number of routes it serves.

Example: If "N_to_C" has a target of 300 veh/h and splits into 3 routes, 
each individual route is generated at 100 veh/h to keep the total injection correct.

How it works under the hood:
1. Reads the parsed .net.xml file.
2. Identifies entry edges by looking for "dead_end" nodes.
3. Uses a BFS to map out all valid through-routes across the network.
4. Calculates the required flow per route and builds the XML.

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

# Target vehicles per hour per ENTRY EDGE for each preset
DEMAND_LEVELS: Dict[str, int] = {
    "low":       100,
    "moderate":  300,
    "high":      600,
    "congested": 900,
}

# Standard passenger car specs used for all generated traffic
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

# ==========================================================================
# Helper utilities for XML formatting
# ==========================================================================

def _pretty_xml(element: ET.Element) -> str:
    """Formats the raw XML string with proper indentation so it's human-readable."""
    raw = ET.tostring(element, encoding="unicode")
    reparsed = minidom.parseString(raw)
    return reparsed.toprettyxml(indent="    ")

def _write_xml(path: str, element: ET.Element) -> None:
    """Ensures the directory exists before dumping the XML."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_pretty_xml(element))

# ==========================================================================
# Route Discovery logic
# ==========================================================================

def _parse_net_edges(net_file: str) -> Tuple[List[str], List[str]]:
    """
    Parses a compiled .net.xml file to separate entry edges from internal ones.
    
    Returns:
      entry_edges: Edges whose 'from' node is a dead_end (network border).
      all_edges: Every valid, non-internal road edge in the network.
    """
    tree = ET.parse(net_file)
    root = tree.getroot()

    # First, find all junctions acting as spawn points
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

        # Skip junction internals, we only care about real roads
        if eid.startswith(":") or function == "internal":
            continue

        from_node = edge_el.get("from", "")
        to_node   = edge_el.get("to",   "")

        all_edges.append(eid)

        # An entry edge starts at a dead_end and points inwards
        if from_node in dead_ends and to_node not in dead_ends:
            entry_edges.append(eid)

    # Safety net: if the user's network didn't explicitly tag dead_ends, 
    # we guess based on our naming convention (N, S, E, W prefixes).
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
    Finds all valid paths through the network using a Breadth-First Search.
    A valid through-route spawns at an entry edge and leaves via a different border node.
    """

    def nodes_from_id(eid: str):
        parts = eid.split("_to_")
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, None

    # Build a simple adjacency graph: from_node -> [(edge_id, to_node)]
    adjacency: Dict[str, List[Tuple[str, str]]] = {}
    for eid in all_edges:
        frm, to = nodes_from_id(eid)
        if frm and to:
            adjacency.setdefault(frm, []).append((eid, to))

    # Identify all borders to know when to terminate the BFS paths
    border_nodes: set = set()
    for eid in entry_edges:
        frm, _ = nodes_from_id(eid)
        if frm:
            border_nodes.add(frm)

    routes: List[Tuple[str, str]] = []
    route_idx = 0
    seen_paths: set = set()
    
    # Failsafe to prevent infinite loops if the network has a roundabout or cycle
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

            # Route complete: we hit a border node that isn't where we started
            if current_node in border_nodes and current_node != start_from:
                key = tuple(path)
                if key not in seen_paths:
                    seen_paths.add(key)
                    route_id = f"route_{route_idx}"
                    routes.append((route_id, " ".join(path)))
                    route_idx += 1
                continue

            # Continue BFS exploration
            for next_edge, next_node in adjacency.get(current_node, []):
                nodes_visited = {nodes_from_id(e)[0] for e in path}
                nodes_visited.add(start_from)
                
                # Prevent U-turns or looping back on ourselves
                if next_node not in nodes_visited:
                    queue.append((next_node, path + [next_edge]))

    return routes


# ==========================================================================
# Main DemandGenerator class
# ==========================================================================

class DemandGenerator:
    """
    Handles the generation of SUMO traffic demand files based on a compiled network.
    
    Args:
        net_file (str): Path to the .net.xml file built by NetworkBuilder.
        output_dir (str, optional): Where to dump the .rou.xml files. Defaults to the net_file's directory.
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

        # Count how many routes branch out from each entry edge.
        # We need this to distribute the target volume evenly later.
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
        Builds the .rou.xml file for a specific congestion level.

        Returns:
            str: The path to the generated file.
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

        # Initialize the XML tree
        root = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set(
            "xsi:noNamespaceSchemaLocation",
            "http://sumo.dlr.de/xsd/routes_file.xsd",
        )

        # Inject the vehicle profile
        vtype_el = ET.SubElement(root, "vType")
        for attr, val in DEFAULT_VTYPE.items():
            vtype_el.set(attr, val)

        # Declare all the valid paths we computed during init
        for route_id, edges in self._routes:
            route_el = ET.SubElement(root, "route")
            route_el.set("id", route_id)
            route_el.set("edges", edges)

        # Create the flow elements
        # Here we apply the flow division logic: if an entry has 3 routes, 
        # each route gets 1/3 of the vph_per_entry target.
        for i, (route_id, edges_str) in enumerate(self._routes):
            first_edge  = edges_str.split()[0]
            n_sharing   = self._routes_per_entry.get(first_edge, 1)
            
            # Ensure we don't end up with zero-flow routes due to rounding
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

        out_path = os.path.join(self.output_dir, f"traffic_{level}.rou.xml")
        _write_xml(out_path, root)

        print(
            f"[DemandGenerator] {level.capitalize()} demand written: {out_path}\n"
            f"  {len(self._routes)} flows, ~{vph_per_entry} veh/h per entry edge"
        )
        return out_path

    def generate_all(self, duration: int = 3600, begin: int = 0) -> List[str]:
        """Convenience method to generate all four preset levels at once."""
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