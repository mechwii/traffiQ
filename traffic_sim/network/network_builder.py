# traffic_sim/network/network_builder.py
"""
NetworkBuilder

This class generates compatible road network files (.net.xml) using the plain XML format that SUMO's netconvert tool can compile.

Both environments must implement identical road network structures. So we need to generate the single intersections (1, 2, and 3 lanes)
and the multi-intersection grids before anything else can exist.

Supported scenarios (from the project specification):
    Lane configurations : 1, 2 or 3 lanes per road
    Intersection counts  : 1, 2, 4, or 8 intersections
    Intersection geometry: four-way (+), T-junction (T), complex (star/offset)

How it works:
    SUMO road networks are defined by two intermediate XML files:
    .edg.xml -> defines directed road edges between nodes

These two files are compiled by the SUMO tool netconvert into a final .net.xml file that the simulator actually loads.

This module:
  1. Writes the .nod.xml and .edg.xml files for each scenario.
  2. Calls netconvert via subprocess to produce the .net.xml.
  3. Saves everything in the configs/ directory.

Usage
-----
    from traffic_sim.network.network_builder import NetworkBuilder

    builder = NetworkBuilder(output_dir="configs")

    # Build a single four-way intersection with 2 lanes per road
    builder.build(
        intersection_type="four_way",
        num_intersections=1,
        num_lanes=2
    )

    # Build all scenarios at once
    builder.build_all()
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Tuple, Dict

# Road segment length in metres (distance from junction to network border)
ROAD_LENGTH = 100

# Lane width in metres
LANE_WIDTH = 3.2

# Maximum vehicle speed on all roads (m/s).  50 km/h ≈ 13.89 m/s
MAX_SPEED = 13.89

# ============ HELPER UTILITIES ============

def _pretty_xml(element: ET.Element) -> str:
    """Return a nicely indented XML string from an ElementTree element."""
    raw = ET.tostring(element, encoding="unicode")
    reparsed = minidom.parseString(raw)
    return reparsed.toprettyxml(indent="    ")


def _write_xml(path: str, element: ET.Element) -> None:
    """Write a pretty-printed XML element to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    content = _pretty_xml(element)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)

# ============ NODE / EDGE DATA CLASSES ============

class Node:
    """Represents a SUMO junction node."""

    def __init__(self, node_id: str, x: float, y: float, node_type: str = "priority"):
        self.node_id = node_id
        self.x = x # pos X of the node
        self.y = y # pos Y of the node
        # Junction type: "priority" (right-of-way rules), "dead_end" (border => end of the road)
        self.node_type = node_type

    def to_xml_element(self) -> ET.Element:
        """Converting Node object to XML element"""
        return ET.Element(
            "node", # XML tag's name <node>
            attrib={
                "id": self.node_id,
                "x": str(self.x),
                "y": str(self.y),
                "type": self.node_type,
            },
        )

class Edge:
    """Represents a directed SUMO road edge."""

    def __init__(
        self,
        edge_id: str,
        from_node: str,
        to_node: str,
        num_lanes: int = 1,
        speed: float = MAX_SPEED,
        priority: int = 1,
    ):
        self.edge_id = edge_id
        self.from_node = from_node # Starting node
        self.to_node = to_node # Ending node
        self.num_lanes = num_lanes # Lines number (one by default)
        self.speed = speed 
        self.priority = priority # The priority value for the roard => the higher the value, the higher the priority of the lane.

    def to_xml_element(self) -> ET.Element:
        """Converting Edge object to XML element"""
        return ET.Element(
            "edge",
            attrib={
                "id": self.edge_id,
                "from": self.from_node,
                "to": self.to_node,
                "numLanes": str(self.num_lanes),
                "speed": str(self.speed),
                "priority": str(self.priority),
            },
        )

# ============ LAYOUT GENERATOR ============
class _LayoutGenerator:
    """
    Internal class that computes node and edge lists for every
    supported road layout.

    All coordinates are in metres.  The origin (0, 0) is the centre of
    the first (or only) intersection.
    """

    def __init__(self, num_lanes : int):
        self.num_lanes = num_lanes
    
    # ---- Public entry points (one per intersection type / count) ----

    def single_four_way(self) -> Tuple[List[Node], List[Edge]]:
        """
        Single four-way (+) intersection.

        Layout (top view):
                 N
                 |
            W ---+--- E
                 |
                 S

        Nodes: centre (C), north (N), south (S), east (E), west (W)
        Edges: bidirectional arms from centre to each border node
        """
        L = ROAD_LENGTH

        # Node(direction, horizontal (x), vertical (Y), node_type)
        nodes = [
            Node("C",  0,  0, "priority"),
            Node("N",  0,  L, "dead_end"),
            Node("S",  0, -L, "dead_end"),
            Node("E",  L,  0, "dead_end"),
            Node("W", -L,  0, "dead_end"),
        ]
        edges = self._bidirectional_arms("C", ["N", "S", "E", "W"])
        return nodes, edges

    def single_t_junction(self) -> Tuple[List[Node], List[Edge]]:
        """
        Single T-junction intersection (missing the southern arm).

        Layout:
                 N
                 |
            W ---+--- E
               (no S)
        """
        L = ROAD_LENGTH
        nodes = [
            Node("C",  0, 0, "priority"),
            Node("N",  0, L, "dead_end"),
            Node("E",  L, 0, "dead_end"),
            Node("W", -L, 0, "dead_end"),
        ]
        edges = self._bidirectional_arms("C", ["N", "E", "W"])
        return nodes, edges
    
    def single_complex(self) -> Tuple[List[Node], List[Edge]]:
        """
        Complex / irregular intersection: a five-arm star layout.

        Layout:
              NW    NE
               \   /
            W ---C--- E
                 |
                 S
        """
        L = ROAD_LENGTH
        diag = int(L * 0.707)   # L * cos(45°)
        nodes = [
            Node("C",     0,     0, "priority"),
            Node("NW", -diag,  diag, "dead_end"),
            Node("NE",  diag,  diag, "dead_end"),
            Node("E",      L,     0, "dead_end"),
            Node("W",     -L,     0, "dead_end"),
            Node("S",      0,    -L, "dead_end"),
        ]
        edges = self._bidirectional_arms("C", ["NW", "NE", "E", "W", "S"])
        return nodes, edges
    
    def multi_intersection(
        self, count: int,
        intersection_type: str = "four_way"
    ) -> Tuple[List[Node], List[Edge]]:
        """
        Build a grid of count four-way intersection (maybe other types) connected by shared road
        segments. Multi-intersection networks always use four-way (+) (for now) geometry
        as shown in the project specification diagrams.

        For count=2:
        + ------- +

        For count=4:
        + --- +
        |     |
        + --- +

        For count=8:   2 * 4 grid
        +---+---+---+---+
        |   |   |   |   |
        +---+---+---+---+

        Each intersection gets its own set of external border nodes.
        Intersections are connected by internal edges.
        """
        if count == 2:
            return self._linear_chain(2)
        elif count == 4:
            return self._grid(2, 2)
        elif count == 8:
            return self._grid(2, 4)
        else:
            raise ValueError(f"Unsupported intersection count: {count}")

    # ---- Internal helpers ----

    def _bidirectional_arms(
        self, centre_id: str, arm_ids: List[str]
    ) -> List[Edge]:
        """
        Create one inbound and one outbound edge for every arm.
        Edge naming convention:  "<from>_to_<to>"
        """
        edges = []
        for arm in arm_ids:
            # Outbound: centre -> border
            edges.append(
                Edge(
                    f"{centre_id}_to_{arm}",
                    centre_id,
                    arm,
                    num_lanes=self.num_lanes,
                )
            )
            # Inbound: border -> centre
            edges.append(
                Edge(
                    f"{arm}_to_{centre_id}",
                    arm,
                    centre_id,
                    num_lanes=self.num_lanes,
                )
            )
        return edges
    
    def _linear_chain(self, count: int) -> Tuple[List[Node], List[Edge]]:
        """
        Build count intersections in a horizontal line.
        Spacing between intersections is 2 * ROAD_LENGTH.
        """

        spacing = 2 * ROAD_LENGTH
        L = ROAD_LENGTH

        nodes: List[Node] = []
        edges: List[Edge] = []

        for i in range(count):
            # J0                   J1
            cx = i * spacing
            jid = f"J{i}"          # junction node id
            nodes.append(Node(jid, cx, 0, "priority"))

            """
            North / South border nodes for this junction
            
            N0                   N1
            |                    |
            J0                   J1
            |                    |
            S0                   S1
            """
            nodes.append(Node(f"N{i}", cx,  L, "dead_end"))
            nodes.append(Node(f"S{i}", cx, -L, "dead_end"))

            # N–S arms
            edges += self._bidirectional_arms(jid, [f"N{i}", f"S{i}"])

            """
            Western border node (only for the leftmost junction)

            N0                   N1
            |                    |
        W0 --J0                   J1-- E1
            |                    |
            S0                   S1
            """
            if i == 0:
                nodes.append(Node("W0", -L, 0, "dead_end"))
                edges += self._bidirectional_arms(jid, ["W0"])

            # Eastern border node (only for the rightmost junction)
            if i == count - 1:
                nodes.append(Node(f"E{i}", cx + L, 0, "dead_end"))
                edges += self._bidirectional_arms(jid, [f"E{i}"])

            """
            Internal horizontal road connecting this junction to the next

            N0                   N1
            |                    |
        W0 --J0===================J1-- E1
            |                    |
            S0                   S1
            
            """
            if i < count - 1:
                next_jid = f"J{i+1}"
                edges.append(
                    Edge(f"{jid}_to_{next_jid}", jid, next_jid,
                         num_lanes=self.num_lanes)
                )
                edges.append(
                    Edge(f"{next_jid}_to_{jid}", next_jid, jid,
                         num_lanes=self.num_lanes)
                )

        return nodes, edges

    def _grid(self, rows: int, cols: int) -> Tuple[List[Node], List[Edge]]:
        """
        Build a rows * cols grid of four-way intersections.
        Nodes are named J_{row}_{col}.
        """
        spacing = 2 * ROAD_LENGTH
        L = ROAD_LENGTH
        nodes: List[Node] = []
        edges: List[Edge] = []

        # Create all junction nodes
        junctions: Dict[Tuple[int, int], str] = {}
        for r in range(rows):
            for c in range(cols):
                jid = f"J_{r}_{c}"
                junctions[(r, c)] = jid
                nodes.append(Node(jid, c * spacing, r * spacing, "priority"))

        # Border nodes and edges
        for r in range(rows):
            for c in range(cols):
                jid = junctions[(r, c)]
                cx = c * spacing
                cy = r * spacing

                # Top border (only top row)
                if r == rows - 1:
                    nid = f"N_{r}_{c}"
                    nodes.append(Node(nid, cx, cy + L, "dead_end"))
                    edges += self._bidirectional_arms(jid, [nid])

                # Bottom border (only bottom row)
                if r == 0:
                    sid = f"S_{r}_{c}"
                    nodes.append(Node(sid, cx, cy - L, "dead_end"))
                    edges += self._bidirectional_arms(jid, [sid])

                # Left border (only left column)
                if c == 0:
                    wid = f"W_{r}_{c}"
                    nodes.append(Node(wid, cx - L, cy, "dead_end"))
                    edges += self._bidirectional_arms(jid, [wid])

                # Right border (only right column)
                if c == cols - 1:
                    eid = f"E_{r}_{c}"
                    nodes.append(Node(eid, cx + L, cy, "dead_end"))
                    edges += self._bidirectional_arms(jid, [eid])

                # Horizontal internal road -> right neighbour
                if c < cols - 1:
                    rjid = junctions[(r, c + 1)]
                    edges.append(
                        Edge(f"{jid}_to_{rjid}", jid, rjid,
                             num_lanes=self.num_lanes)
                    )
                    edges.append(
                        Edge(f"{rjid}_to_{jid}", rjid, jid,
                             num_lanes=self.num_lanes)
                    )

                # Vertical internal road -> top neighbour
                if r < rows - 1:
                    tjid = junctions[(r + 1, c)]
                    edges.append(
                        Edge(f"{jid}_to_{tjid}", jid, tjid,
                             num_lanes=self.num_lanes)
                    )
                    edges.append(
                        Edge(f"{tjid}_to_{jid}", tjid, jid,
                             num_lanes=self.num_lanes)
                    )

        return nodes, edges
    
# ============ MAIN CLASS ============
class NetworkBuilder:
    """
    Generates SUMO road network files for all scenarios defined in the
    project specification.

    Parameters:
        output_dir : str
            Root directory where scenario sub-folders will be created.
            Default: "configs"
        sumo_home : str | None
            Path to the SUMO installation directory (the folder that contains
            'bin/netconvert').  If None, the SUMO_HOME environment variable
            is used.

    Example:
        builder = NetworkBuilder(output_dir="configs")
        builder.build("four_way", num_intersections=1, num_lanes=1)
        builder.build_all()
    """

    # All scenarios from the specification.
    # Single intersections support all three geometry types × all lane counts.
    # Multi-intersection grids support all lane counts (always four-way geometry).
    SCENARIOS = [
        # -- Single intersection, all geometries, all lane counts --
        ("four_way",   1, 1), ("four_way",   1, 2), ("four_way",   1, 3),
        ("t_junction", 1, 1), ("t_junction", 1, 2), ("t_junction", 1, 3),
        ("complex",    1, 1), ("complex",    1, 2), ("complex",    1, 3),
        # -- Multi-intersection grids (four-way), all lane counts --
        ("four_way",   2, 1), ("four_way",   2, 2), ("four_way",   2, 3),
        ("four_way",   4, 1), ("four_way",   4, 2), ("four_way",   4, 3),
        ("four_way",   8, 1), ("four_way",   8, 2), ("four_way",   8, 3),
    ]

    def __init__(self, output_dir: str = "configs", sumo_home: str = None):
        self.output_dir = output_dir
        self.sumo_home = sumo_home or os.environ.get("SUMO_HOME", "")
    
    # ---- Callable functions ----
    def build(
        self,
        intersection_type: str = "four_way",
        num_intersections: int = 1,
        num_lanes: int = 1,
    ) -> str:
        """
        Build one network scenario and return the path to the .net.xml file.

        Parameters:
            intersection_type : str
                One of "four_way", "t_junction", "complex".
            num_intersections : int
                1, 2, 4, or 8.
            num_lanes : int
                1, 2, or 3.

        Returns:
            str
                Absolute path to the generated .net.xml file.
        """
        scenario_name = self._scenario_name(
            intersection_type, num_intersections, num_lanes
        )

        # We create the directory where the files will be saved
        scenario_dir = os.path.join(self.output_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        print(f"[NetworkBuilder] Building scenario: {scenario_name}")

        # 1. Generate node and edge lists
        nodes, edges = self._get_layout(
            intersection_type, num_intersections, num_lanes
        )

        # 2. Write .nod.xml
        nod_path = os.path.join(scenario_dir, "network.nod.xml")
        self._write_nodes(nod_path, nodes)

        # 3. Write .edg.xml
        edg_path = os.path.join(scenario_dir, "network.edg.xml")
        self._write_edges(edg_path, edges)

        # 4. Run netconvert to produce .net.xml
        net_path = os.path.join(scenario_dir, "network.net.xml")
        self._run_netconvert(nod_path, edg_path, net_path)

        print(f"[NetworkBuilder] SUCCESS => Network saved to: {net_path}")
        return net_path
    

    def build_all(self) -> List[str]:
        """
        Build every scenario defined in SCENARIOS and return the list
        of generated .net.xml file paths.
        """
        paths = []
        for intersection_type, num_intersections, num_lanes in self.SCENARIOS:
            path = self.build(intersection_type, num_intersections, num_lanes)
            paths.append(path)
        print(f"\n[NetworkBuilder] SUCCESS => All {len(paths)} scenarios built.")
        return paths

    def get_scenario_dir(
        self,
        intersection_type: str,
        num_intersections: int,
        num_lanes: int,
    ) -> str:
        """Return the directory path for a given scenario (without building)."""
        return os.path.join(
            self.output_dir,
            self._scenario_name(intersection_type, num_intersections, num_lanes),
        )

    # ---- Internal helpers ----
    @staticmethod
    def _scenario_name(
        intersection_type: str, num_intersections: int, num_lanes: int
    ) -> str:
        """
        Convert scenario parameters to a filesystem-safe folder name.
        Example: "four_way_1int_2lanes"
        """
        return f"{intersection_type}_{num_intersections}int_{num_lanes}lanes"


    def _get_layout(
        self,
        intersection_type: str,
        num_intersections: int,
        num_lanes: int,
    ) -> Tuple[List[Node], List[Edge]]:
        """Delegate to the correct _LayoutGenerator method."""
        gen = _LayoutGenerator(num_lanes)

        if num_intersections == 1:
            if intersection_type == "four_way":
                return gen.single_four_way()
            elif intersection_type == "t_junction":
                return gen.single_t_junction()
            elif intersection_type == "complex":
                return gen.single_complex()
            else:
                raise ValueError(
                    f"Unknown intersection_type: '{intersection_type}'. "
                    "Choose from: four_way, t_junction, complex."
                )
        else:
            # Multi-intersection layouts are always four-way grids
            # return gen.multi_intersection(num_intersections, intersection_type)
            return gen.multi_intersection(num_intersections)

    @staticmethod
    def _write_nodes(path: str, nodes: List[Node]) -> None:
        """Write a SUMO .nod.xml file."""
        root = ET.Element("nodes")
        for node in nodes:
            root.append(node.to_xml_element())
        _write_xml(path, root)
        print(f"NODE EVENT => Nodes file written: {path}  ({len(nodes)} nodes)")

    @staticmethod
    def _write_edges(path: str, edges: List[Edge]) -> None:
        """Write a SUMO .edg.xml file."""
        root = ET.Element("edges")
        for edge in edges:
            root.append(edge.to_xml_element())
        _write_xml(path, root)
        print(f"EDGE EVENT => Edges file written: {path}  ({len(edges)} edges)")

    def _run_netconvert(
        self, nod_path: str, edg_path: str, net_path: str
    ) -> None:
        """
        Invoke SUMO's netconvert command-line tool to compile .nod.xml
        and .edg.xml into a final .net.xml file.

        netconvert is located at $SUMO_HOME/bin/netconvert on most
        installations (or simply 'netconvert' if it is on the PATH).
        """
        # Try to find netconvert
        if self.sumo_home:
            netconvert_bin = os.path.join(self.sumo_home, "bin", "netconvert")
        else:
            netconvert_bin = "netconvert"   # assume it's on the system PATH

        cmd = [
            netconvert_bin,
            "--node-files", nod_path,
            "--edge-files", edg_path,
            "--output-file", net_path,
            "--no-warnings",            # suppress minor warnings
            "--offset.disable-normalization",  # keep our coordinates as-is
        ]

        print(f"COMMAND EXECUTION => Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout)
        except FileNotFoundError:
            raise EnvironmentError(
                "netconvert not found.\n"
                "Please either:\n"
                "  1. Set the SUMO_HOME environment variable to your SUMO "
                "installation directory, OR\n"
                "  2. Add the SUMO bin/ folder to your system PATH.\n"
                "Download SUMO from: https://sumo.dlr.de/docs/Downloads.php"
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"netconvert failed with exit code {exc.returncode}.\n"
                f"stderr: {exc.stderr}"
            )