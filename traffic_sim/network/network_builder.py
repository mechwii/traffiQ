# traffic_sim/network/network_builder.py
"""
NetworkBuilder

This class generates road network files (.net.xml) for all scenarios defined
in the project specification. Both simulation environments must use identical
network structures, so we generate everything programmatically rather than
hand-crafting XML files.

Supported scenarios:
    Lane configurations  : 1, 2, or 3 lanes per road
    Intersection counts  : 1, 2, 4, or 8 intersections
    Intersection geometry: four-way (+), T-junction (T), complex (star/offset)

How SUMO network generation works:
    SUMO does not accept hand-drawn maps directly. Instead, it uses two
    intermediate XML files as input to its netconvert compilation tool:
        .nod.xml -> defines junction nodes (intersections and border endpoints)
        .edg.xml -> defines directed road edges between those nodes
    netconvert reads both files and produces a final .net.xml that the
    simulator actually loads. This module generates the two intermediate files
    and then calls netconvert automatically via subprocess.

Usage:
    from traffic_sim.network.network_builder import NetworkBuilder

    builder = NetworkBuilder(output_dir="configs")

    # Build a single four-way intersection with 2 lanes per road
    builder.build(intersection_type="four_way", num_intersections=1, num_lanes=2)

    # Build every scenario defined in the spec at once
    builder.build_all()
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Tuple, Dict

# Distance in metres from the junction centre to the network border nodes.
# This controls how long the approach roads are.
ROAD_LENGTH = 100

# Lane width in metres.
LANE_WIDTH = 3.2

# Maximum vehicle speed on all roads (m/s). 13.89 m/s = 50 km/h.
MAX_SPEED = 13.89

    # ================================================================== #
    #  HELPER UTILITIES                                                  #
    # ================================================================== #

def _pretty_xml(element: ET.Element) -> str:
    """Return a nicely indented XML string from an ElementTree element."""
    raw = ET.tostring(element, encoding="unicode")
    reparsed = minidom.parseString(raw)
    return reparsed.toprettyxml(indent="    ")


def _write_xml(path: str, element: ET.Element) -> None:
    """Write a pretty-printed XML element to the given file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_pretty_xml(element))

    # ================================================================== #
    #  NODE / EDGE DATA CLASSES                                                 #
    # ================================================================== #

class Node:
    """Represents a SUMO junction node (an intersection or a dead-end border point)."""

    def __init__(self, node_id: str, x: float, y: float, node_type: str = "priority"):
        self.node_id   = node_id
        self.x         = x
        self.y         = y
        # "priority" means the junction uses right-of-way rules.
        # "dead_end" marks a network border -> a road that starts or ends here.
        self.node_type = node_type

    def to_xml_element(self) -> ET.Element:
        """Convert this Node to the <node> XML element SUMO expects."""
        return ET.Element(
            "node",
            attrib={
                "id":   self.node_id,
                "x":    str(self.x),
                "y":    str(self.y),
                "type": self.node_type,
            },
        )


class Edge:
    """Represents a directed road edge between two nodes in the SUMO network."""

    def __init__(
        self,
        edge_id:   str,
        from_node: str,
        to_node:   str,
        num_lanes: int   = 1,
        speed:     float = MAX_SPEED,
        priority:  int   = 1,
    ):
        self.edge_id   = edge_id
        self.from_node = from_node
        self.to_node   = to_node
        self.num_lanes = num_lanes
        self.speed     = speed
        # SUMO uses priority to resolve right-of-way at unmanaged junctions.
        # A higher value means higher priority. We keep it uniform (1) everywhere
        # so the AI has full control rather than SUMO's default priority rules.
        self.priority  = priority

    def to_xml_element(self) -> ET.Element:
        """Convert this Edge to the <edge> XML element SUMO expects."""
        return ET.Element(
            "edge",
            attrib={
                "id":       self.edge_id,
                "from":     self.from_node,
                "to":       self.to_node,
                "numLanes": str(self.num_lanes),
                "speed":    str(self.speed),
                "priority": str(self.priority),
            },
        )

    # ================================================================== #
    #  LAYOUT GENERATOR                                                 #
    # ================================================================== #
class _LayoutGenerator:
    """
    Internal helper that computes node and edge lists for every supported layout.

    All coordinates are in metres. The origin (0, 0) is the centre of the
    first (or only) intersection.
    """

    def __init__(self, num_lanes: int):
        self.num_lanes = num_lanes

    def single_four_way(self) -> Tuple[List[Node], List[Edge]]:
        """
        Single four-way (+) intersection.

        Layout (top view):
                 N
                 |
            W ---+--- E
                 |
                 S

        One centre junction node (C) and four dead-end border nodes (N, S, E, W).
        Edges are bidirectional on all four arms.
        """
        L = ROAD_LENGTH
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
        Single T-junction intersection (three arms, missing the southern one).

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
        Complex / irregular intersection with five arms in a star layout.

        Layout:
              NW    NE
               \\   /
            W ---C--- E
                 |
                 S

        The diagonal arms use cos(45°) * ROAD_LENGTH to keep consistent road length.
        """
        L    = ROAD_LENGTH
        diag = int(L * 0.707)  # L * cos(45 degrees)
        nodes = [
            Node("C",      0,     0, "priority"),
            Node("NW", -diag,  diag, "dead_end"),
            Node("NE",  diag,  diag, "dead_end"),
            Node("E",      L,     0, "dead_end"),
            Node("W",     -L,     0, "dead_end"),
            Node("S",      0,    -L, "dead_end"),
        ]
        edges = self._bidirectional_arms("C", ["NW", "NE", "E", "W", "S"])
        return nodes, edges

    def multi_intersection(
        self,
        count: int,
        intersection_type: str = "four_way",
    ) -> Tuple[List[Node], List[Edge]]:
        """
        Build a grid of connected four-way intersections.

        count=2  -> linear chain:  + --- +
        count=4  -> 2x2 grid:      + - +
                                   |   |
                                   + - +
        count=8  -> 2x4 grid:      + - + - + - +
                                   |   |   |   |
                                   + - + - + - +

        Each junction has its own set of external border nodes. Adjacent
        junctions are connected by bidirectional internal edges.
        """
        if count == 2:
            return self._linear_chain(2)
        elif count == 4:
            return self._grid(2, 2)
        elif count == 8:
            return self._grid(2, 4)
        else:
            raise ValueError(f"Unsupported intersection count: {count}")

    # ================================================================== #
    #  Internal helpers                                                 #
    # ================================================================== #

    def _bidirectional_arms(
        self, centre_id: str, arm_ids: List[str]
    ) -> List[Edge]:
        """
        Create one inbound and one outbound edge for every arm of an intersection.

        Edge naming follows the convention "<from>_to_<to>" so the destination
        is always readable directly from the edge ID.
        """
        edges = []
        for arm in arm_ids:
            # Outbound: centre -> border
            edges.append(Edge(f"{centre_id}_to_{arm}", centre_id, arm, num_lanes=self.num_lanes))
            # Inbound: border -> centre
            edges.append(Edge(f"{arm}_to_{centre_id}", arm, centre_id, num_lanes=self.num_lanes))
        return edges

    def _linear_chain(self, count: int) -> Tuple[List[Node], List[Edge]]:
        """
        Build 'count' intersections in a horizontal line.

        Junctions are spaced 2 * ROAD_LENGTH apart so each junction has a
        full ROAD_LENGTH approach road on each side before the next junction.
        Only the leftmost junction gets a western border node, only the
        rightmost gets an eastern one, and each gets its own north/south nodes.

        Visual for count=2:
                N0       N1
                |        |
            W0 -J0 ----- J1- E1
                |        |
                S0       S1
        """
        spacing = 2 * ROAD_LENGTH
        L       = ROAD_LENGTH
        nodes: List[Node] = []
        edges: List[Edge] = []

        for i in range(count):
            cx  = i * spacing
            jid = f"J{i}"
            nodes.append(Node(jid, cx, 0, "priority"))

            # Every junction gets its own north and south border nodes.
            nodes.append(Node(f"N{i}", cx,  L, "dead_end"))
            nodes.append(Node(f"S{i}", cx, -L, "dead_end"))
            edges += self._bidirectional_arms(jid, [f"N{i}", f"S{i}"])

            # Western border only for the leftmost junction.
            if i == 0:
                nodes.append(Node("W0", -L, 0, "dead_end"))
                edges += self._bidirectional_arms(jid, ["W0"])

            # Eastern border only for the rightmost junction.
            if i == count - 1:
                nodes.append(Node(f"E{i}", cx + L, 0, "dead_end"))
                edges += self._bidirectional_arms(jid, [f"E{i}"])

            # Internal horizontal road connecting this junction to the next.
            if i < count - 1:
                next_jid = f"J{i+1}"
                edges.append(Edge(f"{jid}_to_{next_jid}", jid, next_jid, num_lanes=self.num_lanes))
                edges.append(Edge(f"{next_jid}_to_{jid}", next_jid, jid, num_lanes=self.num_lanes))

        return nodes, edges

    def _grid(self, rows: int, cols: int) -> Tuple[List[Node], List[Edge]]:
        """
        Build a rows x cols grid of four-way intersections.

        Junction naming: J_{row}_{col} (e.g. J_0_0 is bottom-left).
        Border nodes are only created on the outer edge of the grid:
            top row    -> north border nodes
            bottom row -> south border nodes
            left col   -> west border nodes
            right col  -> east border nodes
        Internal edges connect each junction to its right and top neighbour.
        """
        spacing = 2 * ROAD_LENGTH
        L       = ROAD_LENGTH
        nodes: List[Node] = []
        edges: List[Edge] = []

        # Create all junction nodes first so we can reference them by (row, col).
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
                cx  = c * spacing
                cy  = r * spacing

                # North border nodes only on the top row.
                if r == rows - 1:
                    nid = f"N_{r}_{c}"
                    nodes.append(Node(nid, cx, cy + L, "dead_end"))
                    edges += self._bidirectional_arms(jid, [nid])

                # South border nodes only on the bottom row.
                if r == 0:
                    sid = f"S_{r}_{c}"
                    nodes.append(Node(sid, cx, cy - L, "dead_end"))
                    edges += self._bidirectional_arms(jid, [sid])

                # West border nodes only on the left column.
                if c == 0:
                    wid = f"W_{r}_{c}"
                    nodes.append(Node(wid, cx - L, cy, "dead_end"))
                    edges += self._bidirectional_arms(jid, [wid])

                # East border nodes only on the right column.
                if c == cols - 1:
                    eid = f"E_{r}_{c}"
                    nodes.append(Node(eid, cx + L, cy, "dead_end"))
                    edges += self._bidirectional_arms(jid, [eid])

                # Internal horizontal road to the right neighbour.
                if c < cols - 1:
                    rjid = junctions[(r, c + 1)]
                    edges.append(Edge(f"{jid}_to_{rjid}", jid, rjid, num_lanes=self.num_lanes))
                    edges.append(Edge(f"{rjid}_to_{jid}", rjid, jid, num_lanes=self.num_lanes))

                # Internal vertical road to the top neighbour.
                if r < rows - 1:
                    tjid = junctions[(r + 1, c)]
                    edges.append(Edge(f"{jid}_to_{tjid}", jid, tjid, num_lanes=self.num_lanes))
                    edges.append(Edge(f"{tjid}_to_{jid}", tjid, jid, num_lanes=self.num_lanes))

        return nodes, edges


class NetworkBuilder:
    """
    Generates SUMO road network files for all scenarios defined in the spec.

    Args:
        output_dir: Root directory where scenario sub-folders will be created.
                    Defaults to "configs".
        sumo_home: Path to the SUMO installation directory (the folder that
                   contains bin/netconvert). Falls back to $SUMO_HOME if None.

    Example:
        builder = NetworkBuilder(output_dir="configs")
        builder.build("four_way", num_intersections=1, num_lanes=1)
        builder.build_all()
    """

    # Complete list of scenarios from the specification.
    # Single intersections support all three geometry types and all lane counts.
    # Multi-intersection grids always use four-way geometry.
    SCENARIOS = [
        ("four_way",   1, 1), ("four_way",   1, 2), ("four_way",   1, 3),
        ("t_junction", 1, 1), ("t_junction", 1, 2), ("t_junction", 1, 3),
        ("complex",    1, 1), ("complex",    1, 2), ("complex",    1, 3),
        ("four_way",   2, 1), ("four_way",   2, 2), ("four_way",   2, 3),
        ("four_way",   4, 1), ("four_way",   4, 2), ("four_way",   4, 3),
        ("four_way",   8, 1), ("four_way",   8, 2), ("four_way",   8, 3),
    ]

    def __init__(self, output_dir: str = "configs", sumo_home: str = None):
        self.output_dir = output_dir
        self.sumo_home  = sumo_home or os.environ.get("SUMO_HOME", "")

    # ================================================================== #
    #  Public API                                                  #
    # ================================================================== #
    
    def build(
        self,
        intersection_type: str = "four_way",
        num_intersections: int = 1,
        num_lanes:         int = 1,
    ) -> str:
        """
        Build one network scenario and return the path to the compiled .net.xml.

        The method generates the intermediate .nod.xml and .edg.xml files,
        then calls netconvert to compile them into the final network file.

        Args:
            intersection_type: One of "four_way", "t_junction", "complex".
            num_intersections: 1, 2, 4, or 8.
            num_lanes: 1, 2, or 3.

        Returns:
            Absolute path to the generated .net.xml file.
        """
        scenario_name = self._scenario_name(intersection_type, num_intersections, num_lanes)
        scenario_dir  = os.path.join(self.output_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        print(f"[NetworkBuilder] Building scenario: {scenario_name}")

        # Step 1 : Generate node and edge lists
        nodes, edges = self._get_layout(
            intersection_type, num_intersections, num_lanes
        )

        # Step 2: Write intermediate .nod.xml and .edg.xml files
        nod_path = os.path.join(scenario_dir, "network.nod.xml")
        self._write_nodes(nod_path, nodes)

        edg_path = os.path.join(scenario_dir, "network.edg.xml")
        self._write_edges(edg_path, edges)

        # Step 3: Run netconvert to compile the .net.xml file
        net_path = os.path.join(scenario_dir, "network.net.xml")
        self._run_netconvert(nod_path, edg_path, net_path)

        print(f"[NetworkBuilder] SUCCESS -> Network saved to: {net_path}")
        return net_path

    def build_all(self) -> List[str]:
        """Build every scenario in SCENARIOS and return the list of .net.xml paths."""
        paths = []
        for intersection_type, num_intersections, num_lanes in self.SCENARIOS:
            path = self.build(intersection_type, num_intersections, num_lanes)
            paths.append(path)
        print(f"\n[NetworkBuilder] SUCCESS -> All {len(paths)} scenarios built.")
        return paths

    def get_scenario_dir(
        self,
        intersection_type: str,
        num_intersections: int,
        num_lanes:         int,
    ) -> str:
        """Return the output directory path for a given scenario without building it."""
        return os.path.join(
            self.output_dir,
            self._scenario_name(intersection_type, num_intersections, num_lanes),
        )

    # ================================================================== #
    # Internal helpers                                                 #
    # ================================================================== #

    @staticmethod
    def _scenario_name(
        intersection_type: str, num_intersections: int, num_lanes: int
    ) -> str:
        """
        Build a filesystem-safe folder name from the scenario parameters.
        Example: ("four_way", 1, 2) -> "four_way_1int_2lanes"
        """
        return f"{intersection_type}_{num_intersections}int_{num_lanes}lanes"

    def _get_layout(
        self,
        intersection_type: str,
        num_intersections: int,
        num_lanes:         int,
    ) -> Tuple[List[Node], List[Edge]]:
        """Dispatch to the correct _LayoutGenerator method based on the scenario parameters."""
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
            # Multi-intersection layouts are always four-way grids.
            return gen.multi_intersection(num_intersections)

    @staticmethod
    def _write_nodes(path: str, nodes: List[Node]) -> None:
        """Write the <nodes> XML file that netconvert expects."""
        root = ET.Element("nodes")
        for node in nodes:
            root.append(node.to_xml_element())
        _write_xml(path, root)
        print(f"  Nodes file written: {path}  ({len(nodes)} nodes)")

    @staticmethod
    def _write_edges(path: str, edges: List[Edge]) -> None:
        """Write the <edges> XML file that netconvert expects."""
        root = ET.Element("edges")
        for edge in edges:
            root.append(edge.to_xml_element())
        _write_xml(path, root)
        print(f"  Edges file written: {path}  ({len(edges)} edges)")

    def _run_netconvert(
        self, nod_path: str, edg_path: str, net_path: str
    ) -> None:
        """
        Call SUMO's netconvert tool to compile the two XML files into a .net.xml.

        netconvert is found either at $SUMO_HOME/bin/netconvert or directly on
        the system PATH if SUMO was installed via a package manager.

        --offset.disable-normalization keeps our manually defined coordinates as-is
        instead of letting netconvert re-center the network.
        """
        netconvert_bin = (
            os.path.join(self.sumo_home, "bin", "netconvert")
            if self.sumo_home
            else "netconvert"
        )

        cmd = [
            netconvert_bin,
            "--node-files",  nod_path,
            "--edge-files",  edg_path,
            "--output-file", net_path,
            "--no-warnings",
            "--offset.disable-normalization", # preserve our custom coordinates without re-centering
        ]

        print(f"  Running: {' '.join(cmd)}")
        try:
            # Using subprocess.run with check=True will raise an exception if netconvert fails,
            # which we can catch and re-raise with a more user-friendly message.
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
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