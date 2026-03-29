# sumo/network/demand_generator.py
"""
DemandGenerator

Generates SUMO-compatible traffic demand files (.rou.xml) for a given road network scenario.

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