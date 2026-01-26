from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EdgeData:
    u: int
    v: int
    capacity: float
    t0: float
    length: float
    b: float
    power: float


@dataclass
class GraphData:
    num_nodes: int
    edges: List[EdgeData]
    od_demand: Dict[Tuple[int, int], float]


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f.readlines()]


def parse_net_tntp(path: str) -> Tuple[int, List[EdgeData]]:
    lines = _read_lines(path)
    data_started = False
    edges: List[EdgeData] = []
    num_nodes = 0
    for line in lines:
        if not line or line.startswith("~"):
            continue
        lower = line.lower()
        if "number of nodes" in lower:
            # supports "<NUMBER OF NODES> 24" and "Number of Nodes> 24"
            if ">" in line:
                num_nodes = int(line.split(">")[-1].strip())
            else:
                num_nodes = int(line.split()[-1].strip())
        if "init_node" in lower or "init node" in lower:
            data_started = True
            continue
        if not data_started:
            continue
        parts = [p for p in line.replace(";", "").split() if p]
        if len(parts) < 6:
            continue
        u = int(parts[0])
        v = int(parts[1])
        capacity = float(parts[2])
        length = float(parts[3])
        t0 = float(parts[4])
        b = float(parts[5]) if len(parts) > 5 else 0.15
        power = float(parts[6]) if len(parts) > 6 else 4.0
        edges.append(
            EdgeData(
                u=u,
                v=v,
                capacity=capacity,
                t0=t0,
                length=length,
                b=b,
                power=power,
            )
        )
    return num_nodes, edges


def parse_trips_tntp(path: str) -> Dict[Tuple[int, int], float]:
    lines = _read_lines(path)
    demands: Dict[Tuple[int, int], float] = {}
    current_origin = None
    for line in lines:
        if not line or line.startswith("~"):
            continue
        if line.lower().startswith("origin"):
            current_origin = int(line.split()[1])
            continue
        if current_origin is None:
            continue
        # format: dest1 : val1; dest2 : val2; ...
        parts = line.split(";")
        for part in parts:
            if ":" not in part:
                continue
            dest_str, val_str = part.split(":")
            dest = int(dest_str.strip())
            val = float(val_str.strip())
            if val > 0:
                demands[(current_origin, dest)] = val
    return demands


def load_graph_data(net_path: str, trips_path: str) -> GraphData:
    num_nodes, edges = parse_net_tntp(net_path)
    od_demand = parse_trips_tntp(trips_path)
    return GraphData(num_nodes=num_nodes, edges=edges, od_demand=od_demand)


if __name__ == "__main__":
    root = Path("./data/SiouxFalls")
    graph = load_graph_data(str(root / "SiouxFalls_net.tntp"), str(root / "SiouxFalls_trips.tntp"))
    print(graph.num_nodes, len(graph.edges), len(graph.od_demand))
