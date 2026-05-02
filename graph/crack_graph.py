from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np


NEIGHBORS_8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _valid(y: int, x: int, h: int, w: int) -> bool:
    return 0 <= y < h and 0 <= x < w


def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    h, w = skeleton.shape
    graph = nx.Graph()
    points = np.argwhere(skeleton > 0)
    for y, x in points:
        graph.add_node((int(y), int(x)))
    for y, x in points:
        for dy, dx in NEIGHBORS_8:
            ny, nx_ = int(y + dy), int(x + dx)
            if _valid(ny, nx_, h, w) and skeleton[ny, nx_] > 0:
                dist = float(np.hypot(dy, dx))
                graph.add_edge((int(y), int(x)), (ny, nx_), weight=dist)
    return graph


def keypoints_from_graph(graph: nx.Graph) -> Dict[str, int]:
    degrees = dict(graph.degree())
    endpoints = sum(1 for _, d in degrees.items() if d == 1)
    junctions = sum(1 for _, d in degrees.items() if d >= 3)
    return {"endpoints": endpoints, "junctions": junctions}


def graph_longest_path_length(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    max_path = 0.0
    for component_nodes in nx.connected_components(graph):
        sub = graph.subgraph(component_nodes).copy()
        for source in sub.nodes():
            lengths = nx.single_source_dijkstra_path_length(sub, source, weight="weight")
            if lengths:
                max_path = max(max_path, max(lengths.values()))
    return float(max_path)


def graph_diameter_safe(graph: nx.Graph) -> float:
    if graph.number_of_nodes() <= 1:
        return 0.0
    diameters = []
    for component_nodes in nx.connected_components(graph):
        sub = graph.subgraph(component_nodes).copy()
        if sub.number_of_nodes() > 1:
            diameters.append(float(nx.diameter(sub)))
    return max(diameters) if diameters else 0.0
