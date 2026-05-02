from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np

from graph.crack_graph import graph_diameter_safe, graph_longest_path_length, keypoints_from_graph


def count_branches_from_graph(graph: nx.Graph) -> int:
    """Count topological branches between keypoints (endpoints and junctions)."""
    if graph.number_of_edges() == 0:
        return 0

    keypoints = keypoints_from_graph(graph)
    endpoints = set()
    junctions = set()

    for node, degree in graph.degree():
        if degree == 1:
            endpoints.add(node)
        elif degree >= 3:
            junctions.add(node)

    # If no keypoints found, entire graph is one branch
    if not endpoints and not junctions:
        return 1 if graph.number_of_edges() > 0 else 0

    # Count unique paths between keypoints
    visited_edges = set()
    branch_count = 0

    def traverse_from_keypoint(start):
        nonlocal branch_count, visited_edges
        for neighbor in graph.neighbors(start):
            edge = tuple(sorted([start, neighbor]))
            if edge in visited_edges:
                continue
            # Follow path until next keypoint
            visited_edges.add(edge)
            current = neighbor
            prev = start
            while current not in endpoints and current not in junctions:
                next_nodes = [n for n in graph.neighbors(current) if n != prev]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                edge = tuple(sorted([current, next_node]))
                visited_edges.add(edge)
                prev, current = current, next_node
            branch_count += 1

    for ep in endpoints:
        traverse_from_keypoint(ep)

    for junc in junctions:
        # Only traverse to unvisited neighbors
        for neighbor in graph.neighbors(junc):
            edge = tuple(sorted([junc, neighbor]))
            if edge not in visited_edges:
                visited_edges.add(edge)
                current = neighbor
                prev = junc
                # Follow path
                while current not in endpoints and current not in junctions:
                    next_nodes = [n for n in graph.neighbors(current) if n != prev]
                    if not next_nodes:
                        break
                    next_node = next_nodes[0]
                    edge = tuple(sorted([current, next_node]))
                    visited_edges.add(edge)
                    prev, current = current, next_node
                branch_count += 1

    return branch_count


def extract_structural_features(graph: nx.Graph) -> Dict[str, float]:
    edge_lengths = [d.get("weight", 1.0) for _, _, d in graph.edges(data=True)]
    total_length = float(np.sum(edge_lengths)) if edge_lengths else 0.0
    num_branches = float(count_branches_from_graph(graph))
    longest_path = graph_longest_path_length(graph)
    diameter = graph_diameter_safe(graph)
    degree_hist = nx.degree_histogram(graph) if graph.number_of_nodes() > 0 else []
    keypoints = keypoints_from_graph(graph)
    return {
        "total_crack_length": total_length,
        "num_branches": num_branches,
        "longest_path": float(longest_path),
        "graph_diameter": float(diameter),
        "mean_node_degree": float(np.mean([d for _, d in graph.degree()])) if graph.number_of_nodes() else 0.0,
        "node_degree_distribution": degree_hist,
        "endpoints": float(keypoints["endpoints"]),
        "junctions": float(keypoints["junctions"]),
    }
