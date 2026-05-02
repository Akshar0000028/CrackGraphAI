"""
Enhanced Crack Graph Extraction with Noise Removal and Pruning

This module provides robust graph extraction from crack skeletons with:
- Spur pruning (removes small branch noise)
- Isolated component filtering
- Edge weight smoothing
- Graph simplification
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean


NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _valid(y: int, x: int, h: int, w: int) -> bool:
    """Check if coordinates are within image bounds."""
    return 0 <= y < h and 0 <= x < w


def skeleton_to_graph_basic(skeleton: np.ndarray) -> nx.Graph:
    """
    Basic skeleton to graph conversion (8-connected).
    
    Args:
        skeleton: Binary skeleton image
        
    Returns:
        NetworkX graph with pixel coordinates as nodes
    """
    h, w = skeleton.shape
    graph = nx.Graph()
    
    # Add all skeleton pixels as nodes
    points = np.argwhere(skeleton > 0)
    for y, x in points:
        graph.add_node((int(y), int(x)))
    
    # Add edges for 8-connected neighbors
    for y, x in points:
        for dy, dx in NEIGHBORS_8:
            ny, nx_ = int(y + dy), int(x + dx)
            if _valid(ny, nx_, h, w) and skeleton[ny, nx_] > 0:
                dist = float(np.hypot(dy, dx))
                graph.add_edge((int(y), int(x)), (ny, nx_), weight=dist)
    
    return graph


def prune_spurs(
    graph: nx.Graph,
    min_branch_length: float = 3.0
) -> nx.Graph:
    """
    Remove short spurs (branches) from the graph.
    
    Spurs are branches that:
    - Start at an endpoint (degree 1)
    - End at a junction (degree >= 3) or another endpoint
    - Have total edge weight less than min_branch_length
    
    Args:
        graph: Input graph
        min_branch_length: Minimum branch length to preserve (in pixels)
        
    Returns:
        Pruned graph (new graph, original unchanged)
    """
    if graph.number_of_nodes() == 0:
        return graph.copy()
    
    pruned = graph.copy()
    
    # Identify keypoints
    endpoints = [n for n, d in pruned.degree() if d == 1]
    junctions = [n for n, d in pruned.degree() if d >= 3]
    keypoints = set(endpoints + junctions)
    
    if not keypoints:
        # Simple path or cycle - check total length
        if pruned.number_of_edges() > 0:
            total_weight = sum(d.get("weight", 1.0) for _, _, d in pruned.edges(data=True))
            if total_weight < min_branch_length:
                return nx.Graph()
        return pruned
    
    # Find short branches
    branches_to_remove = []
    visited_edges = set()
    
    for ep in endpoints:
        for neighbor in pruned.neighbors(ep):
            edge = tuple(sorted([ep, neighbor]))
            if edge in visited_edges:
                continue
            
            # Trace branch from endpoint
            branch_edges = []
            branch_weight = 0.0
            current = neighbor
            prev = ep
            
            while current not in keypoints:
                edge = tuple(sorted([current, prev]))
                visited_edges.add(edge)
                branch_edges.append((prev, current))
                
                weight = pruned[prev][current].get("weight", 1.0)
                branch_weight += weight
                
                # Get next node
                neighbors = [n for n in pruned.neighbors(current) if n != prev]
                if not neighbors:
                    break
                prev, current = current, neighbors[0]
            
            # Add final edge to keypoint
            if current in keypoints:
                edge = tuple(sorted([current, prev]))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    branch_edges.append((prev, current))
                    weight = pruned[prev][current].get("weight", 1.0)
                    branch_weight += weight
            
            # Mark for removal if too short
            if branch_weight < min_branch_length and len(branch_edges) > 0:
                branches_to_remove.extend(branch_edges)
    
    # Remove marked edges
    for u, v in branches_to_remove:
        if pruned.has_edge(u, v):
            pruned.remove_edge(u, v)
    
    # Remove isolated nodes
    isolated = [n for n in pruned.nodes() if pruned.degree(n) == 0]
    pruned.remove_nodes_from(isolated)
    
    return pruned


def filter_small_components(
    graph: nx.Graph,
    min_nodes: int = 3,
    min_total_weight: float = 5.0
) -> nx.Graph:
    """
    Remove small isolated components from the graph.
    
    Args:
        graph: Input graph
        min_nodes: Minimum nodes for a component to be kept
        min_total_weight: Minimum total edge weight for a component
        
    Returns:
        Filtered graph
    """
    if graph.number_of_nodes() == 0:
        return graph.copy()
    
    filtered = nx.Graph()
    
    for component in nx.connected_components(graph):
        if len(component) < min_nodes:
            continue
        
        sub = graph.subgraph(component).copy()
        total_weight = sum(
            d.get("weight", 1.0) for _, _, d in sub.edges(data=True)
        )
        
        if total_weight >= min_total_weight:
            filtered = nx.compose(filtered, sub)
    
    return filtered


def smooth_edges(
    graph: nx.Graph,
    angle_threshold: float = 160.0
) -> nx.Graph:
    """
    Smooth graph by removing zigzag artifacts.
    
    If three consecutive nodes form an angle > angle_threshold,
    consider merging the middle node if it doesn't change topology.
    
    Args:
        graph: Input graph
        angle_threshold: Minimum angle (degrees) for straightening
        
    Returns:
        Smoothed graph
    """
    if graph.number_of_nodes() == 0:
        return graph.copy()
    
    smoothed = graph.copy()
    
    # Find degree-2 nodes (path nodes) that can be smoothed
    degree_2_nodes = [n for n, d in smoothed.degree() if d == 2]
    
    for node in degree_2_nodes:
        if node not in smoothed or smoothed.degree(node) != 2:
            continue
        
        neighbors = list(smoothed.neighbors(node))
        if len(neighbors) != 2:
            continue
        
        n1, n2 = neighbors
        
        # Calculate angle at node
        v1 = np.array([n1[0] - node[0], n1[1] - node[1]])
        v2 = np.array([n2[0] - node[0], n2[1] - node[1]])
        
        # Normalize
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            continue
        
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        # Calculate angle (cosine)
        cos_angle = np.dot(v1, v2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        # If angle is close to 180, the path is nearly straight
        if angle > angle_threshold:
            # Get edge weights
            w1 = smoothed[node][n1].get("weight", 1.0)
            w2 = smoothed[node][n2].get("weight", 1.0)
            
            # Remove middle node and connect neighbors directly
            smoothed.remove_node(node)
            new_weight = w1 + w2
            smoothed.add_edge(n1, n2, weight=new_weight)
    
    return smoothed


def enhance_graph(
    skeleton: np.ndarray,
    prune_length: float = 3.0,
    min_component_nodes: int = 3,
    min_component_weight: float = 5.0,
    smooth_angle: float = 160.0
) -> Tuple[nx.Graph, Dict[str, any]]:
    """
    Extract and enhance crack graph from skeleton with noise removal.
    
    Pipeline:
    1. Basic skeleton to graph conversion
    2. Prune short spurs
    3. Filter small isolated components
    4. Smooth zigzag artifacts
    
    Args:
        skeleton: Binary skeleton image
        prune_length: Minimum branch length to preserve
        min_component_nodes: Min nodes for component to survive
        min_component_weight: Min total edge weight for component
        smooth_angle: Angle threshold for edge smoothing
        
    Returns:
        Tuple of (enhanced_graph, metadata_dict)
    """
    metadata = {
        "original_skeleton_pixels": int(skeleton.sum()),
        "prune_length": prune_length,
        "min_component_nodes": min_component_nodes,
        "min_component_weight": min_component_weight,
    }
    
    # Step 1: Basic conversion
    graph = skeleton_to_graph_basic(skeleton)
    metadata["initial_nodes"] = graph.number_of_nodes()
    metadata["initial_edges"] = graph.number_of_edges()
    
    # Step 2: Prune spurs
    if prune_length > 0:
        graph = prune_spurs(graph, min_branch_length=prune_length)
        metadata["after_pruning_nodes"] = graph.number_of_nodes()
        metadata["after_pruning_edges"] = graph.number_of_edges()
    
    # Step 3: Filter small components
    if min_component_nodes > 0 or min_component_weight > 0:
        graph = filter_small_components(
            graph,
            min_nodes=min_component_nodes,
            min_total_weight=min_component_weight
        )
        metadata["after_filtering_nodes"] = graph.number_of_nodes()
        metadata["after_filtering_edges"] = graph.number_of_edges()
    
    # Step 4: Smooth edges
    if smooth_angle > 0:
        graph = smooth_edges(graph, angle_threshold=smooth_angle)
        metadata["final_nodes"] = graph.number_of_nodes()
        metadata["final_edges"] = graph.number_of_edges()
    
    return graph, metadata


def keypoints_from_graph(graph: nx.Graph) -> Dict[str, int]:
    """
    Extract keypoint counts from graph.
    
    Returns:
        Dict with 'endpoints', 'junctions', 'isolated'
    """
    endpoints = 0
    junctions = 0
    isolated = 0
    
    for node, degree in graph.degree():
        if degree == 0:
            isolated += 1
        elif degree == 1:
            endpoints += 1
        elif degree >= 3:
            junctions += 1
    
    return {
        "endpoints": endpoints,
        "junctions": junctions,
        "isolated": isolated,
    }


def graph_statistics(graph: nx.Graph) -> Dict[str, float]:
    """
    Compute comprehensive graph statistics.
    
    Returns:
        Dict with various graph metrics
    """
    if graph.number_of_nodes() == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "num_components": 0,
            "avg_degree": 0.0,
            "max_degree": 0,
            "total_edge_weight": 0.0,
            "avg_edge_weight": 0.0,
            "network_density": 0.0,
            "avg_clustering": 0.0,
        }
    
    degrees = [d for _, d in graph.degree()]
    edge_weights = [d.get("weight", 1.0) for _, _, d in graph.edges(data=True)]
    
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    
    # Network density (for undirected graph: 2m / (n(n-1)))
    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0
    
    # Clustering coefficient (may fail for path graphs)
    try:
        clustering = nx.average_clustering(graph)
    except:
        clustering = 0.0
    
    return {
        "num_nodes": n,
        "num_edges": m,
        "num_components": nx.number_connected_components(graph),
        "avg_degree": np.mean(degrees) if degrees else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "total_edge_weight": sum(edge_weights),
        "avg_edge_weight": np.mean(edge_weights) if edge_weights else 0.0,
        "network_density": density,
        "avg_clustering": clustering,
    }


def graph_longest_path_length(graph: nx.Graph) -> float:
    """
    Compute longest path length in graph (handling disconnected components).
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Longest weighted path length
    """
    if graph.number_of_nodes() == 0:
        return 0.0
    
    max_path = 0.0
    
    for component_nodes in nx.connected_components(graph):
        if len(component_nodes) <= 1:
            continue
        
        sub = graph.subgraph(component_nodes).copy()
        
        # For each node, compute shortest paths to all others
        for source in sub.nodes():
            try:
                lengths = nx.single_source_dijkstra_path_length(sub, source, weight="weight")
                if lengths:
                    max_path = max(max_path, max(lengths.values()))
            except:
                continue
    
    return float(max_path)


def graph_diameter_safe(graph: nx.Graph) -> float:
    """
    Compute graph diameter safely (handling disconnected components).
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Maximum diameter across all components
    """
    if graph.number_of_nodes() <= 1:
        return 0.0
    
    diameters = []
    
    for component_nodes in nx.connected_components(graph):
        sub = graph.subgraph(component_nodes).copy()
        if sub.number_of_nodes() > 1:
            try:
                diameters.append(float(nx.diameter(sub)))
            except:
                continue
    
    return max(diameters) if diameters else 0.0


# Backward compatibility - use enhanced version by default
def skeleton_to_graph(
    skeleton: np.ndarray,
    prune: bool = True,
    filter_components: bool = True,
    smooth: bool = True
) -> nx.Graph:
    """
    Convert skeleton to graph with optional enhancement.
    
    Args:
        skeleton: Binary skeleton image
        prune: Whether to prune short spurs
        filter_components: Whether to filter small components
        smooth: Whether to smooth zigzag artifacts
        
    Returns:
        NetworkX graph
    """
    if not any([prune, filter_components, smooth]):
        return skeleton_to_graph_basic(skeleton)
    
    graph, _ = enhance_graph(
        skeleton,
        prune_length=3.0 if prune else 0.0,
        min_component_nodes=3 if filter_components else 0,
        min_component_weight=5.0 if filter_components else 0.0,
        smooth_angle=160.0 if smooth else 0.0
    )
    
    return graph
