from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
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


def get_keypoint_coords(graph: nx.Graph) -> Dict[str, List[Tuple[int, int]]]:
    """Return the (y, x) coordinates of endpoints and junctions in the graph.

    Endpoints  – nodes with degree == 1 (crack tips / terminations).
    Junctions  – nodes with degree >= 3 (branching / intersection points).

    Returns
    -------
    dict with keys ``"endpoints"`` and ``"junctions"``, each a list of
    ``(y, x)`` integer tuples.
    """
    endpoints: List[Tuple[int, int]] = []
    junctions: List[Tuple[int, int]] = []
    for node, degree in graph.degree():
        if degree == 1:
            endpoints.append(node)
        elif degree >= 3:
            junctions.append(node)
    return {"endpoints": endpoints, "junctions": junctions}


def draw_keypoints_overlay(
    image: np.ndarray,
    graph: nx.Graph,
    endpoint_color: Tuple[int, int, int] = (0, 0, 255),    # Red   (BGR)
    junction_color: Tuple[int, int, int] = (0, 255, 255),  # Yellow (BGR)
    endpoint_radius: int = 5,
    junction_radius: int = 6,
    thickness: int = -1,  # filled circles
    alpha: float = 1.0,
) -> np.ndarray:
    """Draw endpoint and junction markers on top of a skeleton or any image.

    When a binary/grayscale skeleton array is passed the function converts it
    to a 3-channel BGR image first so the white skeleton lines remain visible
    against the black background and the coloured markers stand out clearly.

    Endpoints are drawn as **red** filled circles.
    Junctions are drawn as **yellow** filled circles.

    Graph node coordinates are ``(y, x)`` integers in the same pixel space as
    *image* — no rescaling is performed here.

    Parameters
    ----------
    image          : Input array.  Can be:
                       • uint8 binary/grayscale skeleton (H × W) or (H × W × 1)
                       • float [0,1] skeleton (H × W)
                       • BGR uint8 colour image (H × W × 3)
    graph          : NetworkX graph built from the skeleton.
    endpoint_color : BGR colour for endpoint markers.
    junction_color : BGR colour for junction markers.
    endpoint_radius: Circle radius in pixels for endpoints.
    junction_radius: Circle radius in pixels for junctions.
    thickness      : cv2 circle thickness (-1 = filled).
    alpha          : Opacity of the marker layer blended over the base image.
                     1.0 = fully opaque markers (recommended for dark skeleton
                     backgrounds so markers are always visible).

    Returns
    -------
    BGR uint8 image with markers drawn on it.
    """
    # ── Normalise input to BGR uint8 ──────────────────────────────────────
    if image.ndim == 2:
        if image.dtype != np.uint8:
            # Float array [0.0, 1.0] → scale to [0, 255]
            img_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            # Binary uint8 with values 0/1 (from skimage skeletonize) → scale to 0/255
            # so skeleton lines are white (255) not near-black (1)
            img_u8 = (image * 255) if image.max() <= 1 else image
            img_u8 = img_u8.astype(np.uint8)
        canvas = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        ch = image[:, :, 0]
        if ch.dtype != np.uint8:
            img_u8 = (np.clip(ch, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            img_u8 = (ch * 255).astype(np.uint8) if ch.max() <= 1 else ch
        canvas = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    else:
        # Already colour
        if image.dtype != np.uint8:
            canvas = (np.clip(image, 0, 255)).astype(np.uint8)
        else:
            canvas = image.copy()

    if graph.number_of_nodes() == 0:
        return canvas

    coords = get_keypoint_coords(graph)

    if alpha >= 1.0:
        # Draw directly — no blending needed, markers are always fully visible
        result = canvas.copy()
        for y, x in coords["endpoints"]:
            cv2.circle(result, (int(x), int(y)), endpoint_radius, endpoint_color, thickness)
        for y, x in coords["junctions"]:
            cv2.circle(result, (int(x), int(y)), junction_radius, junction_color, thickness)
    else:
        # Alpha-blend marker layer over base
        overlay = canvas.copy()
        for y, x in coords["endpoints"]:
            cv2.circle(overlay, (int(x), int(y)), endpoint_radius, endpoint_color, thickness)
        for y, x in coords["junctions"]:
            cv2.circle(overlay, (int(x), int(y)), junction_radius, junction_color, thickness)
        result = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)

    return result


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
