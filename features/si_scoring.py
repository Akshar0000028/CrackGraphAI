"""
Structural Integrity (SI) Scoring System
=========================================
SI = 1.0 - Damage   (1.0 = perfect, 0.0 = failure imminent)

Damage is driven by four observable crack properties:
  1. crack_density   – skeleton pixels / image area
  2. network_density – junctions per unit crack length (branching)
  3. complexity      – branch + junction count
  4. width           – average crack thickness

Risk levels
-----------
>= 0.85  Low
0.70-0.85  Moderate
0.50-0.70  High
0.30-0.50  Critical
< 0.30   Failure Imminent
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────────
# Risk thresholds
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SIRiskThresholds:
    low: float = 0.85
    moderate: float = 0.70
    high: float = 0.50
    critical: float = 0.30

    def classify(self, si_score: float) -> Tuple[str, str, str]:
        if si_score >= self.low:
            return ("Low", "severity-low",
                    "Structure is in good condition. Routine monitoring recommended.")
        elif si_score >= self.moderate:
            return ("Moderate", "severity-moderate",
                    "Minor structural concerns. Schedule inspection within 6 months.")
        elif si_score >= self.high:
            return ("High", "severity-high",
                    "Significant structural concerns. Professional assessment required within 1 month.")
        elif si_score >= self.critical:
            return ("Critical", "severity-critical",
                    "Severe structural damage. Immediate intervention advised.")
        else:
            return ("Failure Imminent", "severity-failure",
                    "Structural failure risk. Immediate evacuation and emergency repairs required.")


# ─────────────────────────────────────────────────────────────────────────────
# Weights  (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SIWeights:
    crack_density: float = 0.35
    network_density: float = 0.25
    complexity: float = 0.25
    width: float = 0.15
    segmentation_quality: float = 0.0   # only used when real GT dice is available

    def validate(self) -> None:
        total = (self.crack_density + self.network_density +
                 self.complexity + self.width + self.segmentation_quality)
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"SIWeights must sum to 1.0, got {total:.4f}")


# Weights when a real GT-based Dice score is available
_WEIGHTS_WITH_SEG = SIWeights(
    crack_density=0.28,
    network_density=0.20,
    complexity=0.20,
    width=0.12,
    segmentation_quality=0.20,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DamageFeatures:
    # raw ratios [0, 1]
    crack_density: float           # mask pixels / total pixels
    skeleton_density: float        # skeleton pixels / total pixels  ← raw, NOT pre-normalised

    # graph topology (raw counts)
    connectivity_ratio: float      # informational only – not used for damage
    num_branches: int
    num_junctions: int
    num_endpoints: int

    # pre-normalised [0, 1]
    complexity_index: float
    network_density_index: float   # junctions per unit crack length, normalised

    # geometry
    total_crack_length: float      # pixels
    max_crack_width_proxy: float   # normalised [0, 1]
    mean_crack_width_proxy: float  # normalised [0, 1]

    # segmentation quality (neutral defaults at inference time)
    dice_score: float = 1.0
    bce_loss: float = 0.0

    # metadata
    image_area_pixels: int = 0
    mask_area_pixels: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Normaliser
# ─────────────────────────────────────────────────────────────────────────────

class FeatureNormalizer:
    """
    Maps raw feature values → [0, 1] damage scores.

    Calibration thresholds (256×256 = 65 536 px):
      MAX_SKELETON_DENSITY = 0.12  →  ~7 864 skeleton px = very severe
      MAX_NETWORK_DENSITY  = 0.015 →  1.5 junctions per 100 px = severe branching
      MAX_COMPLEXITY_SCORE = 30.0  →  branches + 2×junctions
      MAX_WIDTH_PROXY      = 0.25  →  normalised by image diagonal
    """

    MAX_SKELETON_DENSITY: float = 0.03   # 3% skeleton coverage = severe (1966 px in 256x256)
    MAX_NETWORK_DENSITY: float = 0.015   # 1.5 junctions per 100 px = severe branching
    MAX_COMPLEXITY_SCORE: float = 30.0   # branches + 2*junctions
    MAX_WIDTH_PROXY: float = 0.25        # normalised by image diagonal

    @classmethod
    def normalize_density(cls, skeleton_pixels: int, total_pixels: int) -> float:
        if total_pixels == 0:
            return 0.0
        raw = skeleton_pixels / total_pixels
        return float(np.clip(raw / cls.MAX_SKELETON_DENSITY, 0.0, 1.0))

    @classmethod
    def normalize_network_density(cls, num_junctions: int, total_crack_length: float) -> float:
        """Junctions per 100 skeleton pixels → damage [0, 1].
        A straight crack has 0 junctions → 0 damage.
        A heavily branched network → high damage.
        """
        if total_crack_length <= 0:
            return 0.0
        raw = (num_junctions / total_crack_length) * 100.0
        return float(np.clip(raw / (cls.MAX_NETWORK_DENSITY * 100.0), 0.0, 1.0))

    @classmethod
    def normalize_complexity(cls, num_branches: int, num_junctions: int, num_endpoints: int) -> float:
        base = float(num_branches) + 2.0 * float(num_junctions)
        if num_junctions > 0:
            activity = num_endpoints / (2.0 * num_junctions)
            bonus = 1.0 + 0.5 * float(np.tanh(max(0.0, activity - 1.0)))
            base *= bonus
        return float(np.clip(base / cls.MAX_COMPLEXITY_SCORE, 0.0, 1.0))

    @classmethod
    def normalize_crack_width(cls, mask: np.ndarray, skeleton: np.ndarray) -> Tuple[float, float]:
        if mask.sum() == 0 or skeleton.sum() == 0:
            return 0.0, 0.0
        distance = ndimage.distance_transform_edt(mask.astype(bool))
        skel_distances = distance[skeleton > 0]
        if len(skel_distances) == 0:
            return 0.0, 0.0
        diagonal = float(np.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2))
        max_w = 2.0 * float(np.max(skel_distances)) / diagonal
        mean_w = 2.0 * float(np.mean(skel_distances)) / diagonal
        return (
            float(np.clip(max_w / cls.MAX_WIDTH_PROXY, 0.0, 1.0)),
            float(np.clip(mean_w / cls.MAX_WIDTH_PROXY, 0.0, 1.0)),
        )

    @classmethod
    def normalize_segmentation_quality(cls, dice: float, bce: float) -> float:
        if bce <= 0.0:
            bce_quality = 1.0
        else:
            bce_bounded = 2.0 * bce / (bce + 1.0)
            bce_quality = float(np.exp(-bce_bounded))
        quality = 0.7 * float(np.clip(dice, 0.0, 1.0)) + 0.3 * bce_quality
        return float(np.clip(quality, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# SI Generator
# ─────────────────────────────────────────────────────────────────────────────

class SIGenerator:

    def __init__(self, weights: Optional[SIWeights] = None,
                 thresholds: Optional[SIRiskThresholds] = None):
        self.weights = weights or SIWeights()
        self.weights.validate()
        self.thresholds = thresholds or SIRiskThresholds()
        self.normalizer = FeatureNormalizer()

    # ── core scoring ──────────────────────────────────────────────────────

    def compute_si_score(self, features: DamageFeatures) -> Dict[str, float]:
        """
        SI = 1.0 - (w_density*D + w_network*N + w_complexity*C + w_width*W)

        All four components are normalised to [0, 1] before weighting.
        skeleton_density is stored as a raw ratio and normalised here.
        """
        w = self.weights

        # 1. Density: raw skeleton_density → normalised damage
        density_damage = self.normalizer.normalize_density(
            int(features.skeleton_density * features.image_area_pixels),
            features.image_area_pixels,
        )

        # 2. Network density: junctions per unit length
        network_damage = self.normalizer.normalize_network_density(
            features.num_junctions,
            features.total_crack_length,
        )

        # 3. Complexity: already normalised in compute_from_raw
        complexity_damage = float(np.clip(features.complexity_index, 0.0, 1.0))

        # 4. Width: already normalised in compute_from_raw
        width_damage = float(np.clip(features.mean_crack_width_proxy, 0.0, 1.0))

        # 5. Segmentation quality (0 contribution when dice_score == 1.0)
        seg_quality = self.normalizer.normalize_segmentation_quality(
            features.dice_score, features.bce_loss
        )
        seg_damage = (1.0 - seg_quality) * w.segmentation_quality

        total_damage = (
            w.crack_density    * density_damage
            + w.network_density  * network_damage
            + w.complexity       * complexity_damage
            + w.width            * width_damage
            + seg_damage
        )

        si_score = float(np.clip(1.0 - total_damage, 0.0, 1.0))

        return {
            "si_score": si_score,
            "total_damage": float(total_damage),
            "density_damage": float(density_damage),
            "network_damage": float(network_damage),
            "complexity_damage": float(complexity_damage),
            "width_damage": float(width_damage),
            "segmentation_quality": float(seg_quality),
            "risk_level": self.thresholds.classify(si_score)[0],
        }

    # ── from raw arrays ───────────────────────────────────────────────────

    def compute_from_raw(
        self,
        mask: np.ndarray,
        skeleton: np.ndarray,
        graph: "nx.Graph",
        connectivity_ratio: float,
        dice: float = 1.0,
        bce: float = 0.0,
    ) -> Dict[str, float]:
        total_pixels = mask.shape[0] * mask.shape[1]
        skeleton_pixels = int(skeleton.sum())

        num_branches = self._count_branches(graph)
        degrees = dict(graph.degree()) if graph.number_of_nodes() > 0 else {}
        num_junctions = sum(1 for d in degrees.values() if d >= 3)
        num_endpoints = sum(1 for d in degrees.values() if d == 1)

        edge_lengths = [d.get("weight", 1.0) for _, _, d in graph.edges(data=True)]
        total_length = float(sum(edge_lengths))

        # normalise features
        density_norm = skeleton_pixels / max(total_pixels, 1)   # raw ratio [0,1]
        complexity_norm = self.normalizer.normalize_complexity(
            num_branches, num_junctions, num_endpoints
        )
        network_density_norm = self.normalizer.normalize_network_density(
            num_junctions, total_length
        )
        max_width, mean_width = self.normalizer.normalize_crack_width(mask, skeleton)

        features = DamageFeatures(
            crack_density=float(mask.sum()) / max(total_pixels, 1),
            skeleton_density=density_norm,          # raw ratio stored here
            connectivity_ratio=float(connectivity_ratio),
            num_branches=num_branches,
            num_junctions=num_junctions,
            num_endpoints=num_endpoints,
            complexity_index=complexity_norm,
            network_density_index=network_density_norm,
            total_crack_length=total_length,
            max_crack_width_proxy=max_width,
            mean_crack_width_proxy=mean_width,
            dice_score=float(dice),
            bce_loss=float(bce),
            image_area_pixels=total_pixels,
            mask_area_pixels=int(mask.sum()),
        )

        result = self.compute_si_score(features)
        result["raw_features"] = {
            "skeleton_pixels": skeleton_pixels,
            "total_pixels": total_pixels,
            "num_branches": num_branches,
            "num_junctions": num_junctions,
            "num_endpoints": num_endpoints,
            "total_crack_length": total_length,
            "max_width_proxy": max_width,
            "mean_width_proxy": mean_width,
            "connectivity_ratio": float(connectivity_ratio),
        }
        return result

    # ── branch counting ───────────────────────────────────────────────────

    def _count_branches(self, graph: "nx.Graph") -> int:
        if graph.number_of_edges() == 0:
            return 0
        keypoints: set = set()
        for node, degree in graph.degree():
            if degree == 1 or degree >= 3:
                keypoints.add(node)
        if not keypoints:
            return 1 if graph.number_of_edges() > 0 else 0

        visited_edges: set = set()
        branch_count = 0

        def _traverse(start: tuple) -> None:
            nonlocal branch_count
            for neighbor in graph.neighbors(start):
                edge = tuple(sorted([start, neighbor]))
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)
                current, prev = neighbor, start
                while current not in keypoints:
                    nxt = [n for n in graph.neighbors(current) if n != prev]
                    if not nxt:
                        break
                    nxt_node = nxt[0]
                    e = tuple(sorted([current, nxt_node]))
                    visited_edges.add(e)
                    prev, current = current, nxt_node
                branch_count += 1

        for kp in keypoints:
            _traverse(kp)
        return branch_count


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_structural_integrity(
    mask: np.ndarray,
    skeleton: np.ndarray,
    graph: "nx.Graph",
    connectivity_ratio: float,
    dice: float = 1.0,
    bce: float = 0.0,
    weights: Optional[SIWeights] = None,
) -> Dict[str, float]:
    """
    One-call SI computation for the inference pipeline.

    At inference time (no GT): leave dice=1.0, bce=0.0.
    Score is driven purely by observable crack properties.

    With GT available: pass real dice + bce values.
    Automatically switches to weights that include segmentation quality.
    """
    if dice < 1.0 and weights is None:
        weights = _WEIGHTS_WITH_SEG
    generator = SIGenerator(weights=weights)
    return generator.compute_from_raw(mask, skeleton, graph, connectivity_ratio, dice, bce)


def classify_risk(si_score: float) -> Tuple[str, str, str]:
    """Return (level_name, css_class, description) for a given SI score."""
    return SIRiskThresholds().classify(si_score)
