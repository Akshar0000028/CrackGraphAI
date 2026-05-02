"""
Unit tests for the Structural Integrity (SI) scoring system.

Tests are written against the *actual* public API of features/si_scoring.py.
Run with: pytest tests/test_si_scoring.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import numpy as np
import pytest

from features.si_scoring import (
    DamageFeatures,
    FeatureNormalizer,
    SIGenerator,
    SIRiskThresholds,
    SIWeights,
    classify_risk,
    compute_structural_integrity,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_features(**overrides) -> DamageFeatures:
    """Return a DamageFeatures with safe zero-damage defaults, optionally overridden."""
    defaults = dict(
        crack_density=0.0,
        skeleton_density=0.0,
        connectivity_ratio=0.0,
        num_branches=0,
        num_junctions=0,
        num_endpoints=0,
        complexity_index=0.0,
        network_density_index=0.0,
        total_crack_length=0.0,
        max_crack_width_proxy=0.0,
        mean_crack_width_proxy=0.0,
        dice_score=1.0,
        bce_loss=0.0,
        image_area_pixels=65536,  # 256×256
        mask_area_pixels=0,
    )
    defaults.update(overrides)
    return DamageFeatures(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Risk thresholds  (code uses 0.85 / 0.70 / 0.50 / 0.30)
# ─────────────────────────────────────────────────────────────────────────────

class TestSIRiskThresholds:
    """Test risk classification thresholds."""

    def test_low_risk(self):
        level, css, desc = classify_risk(0.95)
        assert level == "Low"
        assert "good condition" in desc.lower()

    def test_moderate_risk(self):
        level, css, desc = classify_risk(0.75)
        assert level == "Moderate"
        assert "minor" in desc.lower()

    def test_high_risk(self):
        level, css, desc = classify_risk(0.55)
        assert level == "High"
        assert "significant" in desc.lower()

    def test_critical_risk(self):
        level, css, desc = classify_risk(0.35)
        assert level == "Critical"
        assert "severe" in desc.lower() or "immediate" in desc.lower()

    def test_failure_imminent(self):
        level, css, desc = classify_risk(0.15)
        assert level == "Failure Imminent"
        assert "emergency" in desc.lower() or "evacuation" in desc.lower()

    def test_boundary_values(self):
        """Exact threshold boundaries match the code constants."""
        thresholds = SIRiskThresholds()
        assert classify_risk(thresholds.low)[0] == "Low"
        assert classify_risk(thresholds.moderate)[0] == "Moderate"
        assert classify_risk(thresholds.high)[0] == "High"
        assert classify_risk(thresholds.critical)[0] == "Critical"
        assert classify_risk(0.0)[0] == "Failure Imminent"

    def test_css_classes_present(self):
        for score in [0.95, 0.75, 0.55, 0.35, 0.15]:
            _, css, _ = classify_risk(score)
            assert css.startswith("severity-")


# ─────────────────────────────────────────────────────────────────────────────
# FeatureNormalizer
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureNormalizer:
    """Test feature normalisation class methods."""

    # normalize_density -------------------------------------------------------

    def test_normalize_density_zero(self):
        assert FeatureNormalizer.normalize_density(0, 65536) == 0.0

    def test_normalize_density_zero_total(self):
        assert FeatureNormalizer.normalize_density(100, 0) == 0.0

    def test_normalize_density_clamps_at_one(self):
        # 100% coverage → must clamp to 1.0
        result = FeatureNormalizer.normalize_density(65536, 65536)
        assert result == 1.0

    def test_normalize_density_mid(self):
        # MAX_SKELETON_DENSITY = 0.03 → 3% coverage = 1.0
        # 1.5% coverage should be ~0.5
        pixels = int(0.015 * 65536)
        result = FeatureNormalizer.normalize_density(pixels, 65536)
        assert 0.45 < result < 0.55

    # normalize_network_density -----------------------------------------------

    def test_normalize_network_density_zero_length(self):
        assert FeatureNormalizer.normalize_network_density(5, 0.0) == 0.0

    def test_normalize_network_density_zero_junctions(self):
        assert FeatureNormalizer.normalize_network_density(0, 100.0) == 0.0

    def test_normalize_network_density_clamps(self):
        # Extreme values must clamp to 1.0
        result = FeatureNormalizer.normalize_network_density(10000, 1.0)
        assert result == 1.0

    # normalize_complexity ----------------------------------------------------

    def test_normalize_complexity_zero(self):
        assert FeatureNormalizer.normalize_complexity(0, 0, 0) == 0.0

    def test_normalize_complexity_clamps(self):
        result = FeatureNormalizer.normalize_complexity(1000, 1000, 1000)
        assert result == 1.0

    def test_normalize_complexity_junctions_weighted(self):
        # 2 junctions contribute 2× more than 1 branch
        branches_only = FeatureNormalizer.normalize_complexity(2, 0, 0)
        junctions_only = FeatureNormalizer.normalize_complexity(0, 1, 0)
        # junctions_only base = 2*1 = 2, branches_only base = 2 → equal before bonus
        assert abs(branches_only - junctions_only) < 0.05

    # normalize_crack_width ---------------------------------------------------

    def test_normalize_crack_width_empty_mask(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        skel = np.zeros((64, 64), dtype=np.uint8)
        max_w, mean_w = FeatureNormalizer.normalize_crack_width(mask, skel)
        assert max_w == 0.0
        assert mean_w == 0.0

    def test_normalize_crack_width_nonzero(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[28:36, 10:54] = 1  # 8-pixel wide crack
        skel = np.zeros((64, 64), dtype=np.uint8)
        skel[32, 10:54] = 1
        max_w, mean_w = FeatureNormalizer.normalize_crack_width(mask, skel)
        assert max_w > 0.0
        assert mean_w > 0.0
        assert max_w <= 1.0
        assert mean_w <= 1.0

    # normalize_segmentation_quality ------------------------------------------

    def test_normalize_segmentation_quality_perfect(self):
        quality = FeatureNormalizer.normalize_segmentation_quality(1.0, 0.0)
        assert quality == 1.0

    def test_normalize_segmentation_quality_worst(self):
        quality = FeatureNormalizer.normalize_segmentation_quality(0.0, 100.0)
        assert quality < 0.3

    def test_normalize_segmentation_quality_range(self):
        for dice in [0.0, 0.5, 1.0]:
            for bce in [0.0, 1.0, 10.0]:
                q = FeatureNormalizer.normalize_segmentation_quality(dice, bce)
                assert 0.0 <= q <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# SIWeights
# ─────────────────────────────────────────────────────────────────────────────

class TestSIWeights:
    """Test weight dataclass validation."""

    def test_default_weights_valid(self):
        """Default weights must sum to 1.0 and pass validation."""
        SIWeights().validate()  # must not raise

    def test_custom_valid_weights(self):
        w = SIWeights(
            crack_density=0.35,
            network_density=0.25,
            complexity=0.25,
            width=0.15,
            segmentation_quality=0.0,
        )
        w.validate()

    def test_invalid_weights_raise(self):
        w = SIWeights(
            crack_density=0.5,
            network_density=0.5,
            complexity=0.5,
            width=0.5,
            segmentation_quality=0.0,
        )
        with pytest.raises(ValueError, match="sum to 1.0"):
            w.validate()

    def test_weights_with_seg_valid(self):
        """Weights that include segmentation quality must also sum to 1.0."""
        w = SIWeights(
            crack_density=0.28,
            network_density=0.20,
            complexity=0.20,
            width=0.12,
            segmentation_quality=0.20,
        )
        w.validate()


# ─────────────────────────────────────────────────────────────────────────────
# SIGenerator.compute_si_score
# ─────────────────────────────────────────────────────────────────────────────

class TestSIGenerator:
    """Test the SI score generator."""

    def test_no_cracks_perfect_score(self):
        """Zero damage features → SI = 1.0, risk = Low."""
        features = _make_features()
        result = SIGenerator().compute_si_score(features)
        assert result["si_score"] == 1.0
        assert result["total_damage"] == 0.0
        assert result["risk_level"] == "Low"

    def test_full_damage_zero_score(self):
        """Maximum damage features → SI = 0.0."""
        features = _make_features(
            skeleton_density=1.0,
            num_junctions=100,
            total_crack_length=10.0,
            complexity_index=1.0,
            mean_crack_width_proxy=1.0,
            image_area_pixels=65536,
        )
        result = SIGenerator().compute_si_score(features)
        assert result["si_score"] == 0.0
        assert result["total_damage"] == 1.0

    def test_score_in_range(self):
        """SI score must always be in [0, 1]."""
        for density in [0.0, 0.01, 0.05, 0.5, 1.0]:
            features = _make_features(
                skeleton_density=density,
                image_area_pixels=65536,
            )
            result = SIGenerator().compute_si_score(features)
            assert 0.0 <= result["si_score"] <= 1.0

    def test_more_damage_lower_score(self):
        """Higher skeleton density must produce a lower SI score."""
        low_dmg = SIGenerator().compute_si_score(
            _make_features(skeleton_density=0.001, image_area_pixels=65536)
        )
        high_dmg = SIGenerator().compute_si_score(
            _make_features(skeleton_density=0.05, image_area_pixels=65536)
        )
        assert high_dmg["si_score"] < low_dmg["si_score"]

    def test_more_junctions_lower_score(self):
        """More junctions (higher network damage) → lower SI score."""
        few = SIGenerator().compute_si_score(
            _make_features(num_junctions=1, total_crack_length=100.0)
        )
        many = SIGenerator().compute_si_score(
            _make_features(num_junctions=20, total_crack_length=100.0)
        )
        assert many["si_score"] < few["si_score"]

    def test_result_keys_present(self):
        """Result dict must contain all expected keys."""
        result = SIGenerator().compute_si_score(_make_features())
        for key in [
            "si_score", "total_damage", "density_damage",
            "network_damage", "complexity_damage", "width_damage",
            "segmentation_quality", "risk_level",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_segmentation_quality_limited_impact(self):
        """Segmentation quality weight is 0 by default → no impact on score."""
        good_seg = SIGenerator().compute_si_score(
            _make_features(dice_score=1.0, bce_loss=0.0)
        )
        poor_seg = SIGenerator().compute_si_score(
            _make_features(dice_score=0.0, bce_loss=10.0)
        )
        # Default weights have segmentation_quality=0.0 → scores must be equal
        assert good_seg["si_score"] == poor_seg["si_score"]


# ─────────────────────────────────────────────────────────────────────────────
# compute_structural_integrity  (public one-call API)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeStructuralIntegrity:
    """Integration tests using numpy arrays and a NetworkX graph."""

    def _straight_crack(self):
        """Return (mask, skeleton, graph) for a simple horizontal crack."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[30:35, 10:54] = 1

        skel = np.zeros((64, 64), dtype=np.uint8)
        skel[32, 10:54] = 1

        g = nx.Graph()
        for x in range(10, 54):
            g.add_node((32, x))
            if x > 10:
                g.add_edge((32, x - 1), (32, x), weight=1.0)
        return mask, skel, g

    def _branched_crack(self):
        """Return (mask, skeleton, graph) for a crack with one branch."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[32, 10:40] = 1
        mask[28:32, 25] = 1

        skel = np.zeros((64, 64), dtype=np.uint8)
        skel[32, 10:40] = 1
        skel[28:32, 25] = 1

        g = nx.Graph()
        for x in range(10, 40):
            g.add_node((32, x))
            if x > 10:
                g.add_edge((32, x - 1), (32, x), weight=1.0)
        for y in range(28, 32):
            g.add_node((y, 25))
            if y > 28:
                g.add_edge((y - 1, 25), (y, 25), weight=1.0)
        g.add_edge((32, 25), (31, 25), weight=1.0)
        return mask, skel, g

    def test_returns_required_keys(self):
        mask, skel, g = self._straight_crack()
        result = compute_structural_integrity(mask, skel, g, connectivity_ratio=0.8)
        for key in ["si_score", "total_damage", "risk_level"]:
            assert key in result

    def test_score_in_range(self):
        mask, skel, g = self._straight_crack()
        result = compute_structural_integrity(mask, skel, g, connectivity_ratio=0.8)
        assert 0.0 <= result["si_score"] <= 1.0

    def test_empty_image_perfect_score(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        skel = np.zeros((64, 64), dtype=np.uint8)
        g = nx.Graph()
        result = compute_structural_integrity(mask, skel, g, connectivity_ratio=0.0)
        assert result["si_score"] == 1.0

    def test_branched_crack_lower_than_straight(self):
        """A branched crack should score lower than a simple straight crack."""
        m1, s1, g1 = self._straight_crack()
        m2, s2, g2 = self._branched_crack()
        r1 = compute_structural_integrity(m1, s1, g1, connectivity_ratio=0.8)
        r2 = compute_structural_integrity(m2, s2, g2, connectivity_ratio=0.9)
        # Branched crack has junctions → higher network damage → lower SI
        assert r2["si_score"] <= r1["si_score"]

    def test_raw_features_in_result(self):
        mask, skel, g = self._straight_crack()
        result = compute_structural_integrity(mask, skel, g, connectivity_ratio=0.8)
        assert "raw_features" in result
        rf = result["raw_features"]
        assert rf["skeleton_pixels"] > 0
        assert rf["total_crack_length"] > 0

    def test_with_gt_dice_uses_seg_weights(self):
        """Passing dice < 1.0 should switch to weights that include seg quality."""
        mask, skel, g = self._straight_crack()
        # With perfect dice (inference mode)
        r_inference = compute_structural_integrity(
            mask, skel, g, connectivity_ratio=0.8, dice=1.0, bce=0.0
        )
        # With poor dice (training/eval mode) — score should differ
        r_eval = compute_structural_integrity(
            mask, skel, g, connectivity_ratio=0.8, dice=0.3, bce=1.5
        )
        # Poor segmentation quality should reduce SI score
        assert r_eval["si_score"] <= r_inference["si_score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
