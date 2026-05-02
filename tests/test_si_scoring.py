"""
Unit tests for the new Structural Integrity scoring system.

Run with: pytest tests/test_si_scoring.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import networkx as nx
import pytest

from features.si_scoring import (
    SIGenerator,
    SIWeights,
    SIRiskThresholds,
    DamageFeatures,
    FeatureNormalizer,
    compute_structural_integrity,
    classify_risk,
    extract_damage_features,
)


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
        # Test exact threshold values
        assert classify_risk(0.80)[0] == "Low"
        assert classify_risk(0.60)[0] == "Moderate"
        assert classify_risk(0.40)[0] == "High"
        assert classify_risk(0.20)[0] == "Critical"
        assert classify_risk(0.00)[0] == "Failure Imminent"


class TestFeatureNormalizer:
    """Test feature normalization functions."""
    
    def test_normalize_density_zero(self):
        result = FeatureNormalizer.normalize_density(0, 1000)
        assert result == 0.0
    
    def test_normalize_density_max(self):
        # 25% of image covered should cap at 1.0
        result = FeatureNormalizer.normalize_density(2500, 10000)
        assert result == 1.0
    
    def test_normalize_density_mid(self):
        # 12.5% coverage = 0.5 normalized
        result = FeatureNormalizer.normalize_density(1250, 10000)
        assert 0.45 < result < 0.55
    
    def test_normalize_complexity_zero(self):
        result = FeatureNormalizer.normalize_complexity(0, 0, 0)
        assert result == 0.0
    
    def test_normalize_complexity_junctions_weighted(self):
        # Junctions should have 2x weight
        branches_only = FeatureNormalizer.normalize_complexity(5, 0, 0)
        junctions_only = FeatureNormalizer.normalize_complexity(0, 2, 4)  # 2 junctions = 4 complexity units
        # Both should give similar results
        assert abs(branches_only - junctions_only) < 0.1
    
    def test_normalize_connectivity_nonlinear(self):
        # Power function should give higher weight to high connectivity
        low = FeatureNormalizer.normalize_connectivity(0.5)
        high = FeatureNormalizer.normalize_connectivity(1.0)
        assert high > low
        assert high == 1.0
    
    def test_normalize_crack_width_empty(self):
        mask = np.zeros((10, 10))
        skeleton = np.zeros((10, 10))
        max_w, mean_w = FeatureNormalizer.normalize_crack_width(mask, skeleton)
        assert max_w == 0.0
        assert mean_w == 0.0
    
    def test_normalize_segmentation_quality(self):
        # Perfect dice, zero bce
        quality = FeatureNormalizer.normalize_segmentation_quality(1.0, 0.0)
        assert quality == 1.0
        
        # Worst dice, high bce
        quality = FeatureNormalizer.normalize_segmentation_quality(0.0, 10.0)
        assert quality < 0.3


class TestSIGenerator:
    """Test the SI score generator."""
    
    def test_no_cracks_perfect_score(self):
        """No cracks should yield SI = 1.0"""
        features = DamageFeatures(
            crack_density=0.0,
            skeleton_density=0.0,
            connectivity_ratio=0.0,
            num_branches=0,
            num_junctions=0,
            num_endpoints=0,
            complexity_index=0.0,
            network_density=0.0,
            total_crack_length=0.0,
            max_crack_width_proxy=0.0,
            mean_crack_width_proxy=0.0,
            dice_score=1.0,
            bce_loss=0.0,
        )
        
        generator = SIGenerator()
        result = generator.compute_si_score(features)
        
        assert result["si_score"] == 1.0
        assert result["total_damage"] == 0.0
        assert result["risk_level"] == "Low"
    
    def test_full_damage_zero_score(self):
        """Maximum damage should yield SI = 0.0"""
        features = DamageFeatures(
            crack_density=1.0,
            skeleton_density=1.0,
            connectivity_ratio=1.0,
            num_branches=20,
            num_junctions=10,
            num_endpoints=5,
            complexity_index=1.0,
            network_density=1.0,
            total_crack_length=1000.0,
            max_crack_width_proxy=1.0,
            mean_crack_width_proxy=1.0,
            dice_score=0.0,  # Poor segmentation confidence
            bce_loss=10.0,
        )
        
        generator = SIGenerator()
        result = generator.compute_si_score(features)
        
        assert result["si_score"] == 0.0
        assert result["total_damage"] == 1.0
    
    def test_connectivity_penalty(self):
        """Higher connectivity should reduce SI score"""
        base_features = DamageFeatures(
            crack_density=0.1,
            skeleton_density=0.1,
            connectivity_ratio=0.0,  # Low connectivity
            num_branches=2,
            num_junctions=1,
            num_endpoints=3,
            complexity_index=0.2,
            network_density=0.05,
            total_crack_length=50.0,
            max_crack_width_proxy=0.01,
            mean_crack_width_proxy=0.005,
            dice_score=0.95,
            bce_loss=0.02,
        )
        
        generator = SIGenerator()
        
        low_conn = generator.compute_si_score(base_features)
        
        # Increase connectivity
        base_features.connectivity_ratio = 0.9
        high_conn = generator.compute_si_score(base_features)
        
        assert high_conn["si_score"] < low_conn["si_score"]
    
    def test_segmentation_quality_capped(self):
        """Segmentation quality should have limited impact (30% max)"""
        base_features = DamageFeatures(
            crack_density=0.2,
            skeleton_density=0.2,
            connectivity_ratio=0.5,
            num_branches=5,
            num_junctions=2,
            num_endpoints=6,
            complexity_index=0.4,
            network_density=0.1,
            total_crack_length=100.0,
            max_crack_width_proxy=0.02,
            mean_crack_width_proxy=0.01,
            dice_score=1.0,  # Perfect segmentation
            bce_loss=0.0,
        )
        
        generator = SIGenerator()
        good_seg = generator.compute_si_score(base_features)
        
        # Poor segmentation on same damage
        base_features.dice_score = 0.0
        base_features.bce_loss = 10.0
        poor_seg = generator.compute_si_score(base_features)
        
        # Difference should be limited by 30% cap
        diff = good_seg["si_score"] - poor_seg["si_score"]
        assert diff <= 0.35  # Allow small tolerance


class TestIntegration:
    """Integration tests with actual data structures."""
    
    def test_compute_from_raw(self):
        """Test full pipeline with numpy arrays and graph"""
        # Create synthetic crack data
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[30:35, 10:54] = 1  # Horizontal crack
        
        skeleton = np.zeros((64, 64), dtype=np.uint8)
        skeleton[32, 10:54] = 1  # Skeleton line
        
        # Create simple graph
        graph = nx.Graph()
        for x in range(10, 54):
            graph.add_node((32, x))
            if x > 10:
                graph.add_edge((32, x-1), (32, x), weight=1.0)
        
        result = compute_structural_integrity(
            mask=mask,
            skeleton=skeleton,
            graph=graph,
            connectivity_ratio=0.8,
            dice=0.95,
            bce=0.02,
        )
        
        assert "si_score" in result
        assert "total_damage" in result
        assert "risk_level" in result
        assert 0.0 <= result["si_score"] <= 1.0
    
    def test_extract_damage_features(self):
        """Test feature extraction from raw data"""
        # Create synthetic data with branches
        mask = np.zeros((64, 64), dtype=np.uint8)
        # Main horizontal line
        mask[32, 10:40] = 1
        # Branch up
        mask[28:32, 25] = 1
        
        skeleton = np.zeros((64, 64), dtype=np.uint8)
        skeleton[32, 10:40] = 1
        skeleton[28:32, 25] = 1
        
        # Create graph
        graph = nx.Graph()
        # Main line
        for x in range(10, 40):
            graph.add_node((32, x))
            if x > 10:
                graph.add_edge((32, x-1), (32, x), weight=1.0)
        # Branch
        for y in range(28, 32):
            graph.add_node((y, 25))
            if y > 28:
                graph.add_edge((y-1, 25), (y, 25), weight=1.0)
        graph.add_edge((32, 25), (31, 25), weight=1.0)  # Connect branch
        
        features = extract_damage_features(
            mask=mask,
            skeleton=skeleton,
            graph=graph,
            connectivity_ratio=0.9,
            dice=0.95,
            bce=0.02,
        )
        
        assert features.num_branches >= 1
        assert features.num_junctions >= 1
        assert features.total_crack_length > 0


class TestSIWeights:
    """Test weight validation."""
    
    def test_valid_weights(self):
        weights = SIWeights(
            crack_density=0.35,
            connectivity_penalty=0.20,
            complexity_penalty=0.15,
            segmentation_quality=0.30,
        )
        weights.validate()  # Should not raise
    
    def test_invalid_weights_sum(self):
        weights = SIWeights(
            crack_density=0.5,
            connectivity_penalty=0.5,
            complexity_penalty=0.5,
            segmentation_quality=0.5,
        )
        with pytest.raises(ValueError, match="sum to 1.0"):
            weights.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
