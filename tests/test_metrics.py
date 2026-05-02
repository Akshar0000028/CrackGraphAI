"""Tests for metrics and utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from utils.metrics import (
    branch_consistency,
    segmentation_metrics,
    structural_integrity_score,
)


class TestSegmentationMetrics:
    """Test segmentation metric calculations."""

    def test_perfect_prediction(self):
        """Test metrics when prediction matches ground truth."""
        logits = torch.ones(1, 1, 64, 64) * 10  # High positive values
        targets = torch.ones(1, 1, 64, 64)

        metrics = segmentation_metrics(logits, targets)

        assert metrics["dice"] > 0.99
        assert metrics["iou"] > 0.99
        assert metrics["precision"] > 0.99
        assert metrics["recall"] > 0.99

    def test_empty_prediction(self):
        """Test metrics when prediction is empty."""
        logits = torch.ones(1, 1, 64, 64) * -10  # High negative values
        targets = torch.ones(1, 1, 64, 64)

        metrics = segmentation_metrics(logits, targets)

        assert metrics["recall"] < 0.01
        assert metrics["precision"] == 0.0

    def test_empty_ground_truth(self):
        """Test metrics when ground truth is empty."""
        logits = torch.ones(1, 1, 64, 64) * 10
        targets = torch.zeros(1, 1, 64, 64)

        metrics = segmentation_metrics(logits, targets)

        assert metrics["precision"] < 0.01
        assert metrics["recall"] == 0.0


class TestStructuralIntegrityScore:
    """Test SI score calculation."""

    def test_perfect_score(self):
        """Test SI score with perfect metrics."""
        weights = {
            "dice": 0.35,
            "bce": 0.15,
            "connectivity": 0.30,
            "branch_consistency": 0.20,
        }
        score = structural_integrity_score(
            dice=1.0,
            bce=0.0,  # exp(-0) = 1.0
            connectivity=1.0,
            branch_consistency_score=1.0,
            weights=weights,
        )
        assert score == 1.0

    def test_zero_score(self):
        """Test SI score with zero metrics."""
        weights = {
            "dice": 0.35,
            "bce": 0.15,
            "connectivity": 0.30,
            "branch_consistency": 0.20,
        }
        score = structural_integrity_score(
            dice=0.0,
            bce=100.0,  # Bounded BCE: 2*100/(100+1) ≈ 1.98, exp(-1.98) ≈ 0.14, small contribution
            connectivity=0.0,
            branch_consistency_score=0.0,
            weights=weights,
        )
        assert score < 0.05  # Near zero with very high BCE

    def test_score_clipping(self):
        """Test SI score is clipped to [0, 1]."""
        weights = {
            "dice": 0.35,
            "bce": 0.15,
            "connectivity": 0.30,
            "branch_consistency": 0.20,
        }
        # Create a scenario that would exceed 1.0
        score = structural_integrity_score(
            dice=2.0,
            bce=0.0,
            connectivity=2.0,
            branch_consistency_score=2.0,
            weights=weights,
        )
        assert score == 1.0

        # Create a scenario that would be below 0
        score = structural_integrity_score(
            dice=-1.0,
            bce=1000.0,  # Very high BCE for minimal contribution
            connectivity=-1.0,
            branch_consistency_score=-1.0,
            weights=weights,
        )
        assert score == 0.0


class TestBranchConsistency:
    """Test branch consistency calculation."""

    def test_perfect_consistency(self):
        """Test when predicted branches match ground truth."""
        score = branch_consistency(pred_branches=10.0, gt_branches=10.0)
        assert score == 1.0

    def test_no_branches(self):
        """Test when both have no branches."""
        score = branch_consistency(pred_branches=0.0, gt_branches=0.0)
        assert score == 1.0

    def test_no_gt_branches(self):
        """Test when GT has no branches but prediction has some."""
        score = branch_consistency(pred_branches=5.0, gt_branches=0.0)
        assert score == 0.0

    def test_partial_consistency(self):
        """Test partial branch consistency."""
        score = branch_consistency(pred_branches=8.0, gt_branches=10.0)
        expected = 1.0 - abs(8.0 - 10.0) / 10.0
        assert abs(score - expected) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
