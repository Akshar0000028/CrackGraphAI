"""Post-processing utilities for stable mask predictions."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects, skeletonize

logger = logging.getLogger("crackgraphai.postprocessing")


class MaskSmoother:
    """Apply morphological smoothing to predicted masks."""
    
    def __init__(
        self,
        kernel_size: int = 3,
        sigma: float = 1.0,
        apply_closing: bool = True,
        apply_opening: bool = True,
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.apply_closing = apply_closing
        self.apply_opening = apply_opening
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
    
    def smooth(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Smooth probability map with Gaussian filter.
        
        Args:
            prob_map: Probability map [H, W] with values in [0, 1]
            
        Returns:
            Smoothed probability map
        """
        if self.sigma > 0:
            prob_map = ndimage.gaussian_filter(prob_map, sigma=self.sigma)
        
        return prob_map
    
    def morphological_clean(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Cleaned mask
        """
        binary = mask.astype(bool)
        
        # Closing: fill small holes
        if self.apply_closing:
            binary = ndimage.binary_closing(binary, structure=self.kernel)
        
        # Opening: remove small noise
        if self.apply_opening:
            binary = ndimage.binary_opening(binary, structure=self.kernel)
        
        return binary.astype(np.uint8)
    
    def remove_small_components(
        self, 
        mask: np.ndarray, 
        min_size: int = 16,
        connectivity: int = 2,
    ) -> np.ndarray:
        """Remove small connected components."""
        binary = mask.astype(bool)
        
        # Use scipy for connected component analysis
        labeled, num_features = ndimage.label(binary)
        
        if num_features == 0:
            return mask
        
        # Find component sizes
        component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        
        # Keep only components larger than min_size
        keep_mask = component_sizes >= min_size
        
        # Reconstruct mask
        result = np.zeros_like(mask)
        for i, keep in enumerate(keep_mask, 1):
            if keep:
                result[labeled == i] = 1
        
        return result
    
    def fill_holes(self, mask: np.ndarray, max_hole_size: int = 100) -> np.ndarray:
        """Fill small holes in mask."""
        binary = mask.astype(bool)
        
        # Invert and find holes
        inverted = ~binary
        labeled, num_features = ndimage.label(inverted)
        
        if num_features == 0:
            return mask
        
        # Component sizes
        hole_sizes = ndimage.sum(inverted, labeled, range(1, num_features + 1))
        
        # Fill holes smaller than max_hole_size
        filled = binary.copy()
        for i, size in enumerate(hole_sizes, 1):
            if size < max_hole_size:
                filled[labeled == i] = True
        
        return filled.astype(np.uint8)


class UncertaintyEstimator:
    """Estimate prediction uncertainty for reliability assessment."""
    
    def __init__(self, method: str = "entropy"):
        self.method = method
    
    def estimate(self, prob_map: np.ndarray) -> Dict[str, float]:
        """
        Estimate uncertainty from probability map.
        
        Args:
            prob_map: Probability map [H, W] with values in [0, 1]
            
        Returns:
            Dict with uncertainty metrics
        """
        # Clamp probabilities to avoid log(0)
        prob_map = np.clip(prob_map, 1e-7, 1 - 1e-7)
        
        # Entropy-based uncertainty
        entropy = -(prob_map * np.log(prob_map) + 
                    (1 - prob_map) * np.log(1 - prob_map))
        
        # Normalize entropy (max entropy is ln(2) at p=0.5)
        max_entropy = np.log(2)
        normalized_entropy = entropy / max_entropy
        
        # Confidence as 1 - normalized entropy
        confidence = 1 - normalized_entropy
        
        # Edge uncertainty (pixels near 0.5)
        edge_mask = (prob_map > 0.4) & (prob_map < 0.6)
        edge_ratio = np.mean(edge_mask)
        
        # High-confidence predictions
        high_conf_mask = (prob_map > 0.8) | (prob_map < 0.2)
        high_conf_ratio = np.mean(high_conf_mask)
        
        return {
            'mean_uncertainty': float(np.mean(normalized_entropy)),
            'max_uncertainty': float(np.max(normalized_entropy)),
            'uncertainty_map': normalized_entropy,
            'mean_confidence': float(np.mean(confidence)),
            'edge_ratio': float(edge_ratio),
            'high_confidence_ratio': float(high_conf_ratio),
            'reliable': float(np.mean(normalized_entropy)) < 0.3,
        }
    
    def monte_carlo_dropout(
        self,
        model,
        x: 'torch.Tensor',
        num_samples: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty using Monte Carlo Dropout.
        
        Note: Model must have dropout layers enabled during inference.
        """
        import torch
        
        model.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = torch.sigmoid(model(x))
                predictions.append(pred.cpu().numpy())
        
        model.eval()  # Restore eval mode
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return {
            'mean': mean_pred,
            'uncertainty': uncertainty,
            'coefficient_of_variation': uncertainty / (mean_pred + 1e-8),
        }


class ContourRefiner:
    """Refine mask contours for smoother edges."""
    
    def __init__(self, epsilon_factor: float = 0.001):
        self.epsilon_factor = epsilon_factor
    
    def refine(self, mask: np.ndarray) -> np.ndarray:
        """Refine mask contours using polygon approximation."""
        binary = mask.astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return mask
        
        # Create new mask
        refined = np.zeros_like(mask)
        
        for contour in contours:
            # Approximate contour
            epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Draw filled contour
            cv2.drawContours(refined, [approx], -1, 1, -1)
        
        return refined


class PredictionAggregator:
    """Aggregate multiple predictions for stability."""
    
    @staticmethod
    def majority_vote(masks: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """
        Majority voting from multiple binary masks.
        
        Args:
            masks: List of binary masks
            threshold: Fraction of votes needed for positive
            
        Returns:
            Aggregated mask
        """
        stacked = np.stack(masks, axis=0)
        vote_ratio = np.mean(stacked, axis=0)
        return (vote_ratio >= threshold).astype(np.uint8)
    
    @staticmethod
    def probability_average(
        prob_maps: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Weighted average of probability maps."""
        stacked = np.stack(prob_maps, axis=0)
        
        if weights is None:
            return np.mean(stacked, axis=0)
        
        weights = np.array(weights) / sum(weights)
        weights = weights.reshape(-1, 1, 1)
        
        return np.sum(stacked * weights, axis=0)
    
    @staticmethod
    def uncertainty_weighted_average(
        prob_maps: List[np.ndarray],
        uncertainties: List[np.ndarray],
    ) -> np.ndarray:
        """Average weighted by inverse uncertainty."""
        # Lower uncertainty = higher weight
        inv_uncertainties = [1.0 / (u + 1e-8) for u in uncertainties]
        total_inv_unc = sum(inv_uncertainties)
        
        weights = [inv_u / total_inv_unc for inv_u in inv_uncertainties]
        
        return PredictionAggregator.probability_average(prob_maps, weights)
