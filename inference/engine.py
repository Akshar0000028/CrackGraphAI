"""Stable inference engine with TTA, ensembling, and uncertainty quantification."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .cache import ResultCache
from .postprocessing import MaskSmoother, UncertaintyEstimator
from .preprocessing import InputValidator, RobustPreprocessor

logger = logging.getLogger("crackgraphai.inference")


@dataclass
class InferenceConfig:
    """Configuration for stable inference."""
    
    # Model settings
    model_path: str = "checkpoints/best_hybrid_segformer.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # TTA settings
    use_tta: bool = True
    tta_flips: List[str] = None  # ['horizontal', 'vertical', 'both']
    tta_merge_strategy: str = "mean"  # 'mean', 'max', 'vote'
    
    # Stability settings
    num_inference_runs: int = 1  # >1 for Monte Carlo dropout or multi-model
    ensemble_threshold: float = 0.5  # Consensus threshold
    
    # Post-processing
    apply_morphology: bool = True
    morphology_kernel_size: int = 3
    min_component_size: int = 16
    smoothing_sigma: float = 1.0
    
    # Uncertainty
    estimate_uncertainty: bool = True
    uncertainty_threshold: float = 0.3
    
    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Performance
    batch_size: int = 1
    mixed_precision: bool = True
    
    def __post_init__(self):
        if self.tta_flips is None:
            self.tta_flips = ['horizontal', 'vertical']


class StableInferenceEngine:
    """
    Production-grade inference engine with stability guarantees.
    
    Features:
    - Test-Time Augmentation (TTA) for robust predictions
    - Input validation and normalization
    - Uncertainty quantification
    - Result caching
    - Consistent output through ensemble strategies
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
        cache: Optional[ResultCache] = None,
    ):
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize model
        self.model = model.to(self.device)
        self.model.eval()
        
        # Enable inference mode optimizations
        if hasattr(torch, 'inference_mode'):
            self._inference_ctx = torch.inference_mode
        else:
            self._inference_ctx = torch.no_grad
        
        # Initialize components
        self.validator = InputValidator()
        self.preprocessor = RobustPreprocessor(
            target_size=256,
            normalize=True,
            enhance_contrast=True,  # Enable contrast enhancement for better crack detection
        )
        self.smoother = MaskSmoother(
            kernel_size=self.config.morphology_kernel_size,
            sigma=self.config.smoothing_sigma,
        )
        self.uncertainty_estimator = UncertaintyEstimator()
        self.cache = cache or (ResultCache(ttl=self.config.cache_ttl) if self.config.enable_cache else None)
        
        # Precompute TTA transforms
        self.tta_transforms = self._build_tta_transforms()
        
        logger.info(f"StableInferenceEngine initialized on {self.device}")
    
    def _build_tta_transforms(self) -> List[Tuple[str, Callable]]:
        """Build TTA transformation functions."""
        transforms = [('none', lambda x: x)]
        
        for flip in self.config.tta_flips:
            if flip == 'horizontal':
                transforms.append(('hflip', lambda x: torch.flip(x, dims=[3])))
            elif flip == 'vertical':
                transforms.append(('vflip', lambda x: torch.flip(x, dims=[2])))
            elif flip == 'both':
                transforms.append(('hvflip', lambda x: torch.flip(x, dims=[2, 3])))
        
        return transforms
    
    def _compute_cache_key(self, image_bytes: bytes, params: Dict) -> str:
        """Compute deterministic cache key."""
        hasher = hashlib.sha256()
        hasher.update(image_bytes)
        hasher.update(str(params).encode())
        return hasher.hexdigest()
    
    @torch.no_grad()
    def _single_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Run single inference pass."""
        with self._inference_ctx():
            if self.config.mixed_precision and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    logits = self.model(x)
            else:
                logits = self.model(x)
            return torch.sigmoid(logits)
    
    def _apply_tta(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Test-Time Augmentation and merge predictions."""
        if not self.config.use_tta or len(self.tta_transforms) == 1:
            return self._single_inference(x)
        
        predictions = []
        
        for name, transform in self.tta_transforms:
            try:
                # Apply transform
                x_transformed = transform(x)
                
                # Inference
                pred = self._single_inference(x_transformed)
                
                # Undo spatial transforms
                if 'hflip' in name:
                    pred = torch.flip(pred, dims=[3])
                if 'vflip' in name:
                    pred = torch.flip(pred, dims=[2])
                
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"TTA transform '{name}' failed: {e}, using original prediction")
                # Fallback: use original prediction without this transform
                pred = self._single_inference(x)
                predictions.append(pred)
        
        if not predictions:
            # If all transforms failed, return single inference
            return self._single_inference(x)
        
        # Merge predictions
        stacked = torch.stack(predictions, dim=0)
        
        if self.config.tta_merge_strategy == 'mean':
            return stacked.mean(dim=0)
        elif self.config.tta_merge_strategy == 'max':
            return stacked.max(dim=0)[0]
        elif self.config.tta_merge_strategy == 'vote':
            # Majority voting
            votes = (stacked > self.config.ensemble_threshold).float()
            vote_ratio = votes.mean(dim=0)
            return vote_ratio
        else:
            return stacked.mean(dim=0)
    
    def predict(
        self,
        image: Union[np.ndarray, bytes, str],
        return_uncertainty: bool = False,
        custom_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Run stable prediction on input image.
        
        Args:
            image: Input image (numpy array, bytes, or path)
            return_uncertainty: Whether to return uncertainty estimate
            custom_threshold: Override default threshold
            
        Returns:
            Dictionary with mask, probability map, and metadata
        """
        start_time = time.time()
        
        # Handle different input types
        if isinstance(image, str):
            image_bytes = open(image, 'rb').read()
        elif isinstance(image, np.ndarray):
            _, buf = cv2.imencode('.png', image)
            image_bytes = buf.tobytes()
        else:
            image_bytes = image
        
        # Check cache
        cache_key = self._compute_cache_key(image_bytes, {
            'tta': self.config.use_tta,
            'threshold': custom_threshold or self.config.ensemble_threshold,
        })
        
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug("Cache hit, returning cached result")
                cached['from_cache'] = True
                return cached
        
        # Validate input
        validated = self.validator.validate(image_bytes)
        if not validated['valid']:
            raise ValueError(f"Invalid input: {validated['error']}")
        
        # Preprocess
        x = self.preprocessor.process(image_bytes)
        x = x.to(self.device)
        
        # Run inference with TTA
        prob_map = self._apply_tta(x)
        
        # Move to CPU for post-processing
        prob_map = prob_map.squeeze().cpu().numpy()
        
        # Compute uncertainty if requested
        uncertainty = None
        if self.config.estimate_uncertainty and return_uncertainty:
            uncertainty = self.uncertainty_estimator.estimate(prob_map)
        
        # Apply morphological smoothing
        if self.config.apply_morphology:
            prob_map = self.smoother.smooth(prob_map)
        
        # Threshold to binary mask
        threshold = custom_threshold or self.config.ensemble_threshold
        mask = (prob_map > threshold).astype(np.uint8)
        
        # Remove small components
        if self.config.min_component_size > 0:
            mask = self.smoother.remove_small_components(
                mask, 
                min_size=self.config.min_component_size
            )
        
        # Compute metrics
        result = {
            'mask': mask,
            'probability_map': prob_map,
            'threshold': threshold,
            'inference_time': time.time() - start_time,
            'image_size': validated['dimensions'],
            'from_cache': False,
        }
        
        if uncertainty is not None:
            result['uncertainty'] = uncertainty
            result['confidence'] = 1.0 - uncertainty['mean_uncertainty']
            result['reliable'] = uncertainty['mean_uncertainty'] < self.config.uncertainty_threshold
        
        # Cache result
        if self.cache:
            self.cache.set(cache_key, result.copy())
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[np.ndarray, bytes, str]],
        return_uncertainty: bool = False,
    ) -> List[Dict]:
        """Run batch prediction with consistent processing."""
        results = []
        
        # Process in chunks
        batch_size = self.config.batch_size
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            for img in batch:
                try:
                    result = self.predict(img, return_uncertainty=return_uncertainty)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch inference failed for image: {e}")
                    results.append({
                        'error': str(e),
                        'mask': np.zeros((256, 256), dtype=np.uint8),
                        'probability_map': np.zeros((256, 256)),
                    })
        
        return results
    
    def warmup(self, num_runs: int = 3):
        """Warm up model with dummy inference to stabilize performance."""
        dummy = torch.randn(1, 3, 256, 256).to(self.device)
        
        logger.info(f"Warming up model with {num_runs} runs...")
        for _ in range(num_runs):
            _ = self._single_inference(dummy)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        logger.info("Model warmup complete")
