"""Robust input validation and preprocessing for production inference."""

from __future__ import annotations

import io
import logging
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageCms

logger = logging.getLogger("crackgraphai.preprocessing")


class InputValidator:
    """Validates input images for inference."""
    
    # Supported formats
    SUPPORTED_FORMATS = {'JPEG', 'JPG', 'PNG', 'BMP', 'TIFF', 'TIF'}
    
    # Size limits
    MAX_DIMENSION = 8192
    MIN_DIMENSION = 32
    MAX_FILE_SIZE_MB = 50
    
    def __init__(self):
        self.validation_stats = {
            'total': 0,
            'passed': 0,
            'failed': 0,
        }
    
    def validate(self, image_bytes: bytes) -> Dict:
        """
        Validate input image bytes.
        
        Returns:
            Dict with 'valid', 'error', and metadata
        """
        self.validation_stats['total'] += 1
        
        # Check file size
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            return {
                'valid': False,
                'error': f"File too large: {size_mb:.1f}MB (max: {self.MAX_FILE_SIZE_MB}MB)"
            }
        
        try:
            # Try to decode
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                # Try with PIL as fallback
                try:
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    pil_img.verify()  # Verify image integrity
                    
                    # Re-open for actual use (verify closes the file)
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    return {
                        'valid': False,
                        'error': f"Cannot decode image: {e}"
                    }
            
            # Check dimensions
            h, w = img.shape[:2]
            
            if h < self.MIN_DIMENSION or w < self.MIN_DIMENSION:
                return {
                    'valid': False,
                    'error': f"Image too small: {w}x{h} (min: {self.MIN_DIMENSION}x{self.MIN_DIMENSION})"
                }
            
            if h > self.MAX_DIMENSION or w > self.MAX_DIMENSION:
                return {
                    'valid': False,
                    'error': f"Image too large: {w}x{h} (max: {self.MAX_DIMENSION}x{self.MAX_DIMENSION})"
                }
            
            # Check for corrupted/unusual images
            if img.size == 0:
                return {'valid': False, 'error': "Empty image data"}
            
            # Check for uniform images (likely corrupted)
            if np.std(img) < 1.0:
                logger.warning("Image has very low variance, might be corrupted or blank")
            
            # Detect format
            fmt = self._detect_format(image_bytes)
            
            self.validation_stats['passed'] += 1
            
            return {
                'valid': True,
                'dimensions': (w, h),
                'channels': 1 if len(img.shape) == 2 else img.shape[2],
                'format': fmt,
                'file_size_mb': size_mb,
            }
            
        except Exception as e:
            self.validation_stats['failed'] += 1
            return {
                'valid': False,
                'error': f"Validation error: {e}"
            }
    
    def _detect_format(self, image_bytes: bytes) -> str:
        """Detect image format from magic bytes."""
        if image_bytes.startswith(b'\x89PNG'):
            return 'PNG'
        elif image_bytes.startswith(b'\xff\xd8'):
            return 'JPEG'
        elif image_bytes.startswith(b'BM'):
            return 'BMP'
        elif image_bytes.startswith(b'II') or image_bytes.startswith(b'MM'):
            return 'TIFF'
        return 'UNKNOWN'


class RobustPreprocessor:
    """Robust preprocessing pipeline for consistent model inputs."""
    
    def __init__(
        self,
        target_size: int = 256,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        enhance_contrast: bool = False,
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.enhance_contrast = enhance_contrast
    
    def process(self, image_bytes: bytes) -> torch.Tensor:
        """
        Process image bytes to model-ready tensor.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Decode image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            # Fallback to PIL
            pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return self.process_numpy(img)
    
    def process_numpy(self, img: np.ndarray) -> torch.Tensor:
        """Process numpy array to model-ready tensor."""
        # Ensure 3-channel BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Resize with proper interpolation
        h, w = img.shape[:2]
        
        # Use cubic for upscaling, area for downscaling
        if self.target_size > max(h, w):
            interp = cv2.INTER_CUBIC
        else:
            interp = cv2.INTER_AREA
        
        img_resized = cv2.resize(img, (self.target_size, self.target_size), interpolation=interp)
        
        # Optional contrast enhancement
        if self.enhance_contrast:
            lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            img_resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        
        if self.normalize:
            img_normalized = (img_float - self.mean) / self.std
        else:
            img_normalized = img_float
        
        # HWC to CHW
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        
        # To tensor and add batch dimension
        return torch.from_numpy(img_chw).unsqueeze(0)
    
    def process_pil(self, pil_img: Image.Image) -> torch.Tensor:
        """Process PIL image to model-ready tensor."""
        # Convert to RGB
        rgb_img = pil_img.convert('RGB')
        
        # Resize
        rgb_img = rgb_img.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        
        # To numpy
        img_array = np.array(rgb_img, dtype=np.float32) / 255.0
        
        if self.normalize:
            img_array = (img_array - self.mean) / self.std
        
        # HWC to CHW
        img_chw = np.transpose(img_array, (2, 0, 1))
        
        return torch.from_numpy(img_chw).unsqueeze(0)


class ColorSpaceConverter:
    """Handle color space conversions for consistent input."""
    
    @staticmethod
    def ensure_srgb(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Convert image to sRGB color space."""
        if isinstance(image, Image.Image):
            # Check for ICC profile
            if 'icc_profile' in image.info:
                try:
                    # Convert to sRGB
                    srgb_profile = ImageCms.createProfile('sRGB')
                    img_profile = ImageCms.ImageCmsProfile(io.BytesIO(image.info['icc_profile']))
                    image = ImageCms.profileToProfile(image, img_profile, srgb_profile)
                except Exception as e:
                    logger.warning(f"Color profile conversion failed: {e}")
            
            return np.array(image.convert('RGB'))
        
        return image
    
    @staticmethod
    def auto_white_balance(img: np.ndarray) -> np.ndarray:
        """Apply automatic white balance."""
        result = cv2.xphoto.createSimpleWB().balanceWhite(img) \
            if hasattr(cv2, 'xphoto') else img
        
        if result is None:
            # Fallback: gray world assumption
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Shift a and b channels to neutral
            a_mean = np.mean(a)
            b_mean = np.mean(b)
            
            a = a - a_mean + 128
            b = b - b_mean + 128
            
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return result
