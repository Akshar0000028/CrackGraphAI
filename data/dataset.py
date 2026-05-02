from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Sample:
    image_path: Path
    mask_path: Path


def build_transforms(image_size: int, split: str) -> A.Compose:
    common = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(p=0.25, alpha=50, sigma=6),
                A.RandomBrightnessContrast(p=0.4),
                A.GaussNoise(p=0.2),
            ]
            + common
        )
    return A.Compose(common)


def discover_samples(root_dir: str, image_dir_name: str, mask_dir_name: str) -> List[Sample]:
    image_dir = Path(root_dir) / image_dir_name
    mask_dir = Path(root_dir) / mask_dir_name
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    samples: List[Sample] = []
    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if mask_path.exists():
            samples.append(Sample(image_path=image_path, mask_path=mask_path))
    if not samples:
        raise RuntimeError(f"No paired images/masks found under {image_dir} and {mask_dir}.")
    return samples


def split_samples(samples: List[Sample], train_ratio: float, val_ratio: float, test_ratio: float):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")
    train_samples, test_samples = train_test_split(samples, test_size=test_ratio, random_state=42, shuffle=True)
    val_size_relative = val_ratio / (train_ratio + val_ratio)
    train_samples, val_samples = train_test_split(
        train_samples, test_size=val_size_relative, random_state=42, shuffle=True
    )
    return train_samples, val_samples, test_samples


class CrackSegmentationDataset(Dataset):
    def __init__(self, samples: List[Sample], image_size: int, split: str) -> None:
        self.samples = samples
        self.transforms = build_transforms(image_size=image_size, split=split)
        self.split = split
        self._validate_samples()

    def _validate_samples(self) -> None:
        """Validate that all samples can be loaded."""
        import logging
        logger = logging.getLogger("crackgraphai.dataset")
        
        invalid_samples = []
        for i, sample in enumerate(self.samples):
            try:
                image = cv2.imread(str(sample.image_path))
                mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    logger.warning(f"Cannot read image: {sample.image_path}")
                    invalid_samples.append(i)
                    continue
                
                if mask is None:
                    logger.warning(f"Cannot read mask: {sample.mask_path}")
                    invalid_samples.append(i)
                    continue
                
                # Check for corrupted/empty images
                if image.size == 0 or mask.size == 0:
                    logger.warning(f"Empty image or mask: {sample.image_path}")
                    invalid_samples.append(i)
                    continue
                
                # Check for uniform images (likely corrupted)
                if np.std(image) < 1.0:
                    logger.warning(f"Image has very low variance (possibly corrupted): {sample.image_path}")
                    invalid_samples.append(i)
                    continue
                    
            except Exception as e:
                logger.warning(f"Error validating sample {sample.image_path}: {e}")
                invalid_samples.append(i)
        
        # Remove invalid samples
        if invalid_samples:
            logger.info(f"Removing {len(invalid_samples)} invalid samples from {self.split} set")
            self.samples = [s for i, s in enumerate(self.samples) if i not in invalid_samples]
        
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {self.split} set")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        try:
            image = cv2.cvtColor(cv2.imread(str(sample.image_path)), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                raise RuntimeError(f"Failed reading sample: {sample}")
            
            mask = (mask > 127).astype(np.float32)
            transformed = self.transforms(image=image, mask=mask)
            image_t = transformed["image"].float()
            mask_t = transformed["mask"].unsqueeze(0).float()
            return image_t, mask_t
        except Exception as e:
            import logging
            logger = logging.getLogger("crackgraphai.dataset")
            logger.error(f"Error loading sample {idx}: {e}")
            raise
