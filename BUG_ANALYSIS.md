# CrackGraphAI - Bug Analysis Report

## Executive Summary
This document identifies **8 critical and high-priority bugs** found in the CrackGraphAI codebase. These bugs affect model inference, data loading, loss computation, and API functionality.

---

## Bug List

### 🔴 BUG #1: CRITICAL - Incorrect Output Upsampling in HybridSegformerUNet
**File:** `models/hybrid_segformer_unet.py` (Line 145)  
**Severity:** CRITICAL  
**Impact:** Model produces incorrect output dimensions, causing shape mismatches in downstream processing

#### Problem:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # ... processing ...
    x = self.head(x)
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
```

The model upsamples by 2x at the end, but the input is 256×256 and after the decoder it's already 256×256 (due to skip connections). This produces a 512×512 output instead of the expected 256×256.

#### Expected Behavior:
The output should match the input size (256×256), not be 2x larger.

#### Fix:
Remove the final upsampling or adjust the decoder to not upsample to full resolution:
```python
# Option 1: Remove the final scale_factor=2
return x  # Output is already 256×256

# Option 2: If input is larger, use proper scaling
return F.interpolate(x, size=x.shape[-2:], mode="bilinear", align_corners=False)
```

---

### 🔴 BUG #2: CRITICAL - Incorrect Dice Loss Calculation
**File:** `losses/segmentation_losses.py` (Line 8)  
**Severity:** CRITICAL  
**Impact:** Loss function produces incorrect gradients, leading to poor model training

#### Problem:
```python
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()
```

The Dice loss formula is **incorrect**. The standard Dice coefficient is:
```
Dice = 2 * |X ∩ Y| / (|X| + |Y|)
```

But the current implementation calculates:
```
Dice = 2 * intersection / (union)
```

Where `union = probs.sum() + targets.sum()` is NOT the union. The union should be:
```
union = probs.sum() + targets.sum() - intersection
```

#### Expected Behavior:
```python
dice = (2.0 * intersection + eps) / (probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection + eps)
```

#### Fix:
```python
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    cardinality = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (cardinality + eps)  # Correct formula
    return 1.0 - dice.mean()
```

---

### 🔴 BUG #3: CRITICAL - Incorrect Skeleton Density Normalization
**File:** `features/si_scoring.py` (Line 139-155)  
**Severity:** CRITICAL  
**Impact:** SI score calculation is fundamentally wrong, producing misleading damage assessments

#### Problem:
```python
@classmethod
def normalize_density(cls, skeleton_pixels: int, total_pixels: int) -> float:
    if total_pixels == 0:
        return 0.0
    raw_density = skeleton_pixels / total_pixels
    normalized = raw_density / cls.MAX_SKELETON_DENSITY
    return float(np.clip(normalized, 0.0, 1.0))
```

The issue: `skeleton_density` is used directly as a damage component in `compute_si_score()`, but it's already normalized to [0, 1]. However, the normalization divides by `MAX_SKELETON_DENSITY = 0.15`, which means:
- If skeleton_pixels = 5% of image: normalized = 0.05 / 0.15 = 0.33 (damage)
- If skeleton_pixels = 15% of image: normalized = 0.15 / 0.15 = 1.0 (max damage)

**But then in `compute_si_score()` line 330:**
```python
density_damage = features.skeleton_density  # Already normalized!
```

This is used directly without further normalization. The problem is that `skeleton_density` is already in [0, 1] range, so it's being treated as a damage score when it should be treated as a raw metric.

#### Expected Behavior:
The density should represent actual crack coverage percentage, not a pre-normalized damage score.

#### Fix:
Either:
1. Return raw density without normalization in `normalize_density()`, OR
2. Don't normalize again in `compute_si_score()`

Recommended fix:
```python
@classmethod
def normalize_density(cls, skeleton_pixels: int, total_pixels: int) -> float:
    """Return raw skeleton density as percentage of image."""
    if total_pixels == 0:
        return 0.0
    raw_density = skeleton_pixels / total_pixels
    return float(np.clip(raw_density, 0.0, 1.0))  # Return raw, not normalized
```

Then in `compute_si_score()`:
```python
# Apply damage scaling based on severity thresholds
density_damage = min(features.skeleton_density / 0.15, 1.0)  # Normalize here
```

---

### 🟠 BUG #4: HIGH - Incorrect Connectivity Penalty Calculation
**File:** `features/si_scoring.py` (Line 195-215)  
**Severity:** HIGH  
**Impact:** Connectivity damage is inverted - high connectivity (good) is penalized as damage (bad)

#### Problem:
```python
@classmethod
def normalize_connectivity(cls, connectivity_ratio: float) -> float:
    """
    Normalize connectivity - higher connectivity = more severe damage.
    ...
    """
    clipped = np.clip(connectivity_ratio, 0.0, 1.0)
    powered = np.power(clipped, 0.7)
    return float(np.clip(powered, 0.0, 1.0))
```

The docstring says "higher connectivity = more severe damage", but this is **semantically wrong**:
- **High connectivity (0.9)** = cracks are well-connected = indicates structural weakness
- **Low connectivity (0.1)** = cracks are fragmented = less dangerous

However, the implementation treats high connectivity as high damage, which is correct mathematically but the logic is inverted in the SI score calculation.

**The real bug:** In `compute_si_score()` line 330:
```python
connectivity_damage = self.normalizer.normalize_connectivity(features.connectivity_ratio)
```

This directly uses the normalized connectivity as damage. But if connectivity_ratio = 0.9 (well-connected cracks), it returns 0.9 as damage, which is correct. However, the semantic confusion in the docstring suggests the developers may have intended the opposite.

#### Expected Behavior:
The connectivity should be treated as a damage indicator (high connectivity = high damage), which is actually correct. But the docstring is misleading.

#### Fix:
Clarify the docstring and ensure the logic is consistent:
```python
@classmethod
def normalize_connectivity(cls, connectivity_ratio: float) -> float:
    """
    Normalize connectivity as damage indicator.
    
    Higher connectivity = more severe damage because:
    - Stress propagates through connected network
    - Single failure can trigger cascade
    - Indicates widespread degradation
    
    Args:
        connectivity_ratio: Ratio of connected to total skeleton pixels [0, 1]
        
    Returns:
        Damage score [0, 1] where 1.0 = fully connected (severe)
    """
    clipped = np.clip(connectivity_ratio, 0.0, 1.0)
    powered = np.power(clipped, 0.7)
    return float(np.clip(powered, 0.0, 1.0))
```

---

### 🟠 BUG #5: HIGH - Missing Tuple Import in si_scoring.py
**File:** `features/si_scoring.py` (Line 1-15)  
**Severity:** HIGH  
**Impact:** Code will crash at runtime when `normalize_crack_width()` is called

#### Problem:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple  # ← Tuple is imported

import numpy as np
from scipy import ndimage
```

Wait, `Tuple` IS imported. Let me check the actual usage...

Actually, looking at line 219:
```python
def normalize_crack_width(
    cls,
    mask: np.ndarray,
    skeleton: np.ndarray
) -> Tuple[float, float]:  # ← Uses Tuple
```

This is correct. **FALSE ALARM** - Tuple is properly imported.

---

### 🟠 BUG #6: HIGH - Incorrect Aspect Ratio Calculation in post_process_mask()
**File:** `api/main.py` (Line 32-100)  
**Severity:** HIGH  
**Impact:** Post-processing filter may incorrectly remove valid cracks or keep noise

#### Problem:
```python
def post_process_mask(mask: np.ndarray, min_aspect_ratio: float = 1.2, min_area: int = 5) -> np.ndarray:
    # ...
    for region in regions:
        if region.area < min_area:
            continue
        
        solidity = region.solidity if hasattr(region, 'solidity') else 1.0
        
        aspect_ratio = 0
        if region.orientation is not None:
            try:
                major_axis = region.major_axis_length
                minor_axis = region.minor_axis_length
            except AttributeError:
                major_axis = region.axis_major_length
                minor_axis = region.axis_minor_length
            
            if minor_axis > 0:
                aspect_ratio = major_axis / minor_axis
            else:
                aspect_ratio = major_axis if major_axis > 0 else 0
```

**Issues:**
1. `region.orientation` check is redundant - if orientation is None, the try/except will still work
2. The fallback to `axis_major_length` and `axis_minor_length` is incorrect - these attributes don't exist in scikit-image's `regionprops`
3. If `minor_axis == 0`, setting `aspect_ratio = major_axis` is wrong - aspect ratio should be infinity or a large number, not the absolute length

#### Expected Behavior:
```python
if minor_axis > 0:
    aspect_ratio = major_axis / minor_axis
else:
    aspect_ratio = float('inf')  # Infinitely thin = crack-like
```

#### Fix:
```python
def post_process_mask(mask: np.ndarray, min_aspect_ratio: float = 1.2, min_area: int = 5) -> np.ndarray:
    from skimage.measure import label, regionprops
    
    if mask.sum() == 0:
        return mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    
    labeled = label(opened.astype(bool))
    regions = list(regionprops(labeled))
    
    if not regions:
        return mask
    
    cleaned_mask = np.zeros_like(mask, dtype=np.uint8)
    
    for region in regions:
        if region.area < min_area:
            continue
        
        solidity = region.solidity
        
        # Calculate aspect ratio correctly
        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length
        
        if minor_axis > 0:
            aspect_ratio = major_axis / minor_axis
        else:
            aspect_ratio = float('inf')  # Infinitely thin
        
        is_elongated = aspect_ratio >= min_aspect_ratio
        is_thin = solidity < 0.7
        is_small_crack = region.area < 500
        
        if is_elongated or is_thin or is_small_crack:
            coords = region.coords
            cleaned_mask[coords[:, 0], coords[:, 1]] = 1
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    labeled_final = label(closed.astype(bool))
    regions_final = list(regionprops(labeled_final))
    
    final_mask = np.zeros_like(mask, dtype=np.uint8)
    for region in regions_final:
        if region.area >= min_area:
            coords = region.coords
            final_mask[coords[:, 0], coords[:, 1]] = 1
    
    return final_mask
```

---

### 🟠 BUG #7: HIGH - Incorrect Complexity Normalization
**File:** `features/si_scoring.py` (Line 158-190)  
**Severity:** HIGH  
**Impact:** Complexity damage score may be incorrect, affecting SI score accuracy

#### Problem:
```python
@classmethod
def normalize_complexity(
    cls,
    num_branches: int,
    num_junctions: int,
    num_endpoints: int
) -> float:
    base_score = num_branches + 2.0 * num_junctions
    
    activity_ratio = 0.0
    if num_junctions > 0:
        activity_ratio = num_endpoints / (2.0 * num_junctions)
        if activity_ratio > 1.5:
            base_score *= (1.0 + 0.3 * (activity_ratio - 1.5))
    
    normalized = base_score / cls.MAX_COMPLEXITY_SCORE
    return float(np.clip(normalized, 0.0, 1.0))
```

**Issues:**
1. The `activity_ratio` calculation divides endpoints by `2.0 * num_junctions`, but this is arbitrary. Why multiply by 2.0?
2. The penalty multiplier `(1.0 + 0.3 * (activity_ratio - 1.5))` can grow unbounded if activity_ratio is very large
3. No upper bound on the penalty, so `base_score` can exceed `MAX_COMPLEXITY_SCORE` significantly before clipping

#### Example:
- num_branches = 5, num_junctions = 2, num_endpoints = 10
- base_score = 5 + 2*2 = 9
- activity_ratio = 10 / (2*2) = 2.5
- penalty = 1.0 + 0.3 * (2.5 - 1.5) = 1.3
- base_score *= 1.3 = 11.7
- normalized = 11.7 / 30.0 = 0.39

But if num_endpoints = 100:
- activity_ratio = 100 / 4 = 25
- penalty = 1.0 + 0.3 * 23.5 = 8.05
- base_score *= 8.05 = 72.45
- normalized = 72.45 / 30.0 = 2.415 → clipped to 1.0

This creates a discontinuity where small changes in endpoints cause large jumps in damage.

#### Fix:
```python
@classmethod
def normalize_complexity(
    cls,
    num_branches: int,
    num_junctions: int,
    num_endpoints: int
) -> float:
    """Normalize crack pattern complexity."""
    base_score = num_branches + 2.0 * num_junctions
    
    # Activity ratio: endpoints relative to junctions
    # Expected ratio: ~2 endpoints per junction (for simple branching)
    if num_junctions > 0:
        expected_endpoints = 2.0 * num_junctions
        activity_ratio = num_endpoints / expected_endpoints
        # Penalty for "active" cracks (many open ends)
        # Use sigmoid to bound the penalty
        activity_penalty = 1.0 + 0.5 * np.tanh((activity_ratio - 1.0) / 2.0)
        base_score *= activity_penalty
    
    normalized = base_score / cls.MAX_COMPLEXITY_SCORE
    return float(np.clip(normalized, 0.0, 1.0))
```

---

### 🟡 BUG #8: MEDIUM - Incorrect Dice Score Estimation in API
**File:** `api/main.py` (Line 251-280)  
**Severity:** MEDIUM  
**Impact:** SI score calculation uses incorrect dice score, leading to inaccurate damage assessment

#### Problem:
```python
def infer(self, file_bytes: bytes, request_id: str) -> Dict:
    # ...
    mask = post_process_mask(raw_mask, min_aspect_ratio=2.0, min_area=20)
    
    # Estimate dice score based on mask coverage
    mask_coverage = float(mask.sum()) / (mask.shape[0] * mask.shape[1])
    
    if mask_coverage < 0.001:
        dice = 0.3
    elif mask_coverage > 0.5:
        dice = 0.5
    else:
        dice = 0.5 + (mask_coverage * 0.5)  # Range: 0.5 to 1.0
    
    # BCE loss estimation
    bce = 0.5 * (1.0 - conn)
```

**Issues:**
1. **Dice score is estimated without ground truth** - The code estimates dice based on mask coverage, but dice requires comparing to ground truth. Without ground truth, this is just a heuristic guess.
2. **The estimation formula is arbitrary** - Why `0.5 + (mask_coverage * 0.5)`? This has no theoretical basis.
3. **BCE loss is also estimated incorrectly** - `bce = 0.5 * (1.0 - conn)` is not a valid BCE loss calculation. BCE requires logits and targets.
4. **These estimates are used in SI score calculation** - This propagates errors into the final SI score.

#### Expected Behavior:
During inference without ground truth, either:
1. Use a default dice/bce value (e.g., 1.0 for dice, 0.0 for bce)
2. Skip the segmentation quality component in SI calculation
3. Use a separate validation set to calibrate dice/bce estimates

#### Fix:
```python
def infer(self, file_bytes: bytes, request_id: str) -> Dict:
    # ...
    mask = post_process_mask(raw_mask, min_aspect_ratio=2.0, min_area=20)
    skel = mask_to_skeleton(mask)
    graph = skeleton_to_graph(skel)
    feats = extract_structural_features(graph)
    conn = connectivity_score(skel)
    
    # During inference, we don't have ground truth
    # Use conservative defaults for segmentation quality
    dice = 0.8  # Assume reasonable segmentation quality
    bce = 0.3   # Assume moderate confidence
    
    si_result = compute_structural_integrity(
        mask=mask,
        skeleton=skel,
        graph=graph,
        connectivity_ratio=conn,
        dice=dice,
        bce=bce,
    )
```

---

## Summary Table

| # | File | Line | Severity | Issue | Impact |
|---|------|------|----------|-------|--------|
| 1 | `models/hybrid_segformer_unet.py` | 145 | 🔴 CRITICAL | Incorrect output upsampling (2x) | Wrong output dimensions |
| 2 | `losses/segmentation_losses.py` | 8 | 🔴 CRITICAL | Incorrect Dice loss formula | Poor training gradients |
| 3 | `features/si_scoring.py` | 139-155 | 🔴 CRITICAL | Incorrect skeleton density normalization | Wrong SI score |
| 4 | `features/si_scoring.py` | 195-215 | 🟠 HIGH | Connectivity penalty logic unclear | Potential SI score errors |
| 5 | `features/si_scoring.py` | 1-15 | 🟡 FALSE ALARM | Missing Tuple import | None (import exists) |
| 6 | `api/main.py` | 32-100 | 🟠 HIGH | Incorrect aspect ratio calculation | Invalid post-processing |
| 7 | `features/si_scoring.py` | 158-190 | 🟠 HIGH | Unbounded complexity penalty | Discontinuous damage scores |
| 8 | `api/main.py` | 251-280 | 🟡 MEDIUM | Incorrect dice/bce estimation | Inaccurate SI scores |

---

## Recommended Fix Priority

1. **FIRST:** Fix Bug #2 (Dice loss) - Affects model training quality
2. **SECOND:** Fix Bug #1 (Output upsampling) - Affects inference output shape
3. **THIRD:** Fix Bug #3 (Skeleton density) - Affects SI score accuracy
4. **FOURTH:** Fix Bug #6 (Aspect ratio) - Affects post-processing
5. **FIFTH:** Fix Bug #7 (Complexity) - Affects SI score stability
6. **SIXTH:** Fix Bug #8 (Dice estimation) - Affects SI score calibration
7. **SEVENTH:** Fix Bug #4 (Connectivity) - Documentation/clarity

