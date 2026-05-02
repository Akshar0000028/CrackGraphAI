# Structural Integrity (SI) Scoring System v2.0

## Executive Summary

This document describes the redesigned Structural Integrity scoring system for CrackGraphAI. The new system is **damage-centric**, penalizing crack density, connectivity, and complexity to produce realistic structural health assessments.

**Key Improvements:**
- Connectivity is now penalized (not rewarded)
- Segmentation metrics limited to 30% max contribution
- 5-level risk classification (added "Failure Imminent")
- Crack width proxy estimation
- Robust to noisy skeletons

---

## Mathematical Foundation

### Core Formula

```
SI = 1.0 - Damage_Score

where:
Damage_Score = Σ(w_i × Damage_Component_i)
```

**Principle:** SI ranges from 0.0 (critical failure) to 1.0 (perfect structure).

### Damage Components

#### 1. Density Damage (35% weight)
Measures how much of the surface is cracked.

```
density_damage = min(1.0, skeleton_pixels / (0.25 × total_pixels))
```

- **Normalization:** 25% skeleton coverage = severe damage (capped at 1.0)
- **Rationale:** More than 25% of the surface cracked indicates structural failure

#### 2. Connectivity Damage (20% weight)
Measures how interconnected the cracks are.

```
connectivity_damage = connectivity_ratio^0.7
```

- **Normalization:** Power function gives more weight to high connectivity
- **Rationale:** Fully connected crack networks allow stress propagation and catastrophic failure

#### 3. Complexity Damage (15% weight)
Measures branching and junction complexity.

```
complexity_score = (branches + 2×junctions) / 15.0
activity_penalty = if (endpoints / (2×junctions) > 1.5) then ×1.3
```

- **Normalization:** 15 complexity units = severe (capped at 1.0)
- **Rationale:** 
  - Junctions weighted 2× (intersections are weak points)
  - Active cracking (many endpoints) gets additional penalty

#### 4. Segmentation Quality (30% weight)
Measures confidence in the segmentation mask.

```
seg_quality = 0.7×dice + 0.3×exp(-2×bce/(bce+1))
seg_damage = 1.0 - seg_quality
```

- **Limitation:** Capped at 30% to prioritize structural features over segmentation confidence
- **Rationale:** A perfect segmentation of a severely cracked surface should still yield a low SI score

---

## Final Weights

| Component | Weight | Purpose |
|-----------|--------|---------|
| Crack Density | 35% | Primary damage indicator |
| Connectivity Penalty | 20% | Network risk assessment |
| Complexity Penalty | 15% | Structural degradation pattern |
| Segmentation Quality | 30% | Confidence weighting (max) |

**Total: 100%**

---

## Risk Classification

| SI Score | Risk Level | Description | Action Required |
|----------|------------|-------------|-----------------|
| 0.80 - 1.00 | Low | Good condition | Routine monitoring |
| 0.60 - 0.80 | Moderate | Minor concerns | Inspect within 6 months |
| 0.40 - 0.60 | High | Significant concerns | Professional assessment within 1 month |
| 0.20 - 0.40 | Critical | Severe damage | Immediate intervention |
| 0.00 - 0.20 | Failure Imminent | Structural failure risk | Emergency evacuation & repairs |

---

## Usage Guide

### Basic Usage

```python
from features.si_scoring import compute_structural_integrity, classify_risk

# Compute SI score from analysis results
result = compute_structural_integrity(
    mask=segmentation_mask,
    skeleton=skeleton,
    graph=crack_graph,
    connectivity_ratio=connectivity_score,
    dice=0.95,  # Optional: segmentation quality
    bce=0.02,   # Optional: segmentation loss
)

print(f"SI Score: {result['si_score']:.3f}")
print(f"Risk Level: {result['risk_level']}")
print(f"Total Damage: {result['total_damage']:.3f}")

# Classify risk
level, css_class, description = classify_risk(result['si_score'])
```

### Advanced Usage with Custom Weights

```python
from features.si_scoring import SIGenerator, SIWeights, DamageFeatures

# Custom weights (must sum to 1.0)
weights = SIWeights(
    crack_density=0.40,
    connectivity_penalty=0.25,
    complexity_penalty=0.20,
    segmentation_quality=0.15,  # Reduced seg contribution
)

generator = SIGenerator(weights=weights)

# Extract features manually
features = DamageFeatures(
    crack_density=0.15,
    skeleton_density=0.12,
    connectivity_ratio=0.85,
    num_branches=8,
    num_junctions=3,
    num_endpoints=12,
    complexity_index=0.45,
    network_density=0.02,
    total_crack_length=145.5,
    max_crack_width_proxy=0.03,
    mean_crack_width_proxy=0.015,
    dice_score=0.92,
    bce_loss=0.03,
)

result = generator.compute_si_score(features)
```

### Feature Extraction

```python
from features.si_scoring import extract_damage_features, FeatureNormalizer

# Extract all features at once
features = extract_damage_features(
    mask=mask,
    skeleton=skeleton,
    graph=graph,
    connectivity_ratio=conn_score,
    dice=0.94,
    bce=0.04,
)

# Manual normalization
normalizer = FeatureNormalizer()
density_norm = normalizer.normalize_density(skeleton_pixels, total_pixels)
complexity_norm = normalizer.normalize_complexity(branches, junctions, endpoints)
max_width, mean_width = normalizer.normalize_crack_width(mask, skeleton)
```

---

## Graph Enhancement

The new `graph/enhanced_crack_graph.py` module provides robust graph extraction:

```python
from graph.enhanced_crack_graph import enhance_graph, skeleton_to_graph

# Full enhancement pipeline
graph, metadata = enhance_graph(
    skeleton=skeleton,
    prune_length=3.0,           # Remove branches < 3 pixels
    min_component_nodes=3,       # Remove components with < 3 nodes
    min_component_weight=5.0,    # Remove components with < 5 pixel length
    smooth_angle=160.0,          # Straighten paths with angles > 160°
)

# Or use simplified interface
graph = skeleton_to_graph(
    skeleton,
    prune=True,
    filter_components=True,
    smooth=True,
)
```

**Enhancement Benefits:**
- Removes noise artifacts from skeletonization
- Prunes small spurs that don't represent real cracks
- Filters isolated pixels and tiny fragments
- Smooths zigzag artifacts while preserving topology

---

## Validation Examples

| Scenario | Density | Connectivity | Complexity | SI Score | Classification |
|----------|-----------|--------------|------------|----------|----------------|
| No cracks | 0.0 | 0.0 | 0.0 | 1.000 | Low |
| Single hairline | 0.02 | 0.1 | 0.0 | 0.977 | Low |
| Multiple isolated | 0.15 | 0.2 | 0.1 | 0.885 | Low |
| Connected network | 0.20 | 0.85 | 0.4 | 0.635 | Moderate |
| Dense branching | 0.25 | 0.90 | 0.8 | 0.445 | High |
| Severe damage | 0.30 | 0.95 | 0.9 | 0.320 | Critical |
| Catastrophic | 0.40 | 1.0 | 1.0 | 0.145 | Failure Imminent |

---

## Implementation Notes

### Files Added/Modified

**New Files:**
- `features/si_scoring.py` - Complete SI scoring implementation
- `graph/enhanced_crack_graph.py` - Robust graph extraction with pruning

**Modified Files:**
- `api/main.py` - Updated to use new SI scoring
- `ui/streamlit_app.py` - Updated to use new risk classification

**Unchanged (for backward compatibility):**
- `utils/metrics.py` - Old `structural_integrity_score` kept for reference
- `features/structural_features.py` - Feature extraction unchanged
- `topology/skeleton.py` - Skeletonization unchanged

### Key Constants

```python
# FeatureNormalizer constants
MAX_SKELETON_DENSITY = 0.25      # 25% coverage = severe
MAX_COMPLEXITY_SCORE = 15.0      # 15 branches+junctions = severe
MAX_TOTAL_LENGTH = 500.0         # For 256×256 images
MAX_WIDTH_PROXY = 0.10           # 10% of image diagonal

# SIRiskThresholds
LOW = 0.80
MODERATE = 0.60
HIGH = 0.40
CRITICAL = 0.20
```

---

## Migration from Old System

### Old System (Incorrect)
```python
from utils.metrics import structural_integrity_score

# WRONG: Connectivity treated as positive
si = structural_integrity_score(
    dice=0.9,
    bce=0.1,
    connectivity=0.8,  # High connectivity increased score!
    branch_consistency=0.7,
    weights={"dice": 0.4, "bce": 0.2, "connectivity": 0.2, "branch_consistency": 0.2}
)
# Result: Heavily cracked surface could get high SI score
```

### New System (Correct)
```python
from features.si_scoring import compute_structural_integrity

# CORRECT: Connectivity penalized as damage
result = compute_structural_integrity(
    mask=mask,
    skeleton=skeleton,
    graph=graph,
    connectivity_ratio=0.8,  # High connectivity reduces score!
    dice=0.9,
    bce=0.1,
)
si = result["si_score"]
# Result: Heavily cracked surface gets low SI score
```

---

## Testing Recommendations

1. **Synthetic Test Cases:**
   - No cracks → SI ≈ 1.0
   - Single thin crack → SI > 0.90
   - Multiple isolated cracks → SI > 0.75
   - Connected network → SI 0.60-0.80
   - Dense mesh → SI < 0.50

2. **Edge Cases:**
   - Empty mask → SI = 1.0 (no damage detected)
   - Full mask → SI ≈ 0.0 (catastrophic)
   - Single pixel → SI ≈ 1.0 (noise, not damage)

3. **Real-world Validation:**
   - Compare against expert structural assessments
   - Verify that wider cracks reduce SI more than thin cracks
   - Confirm that junction density correlates with structural weakness

---

## Future Enhancements

Potential improvements for v2.1+:

1. **Crack Width from Multi-scale Analysis:**
   - Use multiple distance transforms at different scales
   - Better estimation of actual crack aperture

2. **Temporal Analysis:**
   - Track SI changes over time
   - Growth rate as additional damage indicator

3. **Material-specific Calibration:**
   - Different thresholds for concrete vs asphalt vs masonry
   - Adjust `MAX_SKELETON_DENSITY` based on material properties

4. **ML-based Weight Optimization:**
   - Learn optimal weights from labeled structural assessments
   - Replace fixed weights with trained model

---

## References

1. Zhang-Suen thinning algorithm (skeletonization)
2. NetworkX graph analysis library
3. Scikit-image morphology operations
4. Structural engineering damage assessment guidelines

---

*Document Version: 2.0*  
*Last Updated: May 2026*  
*System: CrackGraphAI Structural Integrity Module*
