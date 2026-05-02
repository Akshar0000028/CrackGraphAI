# CrackGraphAI: Production-Grade Crack Segmentation and Structural Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**CrackGraphAI** is an end-to-end deep learning pipeline for automated crack detection and structural integrity assessment in infrastructure images. It combines state-of-the-art computer vision techniques with graph-based structural analysis to provide comprehensive crack characterization.

## Pipeline Overview

```
Input Image
    ↓
Preprocessing (Resize, Normalize, Augment)
    ↓
Hybrid CNN+Transformer Segmentation (ResNet34 + SegFormer)
    ↓
Topology-Constrained Learning (Dice + BCE + Topology Loss)
    ↓
Skeletonization (Morphological Thinning)
    ↓
Graph Modeling (NetworkX Conversion)
    ↓
Structural Feature Extraction
    ↓
Structural Integrity (SI) Score Calculation
```

## Key Features

- **Hybrid Architecture**: Combines ResNet CNN with SegFormer transformer for superior segmentation
- **Topology-Aware Training**: Custom loss function preserving crack connectivity
- **Graph-Based Analysis**: Converts cracks to graph structures for topological analysis
- **Production-Ready API**: FastAPI backend with rate limiting, auth, and monitoring
- **Modern UI**: Interactive Streamlit dashboard with real-time visualizations
- **Docker Support**: Containerized deployment with GPU support
- **Model Export**: TorchScript and ONNX export for edge deployment
- **Comprehensive Metrics**: IoU, Dice, Precision, Recall, Connectivity, SI Score

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Inference](#inference)
- [API & UI](#api--ui)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Architecture Details](#architecture-details)
- [API Reference](#api-reference)
- [Monitoring & Metrics](#monitoring--metrics)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Docker (optional, for containerized deployment)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/CrackGraphAI.git
cd CrackGraphAI
```

2. **Create virtual environment:**
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment (optional):**
```bash
cp .env.example .env
# Edit .env with your production settings
```

## Quick Start

### 1. Prepare Dataset

Place your crack segmentation dataset in the following structure:

```
crack_segmentation_dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── masks/
    ├── image_001.png
    ├── image_002.png
    └── ...
```

### 2. Train the Model

```bash
# Train the hybrid model (recommended)
python scripts/train.py --config configs/config.yaml --model hybrid

# Train baseline models for comparison
python scripts/train.py --config configs/config.yaml --model unetpp
python scripts/train.py --config configs/config.yaml --model deeplabv3plus
python scripts/train.py --config configs/config.yaml --model segformer
```

### 3. Run Inference

```bash
python scripts/infer.py \
    --config configs/config.yaml \
    --weights checkpoints/best_hybrid_segformer.pth \
    --image path/to/image.jpg \
    --output-dir outputs
```

### 4. Start API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 5. Launch Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
CrackGraphAI/
├── api/                      # FastAPI production server
│   └── main.py              # API endpoints and inference service
├── configs/                  # Configuration files
│   └── config.yaml          # Main configuration
├── data/                     # Data loading and preprocessing
│   └── dataset.py           # Dataset class and augmentations
├── docker-compose.yml        # Docker orchestration
├── Dockerfile                # Container definition
├── features/                 # Structural feature extraction
│   └── structural_features.py
├── graph/                    # Graph modeling
│   └── crack_graph.py       # Skeleton-to-graph conversion
├── losses/                   # Custom loss functions
│   └── segmentation_losses.py
├── models/                   # Model architectures
│   ├── baselines.py         # U-Net++, DeepLabV3+, SegFormer
│   └── hybrid_segformer_unet.py
├── scripts/                  # Training and inference scripts
│   ├── benchmark_models.py
│   ├── eval_onnx.py
│   ├── export_model.py
│   ├── infer.py
│   └── train.py
├── tests/                    # Unit tests
│   ├── test_api.py
│   └── test_metrics.py
├── topology/                 # Skeletonization
│   └── skeleton.py
├── training/                 # Training loop
│   └── trainer.py
├── ui/                       # Streamlit interface
│   └── streamlit_app.py
└── utils/                    # Utilities
    ├── config.py
    ├── metrics.py
    └── repro.py
```

## Dataset Setup

The project expects a standard image segmentation dataset format. You can use public datasets like:

- CrackTree (Structural crack dataset)
- DeepCrack
- CRACK500
- Custom datasets

### Dataset Configuration

Update `configs/config.yaml`:

```yaml
data:
  root_dir: crack_segmentation_dataset
  image_dir_name: images
  mask_dir_name: masks
  image_size: 256              # Input image size
  train_ratio: 0.70            # Training split
  val_ratio: 0.15              # Validation split
  test_ratio: 0.15             # Test split
  num_workers: 4               # DataLoader workers
```

## Training

### Training Configuration

Edit `configs/config.yaml`:

```yaml
training:
  batch_size: 8
  epochs: 80
  lr: 0.0002                   # Learning rate
  weight_decay: 0.0001         # AdamW weight decay
  amp: true                    # Automatic Mixed Precision
  early_stopping_patience: 12  # Early stopping patience
  checkpoint_dir: checkpoints
  best_model_name: best_hybrid_segformer.pth
  lambda_topology: 0.2         # Topology loss weight
```

### Model Options

| Model | Architecture | Description |
|-------|--------------|-------------|
| `hybrid` | ResNet34 + SegFormer | **Recommended** - Best accuracy |
| `unetpp` | U-Net++ | Baseline with nested skip connections |
| `deeplabv3plus` | DeepLabV3+ | Atrous convolution based |
| `segformer` | SegFormer B0 | Pure transformer encoder |

### Training from Scratch

```bash
python scripts/train.py \
    --config configs/config.yaml \
    --model hybrid \
    --resume  # Optional: resume from checkpoint
```

### Expected Results

| Model | IoU | Dice | Precision | Recall | SI Score |
|-------|-----|------|-----------|--------|----------|
| U-Net++ | ~0.72 | ~0.84 | ~0.78 | ~0.81 | ~0.75 |
| DeepLabV3+ | ~0.75 | ~0.86 | ~0.80 | ~0.83 | ~0.78 |
| SegFormer | ~0.78 | ~0.88 | ~0.82 | ~0.85 | ~0.81 |
| **Hybrid (Final)** | **~0.82** | **~0.90** | **~0.85** | **~0.88** | **~0.85** |

## Inference

### Single Image Inference

```bash
python scripts/infer.py \
    --config configs/config.yaml \
    --weights checkpoints/best_hybrid_segformer.pth \
    --image path/to/image.jpg \
    --output-dir outputs \
    --visualize
```

### Batch Inference

```bash
python scripts/infer.py \
    --config configs/config.yaml \
    --weights checkpoints/best_hybrid_segformer.pth \
    --image-dir path/to/images/ \
    --output-dir outputs
```

### Output Files

- `pred_mask.npy` - Binary segmentation mask
- `skeleton.npy` - Thinned crack skeleton
- `visualization.png` - Overlay visualization
- `features.json` - Extracted structural features
- `si_score.txt` - Structural integrity score

### TTA (Test-Time Augmentation)

Enable in `configs/config.yaml`:

```yaml
inference:
  use_tta: true
  tta_transforms: [none, hflip, vflip]
```

## API & UI

### FastAPI Production Server

The API provides RESTful endpoints for production deployment.

**Start the server:**

```bash
# Development
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@crack_image.jpg" \
  -H "Authorization: Bearer your-api-key"
```

**Batch Prediction:**
```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -F "images=@crack1.jpg" \
  -F "images=@crack2.jpg" \
  -H "Authorization: Bearer your-api-key"
```

### Streamlit Web Interface

The modern React-style dashboard provides:
- Drag-and-drop image upload
- Real-time analysis with progress indicators
- Interactive visualizations with Plotly
- SI Score gauge charts
- Feature analysis charts
- Download results (PNG, JSON)
- Analysis history tracking

**Launch:**
```bash
streamlit run ui/streamlit_app.py
```

Access at http://localhost:8501

## Deployment

### Docker (Recommended)

**Build and run:**
```bash
# Basic deployment
docker-compose up -d

# With monitoring stack (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# With Redis caching
docker-compose --profile cache up -d
```

**Manual Docker build:**
```bash
docker build -t crackgraphai:latest .
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -e API_KEY=your-secure-key \
  crackgraphai:latest
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `dev-key-change-in-production` | API authentication key |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CONFIG_PATH` | `configs/config.yaml` | Model config path |
| `WEIGHTS_PATH` | `checkpoints/best_hybrid_segformer.pth` | Model weights |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device selection |
| `REDIS_URL` | - | Optional Redis cache |
| `DATABASE_URL` | - | Optional PostgreSQL storage |

### Kubernetes

Apply the provided manifests:

```bash
kubectl apply -f k8s/

# Or use Helm
helm install crackgraphai ./helm-chart
```

### Model Export for Edge Deployment

Export to production formats:

```bash
python scripts/export_model.py \
    --config configs/config.yaml \
    --weights checkpoints/best_hybrid_segformer.pth
```

Outputs:
- `checkpoints/hybrid_segformer.ts` - TorchScript (PyTorch runtime)
- `checkpoints/hybrid_segformer.onnx` - ONNX (cross-platform)

**Run ONNX inference:**
```bash
python scripts/eval_onnx.py \
    --onnx checkpoints/hybrid_segformer.onnx \
    --image path/to/image.jpg
```

## Configuration

The project uses a centralized YAML configuration in `configs/config.yaml`:

```yaml
project:
  name: crackgraphai
  seed: 42                     # Reproducibility seed

data:
  root_dir: crack_segmentation_dataset
  image_size: 256
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

training:
  batch_size: 8
  epochs: 80
  lr: 0.0002
  weight_decay: 0.0001
  amp: true                    # Mixed precision training
  early_stopping_patience: 12

model:
  architecture: hybrid_segformer_unet
  cnn_backbone: resnet34
  transformer_backbone: mit_b0
  in_channels: 3
  classes: 1
  decoder_channels: [256, 128, 64, 32]

inference:
  threshold: 0.5
  use_tta: true                # Test-time augmentation
  tta_transforms: [none, hflip, vflip]

si_score:
  weights:
    dice: 0.35
    bce: 0.15
    connectivity: 0.30
    branch_consistency: 0.20
```

## Architecture Details

### Hybrid CNN+Transformer Architecture

The core innovation of CrackGraphAI is the **HybridSegformerUNet** (`models/hybrid_segformer_unet.py:87`), which combines the strengths of CNNs (local feature extraction) and Transformers (global context modeling).

```
Input (3×256×256)
    ↓
┌─────────────────────────┐     ┌─────────────────────────┐
│     CNN BRANCH          │     │   TRANSFORMER BRANCH    │
│     (ResNet34)          │     │   (SegFormer B0)        │
│                         │     │                         │
│  Input: 3×256×256       │     │  Input: 3×256×256       │
│  s0: 64×128×128  ───────┼─────┼─────────────────────────┤
│  s1: 64×64×64    ───────┼─────┼─────────────────────────┤
│  s2: 128×32×32   ───────┼─────┼─────────────────────────┤
│  s3: 256×16×16   ───────┼─────┼─────────────────────────┤
│  s4: 512×8×8     ───────┼─────┼────────────────────► tf: 512×8×8
└────────┬────────────────┘     └────────┬────────────────┘
         │                               │
         │    ┌──────────────────────────┘
         │    │
         ▼    ▼
    ┌─────────────────────────────────────────┐
    │  FEATURE FUSION (Concat + 1×1 Conv)     │
    │  Input: s4 (512) + tf (512) = 1024     │
    │  Output: 512×8×8                       │
    └──────────────────┬────────────────────┘
                       │
    ┌──────────────────▼────────────────────┐
    │         U-Net DECODER                 │
    │  dec4: 512→256 (w/ skip s3) → 16×16  │
    │  dec3: 256→128 (w/ skip s2) → 32×32  │
    │  dec2: 128→64  (w/ skip s1) → 64×64  │
    │  dec1: 64→32   (w/ skip s0) → 128×128 │
    └──────────────────┬────────────────────┘
                       │
    ┌──────────────────▼────────────────────┐
    │     OUTPUT HEAD (1×1 Conv)           │
    │     32 channels → 1 channel           │
    │     Bilinear upsample 2× → 256×256    │
    └───────────────────────────────────────┘
```

#### Why This Architecture Works for Crack Detection

**CNN Branch (ResNet34):**
- **Local feature extraction**: Detects fine crack details, edges, and textures
- **Hierarchical features**: Skip connections preserve multi-scale crack information
- **Proven backbone**: ResNet34 provides stable gradients and fast convergence

**Transformer Branch (SegFormer B0):**
- **Global context**: Self-attention captures long-range dependencies between crack segments
- **Position-independent**: Better at handling disconnected crack parts
- **Efficient**: Mix-Transformer (MiT) design reduces computational cost vs ViT

**Fusion Strategy:**
- Concatenation at bottleneck (8×8 resolution) combines high-level CNN features with transformer context
- Skip connections preserve spatial precision from early CNN layers
- Decoder progressively upsamples while maintaining crack connectivity

### Structural Integrity Score (SI Score)

The SI Score (`utils/metrics.py`) is a composite metric combining multiple quality indicators:

```
SI = 0.35 × Dice + 0.15 × exp(-BCE) + 0.30 × Connectivity + 0.20 × BranchConsistency
```

| Component | Weight | Purpose | Implementation |
|-----------|--------|---------|----------------|
| **Dice** | 35% | Segmentation overlap quality | `2\|pred ∩ truth\| / (\|pred\| + \|truth\|)` |
| **BCE** | 15% | Confidence calibration | Cross-entropy on predictions |
| **Connectivity** | 30% | Skeleton continuity | Ratio of skeleton pixels to connected components |
| **Branch Consistency** | 20% | Topology preservation | Graph branch count vs ground truth |

**Score Interpretation:**

| Score | Level | Color | Action Required |
|-------|-------|-------|-----------------|
| 0.80 - 1.00 | Excellent | Green | Routine monitoring |
| 0.60 - 0.80 | Good | Light Green | Periodic inspection |
| 0.40 - 0.60 | Moderate | Yellow | Schedule maintenance |
| 0.20 - 0.40 | Poor | Orange | Immediate inspection |
| 0.00 - 0.20 | Critical | Red | Urgent structural assessment |

## What the Code Does

### Module-by-Module Breakdown

#### 1. Data Pipeline (`data/dataset.py`)

**Purpose**: Load, transform, and batch crack segmentation datasets.

**Key Components**:
- `CrackDataset` class: PyTorch Dataset for image/mask pairs
- Augmentations using Albumentations:
  - Random flips, rotations, elastic transforms
  - Brightness/contrast adjustments
  - GaussNoise for robustness
- Automatic train/val/test splitting (70/15/15)
- Image normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Code Flow**:
```python
Dataset Loading
    ↓
Albumentations Transform
    ↓
ToTensor + Normalize
    ↓
DataLoader (batching)
```

---

#### 2. Model Architecture (`models/hybrid_segformer_unet.py`)

**Purpose**: Define the hybrid CNN+Transformer segmentation model.

**Classes**:

**`HybridSegformerUNet`** (lines 87-145):
- **Initialization**: Creates dual-branch encoder (ResNet + SegFormer)
- **Forward pass**:
  1. CNN extracts 5 skip features (s0-s4)
  2. Transformer processes input to feature map (tf)
  3. Fusion: Concatenate s4 + tf, apply 1×1 conv
  4. Decoder: 4 upsampling blocks with skip connections
  5. Output: 1×1 conv + bilinear upsample to input size

**`ResNetEncoder`** (lines 58-85):
- Wraps torchvision ResNet34/50
- Returns skip connections at 5 resolutions
- Pre-trained on ImageNet

**`TransformerBottleneck`** (lines 33-56):
- Fallback transformer when SegFormer unavailable
- Uses PyTorch TransformerEncoder
- Projects to tokens, processes, reshapes back

**`DecoderBlock`** (lines 19-30):
- Upsamples input to skip connection size
- Concatenates input + skip
- Two 3×3 convolutions with BatchNorm and ReLU

---

#### 3. Loss Functions (`losses/segmentation_losses.py`)

**Purpose**: Multi-objective training with topology awareness.

**Loss Components**:
1. **Dice Loss**: Handles class imbalance (cracks are small % of image)
2. **BCE Loss**: Pixel-wise binary classification
3. **Topology Loss**: Penalizes broken skeletons

**Combined Loss**:
```python
Total Loss = Dice + BCE + λ_topology × TopologyLoss
```

---

#### 4. Training Loop (`training/trainer.py`)

**Purpose**: Orchestrates the training process with modern optimizations.

**Features**:
- **AdamW optimizer**: Weight decay decoupled from gradients
- **Cosine Annealing LR**: Smooth learning rate decay
- **AMP (Automatic Mixed Precision)**: FP16 training for speed
- **Early Stopping**: Prevents overfitting (patience=12)
- **Checkpointing**: Saves best model by validation IoU

**Training Step**:
```python
1. Load batch (images, masks)
2. Forward pass → predictions
3. Compute multi-loss (Dice + BCE + Topology)
4. Backward pass with gradient scaling (AMP)
5. Optimizer step
6. Update learning rate (cosine schedule)
7. Log metrics (IoU, Dice, Loss)
```

---

#### 5. Skeletonization (`topology/skeleton.py`)

**Purpose**: Convert binary masks to 1-pixel wide skeletons.

**Algorithm**: Zhang-Suen thinning algorithm (scikit-image)

**Connectivity Scoring**:
```python
connectivity_score = num_skeleton_pixels / num_connected_components
```
- High score (>0.8): Well-connected crack
- Low score (<0.5): Fragmented/broken crack

**Output**: Binary image where white pixels = crack centerline

---

#### 6. Graph Conversion (`graph/crack_graph.py`)

**Purpose**: Convert skeleton to NetworkX graph for topological analysis.

**Process**:
```
Skeleton (numpy array)
    ↓
Find intersection points (degree ≥ 3)
Find endpoints (degree = 1)
    ↓
Split skeleton at keypoints
    ↓
Create nodes (keypoints) + edges (paths between them)
    ↓
NetworkX Graph with edge weights (pixel distances)
```

**Key Functions**:
- `skeleton_to_graph()`: Main conversion function
- `keypoints_from_graph()`: Identify endpoints and junctions
- `graph_diameter_safe()`: Longest shortest path (crack extent)
- `graph_longest_path_length()`: Maximum path length

---

#### 7. Feature Extraction (`features/structural_features.py`)

**Purpose**: Extract quantitative crack characteristics from graph.

**Extracted Features**:

| Feature | Description | Use Case |
|---------|-------------|----------|
| `total_crack_length` | Sum of all edge weights | Overall damage extent |
| `num_branches` | Count of topological branches | Complexity assessment |
| `longest_path` | Maximum path length in graph | Critical crack length |
| `graph_diameter` | Longest shortest path | Spatial extent |
| `mean_node_degree` | Average connections per node | Connectivity density |
| `endpoints` | Degree-1 nodes | Crack tips |
| `junctions` | Degree-3+ nodes | Branch points |

**Branch Counting Algorithm** (`count_branches_from_graph`):
Traverses graph from endpoints, counting unique paths between keypoints.

---

#### 8. Metrics (`utils/metrics.py`)

**Purpose**: Evaluation metrics and SI Score calculation.

**Functions**:
- `compute_iou()`: Intersection over Union
- `compute_dice()`: Dice coefficient (F1 score)
- `compute_precision_recall()`: Classification metrics
- `structural_integrity_score()`: Composite SI score

---

#### 9. API Server (`api/main.py`)

**Purpose**: Production-ready FastAPI inference service.

**Architecture**:

**`InferenceService` class** (lines 75-184):
- **Lifecycle**: Load model at startup, cache in memory
- **Methods**:
  - `_preprocess()`: BGR → RGB → Resize → Normalize → Tensor
  - `_predict()`: Forward pass with sigmoid + threshold
  - `_encode_png()`: NumPy → PNG → Base64
  - `infer()`: Full pipeline (preprocess → predict → skeleton → graph → features → SI)

**Endpoints**:

| Endpoint | Handler | Description |
|----------|---------|-------------|
| `/health` | `health()` | Model loaded check |
| `/ready` | `ready()` | Kubernetes probe |
| `/metrics` | `metrics()` | Prometheus metrics |
| `/predict` | `predict()` | Single image inference |
| `/predict_batch` | `predict_batch()` | Batch inference |

**Middleware**:
- **LoggingMiddleware**: Request ID assignment + latency logging
- **RateLimiter**: 10/min for single, 5/min for batch
- **Auth**: Bearer token verification (skip in dev mode)

**Error Handling**:
- 400: Invalid file type/size
- 401: Authentication failed
- 413: File too large (>10MB)
- 503: Service unavailable (model not loaded)

---

#### 10. Streamlit UI (`ui/streamlit_app.py`)

**Purpose**: Interactive web interface for visual analysis.

**Components**:

**`AnalysisResult` dataclass** (lines 96-108):
- Structured container for API response data
- Includes request_id, scores, images, features

**UI Sections**:
1. **Header**: Title + description
2. **Sidebar**: Settings (API URL, API key, analysis options)
3. **Upload**: Drag-and-drop image upload
4. **Preview**: Original image display
5. **Analysis Button**: Triggers API call
6. **Results**:
   - SI Score gauge chart (Plotly)
   - Metric cards (connectivity, latency, etc.)
   - Visual comparison (original/overlay/skeleton)
   - Feature grid (7 structural metrics)
   - Download buttons (mask, skeleton, JSON)
7. **History**: Previous analyses table

**Key Functions**:
- `call_predict_api()`: HTTP POST to FastAPI
- `check_api_health()`: Verify backend status
- `create_gauge_chart()`: Plotly gauge visualization
- `overlay_mask_on_image()`: Blend mask with original
- `get_severity_level()`: Color coding based on SI score

**Design**:
- Gradient background (purple/blue)
- Glass-morphism cards
- Responsive layout
- Progress indicators during analysis

---

#### 11. Scripts

**`scripts/train.py`**:
- CLI entry point for training
- Args: `--config`, `--model`, `--resume`
- Saves checkpoints to `checkpoints/`

**`scripts/infer.py`**:
- CLI inference on single image or directory
- Outputs: mask.npy, skeleton.npy, visualization.png
- Supports CPU/GPU inference

**`scripts/export_model.py`**:
- Exports trained model to production formats
- TorchScript: For PyTorch runtime
- ONNX: For cross-platform deployment

**`scripts/benchmark_models.py`**:
- Compares all model architectures
- Outputs JSON with metrics per model

**`scripts/eval_onnx.py`**:
- Runs inference using ONNX Runtime
- Validates exported model accuracy

---

#### 12. Configuration (`utils/config.py`, `configs/config.yaml`)

**Purpose**: Centralized, reproducible configuration.

**Structure**:
```yaml
project:      # Name, seed
data:         # Paths, splits, image_size
training:     # Hyperparameters, AMP, early stopping
model:        # Architecture details
inference:    # Threshold, TTA settings
si_score:     # Metric weights
```

**Benefits**:
- Single source of truth
- Version controlled
- Easy A/B testing (swap config files)
- No hardcoded parameters

---

### Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                          │
├─────────────────────────────────────────────────────────────────┤
│ Raw Images → Dataset → Augmentations → DataLoader → Model      │
│                                              ↓                  │
│                                         Forward Pass            │
│                                              ↓                  │
│         Predictions ──────┬───────→ Ground Truth               │
│              ↓              │              ↓                    │
│         Multi-Loss ←──────┘          Compute Metrics           │
│              ↓                                                  │
│         Backward Pass → Optimizer Step → Checkpoint             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        INFERENCE PHASE                          │
├─────────────────────────────────────────────────────────────────┤
│ Input Image → Preprocess → Model → Sigmoid → Threshold → Mask   │
│                                                              ↓  │
│                                                        Skeleton │
│                                                              ↓  │
│                                                    Graph (NetworkX)
│                                                              ↓  │
│                                                    Features      │
│                                                              ↓  │
│                                                    SI Score      │
└─────────────────────────────────────────────────────────────────┘
```

## API Reference

### Endpoints

| Endpoint | Method | Description | Rate Limit |
|----------|--------|-------------|------------|
| `/health` | GET | Service health check | None |
| `/ready` | GET | Kubernetes readiness probe | None |
| `/metrics` | GET | Prometheus metrics | None |
| `/predict` | POST | Single image prediction | 10/min |
| `/predict_batch` | POST | Batch prediction (max 10) | 5/min |

### Request/Response Examples

**POST /predict Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@crack.jpg" \
  -H "Authorization: Bearer your-api-key"
```

**Response:**
```json
{
  "request_id": "a1b2c3d4",
  "segmentation_mask_png_b64": "iVBORw0KGgo...",
  "skeleton_png_b64": "iVBORw0KGgo...",
  "graph_features": {
    "total_crack_length": 1250.5,
    "num_branches": 8,
    "longest_path": 420.3,
    "graph_diameter": 385.2,
    "mean_node_degree": 2.1,
    "endpoints": 6,
    "junctions": 4
  },
  "connectivity_score": 0.82,
  "si_score": 0.78,
  "latency_seconds": 0.45
}
```

## Monitoring & Metrics

### Prometheus Metrics

Available at `http://localhost:8000/metrics`:

- `predictions_total` - Total predictions by status (success/error)
- `prediction_latency_seconds` - Latency histogram
- `batch_size` - Batch size distribution

### Grafana Dashboard

Access at `http://localhost:3000` (admin/admin by default)

Pre-configured dashboards:
- Prediction throughput
- Latency trends
- Error rates
- GPU utilization

### Health Checks

```bash
# Liveness
curl http://localhost:8000/health

# Readiness (K8s)
curl http://localhost:8000/ready

# Metrics
curl http://localhost:8000/metrics
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# With coverage
pytest tests/ --cov=api --cov-report=html
```

### Code Structure

**Adding a new model:**
1. Create class in `models/your_model.py`
2. Register in `models/__init__.py`
3. Add training config in `configs/config.yaml`

**Adding a new feature:**
1. Implement in `features/`
2. Update API response model in `api/main.py`
3. Add UI visualization in `ui/streamlit_app.py`

### Benchmarking

```bash
# Compare all models
python scripts/benchmark_models.py \
    --config configs/config.yaml \
    --test-dir crack_segmentation_dataset/test \
    --output results.json
```

## Troubleshooting

### Common Issues

**"Service unavailable" errors:**
```bash
# Check if model is loaded
curl http://localhost:8000/health

# Verify checkpoint exists
ls -la checkpoints/best_hybrid_segformer.pth
```

**High latency:**
- Enable GPU: `CUDA_VISIBLE_DEVICES=0`
- Check batch size in config
- Consider model quantization

**Out of memory:**
```yaml
# Reduce batch size in configs/config.yaml
training:
  batch_size: 4  # Reduce from 8
```

**Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Performance Tuning

**Multi-worker (CPU only):**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**With Gunicorn:**
```bash
gunicorn api.main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Security Checklist

Before production deployment:

- [ ] Change default `API_KEY` from `dev-key-change-in-production`
- [ ] Run as non-root user in container
- [ ] Enable HTTPS (via reverse proxy)
- [ ] Configure rate limits appropriately
- [ ] Set file size limits (10MB default)
- [ ] Enable input validation
- [ ] Configure logging
- [ ] Keep secrets out of code/repo

## Citation

If you use CrackGraphAI in your research, please cite:

```bibtex
@software{crackgraphai2024,
  title={CrackGraphAI: Production-Grade Crack Segmentation and Structural Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/CrackGraphAI}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SegFormer architecture from [NVIDIA SegFormer](https://github.com/NVlabs/SegFormer)
- Segmentation Models PyTorch library
- FastAPI and Streamlit communities
- NetworkX for graph algorithms

---

**Support:** For issues and feature requests, please use the [GitHub Issues](https://github.com/yourusername/CrackGraphAI/issues) page.
