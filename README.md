# CrackGraphAI

**Production-grade crack segmentation and structural integrity analysis powered by a hybrid CNN–Transformer architecture.**

CrackGraphAI detects cracks in surface images, skeletonises them into a topological graph, and produces a calibrated **Structural Integrity (SI) score** with a five-level risk classification — all through a single REST API call.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
  - [Model: HybridSegformerUNet](#model-hybridsegformerunet)
  - [Inference Pipeline](#inference-pipeline)
  - [Structural Integrity Scoring](#structural-integrity-scoring)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
  - [Docker (Recommended)](#docker-recommended)
- [Configuration](#configuration)
- [API Reference](#api-reference)
  - [Authentication](#authentication)
  - [Endpoints](#endpoints)
  - [Response Schema](#response-schema)
- [Training](#training)
  - [Dataset](#dataset)
  - [Train a Model](#train-a-model)
  - [Baseline Comparison](#baseline-comparison)
  - [Export Model](#export-model)
- [Web UI](#web-ui)
- [Database](#database)
- [Monitoring](#monitoring)
- [Production Deployment](#production-deployment)
  - [Environment Variables](#environment-variables)
  - [Docker Compose (Full Stack)](#docker-compose-full-stack)
  - [Security Checklist](#security-checklist)
- [CI/CD](#cicd)
- [Testing](#testing)
- [Risk Classification](#risk-classification)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

CrackGraphAI is an end-to-end system for automated structural crack analysis. Given a photo of a concrete surface, pavement, wall, or any structure, it:

1. **Segments** crack pixels using a hybrid deep learning model
2. **Post-processes** the mask to remove non-crack noise (blobs, isolated dots)
3. **Skeletonises** the mask into a 1-pixel-wide topological representation
4. **Builds a graph** of the crack network (nodes = pixels, edges = connectivity)
5. **Extracts features** — crack density, branching, junctions, endpoints, width
6. **Computes an SI score** (0.0 = failure imminent → 1.0 = healthy) with a risk level
7. **Returns** all of the above as a structured JSON response with base64-encoded PNG overlays

---

## Key Features

- **Hybrid CNN + Transformer model** — ResNet34 encoder fused with SegFormer (MiT-B0) for both local texture and global context
- **Test-Time Augmentation (TTA)** — averages predictions over original + horizontal + vertical flips for robustness
- **Topology-aware training loss** — Dice + BCE + differentiable skeleton loss to preserve crack connectivity
- **Structural Integrity scoring** — physics-inspired, four-component damage model with calibrated thresholds
- **Graph-based crack analysis** — NetworkX graph with branch counting, junction detection, longest path, diameter
- **CLAHE preprocessing** — contrast-limited adaptive histogram equalisation for dark/low-contrast surfaces
- **Morphological post-processing** — removes blob artefacts while preserving fine crack tips
- **Production FastAPI** — rate limiting, Bearer auth, Prometheus metrics, request IDs, 30 s timeout
- **ONNX + TorchScript export** — deploy without PyTorch runtime
- **Full observability stack** — Prometheus + Grafana + Node Exporter
- **PostgreSQL database** — stores predictions, metrics, and analysis history
- **Redis caching** — optional distributed result cache
- **Nginx reverse proxy** — TLS termination, gzip, rate limiting at the edge
- **GitHub Actions CI/CD** — test → build → security scan → deploy pipeline

---

## Architecture

### Model: HybridSegformerUNet

```
Input (3 × 256 × 256)
        │
        ├──────────────────────────────────────────────┐
        │  ResNet34 Encoder                            │  SegFormer MiT-B0
        │  s0: H/2  (64 ch)                            │  (timm, pretrained)
        │  s1: H/4  (64 ch)                            │  out: H/8 (256 ch)
        │  s2: H/8  (128 ch)                           │
        │  s3: H/16 (256 ch)                           │
        │  s4: H/32 (512 ch)  ─────────────────────────┤
        │                                              │
        │              Fusion (ConvBNReLU 1×1 → 512)  │
        │                         │                   │
        └─────────────────────────┘                   │
                                  │                   │
                    Decoder (4 × DecoderBlock)         │
                    dec4: H/16 (256 ch)                │
                    dec3: H/8  (128 ch)                │
                    dec2: H/4  (64 ch)                 │
                    dec1: H/2  (32 ch)                 │
                                  │
                    Head (Conv 1×1) + Bilinear upsample
                                  │
                    Output (1 × 256 × 256) — logits
```

The transformer branch processes the full input image and its output is aligned to the deepest CNN feature map before fusion. A `TransformerBottleneck` fallback is used automatically if the timm SegFormer weights are unavailable.

### Inference Pipeline

```
Raw bytes
   → cv2 decode
   → Resize to 256×256
   → CLAHE (LAB L-channel)
   → ImageNet normalise
   → Model forward (+ TTA: hflip, vflip)
   → Sigmoid threshold (0.35)
   → Morphological post-processing
       • Opening (2×2 ellipse) — removes salt-and-pepper noise
       • Connected-component filter — keeps elongated / thin / small regions
       • Closing (3×3 ellipse) — reconnects nearby segments
   → Skeletonize (scikit-image)
   → skeleton_to_graph (8-connectivity NetworkX graph)
   → extract_structural_features
   → compute_structural_integrity (SI score)
   → Encode outputs as base64 PNG
   → Return JSON
```

### Structural Integrity Scoring

SI is defined as `1.0 − Damage`, where Damage is a weighted sum of four normalised components:

| Component | Weight | Description |
|---|---|---|
| Crack Density | 0.35 | Skeleton pixels / image area, normalised at 3% coverage = severe |
| Network Density | 0.25 | Junctions per 100 skeleton pixels (branching severity) |
| Complexity | 0.25 | Branches + 2×junctions, with endpoint-activity bonus |
| Crack Width | 0.15 | Mean half-width via distance transform, normalised by image diagonal |

When a ground-truth mask is available (training/evaluation), a fifth **Segmentation Quality** component (Dice + exp(−BCE)) is added and weights are redistributed automatically.

---

## Project Structure

```
CrackGraphAI/
├── api/
│   ├── main.py                  # FastAPI app — endpoints, middleware, InferenceService
│   └── production_main.py       # Production variant
├── checkpoints/
│   ├── best_hybrid_segformer.pth  # Trained PyTorch weights
│   ├── hybrid_segformer.onnx      # ONNX export
│   └── hybrid_segformer.ts        # TorchScript export
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── crack_segmentation_dataset/
│   ├── images/                  # 11 298 training images (CFD + CRACK500)
│   ├── masks/                   # Corresponding binary masks
│   └── test/                    # Held-out test split
├── data/
│   └── dataset.py               # CrackSegmentationDataset, augmentations, splits
├── db/
│   ├── database.py              # SQLAlchemy engine, connection pool, session factory
│   ├── models.py                # ORM models (predictions, metrics)
│   ├── repository.py            # CRUD operations
│   ├── service.py               # Business logic layer
│   ├── api_integration.py       # FastAPI dependency injection
│   └── init.sql                 # Schema bootstrap SQL
├── features/
│   ├── si_scoring.py            # SI score engine (SIGenerator, FeatureNormalizer)
│   └── structural_features.py   # Graph feature extraction
├── graph/
│   ├── crack_graph.py           # skeleton_to_graph, keypoint detection, overlays
│   └── enhanced_crack_graph.py  # Extended graph utilities
├── inference/
│   ├── engine.py                # Standalone inference engine
│   ├── preprocessing.py         # Image preprocessing utilities
│   ├── postprocessing.py        # Mask post-processing
│   └── cache.py                 # Redis result cache
├── losses/
│   └── segmentation_losses.py   # Dice + BCE + TopologyAwareLoss
├── models/
│   ├── hybrid_segformer_unet.py # HybridSegformerUNet (main model)
│   └── baselines.py             # UNet++, DeepLabV3+, SegFormer baselines
├── monitoring/
│   ├── prometheus.yml           # Scrape config
│   ├── alert_rules.yml          # Alerting rules
│   └── grafana/                 # Dashboard provisioning
├── nginx/
│   └── nginx.conf               # Reverse proxy, rate limiting, TLS config
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── infer.py                 # CLI inference with visualisation + Grad-CAM
│   ├── benchmark_models.py      # Compare all models on test set
│   ├── export_model.py          # Export to ONNX + TorchScript
│   ├── eval_onnx.py             # Evaluate ONNX model
│   ├── optimize_model.py        # Model optimisation utilities
│   ├── quick_inference.py       # Fast single-image inference
│   ├── verify_production.py     # Production readiness checks
│   ├── deploy.sh                # Linux deployment script
│   └── deploy-windows.ps1       # Windows deployment script
├── tests/
│   ├── test_api.py              # FastAPI endpoint tests
│   ├── test_metrics.py          # Metric computation tests
│   └── test_si_scoring.py       # SI scoring unit tests
├── topology/
│   └── skeleton.py              # mask_to_skeleton, connectivity_score
├── training/
│   └── trainer.py               # Trainer (AdamW, CosineAnnealingLR, AMP, early stopping)
├── ui/
│   ├── index.html               # Single-page web UI (Tailwind CSS)
│   └── app.js                   # UI logic — upload, analysis, visualisation, export
├── utils/
│   ├── config.py                # YAML config loader
│   ├── metrics.py               # segmentation_metrics, SI score utilities
│   └── repro.py                 # Reproducibility (seed setting)
├── docker-compose.yml           # Development stack
├── docker-compose.prod.yml      # Production stack (Nginx, Redis, Prometheus, Grafana)
├── docker-compose.db.yml        # Database-only stack
├── Dockerfile                   # Development image
├── Dockerfile.prod              # Production image
├── requirements.txt             # Pinned Python dependencies
├── requirements-db.txt          # Database-specific dependencies
├── configs/config.yaml          # Project configuration
├── pytest.ini                   # Test configuration
└── .env.example                 # Environment variable template
```

---

## Quick Start

### Prerequisites

- Python 3.10 or 3.11
- CUDA-capable GPU recommended (CPU inference is supported but slower)
- Docker + Docker Compose (for containerised deployment)

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/your-org/crackgraphai.git
cd crackgraphai

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure environment variables
cp .env.example .env
# Edit .env — at minimum set API_KEY

# 5. Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Open the web UI
# Navigate to http://localhost:8000/docs  (Swagger UI)
# Or open ui/index.html in a browser pointed at http://localhost:8000
```

### Docker (Recommended)

```bash
# Development — API only
docker-compose up -d

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# With Redis caching
docker-compose --profile cache up -d

# Production full stack
docker-compose -f docker-compose.prod.yml up -d
```

Verify the service is healthy:

```bash
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true,"device":"cuda:0","version":"1.2.0"}
```

---

## Configuration

All settings live in `configs/config.yaml`:

```yaml
project:
  name: crackgraphai
  seed: 42

data:
  root_dir: crack_segmentation_dataset
  image_size: 256          # model input resolution
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  num_workers: 4

training:
  batch_size: 8
  epochs: 80
  lr: 0.0002
  weight_decay: 0.0001
  amp: true                # mixed-precision training
  early_stopping_patience: 12
  lambda_topology: 0.2     # weight of topology loss term

model:
  architecture: hybrid_segformer_unet
  cnn_backbone: resnet34
  transformer_backbone: mit_b0
  in_channels: 3
  classes: 1
  decoder_channels: [256, 128, 64, 32]

inference:
  threshold: 0.35          # sigmoid threshold for binary mask
  use_tta: true
  tta_transforms: [none, hflip, vflip]

si_score:
  weights:
    dice: 0.35
    bce: 0.15
    connectivity: 0.30
    branch_consistency: 0.20
```

---

## API Reference

### Authentication

In **development mode** (default `API_KEY=dev-key-change-in-production`), no authentication header is required.

In **production**, set a secure `API_KEY` in `.env` and include it in every request:

```bash
curl -H "Authorization: Bearer <your-api-key>" ...
```

### Endpoints

| Method | Endpoint | Description | Rate Limit | Auth |
|---|---|---|---|---|
| `GET` | `/health` | Service health check | None | No |
| `GET` | `/ready` | Kubernetes readiness probe | None | No |
| `GET` | `/metrics` | Prometheus metrics | None | No |
| `GET` | `/docs` | Swagger UI | None | No |
| `POST` | `/predict` | Single-image crack analysis | 10 / min | Yes |
| `POST` | `/predict_batch` | Batch analysis (max 10 images) | 5 / min | Yes |

**Constraints:**
- Accepted formats: `image/jpeg`, `image/png`
- Maximum file size: 10 MB per image
- Inference timeout: 30 seconds per image

### Single Image — cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer your-api-key" \
  -F "image=@/path/to/crack.jpg"
```

### Batch — cURL

```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Authorization: Bearer your-api-key" \
  -F "images=@crack1.jpg" \
  -F "images=@crack2.jpg"
```

### Python Client

```python
import requests

with open("crack.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        headers={"Authorization": "Bearer your-api-key"},
        files={"image": ("crack.jpg", f, "image/jpeg")},
    )

result = response.json()
print(f"SI Score : {result['si_score']}")
print(f"Risk     : {result['damage_metrics']['risk_level']}")
print(f"Latency  : {result['latency_seconds']}s")
```

### Response Schema

```json
{
  "request_id": "a1b2c3d4",
  "si_score": 0.742,
  "connectivity_score": 0.891,
  "segmentation_mask_png_b64": "<base64-encoded PNG>",
  "raw_mask_png_b64": "<base64-encoded PNG>",
  "skeleton_png_b64": "<base64-encoded PNG>",
  "keypoints_overlay_png_b64": "<base64-encoded PNG>",
  "graph_features": {
    "total_crack_length": 312.5,
    "num_branches": 4.0,
    "longest_path": 187.3,
    "graph_diameter": 12.0,
    "mean_node_degree": 1.94,
    "node_degree_distribution": [0, 142, 156, 14],
    "endpoints": 8.0,
    "junctions": 3.0
  },
  "damage_metrics": {
    "density_damage": 0.18,
    "network_damage": 0.12,
    "complexity_damage": 0.21,
    "width_damage": 0.09,
    "total_damage": 0.258,
    "risk_level": "Moderate"
  },
  "post_processing": {
    "raw_pixels": 1842,
    "filtered_pixels": 47,
    "final_pixels": 1795,
    "filtering_applied": true
  },
  "latency_seconds": 0.412
}
```

Decode a PNG overlay in Python:

```python
import base64, cv2, numpy as np

png_bytes = base64.b64decode(result["keypoints_overlay_png_b64"])
img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
cv2.imwrite("overlay.png", img)
```

---

## Training

### Dataset

The model is trained on a combined dataset of **11 298 paired image/mask samples**:

- **CFD** (Crack Forest Dataset) — concrete surface cracks
- **CRACK500** — pavement crack images at multiple scales

Dataset layout expected by the trainer:

```
crack_segmentation_dataset/
├── images/   ← RGB images (.jpg)
└── masks/    ← Binary masks (.jpg, white = crack)
```

The dataset is split automatically: 70% train / 15% val / 15% test (stratified, seed 42).

**Augmentations (training only):**
- Horizontal and vertical flip (p=0.5 each)
- Elastic transform (α=50, σ=6, p=0.25)
- Random brightness/contrast (p=0.4)
- Gaussian noise (p=0.2)
- Resize to 256×256 + ImageNet normalisation

### Train a Model

```bash
# Train the hybrid model (default)
python scripts/train.py --config configs/config.yaml --model hybrid

# Train baseline models
python scripts/train.py --model unetpp
python scripts/train.py --model deeplabv3plus
python scripts/train.py --model segformer
```

Training features:
- **Optimiser:** AdamW (lr=2e-4, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR
- **Loss:** Dice + BCE + 0.2 × TopologyAwareLoss
- **Mixed precision:** AMP (automatic mixed precision)
- **Early stopping:** patience=12 epochs on validation Dice
- Best checkpoint saved to `checkpoints/best_hybrid_segformer.pth`

### Baseline Comparison

```bash
python scripts/benchmark_models.py --config configs/config.yaml --checkpoints-dir checkpoints
```

Outputs a table of IoU, Dice, Precision, Recall, BCE, and Connectivity for all available checkpoints.

### Export Model

```bash
# Export to ONNX (opset 18) and TorchScript
python scripts/export_model.py \
  --weights checkpoints/best_hybrid_segformer.pth \
  --onnx-out checkpoints/hybrid_segformer.onnx \
  --ts-out checkpoints/hybrid_segformer.ts

# Evaluate ONNX model
python scripts/eval_onnx.py --onnx checkpoints/hybrid_segformer.onnx
```

### CLI Inference

```bash
# Basic inference
python scripts/infer.py \
  --weights checkpoints/best_hybrid_segformer.pth \
  --image path/to/crack.jpg \
  --output-dir outputs/

# With ground-truth mask (computes real Dice/BCE)
python scripts/infer.py \
  --weights checkpoints/best_hybrid_segformer.pth \
  --image path/to/crack.jpg \
  --gt-mask path/to/mask.jpg \
  --output-dir outputs/ \
  --visualize
```

The `--visualize` flag generates a multi-panel figure: Input | GT Mask | Pred Mask | Skeleton | Grad-CAM | Graph Overlay.

---

## Web UI

A lightweight single-page application is included in `ui/`:

- **Drag-and-drop** image upload (PNG/JPG, max 10 MB)
- **Live API status** indicator
- **Animated SI gauge** with colour-coded risk badge
- **Damage breakdown** bar chart (density, network, complexity, width)
- **Four visualisation panels:** original, segmentation mask, skeleton, keypoints overlay
- **Compare mode** — side-by-side selector for any two outputs
- **Export** — download JSON report, segmentation mask, or keypoints overlay

To use the UI, open `ui/index.html` in a browser while the API is running on `http://localhost:8000`. The API URL can be changed in `app.js`.

---

## Database

CrackGraphAI optionally persists all predictions to PostgreSQL.

```bash
# Start the database
docker-compose -f docker-compose.db.yml up -d

# Initialise schema
python db/init_db.py
```

The database layer uses SQLAlchemy with a `QueuePool` (pool_size=10, max_overflow=20). Configure via environment variables:

```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/crackgraphai
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_RECYCLE=3600
```

---

## Monitoring

The production stack ships with full observability out of the box.

### Prometheus Metrics (`:8000/metrics`)

| Metric | Type | Description |
|---|---|---|
| `predictions_total` | Counter | Total predictions, labelled by `status` (success/error) |
| `prediction_latency_seconds` | Histogram | End-to-end inference latency |
| `batch_size` | Histogram | Batch size distribution |

### Starting the Monitoring Stack

```bash
docker-compose -f docker-compose.prod.yml up -d prometheus grafana node-exporter
```

| Service | URL | Default Credentials |
|---|---|---|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| API Metrics | http://localhost:8000/metrics | — |

Grafana dashboards and Prometheus datasource are provisioned automatically from `monitoring/grafana/`.

---

## Production Deployment

### Environment Variables

Copy `.env.example` to `.env` and set all values before deploying:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_KEY` | **Yes** | `dev-key-change-in-production` | Bearer token for API auth |
| `HOST` | No | `0.0.0.0` | Bind address |
| `PORT` | No | `8000` | Bind port |
| `CONFIG_PATH` | No | `configs/config.yaml` | Config file path |
| `WEIGHTS_PATH` | No | `checkpoints/best_hybrid_segformer.pth` | Model weights path |
| `RATE_LIMIT_PER_MINUTE` | No | `10` | Per-IP rate limit for `/predict` |
| `BATCH_RATE_LIMIT_PER_MINUTE` | No | `5` | Per-IP rate limit for `/predict_batch` |
| `MAX_FILE_SIZE` | No | `10485760` | Max upload size in bytes (10 MB) |
| `CUDA_VISIBLE_DEVICES` | No | `0` | GPU device index |
| `REDIS_URL` | No | — | Redis URL for result caching |
| `DATABASE_URL` | No | — | PostgreSQL connection string |
| `ENABLE_METRICS` | No | `true` | Expose Prometheus metrics endpoint |
| `GRAFANA_PASSWORD` | No | `admin` | Grafana admin password |

### Docker Compose (Full Stack)

```bash
# Build and start everything
docker-compose -f docker-compose.prod.yml up -d

# Check all services are healthy
docker-compose -f docker-compose.prod.yml ps

# View API logs
docker-compose -f docker-compose.prod.yml logs -f api

# Scale (CPU-only, not recommended for GPU)
docker-compose -f docker-compose.prod.yml up -d --scale api=2
```

Services started:

| Container | Port | Description |
|---|---|---|
| `crackgraphai-api-prod` | 8000 | FastAPI inference service |
| `crackgraphai-nginx` | 80, 443 | Reverse proxy + TLS termination |
| `crackgraphai-prometheus` | 9090 | Metrics collection |
| `crackgraphai-grafana` | 3000 | Dashboards |
| `crackgraphai-redis` | 6379 | Result cache |
| `crackgraphai-node-exporter` | 9100 | System metrics |

### Security Checklist

Before going to production, verify:

- [ ] `API_KEY` changed from the default value
- [ ] Container runs as non-root user (`appuser`)
- [ ] HTTPS enabled via Nginx (add SSL certs to `nginx/ssl/` and uncomment the HTTPS block in `nginx/nginx.conf`)
- [ ] `CORS_ORIGINS` restricted to your frontend domain
- [ ] Rate limits tuned for your expected traffic
- [ ] `DB_ECHO=false` (prevents SQL logging in production)
- [ ] Secrets not committed to the repository (`.env` is in `.gitignore`)
- [ ] Trivy security scan passing (runs automatically in CI)
- [ ] Health check responding: `curl http://localhost:8000/health`

---

## CI/CD

GitHub Actions pipeline defined in `.github/workflows/ci-cd.yml`:

| Job | Trigger | Description |
|---|---|---|
| `test` | All pushes and PRs | pytest on Python 3.10 + 3.11, ruff lint, mypy type check, coverage upload |
| `build-docker` | Push to `main`/`develop` | Build and push image to GHCR |
| `security-scan` | After build | Trivy vulnerability scan, results uploaded to GitHub Security |
| `deploy-staging` | Push to `develop` | Deploy to staging environment |
| `deploy-production` | GitHub Release published | Deploy to production environment |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=api --cov=utils --cov=features --cov-report=term-missing

# Run a specific test file
pytest tests/test_si_scoring.py -v
pytest tests/test_api.py -v
```

Test files:

| File | Coverage |
|---|---|
| `tests/test_api.py` | FastAPI endpoints, auth, rate limiting, error handling |
| `tests/test_metrics.py` | IoU, Dice, precision, recall, BCE computation |
| `tests/test_si_scoring.py` | SI score components, risk classification, edge cases |

---

## Risk Classification

| SI Score | Risk Level | Recommended Action |
|---|---|---|
| ≥ 0.85 | **Low** | Routine monitoring |
| 0.70 – 0.85 | **Moderate** | Schedule inspection within 6 months |
| 0.50 – 0.70 | **High** | Professional assessment within 1 month |
| 0.30 – 0.50 | **Critical** | Immediate intervention advised |
| < 0.30 | **Failure Imminent** | Evacuate and initiate emergency repairs |

---

## Requirements

Core dependencies (pinned versions in `requirements.txt`):

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.4.0 | Deep learning framework |
| `torchvision` | 0.19.0 | Image utilities |
| `timm` | 0.9.16 | SegFormer (MiT-B0) backbone |
| `segmentation-models-pytorch` | 0.3.3 | Baseline model builders |
| `opencv-python-headless` | 4.10.0.84 | Image I/O and processing |
| `albumentations` | 1.4.14 | Training augmentations |
| `scikit-image` | 0.24.0 | Skeletonisation |
| `networkx` | 3.3 | Crack graph construction |
| `fastapi` | 0.112.0 | REST API framework |
| `uvicorn` | 0.30.5 | ASGI server |
| `slowapi` | 0.1.9 | Rate limiting |
| `prometheus-client` | 0.20.0 | Metrics exposition |
| `onnx` / `onnxruntime-gpu` | 1.16.2 / 1.18.1 | Model export and runtime |
| `redis` | 5.0.8 | Result caching |
| `sqlalchemy` / `asyncpg` | — | Database ORM and async driver |
| `streamlit` | 1.37.1 | Optional Streamlit UI |

**Python:** 3.10 or 3.11  
**CUDA:** 12.1 (for GPU inference)  
**Docker:** 24+ with NVIDIA Container Toolkit for GPU passthrough

---

## Contributing

1. Fork the repository and create a feature branch from `develop`
2. Install dev dependencies: `pip install ruff mypy black`
3. Run linting before committing: `ruff check . --ignore E501`
4. Ensure all tests pass: `pytest tests/ -v`
5. Open a pull request against `develop` — CI will run automatically

---


