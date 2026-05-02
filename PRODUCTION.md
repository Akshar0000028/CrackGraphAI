# CrackGraphAI Production Deployment Guide

This guide covers deploying CrackGraphAI with **production-grade stability guarantees**.

## Key Stability Features

### 1. Stable Inference Engine (`inference/`)

The production API uses a new `StableInferenceEngine` with the following features:

- **Test-Time Augmentation (TTA)**: Horizontal/vertical flips with mean/max/vote merging
- **Input Validation**: Robust validation for file type, size, corruption detection
- **Preprocessing Normalization**: Consistent color space handling, automatic white balance
- **Post-processing Smoothing**: Gaussian filtering, morphological operations, small component removal
- **Uncertainty Quantification**: Entropy-based confidence scores for reliability assessment
- **Result Caching**: LRU cache with TTL for identical inputs
- **Model Warmup**: Pre-warms GPU to avoid cold-start latency

### 2. Production API (`api/production_main.py`)

- **Circuit Breaker Pattern**: Prevents cascading failures
- **Request Queue**: Load shedding with max concurrent request limits
- **Prometheus Metrics**: Latency, throughput, error rates, cache hit rates
- **GZIP Compression**: Reduces response sizes
- **CORS Support**: Configurable origin whitelist
- **Structured Logging**: Request IDs for traceability

### 3. Deployment Options

#### Docker Compose Production (`docker-compose.prod.yml`)

```yaml
services:
  api:           # Main inference API with stability features
  nginx:         # Reverse proxy with SSL
  prometheus:    # Metrics collection
  grafana:       # Dashboards and alerts
  redis:         # Distributed caching
  node-exporter: # System metrics
```

## Quick Start

### 1. Prepare Model Weights

Place your trained model at:
```
checkpoints/best_hybrid_segformer.pth
```

### 2. Deploy

```bash
# Make deploy script executable
chmod +x scripts/deploy.sh

# Deploy everything
./scripts/deploy.sh deploy

# Or use docker-compose directly
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### 4. Test Inference

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -F "image=@test_image.jpg" \
  -F "return_uncertainty=true"

# Batch prediction
curl -X POST http://localhost:8000/predict_batch \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg"
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `change-me-in-production` | Authentication key |
| `USE_TTA` | `true` | Enable Test-Time Augmentation |
| `CACHE_TTL` | `3600` | Cache time-to-live (seconds) |
| `MORPHOLOGY_KERNEL_SIZE` | `3` | Post-processing kernel size |
| `MIN_COMPONENT_SIZE` | `16` | Minimum crack component size |
| `MIXED_PRECISION` | `true` | Use FP16 for faster inference |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `GRAFANA_PASSWORD` | `admin` | Grafana admin password |

### Inference Config (`InferenceConfig`)

```python
from inference.engine import InferenceConfig

config = InferenceConfig(
    use_tta=True,                    # Enable TTA
    tta_flips=['horizontal', 'vertical'],  # Flip augmentations
    tta_merge_strategy='mean',       # mean/max/vote
    ensemble_threshold=0.5,          # Binary threshold
    apply_morphology=True,           # Smooth masks
    morphology_kernel_size=3,
    min_component_size=16,           # Filter small artifacts
    estimate_uncertainty=True,       # Confidence scores
    uncertainty_threshold=0.3,       # Reliability threshold
    enable_cache=True,               # Result caching
    cache_ttl=3600,
    mixed_precision=True,            # FP16 inference
)
```

## Model Optimization

### Export to ONNX/TensorRT

```bash
# Export and benchmark
python scripts/optimize_model.py \
    --checkpoint checkpoints/best_hybrid_segformer.pth \
    --export-onnx \
    --quantize \
    --export-torchscript \
    --benchmark
```

This produces:
- `optimized_models/model.onnx` - Standard ONNX
- `optimized_models/model_quantized.onnx` - INT8 quantized (50% smaller)
- `optimized_models/model.pt` - TorchScript
- `optimized_models/model.trt` - TensorRT engine (GPU only)

### Benchmark Results Example

```
PyTorch:
  mean_ms: 45.23
  p95_ms: 52.11
  throughput: 22.1

TorchScript:
  mean_ms: 38.45
  p95_ms: 44.89
  throughput: 26.0

ONNX Runtime:
  mean_ms: 25.12
  p95_ms: 28.34
  throughput: 39.8
```

## Monitoring & Alerting

### Prometheus Metrics

| Metric | Description |
|--------|-------------|
| `predictions_total` | Total predictions by status |
| `prediction_latency_seconds` | Latency histogram |
| `active_requests` | Current concurrent requests |
| `request_queue_size` | Pending request queue |
| `model_loaded` | Model availability |

### Grafana Dashboard

Access at http://localhost:3000 with pre-configured dashboards:
- API Performance (latency, throughput, errors)
- Model Health (cache stats, inference stability)
- System Resources (CPU, GPU, memory)

### Alerts

Configured alerts for:
- High error rate (>10% for 2 min)
- High latency (p95 > 5s for 2 min)
- Service down (>1 min)
- Model not loaded
- Circuit breaker open

## API Endpoints

### Production Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + cache stats |
| `/ready` | GET | Kubernetes readiness probe |
| `/metrics` | GET | Prometheus metrics |
| `/predict` | POST | Single image with TTA |
| `/predict_batch` | POST | Batch prediction |
| `/cache/stats` | GET | Cache statistics |
| `/cache/clear` | DELETE | Clear cache |

### Predict Response Format

```json
{
  "request_id": "abc123",
  "segmentation_mask_png_b64": "iVBORw0KGgo...",
  "skeleton_png_b64": "iVBORw0KGgo...",
  "probability_map_png_b64": "iVBORw0KGgo...",
  "graph_features": {
    "total_crack_length": 1250.5,
    "num_branches": 8,
    "longest_path": 450.2,
    "endpoints": 6,
    "junctions": 2
  },
  "connectivity_score": 0.85,
  "si_score": 0.72,
  "latency_seconds": 0.45,
  "from_cache": false,
  "uncertainty": {
    "mean_uncertainty": 0.15,
    "mean_confidence": 0.85,
    "reliable": true
  }
}
```

## Troubleshooting

### High Latency

1. Check if TTA is enabled (adds 2-3x latency)
2. Enable mixed precision: `MIXED_PRECISION=true`
3. Use optimized model (ONNX/TensorRT)
4. Increase GPU memory

### Model Not Loading

```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs api

# Verify weights exist
ls -la checkpoints/

# Test locally
python -c "from inference.engine import StableInferenceEngine; print('OK')"
```

### OOM Errors

1. Reduce `batch_size` in config
2. Disable TTA temporarily
3. Use CPU inference: `CUDA_VISIBLE_DEVICES=""`
4. Enable gradient checkpointing (requires model modification)

## Kubernetes Deployment

For K8s deployment, use the health/ready endpoints:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 5
```

## No-Training Deployment

This production setup is **ready to deploy without training**:

1. Place **any compatible** PyTorch model checkpoint at `checkpoints/best_hybrid_segformer.pth`
2. Deploy with `./scripts/deploy.sh deploy`
3. The stability features (TTA, caching, smoothing) work with any model

The inference engine automatically:
- Validates inputs
- Applies TTA for stability
- Caches results
- Quantifies uncertainty
- Handles errors gracefully

## Summary

This production deployment provides:

- **Stable Outputs**: TTA + smoothing + uncertainty quantification
- **Fast Inference**: Caching + optimizations
- **Fault Tolerance**: Circuit breaker + queue management
- **Observability**: Metrics + logging + alerting
- **Security**: Auth + CORS + input validation
- **Scalability**: Stateless design + load balancing ready
