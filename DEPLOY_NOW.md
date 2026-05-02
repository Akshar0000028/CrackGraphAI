# CrackGraphAI - Production Deployment Guide

## Quick Start (No Training Required)

Your model is **ready to deploy** immediately. The trained weights are already in place at `checkpoints/best_hybrid_segformer.pth`.

---

## Windows Deployment

### Option 1: PowerShell Script (Recommended)

```powershell
# Run deployment script
.\scripts\deploy-windows.ps1 -Action deploy

# Or with GPU support
.\scripts\deploy-windows.ps1 -Action deploy -Gpu
```

### Option 2: Docker Compose (Manual)

```powershell
# Ensure .env file exists (copy from example)
copy .env.example .env

# Build and start services
docker compose -f docker-compose.prod.yml up -d

# Check status
docker compose -f docker-compose.prod.yml ps
```

---

## Linux/Mac Deployment

```bash
# Make deploy script executable and run
chmod +x scripts/deploy.sh
./scripts/deploy.sh deploy
```

---

## Verify Deployment

### Test Health
```powershell
curl http://localhost:8000/health
```

### Run Full Verification
```powershell
python scripts/verify_production.py
```

### Test Inference
```powershell
# Using API
curl -X POST "http://localhost:8000/predict" -F "image=@path/to/crack_image.jpg"

# Or use direct inference (no API needed)
python scripts/quick_inference.py --image path/to/crack_image.jpg --output outputs/
```

---

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | Main inference API |
| API Docs | http://localhost:8000/docs | Swagger/OpenAPI documentation |
| Grafana | http://localhost:3000 | Monitoring dashboard (admin/admin) |
| Prometheus | http://localhost:9090 | Metrics collection |

---

## Production Features

### Stability Guarantees
- **Test-Time Augmentation (TTA)**: Horizontal/vertical flips with mean/max/vote merging
- **Input Validation**: File type, size, corruption detection
- **Post-processing**: Gaussian smoothing, morphological operations, small component removal
- **Uncertainty Quantification**: Entropy-based confidence scores
- **Result Caching**: LRU cache with TTL for identical inputs
- **Model Warmup**: Pre-warms GPU to avoid cold-start latency

### Fault Tolerance
- **Circuit Breaker Pattern**: Prevents cascading failures
- **Request Queue**: Load shedding with max concurrent request limits
- **Health Checks**: Kubernetes-compatible /health and /ready endpoints
- **Graceful Degradation**: Continues operating even if some features fail

### Performance
- **Mixed Precision**: FP16 inference for 2x speedup on GPU
- **Batch Processing**: Efficient multi-image inference
- **ONNX Export**: Cross-platform deployment option

---

## Environment Configuration

Edit `.env` file for production settings:

```env
# Security
API_KEY=your-secure-api-key-change-this

# Inference Settings
USE_TTA=true              # Enable Test-Time Augmentation
CACHE_TTL=3600           # Cache time-to-live in seconds
MORPHOLOGY_KERNEL_SIZE=3  # Post-processing smoothing
MIN_COMPONENT_SIZE=16     # Minimum crack component size

# Performance
MIXED_PRECISION=true      # Use FP16 for faster inference
CUDA_VISIBLE_DEVICES=0    # GPU device selection

# Monitoring
GRAFANA_PASSWORD=admin
```

---

## Model Files

Pre-trained models in `checkpoints/`:

| File | Size | Purpose |
|------|------|---------|
| `best_hybrid_segformer.pth` | ~114 MB | Main PyTorch model |
| `hybrid_segformer.onnx` | ~443 KB | ONNX format for cross-platform |
| `hybrid_segformer.ts` | ~114 MB | TorchScript for production |

---

## Quick Inference (No API)

For single-image processing without starting the API:

```bash
# Single image
python scripts/quick_inference.py --image crack.jpg --output outputs/

# Batch processing
python scripts/quick_inference.py --input-dir images/ --output outputs/

# CPU only
python scripts/quick_inference.py --image crack.jpg --device cpu --output outputs/

# Fast inference (no TTA)
python scripts/quick_inference.py --image crack.jpg --no-tta --output outputs/
```

---

## API Usage Examples

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@crack_image.jpg" \
  -H "Authorization: Bearer your-api-key"
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -F "images=@crack1.jpg" \
  -F "images=@crack2.jpg" \
  -H "Authorization: Bearer your-api-key"
```

### With Custom Threshold
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@crack.jpg" \
  -F "threshold=0.6" \
  -F "return_uncertainty=true"
```

---

## Monitoring

### Prometheus Metrics
Available at http://localhost:8000/metrics:
- `predictions_total` - Total predictions by status
- `prediction_latency_seconds` - Latency histogram
- `active_requests` - Current concurrent requests
- `model_loaded` - Model availability

### Grafana Dashboards
Access at http://localhost:3000
- Pre-configured dashboards for API performance
- Model health monitoring
- System resource tracking

---

## Troubleshooting

### Model Not Loading
```bash
# Check logs
docker compose -f docker-compose.prod.yml logs api

# Verify weights exist
ls -la checkpoints/
```

### High Latency
1. Disable TTA: Set `USE_TTA=false` in .env
2. Enable mixed precision: Set `MIXED_PRECISION=true`
3. Use GPU: Ensure CUDA is available

### Out of Memory
1. Reduce batch size in requests
2. Disable TTA temporarily
3. Use CPU inference: Set `CUDA_VISIBLE_DEVICES=""`

---

## Production Checklist

- [ ] `.env` file configured with secure API key
- [ ] Model weights present in `checkpoints/`
- [ ] Docker and Docker Compose installed
- [ ] Services started with `docker compose -f docker-compose.prod.yml up -d`
- [ ] Health check passes: `curl http://localhost:8000/health`
- [ ] Test inference works: `python scripts/verify_production.py`
- [ ] Monitoring accessible: http://localhost:3000

---

## Support

- Check logs: `docker compose -f docker-compose.prod.yml logs -f api`
- Verify deployment: `python scripts/verify_production.py`
- API documentation: http://localhost:8000/docs
