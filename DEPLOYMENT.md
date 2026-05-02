# CrackGraphAI Production Deployment Guide

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit with your production values
nano .env
```

### 2. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit UI (in another terminal)
streamlit run ui/streamlit_app.py
```

### 3. Docker Deployment

```bash
# Build and run
docker-compose up -d

# With monitoring stack
docker-compose --profile monitoring up -d

# With Redis caching
docker-compose --profile cache up -d
```

### 4. Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f k8s/

# Or use Helm
helm install crackgraphai ./helm-chart
```

## API Authentication

### Dev Mode (Default)
API key is `dev-key-change-in-production` — no auth header required.

### Production Mode
Set a secure API key:
```bash
export API_KEY="your-secure-random-key"
```

Then include in requests:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer $API_KEY" \
  -F "image=@crack.jpg"
```

## API Endpoints

| Endpoint | Method | Description | Rate Limit |
|----------|--------|-------------|------------|
| `/health` | GET | Health check | None |
| `/ready` | GET | K8s readiness probe | None |
| `/metrics` | GET | Prometheus metrics | None |
| `/predict` | POST | Single image prediction | 10/min |
| `/predict_batch` | POST | Batch prediction (max 10) | 5/min |

## Monitoring

### Prometheus Metrics

Available at `http://localhost:8000/metrics`:

- `predictions_total` — Total predictions by status
- `prediction_latency_seconds` — Prediction latency histogram
- `batch_size` — Batch size distribution

### Health Checks

```bash
# Health check
curl http://localhost:8000/health

# Readiness probe (K8s)
curl http://localhost:8000/ready

# Metrics
curl http://localhost:8000/metrics
```

## Security Checklist

- [ ] Changed default `API_KEY` from `dev-key-change-in-production`
- [ ] Running as non-root user in container
- [ ] HTTPS enabled (via reverse proxy)
- [ ] Rate limiting configured appropriately
- [ ] File size limits set (10MB default)
- [ ] Input validation enabled
- [ ] Logging configured
- [ ] Secrets not in code/repo

## Troubleshooting

### "Service unavailable" errors
Check model is loaded: `curl http://localhost:8000/health`

### High latency
- Enable GPU: `CUDA_VISIBLE_DEVICES=0`
- Check Prometheus metrics for bottlenecks
- Consider batching requests

### Out of memory
- Reduce `batch_size` in config
- Use smaller input images
- Enable model quantization

## Performance Tuning

```bash
# Multi-worker (CPU only, not recommended for GPU)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn (production)
gunicorn api.main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## CI/CD

GitHub Actions workflow:
- Runs on push to `main`/`develop`
- Tests with pytest
- Builds Docker image
- Scans with Trivy
- Deploys to staging/production

See `.github/workflows/ci-cd.yml`
