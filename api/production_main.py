"""
Production-grade API with stability guarantees, circuit breaker, and caching.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np
import torch
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Security,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

# Import our stable inference engine
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.engine import StableInferenceEngine, InferenceConfig
from inference.cache import ResultCache
from features.structural_features import extract_structural_features
from graph.crack_graph import skeleton_to_graph, draw_keypoints_overlay
from models.hybrid_segformer_unet import HybridSegformerUNet
from topology.skeleton import connectivity_score, mask_to_skeleton
from utils.config import load_config


def post_process_mask(mask: np.ndarray, min_aspect_ratio: float = 2.0, min_area: int = 20) -> np.ndarray:
    """Post-process segmentation mask to keep only crack-like structures.

    Filters out blob artifacts by keeping only elongated connected components.
    Falls back to largest component if filtering removes everything.

    Args:
        mask: Binary segmentation mask (0 or 1)
        min_aspect_ratio: Minimum aspect ratio (length/width) to keep component
        min_area: Minimum pixel area to keep

    Returns:
        Cleaned binary mask with crack-like structures
    """
    from skimage.measure import label, regionprops

    if mask.sum() == 0:
        return mask

    # Label connected components
    labeled = label(mask.astype(bool))
    regions = list(regionprops(labeled))

    if not regions:
        return mask

    cleaned_mask = np.zeros_like(mask, dtype=np.uint8)
    largest_region = max(regions, key=lambda r: r.area)
    kept_regions = 0

    for region in regions:
        # Skip small components
        if region.area < min_area:
            continue

        # Calculate aspect ratio using oriented bounding box
        aspect_ratio = 0
        if region.orientation is not None:
            # Get major and minor axis lengths (handle deprecated API)
            try:
                major_axis = region.axis_major_length
                minor_axis = region.axis_minor_length
            except AttributeError:
                major_axis = region.major_axis_length
                minor_axis = region.minor_axis_length

            if minor_axis > 0:
                aspect_ratio = major_axis / minor_axis
            else:
                aspect_ratio = major_axis if major_axis > 0 else 0

        # Keep elongated structures OR the largest region (fallback for main crack)
        is_elongated = aspect_ratio >= min_aspect_ratio
        is_main_crack = region.label == largest_region.label

        if is_elongated or is_main_crack:
            coords = region.coords
            cleaned_mask[coords[:, 0], coords[:, 1]] = 1
            kept_regions += 1

    # If nothing passed filters, fall back to largest component
    if cleaned_mask.sum() == 0 and largest_region.area > 10:
        coords = largest_region.coords
        cleaned_mask[coords[:, 0], coords[:, 1]] = 1

    return cleaned_mask

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger("crackgraphai.production")

# Prometheus metrics
PREDICTION_COUNTER = Counter("predictions_total", "Total predictions", ["status", "cache_hit"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", ["cache_hit"])
BATCH_SIZE_HISTOGRAM = Histogram("batch_size", "Batch size distribution", buckets=[1, 2, 5, 10, 20, 50])
ACTIVE_REQUESTS = Gauge("active_requests", "Number of active requests")
QUEUE_SIZE = Gauge("request_queue_size", "Current request queue size")
MODEL_LOADED = Gauge("model_loaded", "Model loaded status")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer(auto_error=False)


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitBreakerState.CLOSED
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - (self.last_failure_time or 0) > self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Service temporarily unavailable (circuit breaker open)",
                    )
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Service temporarily unavailable (circuit breaker half-open limit)",
                    )
                self.half_open_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            
            async with self._lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.CLOSED
                    self.failures = 0
                    self.half_open_calls = 0
                    logger.info("Circuit breaker CLOSED (recovered)")
                else:
                    self.failures = max(0, self.failures - 1)
            
            return result
            
        except Exception as e:
            async with self._lock:
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(f"Circuit breaker OPENED after {self.failures} failures")
            
            raise


class RequestQueue:
    """Manage concurrent request queue for load shedding."""
    
    def __init__(self, max_concurrent: int = 5, max_queue: int = 20):
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue_size = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Try to acquire semaphore with queue management."""
        async with self._lock:
            if self.queue_size >= self.max_queue:
                return False
            self.queue_size += 1
            QUEUE_SIZE.set(self.queue_size)
        
        await self.semaphore.acquire()
        return True
    
    def release(self):
        """Release semaphore."""
        async def _release():
            async with self._lock:
                self.queue_size = max(0, self.queue_size - 1)
                QUEUE_SIZE.set(self.queue_size)
            self.semaphore.release()
        
        asyncio.create_task(_release())


class ProductionInferenceService:
    """Production inference service with stability and fault tolerance."""
    
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        weights_path: str = "checkpoints/best_hybrid_segformer.pth",
        inference_config: Optional[InferenceConfig] = None,
    ):
        self.cfg = load_config(config_path)
        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inference configuration
        self.inference_config = inference_config or InferenceConfig(
            use_tta=True,
            tta_flips=['horizontal', 'vertical'],
            tta_merge_strategy='mean',
            apply_morphology=True,
            morphology_kernel_size=3,
            min_component_size=16,
            estimate_uncertainty=True,
            enable_cache=True,
            cache_ttl=3600,
            mixed_precision=self.device.type == 'cuda',
        )
        
        # Initialize cache
        self.cache = ResultCache(
            ttl=self.inference_config.cache_ttl,
            max_size=1000,
            persistent=False,
        )
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
        )
        
        # Request queue
        self.request_queue = RequestQueue(
            max_concurrent=3,
            max_queue=10,
        )
        
        self.model: Optional[HybridSegformerUNet] = None
        self.engine: Optional[StableInferenceEngine] = None
        self._healthy = False
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model with error handling."""
        try:
            logger.info(f"Loading model from {self.weights_path} to {self.device}")
            
            self.model = HybridSegformerUNet(
                cnn_backbone=self.cfg["model"]["cnn_backbone"],
                transformer_backbone=self.cfg["model"]["transformer_backbone"],
                classes=self.cfg["model"]["classes"],
            ).to(self.device)
            
            if not os.path.exists(self.weights_path):
                logger.warning(f"Weights file not found: {self.weights_path}")
                self._healthy = False
                MODEL_LOADED.set(0)
                return
            
            ckpt = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            
            # Initialize stable inference engine
            self.engine = StableInferenceEngine(
                model=self.model,
                config=self.inference_config,
                cache=self.cache,
            )
            
            # Warmup
            self.engine.warmup(num_runs=3)
            
            self._healthy = True
            MODEL_LOADED.set(1)
            logger.info("Model loaded and warmed up successfully")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self._healthy = False
            MODEL_LOADED.set(0)
            raise
    
    def is_healthy(self) -> bool:
        """Check service health."""
        return self._healthy and self.model is not None and self.engine is not None
    
    def _compute_cache_key(self, image_bytes: bytes, params: Dict) -> str:
        """Compute deterministic cache key."""
        hasher = hashlib.sha256()
        hasher.update(image_bytes)
        hasher.update(str(sorted(params.items())).encode())
        return hasher.hexdigest()
    
    def _encode_png(self, arr: np.ndarray) -> str:
        """Encode array as base64 PNG."""
        arr_u8 = (arr * 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", arr_u8)
        if not ok:
            raise RuntimeError("Failed to encode PNG")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _encode_color_png(self, arr: np.ndarray) -> str:
        """Encode a BGR uint8 colour image as a base64-encoded PNG string."""
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 255)).astype(np.uint8)
        ok, buf = cv2.imencode(".png", arr)
        if not ok:
            raise RuntimeError("Failed to encode PNG")
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    
    async def infer(self, file_bytes: bytes, request_id: str, params: Dict) -> Dict:
        """Run full inference pipeline with stability mechanisms."""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._compute_cache_key(file_bytes, params)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            logger.info(f"[{request_id}] Cache hit, returning cached result")
            cached_result['latency_seconds'] = time.time() - start_time
            cached_result['from_cache'] = True
            PREDICTION_COUNTER.labels(status="success", cache_hit="true").inc()
            PREDICTION_LATENCY.labels(cache_hit="true").observe(time.time() - start_time)
            return cached_result
        
        # Check if engine is ready
        if not self.engine:
            raise RuntimeError("Inference engine not initialized")
        
        # Run inference with TTA and stability features
        result = self.engine.predict(
            image=file_bytes,
            return_uncertainty=params.get('return_uncertainty', True),
            custom_threshold=params.get('threshold'),
        )
        
        # Post-processing pipeline
        raw_mask = result['mask']
        prob_map = result['probability_map']

        # Filter out blob artifacts, keep only crack-like structures
        mask = post_process_mask(raw_mask, min_aspect_ratio=2.0, min_area=20)

        # Log filtering results
        filtered_pixels = int(raw_mask.sum()) - int(mask.sum())
        if filtered_pixels > 0:
            logger.info(f"[{request_id}] Post-processing filtered {filtered_pixels} pixels ({filtered_pixels/max(1,int(raw_mask.sum()))*100:.1f}%)")

        # Skeletonization
        skel = mask_to_skeleton(mask, min_size=self.inference_config.min_component_size)
        
        # Graph analysis
        graph = skeleton_to_graph(skel)
        feats = extract_structural_features(graph)
        conn = connectivity_score(skel)
        
        # PROPER DAMAGE-BASED SCORING
        # Structural Integrity = 1.0 - Damage Severity
        # No cracks = 1.0 (Perfect), Massive cracks = 0.0 (Critical)

        skel_density = feats["total_crack_length"] / (256 * 256) if "total_crack_length" in feats else 0.0
        num_branches = feats.get("num_branches", 0)
        junctions = feats.get("junctions", 0)

        # 1. CRACK DENSITY DAMAGE (0 to 1): How much surface is cracked
        # Scale: 0% coverage = 0 damage, 30%+ coverage = max damage (1.0)
        density_damage = min(1.0, skel_density * 3.33)  # 30% coverage = full damage

        # 2. CONNECTIVITY DAMAGE (0 to 1): How interconnected the cracks are
        # Well-connected cracks form networks = more severe than isolated cracks
        connectivity_damage = float(conn)  # 0 = isolated cracks, 1 = fully connected network

        # 3. COMPLEXITY DAMAGE (0 to 1): Branching and junction complexity
        # More branches/junctions = more complex crack pattern = higher damage
        complexity_score = min(1.0, (num_branches + junctions) / 20.0)  # Normalize by 20

        # WEIGHTED DAMAGE COMBINATION (all terms 0-1, higher = more damage)
        # Weights: Density (50%), Connectivity (30%), Complexity (20%)
        total_damage = (
            0.50 * density_damage +
            0.30 * connectivity_damage +
            0.20 * complexity_score
        )

        # STRUCTURAL INTEGRITY = 1 - DAMAGE (invert so high = good structure)
        si = max(0.0, 1.0 - total_damage)

        # ── Keypoints overlay ─────────────────────────────────────────────
        # Draw endpoints (red) and junctions (yellow) directly on the
        # skeleton image so the crack topology is clearly visible.
        # skel is a uint8 binary array (0/1) in the same 256×256 coordinate
        # space as the graph nodes — no image decoding needed.
        try:
            keypoints_overlay = draw_keypoints_overlay(skel, graph)
        except Exception:
            keypoints_overlay = None

        latency = time.time() - start_time
        
        # Build response
        response = {
            "request_id": request_id,
            "segmentation_mask_png_b64": self._encode_png(mask),
            "raw_mask_png_b64": self._encode_png(raw_mask),
            "skeleton_png_b64": self._encode_png(skel),
            "probability_map_png_b64": self._encode_png(prob_map),
            "keypoints_overlay_png_b64": self._encode_color_png(keypoints_overlay) if keypoints_overlay is not None else None,
            "graph_features": feats,
            "connectivity_score": float(conn),
            "si_score": float(si),
            "damage_metrics": {
                "density_damage": round(density_damage, 3),
                "connectivity_damage": round(connectivity_damage, 3),
                "complexity_damage": round(complexity_score, 3),
                "total_damage": round(total_damage, 3),
            },
            "post_processing": {
                "raw_pixels": int(raw_mask.sum()),
                "filtered_pixels": filtered_pixels,
                "final_pixels": int(mask.sum()),
                "filtering_applied": bool(filtered_pixels > 0),
            },
            "latency_seconds": round(latency, 3),
            "from_cache": False,
        }
        
        # Add uncertainty if available
        if 'uncertainty' in result:
            response['uncertainty'] = {
                'mean_uncertainty': result['uncertainty']['mean_uncertainty'],
                'mean_confidence': result['uncertainty']['mean_confidence'],
                'reliable': result['uncertainty']['reliable'],
            }
        
        # Cache result
        self.cache.set(cache_key, response.copy())
        
        # Update metrics
        PREDICTION_COUNTER.labels(status="success", cache_hit="false").inc()
        PREDICTION_LATENCY.labels(cache_hit="false").observe(latency)
        
        logger.info(f"[{request_id}] Inference completed in {latency:.3f}s")
        
        return response
    
    async def infer_with_protection(self, file_bytes: bytes, request_id: str, params: Dict) -> Dict:
        """Run inference with circuit breaker and queue protection."""
        # Acquire queue slot
        acquired = await self.request_queue.acquire()
        if not acquired:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Server overloaded, please try again later",
            )
        
        ACTIVE_REQUESTS.inc()
        
        try:
            # Run with circuit breaker
            result = await self.circuit_breaker.call(
                self.infer, file_bytes, request_id, params
            )
            return result
        finally:
            self.request_queue.release()
            ACTIVE_REQUESTS.dec()


# Global service instance
service: Optional[ProductionInferenceService] = None


def get_api_key() -> str:
    """Get API key from environment."""
    return os.getenv("API_KEY", "dev-key-change-in-production")


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> str:
    """Verify Bearer token."""
    expected = get_api_key()
    if expected == "dev-key-change-in-production":
        return "dev"
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global service
    logger.info("Starting CrackGraphAI Production API...")
    
    try:
        config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
        weights_path = os.getenv("WEIGHTS_PATH", "checkpoints/best_hybrid_segformer.pth")
        
        # Parse inference config from env
        use_tta = os.getenv("USE_TTA", "true").lower() == "true"
        cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        
        inference_config = InferenceConfig(
            use_tta=use_tta,
            cache_ttl=cache_ttl,
            enable_cache=True,
        )
        
        service = ProductionInferenceService(
            config_path=config_path,
            weights_path=weights_path,
            inference_config=inference_config,
        )
        
        logger.info("Production service initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    finally:
        logger.info("Shutting down CrackGraphAI Production API...")
        if service and service.engine:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("GPU memory cleared")


# Create FastAPI app
app = FastAPI(
    title="CrackGraphAI Production API",
    version="2.0.0",
    description="Production-grade crack segmentation with stability guarantees",
    lifespan=lifespan,
)

# Middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda req, exc: JSONResponse(
    status_code=429,
    content={"detail": "Rate limit exceeded"},
))
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging."""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start_time = time.time()
        
        logger.info(f"[{request_id}] {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        latency = time.time() - start_time
        logger.info(f"[{request_id}] Response {response.status_code} in {latency:.3f}s")
        
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(LoggingMiddleware)


@app.get("/health")
async def health():
    """Health check endpoint."""
    if service is None or not service.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        )
    
    cache_stats = service.cache.stats()
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(service.device),
        "version": "2.0.0",
        "cache_stats": cache_stats,
        "circuit_breaker": service.circuit_breaker.state.value,
    }


@app.get("/ready")
async def ready():
    """Readiness probe for Kubernetes."""
    if service is None or not service.is_healthy():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return {"ready": True}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
@limiter.limit("30/minute")
async def predict(
    request: Request,
    image: UploadFile = File(...),
    threshold: Optional[float] = None,
    return_uncertainty: bool = True,
    token: str = Depends(verify_token),
):
    """
    Single image prediction endpoint with stability guarantees.
    
    Args:
        image: Input image file
        threshold: Optional custom threshold (0-1)
        return_uncertainty: Whether to return uncertainty estimates
    """
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialized",
        )
    
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type: {image.content_type}. Allowed: {allowed_types}",
        )
    
    # Validate file size (10MB limit)
    max_size = 10 * 1024 * 1024
    content = await image.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {len(content)} bytes (max: {max_size})",
        )
    
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    
    params = {
        'threshold': threshold,
        'return_uncertainty': return_uncertainty,
    }
    
    try:
        result = await service.infer_with_protection(content, request_id, params)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Inference failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@app.post("/predict_batch")
@limiter.limit("10/minute")
async def predict_batch(
    request: Request,
    images: List[UploadFile] = File(...),
    threshold: Optional[float] = None,
    return_uncertainty: bool = False,
    token: str = Depends(verify_token),
):
    """Batch prediction endpoint with queue management."""
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialized",
        )
    
    # Limit batch size
    max_batch = 10
    if len(images) > max_batch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch too large: {len(images)} images (max: {max_batch})",
        )
    
    BATCH_SIZE_HISTOGRAM.observe(len(images))
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    
    params = {
        'threshold': threshold,
        'return_uncertainty': return_uncertainty,
    }
    
    outputs = []
    for img in images:
        allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
        if img.content_type not in allowed_types:
            outputs.append({
                "filename": img.filename,
                "error": f"Invalid content type: {img.content_type}",
            })
            continue
        
        content = await img.read()
        max_size = 10 * 1024 * 1024
        if len(content) > max_size:
            outputs.append({
                "filename": img.filename,
                "error": f"File too large: {len(content)} bytes",
            })
            continue
        
        try:
            result = await service.infer_with_protection(
                content, 
                f"{request_id}-{img.filename}",
                params,
            )
            outputs.append({"filename": img.filename, "result": result})
        except HTTPException as e:
            outputs.append({"filename": img.filename, "error": e.detail})
        except Exception as e:
            outputs.append({"filename": img.filename, "error": str(e)})
    
    return {"request_id": request_id, "items": outputs}


@app.get("/cache/stats")
async def cache_stats(token: str = Depends(verify_token)):
    """Get cache statistics."""
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    return service.cache.stats()


@app.delete("/cache/clear")
async def cache_clear(token: str = Depends(verify_token)):
    """Clear inference cache."""
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    service.cache.clear()
    return {"message": "Cache cleared"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "request_id": request_id},
    )
