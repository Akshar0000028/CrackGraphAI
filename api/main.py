from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from features.structural_features import extract_structural_features
from features.si_scoring import compute_structural_integrity, classify_risk
from graph.crack_graph import skeleton_to_graph, draw_keypoints_overlay
from models.hybrid_segformer_unet import HybridSegformerUNet
from topology.skeleton import connectivity_score, mask_to_skeleton
from utils.config import load_config


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def post_process_mask(
    mask: np.ndarray,
    min_aspect_ratio: float = 1.5,
    min_area: int = 20,
) -> np.ndarray:
    """
    Remove non-crack noise from a binary segmentation mask.

    Strategy
    --------
    1. Morphological opening  – removes isolated salt-and-pepper noise.
    2. Connected-component filter – keeps only regions that look like cracks:
         • elongated  (aspect ratio ≥ min_aspect_ratio), OR
         • thin       (solidity < 0.75), OR
         • small      (area < 500 px, could be a short crack tip)
    3. Morphological closing  – reconnects nearby crack segments.
    4. Final tiny-region removal.

    Parameters
    ----------
    mask            : binary mask (0/1 uint8)
    min_aspect_ratio: minimum major/minor axis ratio to keep a region
    min_area        : minimum pixel area to keep (removes single-pixel noise)
    """
    from skimage.measure import label, regionprops

    if mask.sum() == 0:
        return mask

    # Step 1 – morphological opening (small kernel to preserve fine cracks)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_open, iterations=1
    )

    # Step 2 – connected-component analysis
    labeled = label(opened.astype(bool))
    regions = regionprops(labeled)

    if not regions:
        return mask  # nothing survived opening; return original

    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for region in regions:
        if region.area < min_area:
            continue  # too small → noise

        # Aspect ratio: major_axis_length / minor_axis_length
        # Both attributes exist in all scikit-image versions ≥ 0.14
        major = region.major_axis_length
        minor = region.minor_axis_length
        if minor > 0:
            aspect_ratio = major / minor
        else:
            # Degenerate region (single-pixel line) → treat as infinitely thin
            aspect_ratio = float("inf")

        solidity = region.solidity  # area / convex_area

        is_elongated = aspect_ratio >= min_aspect_ratio
        is_thin = solidity < 0.75
        is_small_crack = region.area < 500  # short crack tips

        if is_elongated or is_thin or is_small_crack:
            coords = region.coords
            cleaned[coords[:, 0], coords[:, 1]] = 1

    # Step 3 – morphological closing (reconnect nearby segments)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Step 4 – remove tiny isolated regions created by closing
    labeled_final = label(closed.astype(bool))
    final = np.zeros_like(mask, dtype=np.uint8)
    for region in regionprops(labeled_final):
        if region.area >= min_area:
            coords = region.coords
            final[coords[:, 0], coords[:, 1]] = 1

    return final


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("crackgraphai")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

PREDICTION_COUNTER = Counter("predictions_total", "Total predictions", ["status"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
BATCH_SIZE_HISTOGRAM = Histogram(
    "batch_size", "Batch size distribution", buckets=[1, 2, 5, 10, 20, 50]
)

# ---------------------------------------------------------------------------
# Rate limiter & security
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)
security = HTTPBearer(auto_error=False)


def get_api_key() -> str:
    return os.getenv("API_KEY", "dev-key-change-in-production")


def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> str:
    expected = get_api_key()
    if expected == "dev-key-change-in-production":
        return "dev"  # skip auth in dev mode
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


# ---------------------------------------------------------------------------
# Inference service
# ---------------------------------------------------------------------------

class InferenceService:
    """Production inference service with health tracking."""

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        weights_path: str = "checkpoints/best_hybrid_segformer.pth",
    ) -> None:
        self.cfg = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        self.model: Optional[HybridSegformerUNet] = None
        self._healthy = False
        self._load_model()

    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        try:
            logger.info(f"Loading model from {self.weights_path} on {self.device}")
            self.model = HybridSegformerUNet(
                cnn_backbone=self.cfg["model"]["cnn_backbone"],
                transformer_backbone=self.cfg["model"]["transformer_backbone"],
                classes=self.cfg["model"]["classes"],
            ).to(self.device)
            ckpt = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            self._healthy = True
            logger.info("Model loaded successfully")
        except Exception as exc:
            logger.error(f"Model loading failed: {exc}")
            self._healthy = False
            raise

    def is_healthy(self) -> bool:
        return self._healthy and self.model is not None

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        """
        BGR image → normalised float tensor (1, 3, H, W).

        Steps
        -----
        1. Resize to model input size.
        2. CLAHE contrast enhancement (improves crack visibility on dark surfaces).
        3. BGR → RGB.
        4. ImageNet normalisation.
        """
        size = self.cfg["data"]["image_size"]

        resized = cv2.resize(bgr, (size, size))

        # CLAHE on L channel of LAB colour space
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        x = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std

        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()

    # ------------------------------------------------------------------ #
    # Prediction with TTA
    # ------------------------------------------------------------------ #

    def _predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Run model inference with optional Test-Time Augmentation (TTA).

        TTA averages predictions over the original image plus horizontal
        and vertical flips, which improves robustness on unseen orientations.

        Returns a binary mask (uint8, 0/1) at the model's output resolution.
        """
        use_tta = self.cfg["inference"].get("use_tta", True)
        threshold = float(self.cfg["inference"]["threshold"])

        with torch.no_grad():
            x_dev = x.to(self.device)
            probs = torch.sigmoid(self.model(x_dev))

            if use_tta:
                # Horizontal flip
                hf = torch.flip(x_dev, dims=[3])
                probs_hf = torch.flip(torch.sigmoid(self.model(hf)), dims=[3])
                # Vertical flip
                vf = torch.flip(x_dev, dims=[2])
                probs_vf = torch.flip(torch.sigmoid(self.model(vf)), dims=[2])
                # Average
                probs = (probs + probs_hf + probs_vf) / 3.0

        return (probs.squeeze().cpu().numpy() > threshold).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #

    def _encode_png(self, arr: np.ndarray) -> str:
        """Encode a binary/float array as a base64-encoded PNG string."""
        arr_u8 = (arr * 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", arr_u8)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _encode_color_png(self, arr: np.ndarray) -> str:
        """Encode a BGR uint8 colour image as a base64-encoded PNG string."""
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 255)).astype(np.uint8)
        ok, buf = cv2.imencode(".png", arr)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # ------------------------------------------------------------------ #
    # Full inference pipeline
    # ------------------------------------------------------------------ #

    def infer(self, file_bytes: bytes, request_id: str) -> Dict:
        """
        End-to-end inference: raw image bytes → structured result dict.

        Pipeline
        --------
        decode → preprocess → model (+ TTA) → post-process → skeletonise
        → graph → features → SI score → encode outputs
        """
        t0 = time.time()
        logger.info(f"[{request_id}] Starting inference")

        try:
            # ── Decode ────────────────────────────────────────────────────
            np_arr = np.frombuffer(file_bytes, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Invalid or corrupted image file")

            # ── Model ─────────────────────────────────────────────────────
            x = self._preprocess(bgr)
            raw_mask = self._predict(x)

            # ── Post-process ──────────────────────────────────────────────
            # Remove blob artefacts; keep only crack-like structures
            mask = post_process_mask(raw_mask, min_aspect_ratio=1.5, min_area=20)

            filtered_pixels = int(raw_mask.sum()) - int(mask.sum())
            if filtered_pixels > 0:
                pct = filtered_pixels / max(1, int(raw_mask.sum())) * 100
                logger.info(
                    f"[{request_id}] Post-processing removed {filtered_pixels} px "
                    f"({pct:.1f}% of raw mask)"
                )

            # ── Topology ──────────────────────────────────────────────────
            skel = mask_to_skeleton(mask)
            graph = skeleton_to_graph(skel)
            feats = extract_structural_features(graph)
            conn = connectivity_score(skel)

            # ── SI Score ──────────────────────────────────────────────────
            # At inference time we have no ground-truth mask, so dice=1.0
            # and bce=0.0 (neutral defaults).  The SI score is driven
            # entirely by observable crack properties: density, connectivity,
            # complexity, and width.
            si_result = compute_structural_integrity(
                mask=mask,
                skeleton=skel,
                graph=graph,
                connectivity_ratio=conn,
                dice=1.0,   # no GT available
                bce=0.0,    # no GT available
            )

            si = si_result["si_score"]
            risk_level = si_result["risk_level"]

            logger.info(
                f"[{request_id}] SI={si:.3f} ({risk_level}) | "
                f"density={si_result['density_damage']:.3f} "
                f"network={si_result.get('network_damage', 0):.3f} "
                f"complexity={si_result['complexity_damage']:.3f} "
                f"width={si_result.get('width_damage', 0):.3f}"
            )

            latency = time.time() - t0
            logger.info(f"[{request_id}] Done in {latency:.3f}s")
            PREDICTION_LATENCY.observe(latency)
            PREDICTION_COUNTER.labels(status="success").inc()

            # ── Keypoints overlay ─────────────────────────────────────────
            # Draw endpoints (red) and junctions (yellow) directly on the
            # skeleton image so the crack topology is clearly visible.
            # skel is a uint8 binary array (0/1) in the model's 256×256 space,
            # which is the same coordinate space as the graph nodes.
            keypoints_overlay = draw_keypoints_overlay(skel, graph)

            return {
                "request_id": request_id,
                "segmentation_mask_png_b64": self._encode_png(mask),
                "raw_mask_png_b64": self._encode_png(raw_mask),
                "skeleton_png_b64": self._encode_png(skel),
                "keypoints_overlay_png_b64": self._encode_color_png(keypoints_overlay),
                "graph_features": feats,
                "connectivity_score": round(float(conn), 4),
                "si_score": round(float(si), 4),
                "damage_metrics": {
                    "density_damage": round(si_result["density_damage"], 4),
                    "network_damage": round(si_result.get("network_damage", 0.0), 4),
                    "complexity_damage": round(si_result["complexity_damage"], 4),
                    "width_damage": round(si_result.get("width_damage", 0.0), 4),
                    "total_damage": round(si_result["total_damage"], 4),
                    "risk_level": risk_level,
                },
                "post_processing": {
                    "raw_pixels": int(raw_mask.sum()),
                    "filtered_pixels": filtered_pixels,
                    "final_pixels": int(mask.sum()),
                    "filtering_applied": filtered_pixels > 0,
                },
                "latency_seconds": round(latency, 3),
            }

        except Exception as exc:
            PREDICTION_COUNTER.labels(status="error").inc()
            logger.error(f"[{request_id}] Inference failed: {exc}", exc_info=True)
            raise


# ---------------------------------------------------------------------------
# Global service instance
# ---------------------------------------------------------------------------

service: Optional[InferenceService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    logger.info("Starting CrackGraphAI API …")
    try:
        config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
        weights_path = os.getenv("WEIGHTS_PATH", "checkpoints/best_hybrid_segformer.pth")
        service = InferenceService(config_path, weights_path)
        logger.info("Service initialised successfully")
        yield
    except Exception as exc:
        logger.error(f"Startup failed: {exc}")
        raise
    finally:
        logger.info("Shutting down CrackGraphAI API …")
        if service and service.model and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CrackGraphAI API",
    version="1.2.0",
    description="Production-grade crack segmentation and structural analysis",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        t0 = time.time()
        logger.info(f"[{request_id}] {request.method} {request.url.path}")
        response = await call_next(request)
        logger.info(
            f"[{request_id}] {response.status_code} in {time.time() - t0:.3f}s"
        )
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(LoggingMiddleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    if service is None or not service.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(service.device),
        "version": "1.2.0",
    }


@app.get("/ready")
async def ready():
    if service is None or not service.is_healthy():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return {"ready": True}


@app.get("/metrics")
async def metrics():
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
@limiter.limit("10/minute")
async def predict(
    request: Request,
    image: UploadFile = File(...),
    token: str = Depends(verify_token),
):
    """Single-image crack analysis endpoint."""
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialised",
        )

    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type: {image.content_type}. Allowed: {allowed_types}",
        )

    content = await image.read()
    max_size = 10 * 1024 * 1024  # 10 MB
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(content)} bytes, max {max_size})",
        )

    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    try:
        result = await asyncio.wait_for(
            run_in_threadpool(service.infer, content, request_id),
            timeout=30.0,
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Inference timeout")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timed out after 30 s",
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"[{request_id}] Unexpected error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@app.post("/predict_batch")
@limiter.limit("5/minute")
async def predict_batch(
    request: Request,
    images: list[UploadFile] = File(...),
    token: str = Depends(verify_token),
):
    """Batch crack analysis endpoint (max 10 images)."""
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialised",
        )

    max_batch = 10
    if len(images) > max_batch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch too large ({len(images)} images, max {max_batch})",
        )

    BATCH_SIZE_HISTOGRAM.observe(len(images))
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    max_size = 10 * 1024 * 1024

    outputs = []
    for img in images:
        if img.content_type not in allowed_types:
            outputs.append(
                {"filename": img.filename, "error": f"Invalid type: {img.content_type}"}
            )
            continue

        content = await img.read()
        if len(content) > max_size:
            outputs.append(
                {"filename": img.filename, "error": f"File too large ({len(content)} bytes)"}
            )
            continue

        try:
            result = await asyncio.wait_for(
                run_in_threadpool(
                    service.infer, content, f"{request_id}-{img.filename}"
                ),
                timeout=30.0,
            )
            outputs.append({"filename": img.filename, "result": result})
        except asyncio.TimeoutError:
            outputs.append(
                {"filename": img.filename, "error": "Inference timeout after 30 s"}
            )
        except Exception as exc:
            outputs.append({"filename": img.filename, "error": str(exc)})

    return {"request_id": request_id, "items": outputs}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "request_id": request_id},
    )
