#!/usr/bin/env python3
"""
Quick inference script for CrackGraphAI.
Run inference directly on images without starting the API server.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference.engine import InferenceConfig, StableInferenceEngine
from models.hybrid_segformer_unet import HybridSegformerUNet
from utils.config import load_config


def load_model(config_path: str, weights_path: str, device: str = "auto"):
    """Load the model and create inference engine."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {weights_path}")
    print(f"Using device: {device}")

    cfg = load_config(config_path)

    model = HybridSegformerUNet(
        cnn_backbone=cfg["model"]["cnn_backbone"],
        transformer_backbone=cfg["model"]["transformer_backbone"],
        classes=cfg["model"]["classes"],
    ).to(device)

    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Create inference engine with optimized settings
    inference_config = InferenceConfig(
        model_path=weights_path,
        device=device,
        use_tta=True,
        tta_flips=["horizontal", "vertical"],
        tta_merge_strategy="mean",
        apply_morphology=True,
        morphology_kernel_size=3,
        min_component_size=16,
        estimate_uncertainty=True,
        enable_cache=False,  # Disable cache for single inference
        mixed_precision=device == "cuda",
    )

    engine = StableInferenceEngine(model=model, config=inference_config)

    # Warmup
    print("Warming up model...")
    engine.warmup(num_runs=2)

    return engine, cfg


def process_image(engine: StableInferenceEngine, image_path: str, output_dir: Optional[str] = None) -> Dict:
    """Process a single image."""
    print(f"\nProcessing: {image_path}")

    # Load image
    image_bytes = open(image_path, "rb").read()

    # Run inference
    start_time = time.time()
    result = engine.predict(
        image=image_bytes,
        return_uncertainty=True,
    )
    total_time = time.time() - start_time

    # Print results
    print(f"  Inference time: {result['inference_time']:.3f}s")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Image size: {result['image_size']}")

    if "uncertainty" in result:
        unc = result["uncertainty"]
        print(f"  Mean confidence: {unc['mean_confidence']:.3f}")
        print(f"  Reliable: {unc['reliable']}")

    # Get mask info
    mask = result["mask"]
    crack_pixels = np.sum(mask)
    total_pixels = mask.size
    crack_percentage = (crack_pixels / total_pixels) * 100

    print(f"  Crack coverage: {crack_pixels} pixels ({crack_percentage:.2f}%)")

    # Save outputs if requested
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        base_name = Path(image_path).stem

        # Save mask
        mask_path = output_path / f"{base_name}_mask.png"
        cv2.imwrite(str(mask_path), mask * 255)
        print(f"  Saved mask: {mask_path}")

        # Save probability map
        prob_path = output_path / f"{base_name}_prob.npy"
        np.save(prob_path, result["probability_map"])
        print(f"  Saved probability map: {prob_path}")

        # Save visualization
        vis_path = output_path / f"{base_name}_visualization.png"
        create_visualization(image_path, mask, result["probability_map"], vis_path)
        print(f"  Saved visualization: {vis_path}")

        # Save JSON results
        json_path = output_path / f"{base_name}_results.json"
        save_results_json(result, json_path, total_time)
        print(f"  Saved results: {json_path}")

    return result


def create_visualization(image_path: str, mask: np.ndarray, prob_map: np.ndarray, output_path: Path):
    """Create visualization overlay."""
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        # Try PIL fallback
        pil_img = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Resize to match mask
    h, w = mask.shape
    img = cv2.resize(img, (w, h))

    # Create overlay
    overlay = img.copy()
    overlay[mask > 0] = [0, 0, 255]  # Red for cracks

    # Blend
    alpha = 0.5
    vis = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

    # Create side-by-side
    prob_colored = (prob_map * 255).astype(np.uint8)
    prob_colored = cv2.applyColorMap(prob_colored, cv2.COLORMAP_JET)

    # Stack: original | overlay | probability
    combined = np.hstack([img, vis, prob_colored])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Segmentation", (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Probability", (2 * w + 10, 30), font, 1, (255, 255, 255), 2)

    cv2.imwrite(str(output_path), combined)


def save_results_json(result: Dict, output_path: Path, total_time: float):
    """Save results to JSON."""
    mask = result["mask"]

    # Convert numpy arrays to lists for JSON serialization
    json_result = {
        "inference_time_seconds": result["inference_time"],
        "total_time_seconds": total_time,
        "threshold": result["threshold"],
        "image_size": result["image_size"],
        "crack_pixels": int(np.sum(mask)),
        "total_pixels": int(mask.size),
        "crack_percentage": float(np.sum(mask) / mask.size * 100),
    }

    if "uncertainty" in result:
        json_result["uncertainty"] = {
            "mean_uncertainty": float(result["uncertainty"]["mean_uncertainty"]),
            "mean_confidence": float(result["uncertainty"]["mean_confidence"]),
            "reliable": bool(result["uncertainty"]["reliable"]),
            "edge_ratio": float(result["uncertainty"]["edge_ratio"]),
            "high_confidence_ratio": float(result["uncertainty"]["high_confidence_ratio"]),
        }

    with open(output_path, "w") as f:
        json.dump(json_result, f, indent=2)


def batch_process(engine: StableInferenceEngine, image_paths: list, output_dir: str):
    """Process multiple images."""
    print(f"\nBatch processing {len(image_paths)} images...")

    results = []
    total_start = time.time()

    for i, path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {path}")
        try:
            result = process_image(engine, path, output_dir)
            results.append({"path": path, "status": "success", "result": result})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"path": path, "status": "error", "error": str(e)})

    total_time = time.time() - total_start

    # Print summary
    print(f"\n{'='*60}")
    print("Batch Processing Complete")
    print(f"{'='*60}")
    print(f"Total images: {len(image_paths)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'error')}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {total_time / len(image_paths):.2f}s per image")

    # Save batch summary
    if output_dir:
        summary_path = Path(output_dir) / "batch_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "total_images": len(image_paths),
                "successful": sum(1 for r in results if r["status"] == "success"),
                "failed": sum(1 for r in results if r["status"] == "error"),
                "total_time_seconds": total_time,
                "average_time_seconds": total_time / len(image_paths),
                "results": [{"path": r["path"], "status": r["status"]} for r in results],
            }, f, indent=2)
        print(f"Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick inference for CrackGraphAI - Process images directly without API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python scripts/quick_inference.py --image path/to/crack.jpg --output outputs/

  # Batch processing
  python scripts/quick_inference.py --input-dir path/to/images/ --output outputs/

  # CPU inference
  python scripts/quick_inference.py --image crack.jpg --device cpu --output outputs/

  # Disable TTA for faster inference
  python scripts/quick_inference.py --image crack.jpg --no-tta --output outputs/
        """
    )

    parser.add_argument("--image", "-i", help="Path to single image")
    parser.add_argument("--input-dir", "-d", help="Directory containing images")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument("--config", "-c", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--weights", "-w", default="checkpoints/best_hybrid_segformer.pth", help="Model weights path")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device to use")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Segmentation threshold")

    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error("Either --image or --input-dir must be specified")

    print("=" * 60)
    print("CrackGraphAI Quick Inference")
    print("=" * 60)

    # Load model
    engine, cfg = load_model(args.config, args.weights, args.device)

    # Disable TTA if requested
    if args.no_tta:
        engine.config.use_tta = False
        print("TTA disabled")

    # Set custom threshold
    engine.config.ensemble_threshold = args.threshold

    # Process images
    if args.image:
        process_image(engine, args.image, args.output)
    else:
        # Batch processing
        image_dir = Path(args.input_dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = [
            str(p) for p in image_dir.iterdir()
            if p.suffix.lower() in extensions
        ]

        if not image_paths:
            print(f"No images found in {args.input_dir}")
            sys.exit(1)

        batch_process(engine, image_paths, args.output)

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
