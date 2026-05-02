from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import onnxruntime as ort

from utils.config import load_config


def preprocess(image_bgr: np.ndarray, size: int) -> np.ndarray:
    image = cv2.cvtColor(cv2.resize(image_bgr, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = ((image - mean) / std).transpose(2, 0, 1)[None, ...]
    return x


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = float(np.logical_and(pred_mask, gt_mask).sum())
    return float((2.0 * inter) / (pred_mask.sum() + gt_mask.sum() + 1e-8))


def main():
    parser = argparse.ArgumentParser(description="Evaluate ONNX crack segmentation model against GT masks.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--onnx", default="checkpoints/hybrid_segformer.onnx")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--show-worst", type=int, default=10)
    parser.add_argument("--sweep-start", type=float, default=None)
    parser.add_argument("--sweep-end", type=float, default=None)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_size = int(cfg["data"]["image_size"])
    image_dir = Path(cfg["data"]["root_dir"]) / cfg["data"]["image_dir_name"]
    mask_dir = Path(cfg["data"]["root_dir"]) / cfg["data"]["mask_dir_name"]

    image_paths = sorted(image_dir.glob("CFD_*.jpg"))[: args.num_images]
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    prepared = []
    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            continue
        image = cv2.imread(str(image_path))
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or gt is None:
            continue
        x = preprocess(image, image_size)
        logits = sess.run(None, {"image": x})[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        gt_bin = cv2.resize(gt, (image_size, image_size), interpolation=cv2.INTER_NEAREST) > 127
        prepared.append((image_path.name, probs[0, 0], gt_bin))

    def evaluate_for_threshold(threshold: float):
        rows = []
        for name, prob_map, gt_bin in prepared:
            pred = prob_map > threshold
            rows.append((dice_score(pred, gt_bin), name))
        return rows

    if args.sweep_start is not None and args.sweep_end is not None:
        best_t = None
        best_mean = -1.0
        t = float(args.sweep_start)
        while t <= float(args.sweep_end) + 1e-12:
            rows = evaluate_for_threshold(t)
            if rows:
                dices = np.array([r[0] for r in rows], dtype=np.float32)
                mean_dice = float(dices.mean())
                print({"threshold": round(t, 4), "mean_dice": mean_dice})
                if mean_dice > best_mean:
                    best_mean = mean_dice
                    best_t = t
            t += float(args.sweep_step)
        print({"best_threshold": best_t, "best_mean_dice": best_mean})
        return

    rows = evaluate_for_threshold(float(args.threshold))

    if not rows:
        raise RuntimeError("No valid image/mask pairs evaluated.")

    dices = np.array([r[0] for r in rows], dtype=np.float32)
    rows_sorted = sorted(rows, key=lambda r: r[0])
    k = max(1, min(args.show_worst, len(rows_sorted)))

    print(
        {
            "count": int(len(rows)),
            "mean_dice": float(dices.mean()),
            "min_dice": float(dices.min()),
            "max_dice": float(dices.max()),
            "threshold": float(args.threshold),
        }
    )
    print({"worst_samples": rows_sorted[:k]})


if __name__ == "__main__":
    main()
