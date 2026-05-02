from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from features.si_scoring import compute_structural_integrity
from features.structural_features import extract_structural_features
from graph.crack_graph import skeleton_to_graph
from models.hybrid_segformer_unet import HybridSegformerUNet
from topology.skeleton import connectivity_score, mask_to_skeleton
from utils.config import load_config


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(image: np.ndarray, size: int = 256) -> torch.Tensor:
    """BGR image → normalised float tensor (1, 3, H, W)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size))
    x = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


# ---------------------------------------------------------------------------
# TTA prediction
# ---------------------------------------------------------------------------

def tta_predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    use_tta: bool = True,
) -> torch.Tensor:
    """
    Average predictions over original + horizontal + vertical flips.
    Returns probability map (sigmoid applied).
    """
    preds = [torch.sigmoid(model(x))]
    if use_tta:
        hf = torch.flip(x, dims=[3])
        preds.append(torch.flip(torch.sigmoid(model(hf)), dims=[3]))
        vf = torch.flip(x, dims=[2])
        preds.append(torch.flip(torch.sigmoid(model(vf)), dims=[2]))
    return torch.mean(torch.stack(preds, dim=0), dim=0)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def simple_gradcam(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Lightweight Grad-CAM on the fusion layer."""
    model.eval()
    feat: dict = {}
    grad: dict = {}

    def fwd_hook(_, __, output):
        feat["v"] = output

    def bwd_hook(_, gin, gout):
        grad["v"] = gout[0]

    h_f = model.fusion.register_forward_hook(fwd_hook)
    h_b = model.fusion.register_full_backward_hook(bwd_hook)

    y = model(x)
    torch.sigmoid(y).mean().backward()

    weights = grad["v"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feat["v"]).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam, size=x.shape[-2:], mode="bilinear", align_corners=False
    )
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h_f.remove()
    h_b.remove()
    model.zero_grad(set_to_none=True)
    return cam


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_visualization(
    orig_rgb: np.ndarray,
    gt_mask: np.ndarray | None,
    pred_mask: np.ndarray,
    skeleton: np.ndarray,
    graph,
    gradcam: np.ndarray,
    output_path: Path,
) -> None:
    cols = 6 if gt_mask is not None else 5
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 5))

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Input")
    idx = 1

    if gt_mask is not None:
        axes[idx].imshow(gt_mask, cmap="gray")
        axes[idx].set_title("Ground Truth")
        idx += 1

    axes[idx].imshow(pred_mask, cmap="gray")
    axes[idx].set_title("Pred Mask")
    idx += 1

    axes[idx].imshow(skeleton, cmap="gray")
    axes[idx].set_title("Skeleton")
    idx += 1

    axes[idx].imshow(orig_rgb)
    axes[idx].imshow(gradcam, cmap="jet", alpha=0.4)
    axes[idx].set_title("Grad-CAM")
    idx += 1

    axes[idx].set_title("Graph Overlay")
    axes[idx].imshow(np.zeros_like(pred_mask), cmap="gray")
    for (y1, x1), (y2, x2) in graph.edges():
        axes[idx].plot([x1, x2], [y1, y2], "c-", linewidth=0.5)
    for y, x in graph.nodes():
        axes[idx].plot(x, y, "ro", markersize=1)

    for ax in axes:
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference pipeline: image → mask → graph → SI score"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--gt-mask", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────
    model = HybridSegformerUNet(
        cnn_backbone=cfg["model"]["cnn_backbone"],
        transformer_backbone=cfg["model"]["transformer_backbone"],
        classes=cfg["model"]["classes"],
    ).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────
    raw_bgr = cv2.imread(args.image)
    if raw_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    x = preprocess(raw_bgr, cfg["data"]["image_size"]).to(device)

    with torch.no_grad():
        prob_map = tta_predict(model, x, use_tta=cfg["inference"].get("use_tta", True))

    threshold = float(cfg["inference"]["threshold"])
    pred_mask = (prob_map.squeeze().cpu().numpy() > threshold).astype(np.uint8)

    # ── Topology ──────────────────────────────────────────────────────────
    skeleton = mask_to_skeleton(pred_mask)
    graph = skeleton_to_graph(skeleton)
    feats = extract_structural_features(graph)
    conn = connectivity_score(skeleton)

    # ── SI Score ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_mask: np.ndarray | None = None
    dice: float = 1.0
    bce: float = 0.0

    if args.gt_mask:
        gt_raw = cv2.imread(args.gt_mask, cv2.IMREAD_GRAYSCALE)
        if gt_raw is not None:
            gt_mask = (
                cv2.resize(
                    gt_raw,
                    (cfg["data"]["image_size"], cfg["data"]["image_size"]),
                )
                > 127
            ).astype(np.uint8)

            # Compute real Dice and BCE against GT
            pred_bool = pred_mask.astype(bool)
            gt_bool = gt_mask.astype(bool)
            tp = float(np.logical_and(pred_bool, gt_bool).sum())
            fp = float(np.logical_and(pred_bool, ~gt_bool).sum())
            fn = float(np.logical_and(~pred_bool, gt_bool).sum())
            eps = 1e-8
            dice = float((2.0 * tp) / (2.0 * tp + fp + fn + eps))

            prob_np = np.clip(prob_map.squeeze().cpu().numpy(), 1e-6, 1.0 - 1e-6)
            gt_f = gt_mask.astype(np.float32)
            bce = float(
                -(gt_f * np.log(prob_np) + (1.0 - gt_f) * np.log(1.0 - prob_np)).mean()
            )

    # compute_structural_integrity automatically selects the right weights
    # (damage-only when dice==1.0, damage+seg when real dice is available)
    si_result = compute_structural_integrity(
        mask=pred_mask,
        skeleton=skeleton,
        graph=graph,
        connectivity_ratio=conn,
        dice=dice,
        bce=bce,
    )

    metrics = {
        "connectivity_score": round(float(conn), 4),
        "si_score": round(float(si_result["si_score"]), 4),
        "risk_level": si_result["risk_level"],
        "damage_breakdown": {
            "density": round(si_result["density_damage"], 4),
            "connectivity": round(si_result["connectivity_damage"], 4),
            "complexity": round(si_result["complexity_damage"], 4),
            "width": round(si_result.get("width_damage", 0.0), 4),
            "total": round(si_result["total_damage"], 4),
        },
    }
    if gt_mask is not None:
        metrics["dice"] = round(dice, 4)
        metrics["bce"] = round(bce, 4)
        metrics["segmentation_quality"] = round(si_result["segmentation_quality"], 4)

    # ── Save outputs ──────────────────────────────────────────────────────
    np.save(output_dir / "pred_mask.npy", pred_mask)
    np.save(output_dir / "skeleton.npy", skeleton)

    with open(output_dir / "features.json", "w") as f:
        json.dump(feats, f, indent=2, default=float)

    with open(output_dir / "si_score.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if args.visualize:
        cam = simple_gradcam(model, x)
        save_visualization(
            orig_rgb=cv2.cvtColor(
                cv2.resize(
                    raw_bgr,
                    (cfg["data"]["image_size"], cfg["data"]["image_size"]),
                ),
                cv2.COLOR_BGR2RGB,
            ),
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            skeleton=skeleton,
            graph=graph,
            gradcam=cam,
            output_path=output_dir / "visualization.png",
        )

    print(json.dumps({"features": feats, **metrics}, indent=2, default=float))


if __name__ == "__main__":
    main()
