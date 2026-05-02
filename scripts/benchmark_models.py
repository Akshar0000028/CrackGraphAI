from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CrackSegmentationDataset, discover_samples, split_samples
from models.baselines import build_deeplabv3plus, build_segformer, build_unetplusplus
from models.hybrid_segformer_unet import HybridSegformerUNet
from topology.skeleton import connectivity_score, mask_to_skeleton
from utils.config import load_config
from utils.metrics import segmentation_metrics


def build_model(name: str, cfg: dict):
    if name == "hybrid":
        return HybridSegformerUNet(
            cnn_backbone=cfg["model"]["cnn_backbone"],
            transformer_backbone=cfg["model"]["transformer_backbone"],
            classes=cfg["model"]["classes"],
        )
    if name == "unetpp":
        return build_unetplusplus()
    if name == "deeplabv3plus":
        return build_deeplabv3plus()
    if name == "segformer":
        return build_segformer()
    raise ValueError(name)


def evaluate(model, loader, device):
    model.eval()
    agg = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "bce": 0.0, "connectivity": 0.0}
    n = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, leave=False):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            m = segmentation_metrics(logits, masks)
            pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            conn = []
            for i in range(pred.shape[0]):
                sk = mask_to_skeleton(pred[i, 0].astype("uint8"))
                conn.append(connectivity_score(sk))
            for k in ["iou", "dice", "precision", "recall", "bce"]:
                agg[k] += m[k]
            agg["connectivity"] += float(sum(conn) / max(1, len(conn)))
            n += 1
    return {k: v / max(1, n) for k, v in agg.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = discover_samples(cfg["data"]["root_dir"], cfg["data"]["image_dir_name"], cfg["data"]["mask_dir_name"])
    _, _, test_s = split_samples(samples, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"])
    test_ds = CrackSegmentationDataset(test_s, cfg["data"]["image_size"], split="test")
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])
    rows = []
    for model_name, ckpt_name in [
        ("unetpp", "best_unetpp.pth"),
        ("deeplabv3plus", "best_deeplabv3plus.pth"),
        ("segformer", "best_segformer.pth"),
        ("hybrid", cfg["training"]["best_model_name"]),
    ]:
        ckpt_path = Path(args.checkpoints_dir) / ckpt_name
        if not ckpt_path.exists():
            continue
        model = build_model(model_name, cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        metrics = evaluate(model, test_loader, device)
        rows.append({"model": model_name, **metrics})
    df = pd.DataFrame(rows)
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
