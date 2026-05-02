from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from data.dataset import CrackSegmentationDataset, discover_samples, split_samples
from models.baselines import build_deeplabv3plus, build_segformer, build_unetplusplus
from models.hybrid_segformer_unet import HybridSegformerUNet
from training.trainer import Trainer
from utils.config import load_config
from utils.repro import set_seed


def build_model(model_name: str, cfg: dict) -> torch.nn.Module:
    if model_name == "hybrid":
        return HybridSegformerUNet(
            cnn_backbone=cfg["model"]["cnn_backbone"],
            transformer_backbone=cfg["model"]["transformer_backbone"],
            classes=cfg["model"]["classes"],
        )
    if model_name == "unetpp":
        return build_unetplusplus()
    if model_name == "deeplabv3plus":
        return build_deeplabv3plus()
    if model_name == "segformer":
        return build_segformer()
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crack segmentation models")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="hybrid", choices=["hybrid", "unetpp", "deeplabv3plus", "segformer"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = discover_samples(
        root_dir=cfg["data"]["root_dir"],
        image_dir_name=cfg["data"]["image_dir_name"],
        mask_dir_name=cfg["data"]["mask_dir_name"],
    )
    train_s, val_s, _ = split_samples(
        samples=samples,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
    )
    train_ds = CrackSegmentationDataset(train_s, cfg["data"]["image_size"], split="train")
    val_ds = CrackSegmentationDataset(val_s, cfg["data"]["image_size"], split="val")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    model = build_model(args.model, cfg)
    if args.model != "hybrid":
        cfg["training"]["best_model_name"] = f"best_{args.model}.pth"
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, cfg=cfg, device=device)
    trainer.fit()
    print("Training complete.")


if __name__ == "__main__":
    main()
