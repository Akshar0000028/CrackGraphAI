from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses.segmentation_losses import TopologyAwareLoss
from utils.metrics import segmentation_metrics


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        tr_cfg = cfg["training"]
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=tr_cfg["lr"],
            weight_decay=tr_cfg["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=tr_cfg["epochs"])
        self.loss_fn = TopologyAwareLoss(lambda_topology=tr_cfg["lambda_topology"])
        self.scaler = GradScaler(enabled=tr_cfg["amp"])
        self.best_dice = 0.0
        self.patience = tr_cfg["early_stopping_patience"]
        self.bad_epochs = 0
        self.checkpoint_dir = Path(tr_cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / tr_cfg["best_model_name"]

    def _run_epoch(self, train: bool = True) -> Tuple[float, Dict[str, float]]:
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)
        losses = []
        metrics_acc = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "bce": 0.0}
        progress = tqdm(loader, desc="train" if train else "val", leave=False)
        for images, masks in progress:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            with autocast(device_type="cuda", enabled=self.cfg["training"]["amp"] and self.device.type == "cuda"):
                logits = self.model(images)
                loss, _ = self.loss_fn(logits, masks)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            losses.append(loss.item())
            m = segmentation_metrics(logits.detach(), masks)
            for key in metrics_acc:
                metrics_acc[key] += m[key]
        n = max(1, len(loader))
        return sum(losses) / max(1, len(losses)), {k: v / n for k, v in metrics_acc.items()}

    def fit(self) -> None:
        for epoch in range(1, self.cfg["training"]["epochs"] + 1):
            train_loss, train_metrics = self._run_epoch(train=True)
            val_loss, val_metrics = self._run_epoch(train=False)
            self.scheduler.step()
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_dice={train_metrics['dice']:.4f} val_dice={val_metrics['dice']:.4f}"
            )
            if val_metrics["dice"] > self.best_dice:
                self.best_dice = val_metrics["dice"]
                self.bad_epochs = 0
                torch.save({"model_state_dict": self.model.state_dict(), "val_metrics": val_metrics}, self.best_model_path)
                print(f"Saved best model to {self.best_model_path}")
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
