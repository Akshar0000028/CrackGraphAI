from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    bce = F.binary_cross_entropy_with_logits(logits, targets).item()
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "bce": bce,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def branch_consistency(pred_branches: float, gt_branches: float) -> float:
    if gt_branches <= 0:
        return 1.0 if pred_branches <= 0 else 0.0
    return float(max(0.0, 1.0 - abs(pred_branches - gt_branches) / (gt_branches + 1e-8)))


def structural_integrity_score(
    dice: float,
    bce: float,
    connectivity: float,
    branch_consistency_score: float,
    weights: Dict[str, float],
) -> float:
    # Original formula: SI = w_dice*Dice + w_bce*exp(-BCE) + w_conn*Connectivity + w_branch*BranchConsistency
    # BCE normalization: Apply bounded sigmoid-like normalization to keep BCE in [0, 2] effective range
    # This ensures exp(-BCE) maintains discriminative power across all BCE values
    bce_bounded = 2.0 * bce / (bce + 1.0) if bce > 0 else 0.0  # Maps [0, inf) -> [0, 2)
    bce_norm = float(np.exp(-bce_bounded))
    raw = (
        weights["dice"] * dice
        + weights["bce"] * bce_norm
        + weights["connectivity"] * connectivity
        + weights["branch_consistency"] * branch_consistency_score
    )
    return float(np.clip(raw, 0.0, 1.0))
