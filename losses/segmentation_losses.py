from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss for binary segmentation.

    Formula: 1 - (2 * |pred ∩ target| + eps) / (|pred| + |target| + eps)

    The denominator is the SUM of cardinalities (not union), which is the
    standard Sørensen–Dice coefficient used in segmentation literature.
    """
    probs = torch.sigmoid(logits)
    # Flatten spatial dims so we sum over H×W per sample
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    cardinality = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def soft_skeletonize(probs: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    """
    Differentiable approximation of morphological thinning for topology regularization.
    Iteratively erodes the boundary of the probability map to approximate a skeleton.
    """
    skel = probs.clone()
    for _ in range(iterations):
        # Morphological erosion via min-pooling (negate → max-pool → negate)
        min_pool = -F.max_pool2d(-skel, kernel_size=3, stride=1, padding=1)
        contour = F.relu(skel - min_pool)
        skel = F.relu(skel - contour)
    return skel


def topology_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Topology-aware loss that penalises broken or disconnected skeletons.

    Combines:
    - L1 distance between soft skeletons of prediction and ground truth
    - Connectivity penalty via local average-pooling difference
    """
    probs = torch.sigmoid(logits)
    # targets may already be float [0,1]; clamp for safety
    targets_f = targets.float().clamp(0.0, 1.0)

    skel_pred = soft_skeletonize(probs)
    skel_gt = soft_skeletonize(targets_f)

    l1 = torch.mean(torch.abs(skel_pred - skel_gt))

    # Local connectivity: 5×5 neighbourhood average
    conn_pred = F.avg_pool2d(skel_pred, kernel_size=5, stride=1, padding=2)
    conn_gt = F.avg_pool2d(skel_gt, kernel_size=5, stride=1, padding=2)
    connectivity_penalty = torch.mean(torch.abs(conn_pred - conn_gt))

    return l1 + connectivity_penalty


class TopologyAwareLoss(nn.Module):
    """
    Combined loss: Dice + BCE + λ × Topology.

    - Dice handles class imbalance (cracks are a small fraction of pixels).
    - BCE provides per-pixel gradient signal.
    - Topology loss preserves crack connectivity and skeleton structure.
    """

    def __init__(self, lambda_topology: float = 0.2):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_topology = lambda_topology

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce = self.bce(logits, targets)
        d_loss = dice_loss(logits, targets)
        t_loss = topology_loss(logits, targets)
        total = d_loss + bce + self.lambda_topology * t_loss
        return total, {
            "dice_loss": d_loss.item(),
            "bce_loss": bce.item(),
            "topology_loss": t_loss.item(),
        }
