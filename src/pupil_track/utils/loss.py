"""Combined Dice + BCE loss for binary pupil segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def dice_coeff(pred: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Compute Dice coefficient for binary masks.

    Args:
        pred: (B, H, W) float in [0, 1]
        target: (B, H, W) float in {0, 1}
    """
    assert pred.size() == target.size()
    inter = 2 * (pred * target).sum(dim=(-1, -2))
    sets_sum = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def dice_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Dice loss: 1 - dice_coeff."""
    return 1.0 - dice_coeff(pred, target)


class CombinedLoss(nn.Module):
    """Weighted combination of Dice loss and Binary Cross-Entropy.

    Loss = dice_weight * DiceLoss + bce_weight * BCEWithLogitsLoss
    """

    def __init__(self, dice_weight: float = 0.7, bce_weight: float = 0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw model output
            targets: (B, H, W) float in {0, 1}
        """
        logits_flat = logits.squeeze(1)  # (B, H, W)
        bce = F.binary_cross_entropy_with_logits(logits_flat, targets)
        pred = torch.sigmoid(logits_flat)
        dl = dice_loss(pred, targets)
        return self.dice_weight * dl + self.bce_weight * bce
