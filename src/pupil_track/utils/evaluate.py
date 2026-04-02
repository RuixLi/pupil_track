"""Validation evaluation for binary pupil segmentation."""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .loss import dice_coeff


@torch.inference_mode()
def evaluate(model, dataloader, device, amp: bool = False):
    """Compute mean Dice score on validation set.

    Returns:
        float: Mean Dice coefficient over all validation batches.
    """
    model.eval()
    num_batches = len(dataloader)
    total_dice = 0.0

    with torch.autocast(
        device.type if device.type != "mps" else "cpu", enabled=amp
    ):
        for batch in tqdm(
            dataloader, total=num_batches, desc="Validation", unit="batch", leave=False
        ):
            images = batch["image"].to(device=device, dtype=torch.float32)
            masks_true = batch["mask"].to(device=device, dtype=torch.float32)

            logits = model(images)
            masks_pred = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
            total_dice += dice_coeff(masks_pred, masks_true).item()

    model.train()
    return total_dice / max(num_batches, 1)
