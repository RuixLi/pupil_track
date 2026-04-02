"""Training pipeline for shallow U-Net pupil segmentation.

Usage:
    python -m pupil_track train --dir-img ./data/imgs --dir-mask ./data/masks
    python -m pupil_track train --load best_model.pth --dir-img ./new --dir-mask ./new --freeze-encoder
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from .model import ShallowUNet
from .utils.dataset import PupilDataset
from .utils.evaluate import evaluate
from .utils.loss import CombinedLoss

logger = logging.getLogger(__name__)


def _rotate_checkpoints(ckpt_dir: Path, max_keep: int):
    """Keep only the top-N best checkpoints by Dice score, delete the rest.

    Checkpoint filenames: best_epochNNN_diceX.XXXX.pth
    """
    pattern = re.compile(r"best_epoch(\d+)_dice_(\d+)p(\d+)\.pth$")
    snapshots = []
    for p in ckpt_dir.glob("best_epoch*_dice_*.pth"):
        m = pattern.match(p.name)
        if m:
            score = float(f"{m.group(2)}.{m.group(3)}")
            snapshots.append((score, p))
    # Sort by score descending — keep the best N
    snapshots.sort(key=lambda x: x[0], reverse=True)
    for _, p in snapshots[max_keep:]:
        p.unlink(missing_ok=True)
        logger.info("Removed old snapshot: %s", p.name)


def train_model(
    data_dirs: list[tuple[Path, Path]],
    dir_checkpoint: Path,
    model: ShallowUNet,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    val_percent: float = 0.1,
    amp: bool = False,
    freeze_encoder: bool = False,
    patience: int = 10,
    input_size: int = 128,
    max_snapshots: int = 3,
) -> Path:
    """Train or fine-tune the shallow U-Net.

    Args:
        data_dirs: list of (images_dir, masks_dir) path pairs.
        dir_checkpoint: directory to save checkpoints.

    Returns:
        Path to the best model checkpoint.
    """
    # Freeze encoder for fine-tuning
    if freeze_encoder:
        encoder_layers = ["inc", "down1", "down2", "down3"]
        for name, param in model.named_parameters():
            if any(name.startswith(layer) for layer in encoder_layers):
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Encoder frozen: training {trainable:,} / {total:,} parameters")

    # Create datasets from all data directories
    all_datasets_noaug = []
    all_datasets_aug = []
    for dir_img, dir_mask in data_dirs:
        all_datasets_noaug.append(
            PupilDataset(dir_img, dir_mask, input_size=input_size, augment=False))
        all_datasets_aug.append(
            PupilDataset(dir_img, dir_mask, input_size=input_size, augment=True))

    if len(all_datasets_noaug) == 1:
        dataset_full = all_datasets_noaug[0]
        dataset_aug = all_datasets_aug[0]
    else:
        dataset_full = torch.utils.data.ConcatDataset(all_datasets_noaug)
        dataset_aug = torch.utils.data.ConcatDataset(all_datasets_aug)

    n_total = len(dataset_full)
    n_val = max(1, int(n_total * val_percent))
    n_train = n_total - n_val

    train_indices, val_indices = random_split(
        range(n_total),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    train_set = torch.utils.data.Subset(dataset_aug, train_indices.indices)
    val_set = torch.utils.data.Subset(dataset_full, val_indices.indices)

    # DataLoaders — 0 workers on Windows to avoid spawn issues
    num_workers = 0 if sys.platform == "win32" else min(os.cpu_count() or 1, 4)
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    logger.info(
        f"Training: {n_train} samples, Validation: {n_val} samples, "
        f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}, "
        f"Device: {device.type}, AMP: {amp}"
    )

    # Optimizer, loss, scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-2,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    criterion = CombinedLoss(dice_weight=0.7, bce_weight=0.3)

    # Training loop
    best_val_score = 0.0
    epochs_no_improve = 0
    best_model_path = Path(dir_checkpoint) / "best_model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_processed = 0
        for batch in train_loader:
            images = batch["image"].to(
                device=device, dtype=torch.float32, memory_format=torch.channels_last
            )
            masks = batch["mask"].to(device=device, dtype=torch.float32)

            with torch.autocast(
                device.type if device.type != "mps" else "cpu", enabled=amp
            ):
                logits = model(images)
                loss = criterion(logits, masks)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            n_processed += images.shape[0]
            epoch_loss += loss.item()

        # End-of-epoch validation
        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)
        logger.info(
            f"Epoch {epoch} — loss: {epoch_loss / len(train_loader):.4f}, "
            f"val Dice: {val_score:.4f} (best: {best_val_score:.4f})"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            epochs_no_improve = 0
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            # Save with epoch + score in filename (use 'p' instead of '.')
            dice_str = f"{val_score:.4f}".replace(".", "p")
            snapshot_name = f"best_epoch{epoch:03d}_dice_{dice_str}.pth"
            snapshot_path = Path(dir_checkpoint) / snapshot_name
            torch.save(model.state_dict(), str(snapshot_path))
            # Update best_model.pth as a copy of the latest best
            shutil.copy2(str(snapshot_path), str(best_model_path))
            # Rotate: keep only top-N snapshots by score
            if max_snapshots > 0:
                _rotate_checkpoints(Path(dir_checkpoint), max_snapshots)
            logger.info(f"Best model saved: {snapshot_name} (Dice: {val_score:.4f})")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve}/{patience} epochs")

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping after {epoch} epochs")
            break

    # Save final model
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    final_path = Path(dir_checkpoint) / "final_model.pth"
    torch.save(model.state_dict(), str(final_path))
    logger.info(f"Final model saved to {final_path}")
    logger.info(f"Training complete. Best validation Dice: {best_val_score:.4f}")

    # Save training config alongside the model
    from . import __version__
    train_config = {
        "software": {
            "name": "pupil_track",
            "version": __version__,
            "datetime": datetime.now().isoformat(timespec="seconds"),
        },
        "training": {
            "epochs_run": epoch,
            "epochs_max": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "val_percent": val_percent,
            "amp": amp,
            "freeze_encoder": freeze_encoder,
            "patience": patience,
            "input_size": input_size,
            "max_snapshots": max_snapshots,
            "best_val_dice": round(best_val_score, 4),
        },
        "data": [
            {"images": str(img), "masks": str(msk)} for img, msk in data_dirs
        ],
    }
    config_path = Path(dir_checkpoint) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=2)
    logger.info(f"Training config saved to {config_path}")

    return best_model_path


def get_train_args(args=None):
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(description="Train shallow U-Net for pupil segmentation")
    parser.add_argument(
        "--data-dir", nargs=2, action="append", metavar=("IMG_DIR", "MASK_DIR"),
        help="Image and mask directory pair (can be repeated for multiple datasets)",
    )
    # Legacy single-directory arguments (still supported)
    parser.add_argument("--dir-img", type=str, default=None, help="Images directory (legacy)")
    parser.add_argument("--dir-mask", type=str, default=None, help="Masks directory (legacy)")
    parser.add_argument("--dir-checkpoint", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--epochs", "-e", type=int, default=50)
    parser.add_argument("--batch-size", "-b", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--load", type=str, default=None, help="Load pretrained .pth")
    parser.add_argument("--val", type=float, default=10.0, help="Validation percent (0-100)")
    parser.add_argument("--amp", action="store_true", help="Mixed precision training")
    parser.add_argument("--bilinear", action="store_true", help="Use bilinear upsampling")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder layers")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--input-size", type=int, default=128, help="Input image size")
    parser.add_argument("--max-snapshots", type=int, default=3, help="Keep top-N best checkpoints")
    return parser.parse_args(args)


def run_training(args=None):
    """Entry point for training subcommand."""
    args = get_train_args(args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Build data_dirs from --data-dir and/or legacy --dir-img/--dir-mask
    data_dirs: list[tuple[Path, Path]] = []
    if args.data_dir:
        for img_dir, mask_dir in args.data_dir:
            data_dirs.append((Path(img_dir), Path(mask_dir)))
    if args.dir_img and args.dir_mask:
        data_dirs.append((Path(args.dir_img), Path(args.dir_mask)))
    if not data_dirs:
        raise ValueError("No data directories specified. Use --data-dir or --dir-img/--dir-mask.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = ShallowUNet(n_channels=1, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ShallowUNet: {total_params:,} parameters, bilinear={args.bilinear}")

    if args.load:
        state_dict = torch.load(args.load, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {args.load}")

    model.to(device=device)

    train_model(
        data_dirs=data_dirs,
        dir_checkpoint=Path(args.dir_checkpoint),
        model=model,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_percent=args.val / 100,
        amp=args.amp,
        freeze_encoder=args.freeze_encoder,
        patience=args.patience,
        input_size=args.input_size,
        max_snapshots=args.max_snapshots,
    )
