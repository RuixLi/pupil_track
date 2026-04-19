"""Reproduce silent training crash at model-save time.

Fill in DATA_DIRS and CHECKPOINT_DIR below, then run:

    python tests/reproduce_train_crash.py

Design notes (why the extra instrumentation):
  * `faulthandler` dumps a native traceback on segfault / CUDA fatal errors —
    these normally kill Python with no Python-level traceback at all.
  * All log output is ALSO written to `reproduce_train_crash.log`, line-buffered,
    so even if the terminal is wiped by tqdm / Windows console buffer limits,
    we still know exactly which line was executing when the process died.
  * The save block is wrapped in try/except and each of the three steps
    (torch.save → shutil.copy2 → _rotate_checkpoints) is bracketed by
    "[SAVE] ▶ step" / "[SAVE] ✓ step OK" markers. The last marker in the log
    identifies the failing line.
  * `torch.cuda.synchronize()` is called right before save so that any pending
    async CUDA error surfaces BEFORE save (helps distinguish "save itself
    crashed" from "a stale GPU error exploded during save").
  * We save EVERY epoch (not only on val_score improvement) to maximise the
    number of save attempts per unit time.
"""

import faulthandler
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

# ─── EDIT THESE ───────────────────────────────────────────────────────────────
DATA_DIRS: list[tuple[str, str]] = [
    (r"D:\OneDrive - 同志社大学\Data\pupil_dataset\dataset_260419\images",
     r"D:\OneDrive - 同志社大学\Data\pupil_dataset\dataset_260419\masks"),
    # Add more (img_dir, mask_dir) pairs if needed
    # (img_dir, mask_dir),
    # e.g. (r"D:\data\imgs", r"D:\data\masks"),
]
CHECKPOINT_DIR = r".\checkpoints_crashtest"
EPOCHS = 90
BATCH_SIZE = 8
INPUT_SIZE = 128
AMP = False           # set True if your normal training uses --amp
PIN_MEMORY = True     # matches train.py default
# ──────────────────────────────────────────────────────────────────────────────

LOG_FILE = Path(__file__).resolve().parent / "reproduce_train_crash.log"

# Line-buffered file so every log line hits disk immediately — survives crash.
_log_fh = open(LOG_FILE, "w", buffering=1, encoding="utf-8")

# faulthandler catches SIGSEGV / access violation / CUDA fatal and prints a
# C-level traceback to the given file before the process dies.
faulthandler.enable(file=_log_fh, all_threads=True)

root = logging.getLogger()
root.setLevel(logging.INFO)
root.handlers.clear()
_fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
_sh = logging.StreamHandler(sys.stdout); _sh.setFormatter(_fmt); root.addHandler(_sh)
_fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
_fh.setFormatter(_fmt); root.addHandler(_fh)
logger = logging.getLogger("reproduce_train_crash")

# Make sure we can import the package when running from the tests/ folder
# even if pupil_track isn't pip-installed.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pupil_track.model import ShallowUNet                   # noqa: E402
from pupil_track.utils.dataset import PupilDataset          # noqa: E402
from pupil_track.utils.evaluate import evaluate             # noqa: E402
from pupil_track.utils.loss import CombinedLoss             # noqa: E402
from pupil_track.train import _rotate_checkpoints           # noqa: E402


def run():
    assert DATA_DIRS, (
        "DATA_DIRS is empty — edit the top of this script and add at least one "
        "(img_dir, mask_dir) pair before running."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "torch=%s cuda_build=%s cuda_available=%s device=%s",
        torch.__version__, torch.version.cuda, torch.cuda.is_available(), device,
    )
    if device.type == "cuda":
        logger.info("GPU: %s  driver_cudnn=%s",
                    torch.cuda.get_device_name(0), torch.backends.cudnn.version())

    # ── Datasets / loaders (mirror train.train_model) ─────────────────────────
    datasets_noaug, datasets_aug = [], []
    for img_dir, mask_dir in DATA_DIRS:
        datasets_noaug.append(PupilDataset(img_dir, mask_dir, input_size=INPUT_SIZE, augment=False))
        datasets_aug.append(PupilDataset(img_dir, mask_dir, input_size=INPUT_SIZE, augment=True))

    if len(datasets_noaug) == 1:
        ds_full, ds_aug = datasets_noaug[0], datasets_aug[0]
    else:
        ds_full = torch.utils.data.ConcatDataset(datasets_noaug)
        ds_aug = torch.utils.data.ConcatDataset(datasets_aug)

    n_total = len(ds_full)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    tr_idx, va_idx = random_split(
        range(n_total), [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    train_set = torch.utils.data.Subset(ds_aug, tr_idx.indices)
    val_set = torch.utils.data.Subset(ds_full, va_idx.indices)

    num_workers = 0 if sys.platform == "win32" else min(os.cpu_count() or 1, 4)
    loader_args = dict(batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=PIN_MEMORY)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    logger.info("Train=%d  Val=%d  Batch=%d  Epochs=%d  AMP=%s  pin_memory=%s",
                n_train, n_val, BATCH_SIZE, EPOCHS, AMP, PIN_MEMORY)

    # ── Model / opt / loss (mirror train.train_model) ─────────────────────────
    model = ShallowUNet(n_channels=1, bilinear=False).to(memory_format=torch.channels_last)
    model.to(device=device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    grad_scaler = torch.amp.GradScaler(enabled=AMP)
    criterion = CombinedLoss(dice_weight=0.7, bce_weight=0.3)

    ckpt_dir = Path(CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = ckpt_dir / "best_model.pth"
    logger.info("Checkpoint dir: %s", ckpt_dir.resolve())

    best_val_score = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.perf_counter()
        epoch_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(
                device=device, dtype=torch.float32, memory_format=torch.channels_last
            )
            masks = batch["mask"].to(device=device, dtype=torch.float32)
            with torch.autocast(
                device.type if device.type != "mps" else "cpu", enabled=AMP
            ):
                logits = model(images)
                loss = criterion(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            epoch_loss += loss.item()

        val_score = evaluate(model, val_loader, device, AMP)
        logger.info(
            "Epoch %d/%d done in %.1fs  loss=%.4f  val=%.4f  (best=%.4f)",
            epoch, EPOCHS, time.perf_counter() - t0,
            epoch_loss / max(len(train_loader), 1), val_score, best_val_score,
        )

        # ── Save every epoch, step-by-step, with markers ─────────────────────
        dice_str = f"{max(val_score, 0.0):.4f}".replace(".", "p")
        snapshot_name = f"best_epoch{epoch:03d}_dice_{dice_str}.pth"
        snapshot_path = ckpt_dir / snapshot_name

        try:
            logger.info("[SAVE] ▶ cuda.synchronize() (flush pending async errors)")
            if device.type == "cuda":
                torch.cuda.synchronize()
            logger.info("[SAVE] ✓ cuda.synchronize() OK")

            logger.info("[SAVE] ▶ torch.save(state_dict, %s)", snapshot_path.name)
            torch.save(model.state_dict(), str(snapshot_path))
            size = snapshot_path.stat().st_size
            logger.info("[SAVE] ✓ torch.save OK (%d bytes)", size)

            logger.info("[SAVE] ▶ shutil.copy2(snapshot → best_model.pth)")
            shutil.copy2(str(snapshot_path), str(best_model_path))
            logger.info("[SAVE] ✓ shutil.copy2 OK")

            logger.info("[SAVE] ▶ _rotate_checkpoints(keep=3)")
            _rotate_checkpoints(ckpt_dir, 3)
            logger.info("[SAVE] ✓ _rotate_checkpoints OK")
        except BaseException:
            # BaseException so KeyboardInterrupt / SystemExit also get logged.
            logger.exception("[SAVE] ✗ exception at epoch %d", epoch)
            raise

        if val_score > best_val_score:
            best_val_score = val_score

    logger.info("All %d epochs completed. Best val Dice: %.4f", EPOCHS, best_val_score)


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("reproduce_train_crash.py starting  pid=%d  log=%s", os.getpid(), LOG_FILE)
    logger.info("=" * 70)
    try:
        run()
    except BaseException:
        logger.exception("Unhandled exception in run()")
        raise
    finally:
        for h in root.handlers:
            try: h.flush()
            except Exception: pass
        try: _log_fh.flush()
        except Exception: pass
