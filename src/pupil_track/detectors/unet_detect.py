"""U-Net detection: streaming batched inference for pupil segmentation."""

import logging
from collections.abc import Iterator

import numpy as np
import torch

from ..model import ShallowUNet

logger = logging.getLogger(__name__)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Per-image zero-mean unit-variance normalization."""
    img = frame.astype(np.float32)
    mean = img.mean()
    std = img.std()
    if std < 1e-6:
        std = 1.0
    return (img - mean) / std


def load_model(model_path: str, device: torch.device, bilinear: bool = False) -> ShallowUNet:
    """Load a trained ShallowUNet from .pth checkpoint."""
    model = ShallowUNet(n_channels=1, bilinear=bilinear)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Loaded U-Net from {model_path}")
    return model


def _infer_batch(
    model: ShallowUNet,
    frames: list[np.ndarray],
    device: torch.device,
    threshold: float,
) -> list[np.ndarray]:
    """Run inference on a batch of pre-cropped grayscale frames.

    Frames are already at the correct input_size.
    Returns list of binary masks (uint8, same size as input).
    """
    batch = []
    for f in frames:
        normed = normalize_frame(f)
        batch.append(normed[np.newaxis, ...])  # (1, H, W)

    tensor = torch.from_numpy(np.stack(batch, axis=0)).to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.sigmoid(logits.squeeze(1))  # (B, H, W)
        masks = (probs > threshold).cpu().numpy().astype(np.uint8)

    return list(masks)


def detect_unet(
    frames: list[tuple[int, np.ndarray]] | Iterator[tuple[int, np.ndarray]],
    model_path: str,
    threshold: float = 0.5,
    batch_size: int = 32,
    bilinear: bool = False,
) -> dict[int, np.ndarray]:
    """Run U-Net detection on (index, frame) pairs.

    Accepts either a list or an iterator/generator of (index, frame) pairs.
    Frames should already be cropped and resized to the model's input size.
    When given a generator, frames are consumed in batches — only one batch
    of frames is held in memory at a time.

    Returns:
        dict mapping frame_index -> binary mask (uint8).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device, bilinear=bilinear)

    results = {}
    batch_indices: list[int] = []
    batch_frames: list[np.ndarray] = []

    for idx, frame in frames:
        batch_indices.append(idx)
        batch_frames.append(frame)

        if len(batch_frames) == batch_size:
            masks = _infer_batch(model, batch_frames, device, threshold)
            for i, mask in zip(batch_indices, masks):
                results[i] = mask
            batch_indices.clear()
            batch_frames.clear()

    # Process remaining frames
    if batch_frames:
        masks = _infer_batch(model, batch_frames, device, threshold)
        for i, mask in zip(batch_indices, masks):
            results[i] = mask

    return results


# Keep for external use / benchmarking
predict_batch = _infer_batch
