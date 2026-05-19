"""U-Net detection: streaming batched inference for pupil segmentation."""

import logging
from collections.abc import Iterator

import numpy as np
import torch
from tqdm import tqdm

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


def get_optimal_batch_size(device: torch.device, input_size: int = 128) -> int:
    """Determine optimal batch size based on available GPU memory."""
    if device.type != "cuda":
        return 32  # Conservative for CPU
    
    try:
        gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory // (1024**3)
        # Rough heuristic: larger batches for more memory, but cap at reasonable limits
        if gpu_memory_gb >= 24:      # RTX 4090, etc
            return 128
        elif gpu_memory_gb >= 16:    # RTX 4080, etc  
            return 96
        elif gpu_memory_gb >= 12:    # RTX 4070 Ti, etc
            return 80
        elif gpu_memory_gb >= 8:     # RTX 4070, etc
            return 64
        else:                        # < 8GB
            return 48
    except:
        return 64  # Safe default


def load_model(model_path: str, device: torch.device, bilinear: bool = False) -> ShallowUNet:
    """Load a trained ShallowUNet from .pth checkpoint."""
    model = ShallowUNet(n_channels=1, bilinear=bilinear)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Log device information
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory // (1024**3)
        logger.info(f"Loaded U-Net from {model_path} on GPU: {gpu_name} ({gpu_memory}GB)")
    else:
        logger.info(f"Loaded U-Net from {model_path} on CPU")
    
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

    tensor = torch.from_numpy(np.stack(batch, axis=0)).to(device=device, dtype=torch.float32, non_blocking=True)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.sigmoid(logits.squeeze(1))  # (B, H, W)
        masks = (probs > threshold).cpu().numpy().astype(np.uint8)

    return list(masks)


def detect_unet(
    frames: list[tuple[int, np.ndarray]] | Iterator[tuple[int, np.ndarray]],
    model_path: str,
    threshold: float = 0.5,
    batch_size: int | None = None,  # Auto-detect if None
    bilinear: bool = False,
    total_frames: int | None = None,  # For progress bar with streaming
) -> dict[int, np.ndarray]:
    """Run U-Net detection on (index, frame) pairs.

    Accepts either a list or an iterator/generator of (index, frame) pairs.
    Frames should already be cropped and resized to the model's input size.
    When given a generator, frames are consumed in batches — only one batch
    of frames is held in memory at a time.

    Args:
        frames: Iterator or list of (frame_index, frame_array) pairs
        model_path: Path to trained U-Net model
        threshold: Detection threshold
        batch_size: Batch size for inference (auto-detected if None)
        bilinear: Use bilinear upsampling in U-Net
        total_frames: Total number of frames for progress bar (required for streaming)

    Returns:
        dict mapping frame_index -> binary mask (uint8).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Auto-detect optimal batch size if not specified
    if batch_size is None:
        batch_size = get_optimal_batch_size(device)
    
    # Log device and performance info
    logger.info(f"Starting U-Net detection on {device.type.upper()}")
    if device.type == "cuda":
        logger.info(f"Batch size: {batch_size}, Threshold: {threshold}")
    
    model = load_model(model_path, device, bilinear=bilinear)

    # Determine if we're dealing with a list or iterator
    is_list = hasattr(frames, '__len__')
    if is_list:
        total_frames = len(frames)
        frame_iter = iter(frames)
        logger.info(f"Loaded {total_frames} frames into memory - processing in batches of {batch_size}")
    else:
        frame_iter = frames
        if total_frames is None:
            logger.warning("No total_frames provided for streaming - progress bar disabled")
        logger.info(f"Streaming frames - processing in batches of {batch_size}")
    
    results = {}
    batch_indices: list[int] = []
    batch_frames: list[np.ndarray] = []
    processed_count = 0
    
    # Add progress bar only if we know total frames
    progress_bar = None
    if total_frames:
        progress_bar = tqdm(total=total_frames, desc="U-Net Detection", unit="frames")
    
    try:
        for idx, frame in frame_iter:
            batch_indices.append(idx)
            batch_frames.append(frame)

            if len(batch_frames) == batch_size:
                masks = _infer_batch(model, batch_frames, device, threshold)
                for i, mask in zip(batch_indices, masks):
                    results[i] = mask
                
                processed_count += len(batch_frames)
                if progress_bar:
                    progress_bar.update(len(batch_frames))
                    
                batch_indices.clear()
                batch_frames.clear()

        # Process remaining frames
        if batch_frames:
            masks = _infer_batch(model, batch_frames, device, threshold)
            for i, mask in zip(batch_indices, masks):
                results[i] = mask
            processed_count += len(batch_frames)
            if progress_bar:
                progress_bar.update(len(batch_frames))

    finally:
        if progress_bar:
            progress_bar.close()

    logger.info(f"Detection complete: {len(results)} masks generated")
    return results


# Keep for external use / benchmarking
predict_batch = _infer_batch
