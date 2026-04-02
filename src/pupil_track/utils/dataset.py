"""PyTorch Dataset for pupil segmentation training data."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


def _imread_unicode(path: str | Path, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray | None:
    """cv2.imread replacement that handles non-ASCII paths on Windows."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, flags)


class PupilDataset(Dataset):
    """Binary segmentation dataset: grayscale images + binary masks.

    Expects:
        images_dir/ containing .png or .jpg images
        masks_dir/  containing .png masks (0=bg, 255=pupil)

    Image and mask filenames must match (ignoring extension).
    All images are resized to (input_size, input_size).
    """

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        input_size: int = 128,
        augment: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.input_size = input_size
        self.transform = get_train_transforms() if augment else get_val_transforms()

        # Collect IDs that have BOTH an image and a matching mask
        img_stems = {
            p.stem for p in self.images_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        }
        mask_stems = {
            p.stem for p in self.masks_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        }
        self.ids = sorted(img_stems & mask_stems)
        if not self.ids:
            raise RuntimeError(
                f"No matched image+mask pairs in {self.images_dir} and {self.masks_dir}"
            )
        n_unmatched = len(img_stems - mask_stems)
        if n_unmatched > 0:
            logger.warning(
                f"Skipped {n_unmatched} image(s) without matching masks"
            )
        logger.info(f"Dataset: {len(self.ids)} matched pairs (augment={augment})")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        """Per-image zero-mean unit-variance normalization."""
        mean = img.mean()
        std = img.std()
        if std < 1e-6:
            std = 1.0
        return (img - mean) / std

    def _load_image(self, name: str) -> np.ndarray:
        """Load image as grayscale uint8, resized to input_size."""
        candidates = list(self.images_dir.glob(f"{name}.*"))
        assert len(candidates) == 1, f"Expected 1 image for '{name}', found {len(candidates)}"
        img = _imread_unicode(candidates[0], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = _imread_unicode(candidates[0], cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert img is not None, f"Failed to load image: {candidates[0]}"
        return cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

    def _load_mask(self, name: str) -> np.ndarray:
        """Load mask as binary uint8 (0 or 1), resized to input_size."""
        candidates = list(self.masks_dir.glob(f"{name}.*"))
        assert len(candidates) == 1, f"Expected 1 mask for '{name}', found {len(candidates)}"
        mask = _imread_unicode(candidates[0], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = _imread_unicode(candidates[0], cv2.IMREAD_COLOR)
            if mask is not None:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        assert mask is not None, f"Failed to load mask: {candidates[0]}"
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        # Binarize: anything > 127 is pupil
        return (mask > 127).astype(np.uint8)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = self._load_image(name)   # (H, W) uint8
        mask = self._load_mask(name)   # (H, W) uint8 {0, 1}

        # Apply augmentations (synchronized for image+mask)
        transformed = self.transform(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

        # Normalize image: float32, zero-mean unit-variance
        img = img.astype(np.float32)
        img = self._normalize(img)

        # Add channel dimension: (1, H, W)
        img = img[np.newaxis, ...]

        return {
            "image": torch.from_numpy(img).contiguous(),
            "mask": torch.from_numpy(mask.astype(np.float32)).contiguous(),
        }
