"""Data augmentation transforms using Albumentations."""

import albumentations as A


def get_train_transforms() -> A.Compose:
    """Augmentation pipeline for training.

    Includes geometric transforms (applied to both image and mask)
    and photometric transforms (image only). CoarseDropout with
    fill_value=255 simulates specular reflections (bright spots).
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.12, contrast_limit=0.3, p=0.5
        ),
        A.GaussNoise(var_limit=(1.0, 64.0), p=0.5),
        A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
        A.CoarseDropout(
            max_holes=5,
            max_height=12,
            max_width=12,
            fill_value=255,
            p=0.4,
        ),
    ])


def get_val_transforms() -> A.Compose:
    """No augmentation for validation."""
    return A.Compose([])
