"""Video I/O: load and iterate over video frames."""

from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


class VideoReader:
    """Wrapper around cv2.VideoCapture for frame-by-frame access.

    All frames are returned as uint8 grayscale numpy arrays.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self, idx: int) -> Optional[np.ndarray]:
        """Read a single frame by index. Returns grayscale uint8 or None."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            return None
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def frames(
        self, indices: Optional[np.ndarray] = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over frames. If indices is None, iterate all frames.

        Yields:
            (frame_index, grayscale_frame) tuples.
        """
        if indices is None:
            indices = range(self.n_frames)
        for idx in indices:
            frame = self.read_frame(idx)
            if frame is not None:
                yield idx, frame

    def sample_indices(self, n: int, mode: str = "uniform") -> np.ndarray:
        """Select frame indices for labeling.

        Args:
            n: number of frames to select
            mode: "uniform" (evenly spaced) or "random"
        """
        if mode == "uniform":
            return np.round(np.linspace(0, self.n_frames - 1, n)).astype(int)
        elif mode == "random":
            return np.sort(np.random.choice(self.n_frames, size=min(n, self.n_frames), replace=False))
        else:
            raise ValueError(f"Unknown sample mode: {mode}")

    def close(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        return self.n_frames
