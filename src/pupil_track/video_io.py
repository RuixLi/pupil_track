"""Video I/O: load and iterate over video frames."""

import logging
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Wrapper around cv2.VideoCapture for frame-by-frame access.

    All frames are returned as uint8 grayscale numpy arrays.
    """

    def __init__(self, path: str | Path, enable_threading: bool = True, try_hardware_decode: bool = True):
        self.path = str(path)
        
        # Try hardware-accelerated decoding first
        self._cap = None
        if try_hardware_decode:
            self._cap = self._try_hardware_decode()
        
        # Fallback to software decoding
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.path)
            logger.info("Using software video decoding")
        
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
            
        # Enable multi-threading for better performance
        if enable_threading:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
            # OpenCV will automatically use multiple threads for decoding
            
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _try_hardware_decode(self) -> Optional[cv2.VideoCapture]:
        """Try hardware-accelerated video decoding."""
        # List of hardware decoding backends to try (Windows)
        hw_backends = [
            cv2.CAP_DSHOW,      # DirectShow (Windows)
            cv2.CAP_MSMF,       # Media Foundation (Windows) 
            cv2.CAP_FFMPEG,     # FFmpeg with potential hardware acceleration
        ]
        
        for backend in hw_backends:
            try:
                cap = cv2.VideoCapture(self.path, backend)
                if cap.isOpened():
                    # Test reading one frame to verify it works
                    ret, _ = cap.read()
                    if ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                        logger.info(f"Using hardware-accelerated decoding (backend: {backend})")
                        return cap
                cap.release()
            except:
                pass
        
        return None

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
        
        # Check if indices are sequential - if so, use optimized sequential reading
        indices_array = np.array(indices)
        if len(indices_array) > 1 and np.all(np.diff(indices_array) == 1):
            logger.info(f"Sequential frame access detected - using optimized reading")
            yield from self._frames_sequential(indices_array)
        else:
            # Fall back to random access for non-sequential indices
            for idx in indices:
                frame = self.read_frame(idx)
                if frame is not None:
                    yield idx, frame

    def _frames_sequential(self, indices: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
        """Optimized sequential frame reading - much faster than seeking."""
        start_idx = indices[0]
        end_idx = indices[-1]
        
        # Seek to start position once
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        
        current_idx = start_idx
        for target_idx in indices:
            # Read frames sequentially until we reach target
            while current_idx <= target_idx:
                ret, frame = self._cap.read()
                if not ret:
                    return
                    
                if current_idx == target_idx:
                    # Convert to grayscale
                    if frame.ndim == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    yield current_idx, frame
                    
                current_idx += 1
                
    def read_all_frames(self, indices: Optional[np.ndarray] = None) -> dict[int, np.ndarray]:
        """Bulk read all frames into memory for maximum speed.
        
        Returns:
            dict mapping frame_index -> grayscale frame
        """
        if indices is None:
            indices = np.arange(self.n_frames)
        
        logger.info(f"Bulk reading {len(indices)} frames...")
        
        frames = {}
        for idx, frame in self.frames(indices):
            frames[idx] = frame
            
        logger.info(f"Loaded {len(frames)} frames into memory")
        return frames
        
    def close(self):
        """Close the video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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
