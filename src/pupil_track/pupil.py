"""Object-oriented pupil tracking pipeline.

All pipeline state is held as properties on a single Pupil object.
Methods correspond to pipeline steps and modify the object in-place.

Usage:
    p = Pupil("video.avi", output_dir="./output")
    p.auto_roi()
    masks = p.detect("unet")
    p.postprocess("unet")
    p.save_results("unet")
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import Config
from .infer import crop_and_resize
from .postprocess import export_csv
from .postprocess import postprocess as _postprocess
from .roi import auto_roi as _auto_roi
from .roi import manual_roi as _manual_roi
from .train import run_training as _run_training
from .video_io import VideoReader
from .visualize import plot_timeseries as _plot_timeseries

logger = logging.getLogger(__name__)


class Pupil:
    """Pupil tracking pipeline object.

    Properties
    ----------
    video_path : Path or None
    output_dir : Path          – working directory for all saved outputs
    input_size : int
    config     : Config
    roi        : dict or None      – {"x", "y", "w", "h"}
    masks      : dict[str, dict[int, ndarray]]  – per-method detection masks
    results    : dict[str, tuple[ndarray, ndarray]]  – per-method (raw, smoothed)
    model_path : Path or None
    label_frames : list[tuple[int, ndarray]]
    """

    def __init__(
        self,
        video_path: str | Path | None = None,
        output_dir: str | Path = "./output",
        input_size: int = 128,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_size = input_size
        self.config = Config(input_size=input_size)

        # mutable state
        self._checkpoint_dir: Optional[Path] = None
        self._reader: Optional[VideoReader] = None
        self.roi: Optional[dict] = None
        self.masks: dict[str, dict[int, np.ndarray]] = {}
        self.results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.label_frames: list[tuple[int, np.ndarray]] = []
        self._cached_full_frames: Optional[dict[int, np.ndarray]] = None  # Cache for avoiding re-decoding

        if video_path is not None:
            self.load_video(video_path)

    # ── convenience paths ────────────────────────────────────────────────

    @property
    def video_stem(self) -> str:
        """Video filename without extension, used as prefix for saved files."""
        if self._reader is None:
            raise RuntimeError("No video loaded. Call load_video() first.")
        return Path(self._reader.path).stem

    @property
    def img_dir(self) -> Path:
        return self.output_dir / "images"

    @property
    def mask_dir(self) -> Path:
        return self.output_dir / "masks"

    @property
    def checkpoint_dir(self) -> Path:
        return self._checkpoint_dir if self._checkpoint_dir else self.output_dir / "checkpoints"

    @checkpoint_dir.setter
    def checkpoint_dir(self, value: str | Path | None):
        self._checkpoint_dir = Path(value) if value else None

    @property
    def model_path(self) -> Optional[Path]:
        """U-Net model checkpoint path (stored in config)."""
        p = self.config.model_path
        return Path(p) if p else None

    @model_path.setter
    def model_path(self, value: str | Path | None):
        self.config.model_path = str(value) if value else ""

    @property
    def reader(self) -> VideoReader:
        if self._reader is None:
            raise RuntimeError("No video loaded. Call load_video() first.")
        return self._reader

    @property
    def video_info(self) -> dict:
        r = self.reader
        return dict(
            path=r.path,
            n_frames=r.n_frames,
            fps=r.fps,
            width=r.width,
            height=r.height,
        )

    # ── Step 1: Video I/O ────────────────────────────────────────────────

    def load_video(self, path: str | Path):
        """Open a video file."""
        if self._reader is not None:
            self._reader.close()
        self._reader = VideoReader(str(path))
        logger.info(
            "Loaded video: %s (%d frames, %.1f fps, %dx%d)",
            self._reader.path,
            self._reader.n_frames,
            self._reader.fps,
            self._reader.width,
            self._reader.height,
        )

    def read_frame(self, idx: int) -> np.ndarray:
        """Read a single frame (grayscale uint8)."""
        frame = self.reader.read_frame(idx)
        if frame is None:
            raise ValueError(f"Could not read frame {idx}")
        return frame

    def get_sample_indices(self, n: int, mode: str = "uniform") -> np.ndarray:
        return self.reader.sample_indices(n, mode=mode)

    def get_all_indices(self) -> np.ndarray:
        """Return indices for ALL frames in the video."""
        return np.arange(self.reader.n_frames)

    # ── Step 2: ROI ──────────────────────────────────────────────────────

    def auto_roi(self, frame_idx: int = 0, resize: bool = False,
                  method: str = "dark_blob", **kwargs) -> dict:
        """Detect ROI automatically on the given frame."""
        frame = self.read_frame(frame_idx)
        self.roi = _auto_roi(frame, target_size=self.input_size, method=method,
                             resize=resize, **kwargs)
        self.config.roi = self.roi
        logger.info("Auto ROI: %s", self.roi)
        return self.roi

    def manual_roi(self, frame_idx: int = 0, resize: bool = False,
                   frame: np.ndarray | None = None, scale: float = 0.5) -> dict:
        """Open cv2 window for manual ROI selection."""
        if frame is None:
            frame = self.read_frame(frame_idx)
        self.roi = _manual_roi(frame, target_size=self.input_size, resize=resize, scale=scale)
        self.config.roi = self.roi
        return self.roi

    def set_roi(self, x: int, y: int, w: int, h: int):
        self.roi = {"x": x, "y": y, "w": w, "h": h}
        self.config.roi = self.roi

    def show_roi(self, frame_idx: int = 0, scale: float = 0.5):
        """Display the current ROI as a green rectangle on the frame. Press any key to close."""
        if self.roi is None:
            raise RuntimeError("No ROI set. Call auto_roi(), manual_roi(), or set_roi() first.")
        frame = self.read_frame(frame_idx)
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        r = self.roi
        cv2.rectangle(display, (r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"]), (0, 255, 0), 2)
        h, w = display.shape[:2]
        cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI", int(w * scale), int(h * scale))
        cv2.imshow("ROI", display)
        logger.info("ROI: x=%d, y=%d, w=%d, h=%d — press any key to continue", r["x"], r["y"], r["w"], r["h"])
        cv2.waitKey(0)
        cv2.destroyWindow("ROI")

    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply ROI crop + resize to input_size."""
        return crop_and_resize(frame, self.roi, self.input_size)

    # ── Step 3: Detection ────────────────────────────────────────────────

    def _load_frames(
        self, frame_indices: np.ndarray | None, n_default: int = 30
    ) -> list[tuple[int, np.ndarray]]:
        """Load, crop and resize frames for detection (into memory)."""
        if frame_indices is None:
            frame_indices = self.get_sample_indices(n_default)
        frames = []
        for idx, frame in self.reader.frames(frame_indices):
            frames.append((idx, self.crop_frame(frame)))
        return frames

    def _load_frames_bulk(
        self, frame_indices: np.ndarray | None, n_default: int = 30
    ) -> tuple[list[tuple[int, np.ndarray]], dict[int, np.ndarray]]:
        """Bulk load entire video into memory, then extract ROI crops for maximum speed.
        
        Returns:
            tuple of (cropped_frames_for_detection, full_frames_for_export)
        """
        if frame_indices is None:
            frame_indices = self.get_sample_indices(n_default)
        
        logger.info(f"Optimized bulk loading {len(frame_indices)} frames...")
        
        # Use optimized bulk reading from VideoReader
        all_frames = self.reader.read_all_frames(frame_indices)
        
        logger.info(f"Extracting {len(frame_indices)} ROI crops from memory...")
        
        # Extract ROI crops from in-memory frames (pure memory operations)
        cropped_frames = []
        for idx in frame_indices:
            if idx in all_frames:
                cropped = self.crop_frame(all_frames[idx])
                cropped_frames.append((idx, cropped))
        
        logger.info(f"Prepared {len(cropped_frames)} cropped frames for detection")
        
        # Store full frames for later video export (avoid re-decoding)
        self._cached_full_frames = all_frames
        
        return cropped_frames, all_frames

    def _stream_frames(
        self, frame_indices: np.ndarray | None, n_default: int = 30
    ):
        """Yield (idx, cropped_frame) one at a time — constant memory."""
        if frame_indices is None:
            frame_indices = self.get_sample_indices(n_default)
        for idx, frame in self.reader.frames(frame_indices):
            yield idx, self.crop_frame(frame)

    def detect(
        self,
        method: str,
        frame_indices: np.ndarray | None = None,
        **kwargs,
    ) -> dict[int, np.ndarray]:
        """Run detection and store results in ``self.masks[method]``."""
        if method == "intdiff":
            from .detectors.intdiff import detect_intdiff

            frames = self._load_frames(frame_indices)
            masks = detect_intdiff(
                frames,
                rmin=kwargs.get("rmin", self.config.intdiff_rmin),
                rmax=kwargs.get("rmax", self.config.intdiff_rmax),
                downscale=kwargs.get("downscale", self.config.intdiff_downscale),
            )
        elif method == "starburst":
            from .detectors.starburst import detect_starburst

            frames = self._load_frames(frame_indices)
            masks = detect_starburst(
                frames,
                edge_thresh=kwargs.get("edge_thresh", self.config.starburst_edge_thresh),
                n_rays=kwargs.get("n_rays", self.config.starburst_rays),
                sigma=kwargs.get("sigma", self.config.starburst_sigma),
                min_features=kwargs.get("min_features", self.config.starburst_min_features),
                max_ransac=kwargs.get("max_ransac", self.config.starburst_max_ransac),
                remove_cr=kwargs.get("remove_cr", self.config.starburst_remove_cr),
            )
        elif method == "unet":
            if self.model_path is None:
                raise RuntimeError("No model path set. Train first or set model_path.")
            from .detectors.unet_detect import detect_unet

            # Bulk load entire video + extract ROI crops for maximum speed
            frames, full_frames = self._load_frames_bulk(frame_indices)
            masks = detect_unet(
                frames,
                model_path=str(self.model_path),
                threshold=kwargs.get("threshold", self.config.threshold),
            )
            self.masks[method] = masks
            n_ok = sum(1 for m in masks.values() if m.sum() > 0)
            logger.info("%s: detected %d / %d frames", method, n_ok, len(masks))
            return masks
        else:
            raise ValueError(f"Unknown method: {method}")

        self.masks[method] = masks
        n_ok = sum(1 for m in masks.values() if m.sum() > 0)
        logger.info("%s: detected %d / %d frames", method, n_ok, len(frames))
        return masks

    # ── Step 4: Labeling ─────────────────────────────────────────────────

    def prepare_label_frames(self, n: int = 10) -> list[tuple[int, np.ndarray]]:
        """Sample, crop and save frames for annotation."""
        indices = self.get_sample_indices(n)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        stem = self.video_stem
        self.label_frames = []
        for idx, frame in self.reader.frames(indices):
            cropped = self.crop_frame(frame)
            self.label_frames.append((idx, cropped))
            cv2.imwrite(str(self.img_dir / f"{stem}_{idx:06d}.png"), cropped)
        logger.info("Prepared %d label frames → %s", len(self.label_frames), self.img_dir)
        return self.label_frames

    def annotate(self):
        """Launch the OpenCV annotation GUI (blocking)."""
        if not self.label_frames:
            raise RuntimeError("No frames prepared. Call prepare_label_frames() first.")
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        from .annotate import annotate_frames

        annotate_frames(self.label_frames, self.mask_dir, video_stem=self.video_stem)

    @property
    def n_labeled(self) -> int:
        """Number of saved mask PNGs."""
        if not self.mask_dir.exists():
            return 0
        return len(list(self.mask_dir.glob("*.png")))

    # ── Step 5: Post-processing ──────────────────────────────────────────

    def postprocess(self, method: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract features, fill gaps, smooth. Stores in ``self.results``."""
        if method not in self.masks:
            raise RuntimeError(f"No masks for '{method}'. Run detect() first.")
        fps = self._reader.fps if self._reader else 30.0
        raw, smoothed = _postprocess(
            self.masks[method], fps=fps, smooth_window=self.config.smooth_window
        )
        self.results[method] = (raw, smoothed)
        return raw, smoothed

    def plot(self, method: str, path: str | Path | None = None) -> Path | None:
        if method not in self.results:
            self.postprocess(method)
        _, smoothed = self.results[method]
        if path is None:
            path = self.output_dir / f"{self.video_stem}_plot_{method}.png"
        _plot_timeseries(smoothed, str(path))
        return Path(path)

    def preview(self, method: str, start_frame: int = 0, scale: float = 2.0):
        """Interactive playback with detection overlay.

        After detect() only  → shows raw mask contour
        After postprocess()  → shows fitted ellipse + centroid
        """
        from .visualize import preview as _preview

        masks = self.masks.get(method)
        if masks is None:
            raise RuntimeError(f"No masks for '{method}'. Run detect() first.")

        # Use postprocessed results if available, otherwise None → raw contour mode
        results = None
        if method in self.results:
            _, results = self.results[method]

        _preview(
            self.reader.path,
            masks=masks,
            results=results,
            roi=self.roi,
            input_size=self.input_size,
            start_frame=start_frame,
            scale=scale,
        )

    def _populate_config(self, method: str, paths: dict):
        """Fill config fields from current pipeline state before saving."""
        # Data info
        r = self.reader
        self.config.data_name = Path(r.path).name
        self.config.data_path = r.path
        self.config.data_format = Path(r.path).suffix.lstrip(".")
        self.config.data_width = r.width
        self.config.data_height = r.height
        self.config.data_n_frames = r.n_frames
        self.config.data_fps = r.fps

        # Method
        self.config.method = method

        # Results paths
        self.config.results_csv = str(paths.get("csv", ""))
        self.config.results_masks_npy = str(paths.get("masks_npy", ""))
        self.config.results_plot = str(paths.get("plot", ""))
        self.config.results_preview_video = str(paths.get("preview_video", ""))

    def save_results(self, method: str):
        """Save CSV, masks, plot, config, and preview video into output_dir."""
        if method not in self.results:
            self.postprocess(method)
        _, smoothed = self.results[method]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.video_stem
        paths: dict = {}

        # CSV
        csv_path = self.output_dir / f"{stem}_results_{method}.csv"
        export_csv(smoothed, csv_path)
        paths["csv"] = csv_path
        logger.info("Saved CSV: %s", csv_path)

        # Binary masks (.npy)
        masks = self.masks.get(method)
        if masks:
            masks_path = self.output_dir / f"{stem}_masks_{method}.npy"
            np.save(str(masks_path), masks)
            paths["masks_npy"] = masks_path
            logger.info("Saved masks: %s", masks_path)

        # Plot
        plot_path = self.output_dir / f"{stem}_plot_{method}.png"
        _plot_timeseries(smoothed, str(plot_path))
        paths["plot"] = plot_path
        logger.info("Saved plot: %s", plot_path)

        # Preview video (use cached frames to avoid re-decoding)
        from .visualize import export_video
        video_path = self.output_dir / f"{stem}_preview_{method}.mp4"
        
        # Use cached full frames if available, otherwise fallback to re-reading
        preloaded_frames = getattr(self, '_cached_full_frames', None)
        if preloaded_frames is not None:
            logger.info("Using cached frames for video export - no re-decoding needed")
        else:
            logger.info("No cached frames available - will re-decode video (slower)")
            
        export_video(
            self.reader.path, str(video_path), smoothed,
            masks=masks, roi=self.roi, input_size=self.input_size,
            fps=self.config.export_fps, preloaded_frames=preloaded_frames,
        )
        paths["preview_video"] = video_path
        logger.info("Saved preview video: %s", video_path)

        # Config (populated last, after all paths are known)
        self._populate_config(method, paths)
        config_path = self.output_dir / f"{stem}_config.json"
        self.config.save(config_path)
        logger.info("Saved config: %s", config_path)

        return self.output_dir

    # ── Step 6: Training ─────────────────────────────────────────────────

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 8,
        lr: float = 1e-3,
        patience: int = 10,
        amp: bool = False,
        freeze_encoder: bool = False,
        data_dirs: list[tuple[str, str]] | None = None,
        max_snapshots: int = 3,
        checkpoint_dir: str | Path | None = None,
    ) -> Path:
        """Train ShallowUNet on labeled data. Returns path to best checkpoint."""
        if data_dirs is None:
            data_dirs = [(str(self.img_dir), str(self.mask_dir))]

        # Resolve checkpoint directory: explicit arg > property > first data dir
        if checkpoint_dir is not None:
            self.checkpoint_dir = checkpoint_dir
        elif self._checkpoint_dir is None:
            self.checkpoint_dir = Path(data_dirs[0][0]).parent / "checkpoints"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            "--dir-checkpoint", str(self.checkpoint_dir),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--patience", str(patience),
            "--input-size", str(self.input_size),
            "--max-snapshots", str(max_snapshots),
        ]
        for img_dir, mask_dir in data_dirs:
            argv.extend(["--data-dir", img_dir, mask_dir])
        if amp:
            argv.append("--amp")
        if freeze_encoder:
            argv.append("--freeze-encoder")

        _run_training(argv)

        best = self.checkpoint_dir / "best_model.pth"
        if best.exists():
            self.model_path = best
        return best

    # ── Config persistence ───────────────────────────────────────────────

    def save_config(self, path: str | Path | None = None):
        if path is None:
            path = self.output_dir / f"{self.video_stem}_config.json"
        # Populate data info if a video is loaded
        if self._reader is not None:
            r = self._reader
            self.config.data_name = Path(r.path).name
            self.config.data_path = r.path
            self.config.data_format = Path(r.path).suffix.lstrip(".")
            self.config.data_width = r.width
            self.config.data_height = r.height
            self.config.data_n_frames = r.n_frames
            self.config.data_fps = r.fps
        self.config.save(path)

    def load_config(self, path: str | Path | None = None):
        """Load config from JSON. If path is omitted, auto-finds {stem}_config.json."""
        if path is None:
            path = self.output_dir / f"{self.video_stem}_config.json"
            if not path.exists():
                logger.info("No config found at %s", path)
                return
        self.config = Config.load(path)
        self.roi = self.config.roi
        self.input_size = self.config.input_size
        logger.info("Loaded config from %s", path)

        # Restore masks from .npy if available
        if self.config.results_masks_npy:
            npy = Path(self.config.results_masks_npy)
            if npy.exists():
                try:
                    loaded = np.load(str(npy), allow_pickle=True).item()
                    method = self.config.method
                    self.masks[method] = loaded
                    logger.info("Loaded masks from %s (%d frames)", npy, len(loaded))
                except Exception as e:
                    logger.warning("Failed to load masks: %s", e)

    # ── Cleanup ──────────────────────────────────────────────────────────

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        video = self._reader.path if self._reader else "None"
        return f"Pupil(video={video}, output_dir={self.output_dir})"
