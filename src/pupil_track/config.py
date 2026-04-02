"""Pipeline configuration with JSON serialization.

The config is organized into logical sections for readability and
reproducibility.  Internally all fields are flat dataclass attributes
for easy access (``self.config.roi``, ``self.config.model_path``, …).
``save()`` writes structured JSON; ``load()`` reads both the new
structured format and the legacy flat format.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import __version__


@dataclass
class Config:
    """Pipeline configuration for pupil tracking.

    Sections (as they appear in the saved JSON):
        data        — video name, path, format, resolution, length
        software    — package name, version, save timestamp
        roi         — crop region and target size
        model       — U-Net checkpoint path and threshold
        postprocess — smoothing, corrected frames
        results     — paths to saved output files
        detectors   — method selection + intdiff / starburst parameters
    """

    # --- Data (populated on save, from Pupil state) ---
    data_name: str = ""
    data_path: str = ""
    data_format: str = ""
    data_width: int = 0
    data_height: int = 0
    data_n_frames: int = 0
    data_fps: float = 0.0

    # --- ROI ---
    roi: Optional[dict] = None  # {"x", "y", "w", "h"}
    roi_resize: bool = False
    input_size: int = 128

    # --- Detection method ---
    method: str = "unet"  # "unet", "intdiff", "starburst"

    # --- Model / Inference ---
    model_path: str = ""
    threshold: float = 0.5

    # --- Integrodifferential ---
    intdiff_rmin: int = 10
    intdiff_rmax: int = 20
    intdiff_downscale: float = 0.5
    intdiff_nsides: int = 600

    # --- Starburst ---
    starburst_edge_thresh: int = 10
    starburst_rays: int = 36
    starburst_sigma: float = 4.0
    starburst_min_features: int = 10
    starburst_max_ransac: int = 10000
    starburst_remove_cr: bool = False

    # --- Post-processing ---
    smooth_window: int = 10
    outlier_std_factor: float = 3.0
    corrected_frames: list[int] = field(default_factory=list)

    # --- Visualization / Export ---
    export_fps: int = 30

    # --- Results paths (populated on save) ---
    results_csv: str = ""
    results_masks_npy: str = ""
    results_plot: str = ""
    results_preview_video: str = ""

    # ── Serialize ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return organized nested dict for JSON serialization."""
        return {
            "data": {
                "name": self.data_name,
                "path": self.data_path,
                "format": self.data_format,
                "width": self.data_width,
                "height": self.data_height,
                "n_frames": self.data_n_frames,
                "fps": self.data_fps,
            },
            "software": {
                "name": "pupil_track",
                "version": __version__,
                "datetime": datetime.now().isoformat(timespec="seconds"),
            },
            "roi": {
                "x": self.roi["x"] if self.roi else None,
                "y": self.roi["y"] if self.roi else None,
                "w": self.roi["w"] if self.roi else None,
                "h": self.roi["h"] if self.roi else None,
                "resize": self.roi_resize,
                "input_size": self.input_size,
            },
            "model": {
                "path": self.model_path,
                "threshold": self.threshold,
            },
            "postprocess": {
                "smooth_window": self.smooth_window,
                "outlier_std_factor": self.outlier_std_factor,
                "corrected_frames": self.corrected_frames,
                "export_fps": self.export_fps,
            },
            "results": {
                "csv": self.results_csv,
                "masks_npy": self.results_masks_npy,
                "plot": self.results_plot,
                "preview_video": self.results_preview_video,
            },
            "detectors": {
                "method": self.method,
                "intdiff": {
                    "rmin": self.intdiff_rmin,
                    "rmax": self.intdiff_rmax,
                    "downscale": self.intdiff_downscale,
                    "nsides": self.intdiff_nsides,
                },
                "starburst": {
                    "edge_thresh": self.starburst_edge_thresh,
                    "rays": self.starburst_rays,
                    "sigma": self.starburst_sigma,
                    "min_features": self.starburst_min_features,
                    "max_ransac": self.starburst_max_ransac,
                    "remove_cr": self.starburst_remove_cr,
                },
            },
        }

    def save(self, path: str | Path):
        """Save configuration to organized JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    # ── Deserialize ───────────────────────────────────────────────────

    @classmethod
    def _from_structured(cls, data: dict) -> "Config":
        """Load from the new organized JSON format."""
        kw: dict = {}

        # data section
        d = data.get("data", {})
        kw["data_name"] = d.get("name", "")
        kw["data_path"] = d.get("path", "")
        kw["data_format"] = d.get("format", "")
        kw["data_width"] = d.get("width", 0)
        kw["data_height"] = d.get("height", 0)
        kw["data_n_frames"] = d.get("n_frames", 0)
        kw["data_fps"] = d.get("fps", 0.0)

        # roi section
        r = data.get("roi", {})
        if r.get("x") is not None:
            kw["roi"] = {"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"]}
        kw["roi_resize"] = r.get("resize", False)
        kw["input_size"] = r.get("input_size", 128)

        # model section
        m = data.get("model", {})
        kw["model_path"] = m.get("path", "")
        kw["threshold"] = m.get("threshold", 0.5)

        # postprocess section
        p = data.get("postprocess", {})
        kw["smooth_window"] = p.get("smooth_window", 10)
        kw["outlier_std_factor"] = p.get("outlier_std_factor", 3.0)
        kw["corrected_frames"] = p.get("corrected_frames", [])
        kw["export_fps"] = p.get("export_fps", 30)

        # results section
        res = data.get("results", {})
        kw["results_csv"] = res.get("csv", "")
        kw["results_masks_npy"] = res.get("masks_npy", "")
        kw["results_plot"] = res.get("plot", "")
        kw["results_preview_video"] = res.get("preview_video", "")

        # detectors section
        det = data.get("detectors", {})
        kw["method"] = det.get("method", "unet")
        intdiff = det.get("intdiff", {})
        kw["intdiff_rmin"] = intdiff.get("rmin", 10)
        kw["intdiff_rmax"] = intdiff.get("rmax", 20)
        kw["intdiff_downscale"] = intdiff.get("downscale", 0.5)
        kw["intdiff_nsides"] = intdiff.get("nsides", 600)
        sb = det.get("starburst", {})
        kw["starburst_edge_thresh"] = sb.get("edge_thresh", 10)
        kw["starburst_rays"] = sb.get("rays", 36)
        kw["starburst_sigma"] = sb.get("sigma", 4.0)
        kw["starburst_min_features"] = sb.get("min_features", 10)
        kw["starburst_max_ransac"] = sb.get("max_ransac", 10000)
        kw["starburst_remove_cr"] = sb.get("remove_cr", False)

        return cls(**kw)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls._from_structured(data)
