"""Annotation GUI: brush and ellipse tools for creating binary masks."""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AnnotationGUI:
    """OpenCV-based annotation tool for painting binary pupil masks.

    Controls:
        Left mouse button: paint (foreground = pupil)
        Right mouse button: erase (background)
        Scroll wheel / +/-: adjust brush radius
        E: toggle ellipse mode (click center, drag to define)
        N: next frame
        P: previous frame
        S: save current mask
        Q: quit and save all
    """

    def __init__(self, frames: list[tuple[int, np.ndarray]], output_dir: str | Path,
                 video_stem: str = "video"):
        self.frames = frames  # list of (index, grayscale_frame)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_stem = video_stem

        self.current = 0
        self.brush_radius = 5
        self.drawing = False
        self.erasing = False
        self.masks = {}

        # Initialize masks (try loading existing ones)
        for idx, frame in self.frames:
            mask_path = self.output_dir / f"{self.video_stem}_{idx:06d}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    h, w = frame.shape[:2]
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    self.masks[idx] = (mask > 127).astype(np.uint8) * 255
                    continue
            self.masks[idx] = np.zeros_like(frame)

    def _current_frame(self) -> tuple[int, np.ndarray]:
        return self.frames[self.current]

    def _display(self):
        idx, frame = self._current_frame()
        mask = self.masks[idx]

        # Overlay mask in red on the frame
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        overlay = display.copy()
        overlay[mask > 0] = [0, 0, 255]
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

        # Info text
        info = f"Frame {idx} [{self.current + 1}/{len(self.frames)}] Brush: {self.brush_radius}"
        cv2.putText(display, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(display, "L:paint R:erase +/-:size N/P:nav S:save Q:quit",
                    (5, display.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        cv2.imshow("Annotate", display)

    def _mouse_callback(self, event, x, y, flags, param):
        idx, _ = self._current_frame()
        mask = self.masks[idx]

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.circle(mask, (x, y), self.brush_radius, 255, -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.erasing = True
            cv2.circle(mask, (x, y), self.brush_radius, 0, -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.circle(mask, (x, y), self.brush_radius, 255, -1)
            elif self.erasing:
                cv2.circle(mask, (x, y), self.brush_radius, 0, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
        elif event == cv2.EVENT_RBUTTONUP:
            self.erasing = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.brush_radius = min(self.brush_radius + 1, 50)
            else:
                self.brush_radius = max(self.brush_radius - 1, 1)

        self._display()

    def _save_mask(self, idx: int):
        mask = self.masks[idx]
        path = self.output_dir / f"{self.video_stem}_{idx:06d}.png"
        cv2.imwrite(str(path), mask)

    def _save_all(self):
        for idx in self.masks:
            self._save_mask(idx)
        logger.info(f"Saved {len(self.masks)} masks to {self.output_dir}")

    def run(self):
        """Run the annotation GUI."""
        cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotate", self._mouse_callback)
        self._display()

        while True:
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                self._save_all()
                break
            elif key == ord("s"):
                idx, _ = self._current_frame()
                self._save_mask(idx)
                logger.info(f"Saved mask for frame {idx}")
            elif key == ord("n"):
                if self.current < len(self.frames) - 1:
                    self.current += 1
                    self._display()
            elif key == ord("p"):
                if self.current > 0:
                    self.current -= 1
                    self._display()
            elif key == ord("+") or key == ord("="):
                self.brush_radius = min(self.brush_radius + 1, 50)
                self._display()
            elif key == ord("-"):
                self.brush_radius = max(self.brush_radius - 1, 1)
                self._display()

        cv2.destroyAllWindows()


def annotate_frames(
    frames: list[tuple[int, np.ndarray]],
    output_dir: str | Path,
    video_stem: str = "video",
):
    """Launch the annotation GUI.

    Args:
        frames: list of (frame_index, grayscale_frame) tuples
        output_dir: directory to save mask PNGs
        video_stem: video filename stem used as filename prefix
    """
    gui = AnnotationGUI(frames, output_dir, video_stem=video_stem)
    gui.run()
