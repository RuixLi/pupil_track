"""PyQt5 GUI for the pupil tracking pipeline.

Launch with:  python -m pupil_track gui
         or:  python run_demo.py
"""

import logging
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .pupil import Pupil

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Dark theme
# ═══════════════════════════════════════════════════════════════════════════════

DARK_STYLE = """
* { font-size: 30px; font-family: "Segoe UI"; }
QMainWindow, QWidget { background-color: #1e1e1e; color: #d4d4d4; }
QGroupBox {
    border: 1px solid #3d3d3d; border-radius: 4px; margin-top: 10px;
    padding-top: 14px; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton {
    background-color: #0e639c; color: white; border: none;
    padding: 8px 20px; border-radius: 3px; min-height: 32px;
}
QPushButton:hover { background-color: #1177bb; }
QPushButton:pressed { background-color: #094771; }
QPushButton:disabled { background-color: #3d3d3d; color: #808080; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #2d2d2d; border: 1px solid #3d3d3d;
    border-radius: 3px; padding: 6px 10px; color: #d4d4d4;
    min-height: 28px;
}
QListWidget {
    background-color: #252526; border: 1px solid #3d3d3d;
    outline: none; font-size: 25px; font-weight: bold;
}
QListWidget::item { padding: 12px 16px; border-bottom: 1px solid #2d2d2d; }
QListWidget::item:selected { background-color: #094771; color: white; }
QListWidget::item:hover:!selected { background-color: #2a2d2e; }
QListWidget::item:disabled { color: #666666; }
QTextEdit {
    background-color: #1e1e1e; border: 1px solid #3d3d3d;
    color: #9cdcfe; font-family: Consolas, monospace; font-size: 25px;
}
QSlider::groove:horizontal {
    background: #3d3d3d; height: 6px; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #0e639c; width: 18px; margin: -6px 0; border-radius: 9px;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; }
QProgressBar {
    border: 1px solid #3d3d3d; border-radius: 3px;
    background-color: #2d2d2d; text-align: center; color: white;
    min-height: 24px;
}
QProgressBar::chunk { background-color: #0e639c; border-radius: 2px; }
QLabel#title { font-size: 42px; font-weight: bold; color: #e0e0e0; }
QLabel#info  { color: #808080; font-size: 33px; }
QLabel#status_ok   { color: #4ec9b0; }
QLabel#status_warn { color: #dcdcaa; }
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def ndarray_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Convert uint8 numpy array (gray or BGR) to QPixmap."""
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        return QPixmap.fromImage(QImage(arr.data, w, h, w, QImage.Format_Grayscale8))
    h, w, c = arr.shape
    if c == 3:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888))
    return QPixmap.fromImage(QImage(arr.data, w, h, w, QImage.Format_Grayscale8))


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """Draw raw mask contour on a grayscale frame. Returns BGR."""
    display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if mask is not None and mask.sum() > 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, color, 1)
    return display


def enhance_contrast(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE contrast enhancement.

    Works on grayscale (H,W) or BGR (H,W,3) images.
    clip_limit controls the enhancement strength (1.0 = mild, 10.0 = strong).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    if img.ndim == 2:
        return clahe.apply(img)
    # BGR: convert to LAB, enhance L channel, convert back
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ═══════════════════════════════════════════════════════════════════════════════
# Background worker
# ═══════════════════════════════════════════════════════════════════════════════


class Worker(QThread):
    """Run a callable in a background thread."""

    finished = pyqtSignal(object)  # result
    error = pyqtSignal(str)  # traceback string

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# Image viewer widget
# ═══════════════════════════════════════════════════════════════════════════════


class ImageViewer(QLabel):
    """Displays a numpy image scaled to fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(128, 128)
        self.setStyleSheet("background-color: #1e1e1e;")
        self._source: QPixmap | None = None

    def set_image(self, arr: np.ndarray):
        self._source = ndarray_to_qpixmap(arr)
        self._rescale()

    def set_max_display(self, max_w: int, max_h: int):
        """Set the maximum display size (keeps aspect ratio via scaling)."""
        self.setMaximumSize(max_w, max_h)

    def clear_image(self):
        self._source = None
        self.clear()

    def _rescale(self):
        if self._source and not self._source.isNull():
            scaled = self._source.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════════
# Annotation canvas
# ═══════════════════════════════════════════════════════════════════════════════


class AnnotationCanvas(QWidget):
    """Paint-on-image widget for mask annotation."""

    mask_changed = pyqtSignal()
    brush_changed = pyqtSignal(int)  # emitted when wheel changes brush size

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(256, 256)
        self._image: np.ndarray | None = None  # grayscale uint8
        self._mask: np.ndarray | None = None  # uint8 0/255
        self._brush_radius = 20
        self._drawing = False
        self._erasing = False
        self.setMouseTracking(True)
        self._cursor_pos = None

    def set_data(self, image: np.ndarray, mask: np.ndarray | None = None):
        self._image = image.copy()
        if mask is None:
            self._mask = np.zeros_like(image, dtype=np.uint8)
        else:
            self._mask = mask.copy()
        self.update()

    @property
    def mask(self) -> np.ndarray | None:
        return self._mask

    @property
    def brush_radius(self) -> int:
        return self._brush_radius

    @brush_radius.setter
    def brush_radius(self, v: int):
        self._brush_radius = max(1, min(v, 100))
        self.update()

    def _img_to_widget(self, ix: float, iy: float):
        """Convert image coords to widget coords."""
        if self._image is None:
            return 0, 0
        ih, iw = self._image.shape[:2]
        ww, wh = self.width(), self.height()
        scale = min(ww / iw, wh / ih)
        ox = (ww - iw * scale) / 2
        oy = (wh - ih * scale) / 2
        return int(ix * scale + ox), int(iy * scale + oy)

    def _widget_to_img(self, wx: int, wy: int):
        """Convert widget coords to image coords."""
        if self._image is None:
            return -1, -1
        ih, iw = self._image.shape[:2]
        ww, wh = self.width(), self.height()
        scale = min(ww / iw, wh / ih)
        ox = (ww - iw * scale) / 2
        oy = (wh - ih * scale) / 2
        ix = (wx - ox) / scale
        iy = (wy - oy) / scale
        return int(ix), int(iy)

    def _paint_at(self, wx, wy, erase=False):
        ix, iy = self._widget_to_img(wx, wy)
        if self._mask is None:
            return
        ih, iw = self._mask.shape[:2]
        if 0 <= ix < iw and 0 <= iy < ih:
            ww, wh = self.width(), self.height()
            scale = min(ww / iw, wh / ih)
            r = max(1, int(self._brush_radius / scale))
            color = 0 if erase else 255
            cv2.circle(self._mask, (ix, iy), r, color, -1)
            self.mask_changed.emit()
            self.update()

    def paintEvent(self, event):
        if self._image is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        ih, iw = self._image.shape[:2]
        ww, wh = self.width(), self.height()
        scale = min(ww / iw, wh / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        ox, oy = (ww - dw) // 2, (wh - dh) // 2

        # draw image
        pix = ndarray_to_qpixmap(self._image)
        painter.drawPixmap(ox, oy, dw, dh, pix)

        # draw mask overlay (semi-transparent red)
        if self._mask is not None and self._mask.sum() > 0:
            red = np.zeros((*self._mask.shape, 4), dtype=np.uint8)
            red[self._mask > 0] = [255, 0, 0, 100]
            red = np.ascontiguousarray(red)
            h, w = red.shape[:2]
            qimg = QImage(red.data, w, h, 4 * w, QImage.Format_RGBA8888)
            painter.drawPixmap(ox, oy, dw, dh, QPixmap.fromImage(qimg))

        # brush cursor
        if self._cursor_pos:
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(
                self._cursor_pos[0] - self._brush_radius,
                self._cursor_pos[1] - self._brush_radius,
                self._brush_radius * 2,
                self._brush_radius * 2,
            )

        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drawing = True
            self._paint_at(event.x(), event.y(), erase=False)
        elif event.button() == Qt.RightButton:
            self._erasing = True
            self._paint_at(event.x(), event.y(), erase=True)

    def mouseMoveEvent(self, event):
        self._cursor_pos = (event.x(), event.y())
        if self._drawing:
            self._paint_at(event.x(), event.y(), erase=False)
        elif self._erasing:
            self._paint_at(event.x(), event.y(), erase=True)
        self.update()

    def mouseReleaseEvent(self, event):
        self._drawing = False
        self._erasing = False

    def wheelEvent(self, event):
        delta = 1 if event.angleDelta().y() > 0 else -1
        self.brush_radius += delta
        self.brush_changed.emit(self._brush_radius)

    def leaveEvent(self, event):
        self._cursor_pos = None
        self.update()


# ═══════════════════════════════════════════════════════════════════════════════
# Logging handler → QTextEdit
# ═══════════════════════════════════════════════════════════════════════════════


class QTextEditLogHandler(logging.Handler):
    """Redirect log messages to a QTextEdit widget."""

    def __init__(self, widget: QTextEdit):
        super().__init__()
        self._widget = widget

    def emit(self, record):
        msg = self.format(record)
        self._widget.append(msg)
        # auto-scroll
        sb = self._widget.verticalScrollBar()
        sb.setValue(sb.maximum())


# ═══════════════════════════════════════════════════════════════════════════════
# Step pages
# ═══════════════════════════════════════════════════════════════════════════════


class VideoPage(QWidget):
    """Step 1 — Load video, set save directory, inspect basic properties."""

    video_loaded = pyqtSignal()  # emitted after a video is successfully loaded

    def __init__(self, pupil: Pupil, parent=None):
        super().__init__(parent)
        self.pupil = pupil
        self._enhance_fn = None  # set by MainWindow after ROIPage is created
        layout = QVBoxLayout(self)

        title = QLabel("Step 1: Video I/O")
        title.setObjectName("title")
        layout.addWidget(title)

        # video file controls
        ctrl = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Video file path...")
        ctrl.addWidget(self.path_edit, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse)
        ctrl.addWidget(browse_btn)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load)
        ctrl.addWidget(load_btn)
        layout.addLayout(ctrl)

        # save directory
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Save dir:"))
        self.save_dir_edit = QLineEdit()
        self.save_dir_edit.setPlaceholderText("Output directory (defaults to video folder)")
        save_row.addWidget(self.save_dir_edit, 1)
        save_browse_btn = QPushButton("Browse")
        save_browse_btn.clicked.connect(self._browse_save_dir)
        save_row.addWidget(save_browse_btn)
        layout.addLayout(save_row)

        # info
        self.info_label = QLabel("")
        self.info_label.setObjectName("info")
        layout.addWidget(self.info_label)

        # frame slider + play
        slider_row = QHBoxLayout()
        self.play_btn = QPushButton(">>")
        self.play_btn.setFixedWidth(80)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._toggle_play)
        slider_row.addWidget(self.play_btn)
        slider_row.addWidget(QLabel("Frame:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._show_frame)
        slider_row.addWidget(self.slider, 1)
        self.frame_label = QLabel("0")
        self.frame_label.setFixedWidth(60)
        slider_row.addWidget(self.frame_label)
        layout.addLayout(slider_row)

        self.viewer = ImageViewer()
        layout.addWidget(self.viewer, 1)

        # play timer
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)  # ~30 fps
        self._play_timer.timeout.connect(self._play_step)
        self.setFocusPolicy(Qt.StrongFocus)

    def _toggle_play(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_btn.setText(">>")
        else:
            self._play_timer.start()
            self.play_btn.setText("<>")

    def _play_step(self):
        val = self.slider.value()
        if val < self.slider.maximum():
            self.slider.setValue(val + 1)
        else:
            self._play_timer.stop()
            self.play_btn.setText(">>")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and self.slider.isEnabled():
            self._toggle_play()
        else:
            super().keyPressEvent(event)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.avi *.mp4 *.mkv *.mov);;All (*)"
        )
        if path:
            self.path_edit.setText(path)

    def _browse_save_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if folder:
            self.save_dir_edit.setText(folder)

    def _load(self):
        path = self.path_edit.text().strip()
        if not path:
            return
        try:
            self.pupil.load_video(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        # Set save_dir: use user-specified or default to video's parent folder
        save_dir = self.save_dir_edit.text().strip()
        if not save_dir:
            save_dir = str(Path(path).parent)
            self.save_dir_edit.setText(save_dir)
        self.pupil.output_dir = Path(save_dir)
        self.pupil.output_dir.mkdir(parents=True, exist_ok=True)

        # Config is loaded in MainWindow._restore_from_config() after reset
        info = self.pupil.video_info
        self.info_label.setText(
            f"{info['n_frames']} frames  |  {info['fps']:.1f} fps  |  "
            f"{info['width']}x{info['height']}  |  Save: {save_dir}"
        )
        # scale viewer to match video aspect ratio, fit within available space
        vw, vh = info["width"], info["height"]
        max_h = 1200
        scale = min(max_h / vh, 1.0)
        self.viewer.set_max_display(int(vw * scale), int(vh * scale))

        self.slider.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.slider.setRange(0, info["n_frames"] - 1)
        self.slider.setValue(0)
        self._show_frame(0)
        self.video_loaded.emit()

    def _show_frame(self, idx):
        self.frame_label.setText(str(idx))
        try:
            frame = self.pupil.read_frame(idx)
            if self._enhance_fn:
                frame = self._enhance_fn(frame)
            self.viewer.set_image(frame)
        except Exception:
            pass


class ROIPage(QWidget):
    """Step 2 — ROI selection and cropping, with optional contrast enhancement."""

    # Signal emitted when contrast settings change — other pages can connect
    contrast_changed = pyqtSignal(bool, float)  # (enabled, clip_limit)

    def __init__(self, pupil: Pupil, parent=None):
        super().__init__(parent)
        self.pupil = pupil
        self._enhance_enabled = False
        self._clip_limit = 2.0
        layout = QVBoxLayout(self)

        title = QLabel("Step 2: ROI Selection")
        title.setObjectName("title")
        layout.addWidget(title)

        # ROI buttons
        ctrl = QHBoxLayout()
        self.auto_btn = QPushButton("Auto ROI")
        self.auto_btn.clicked.connect(self._auto)
        ctrl.addWidget(self.auto_btn)

        ctrl.addWidget(QLabel("Method:"))
        self.auto_method_combo = QComboBox()
        self.auto_method_combo.addItems(["dark_blob", "hough", "center_dark"])
        ctrl.addWidget(self.auto_method_combo)

        self.manual_btn = QPushButton("Manual ROI (cv2)")
        self.manual_btn.clicked.connect(self._manual)
        ctrl.addWidget(self.manual_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # ROI options: resize checkbox + target size dropdown
        roi_opts = QHBoxLayout()
        self.resize_check = QCheckBox("Resize ROI")
        self.resize_check.setToolTip(
            "When checked, the user-drawn ROI is kept at its original size\n"
            "and resized to the target size during processing.\n"
            "When unchecked, the ROI is forced to target size pixels."
        )
        roi_opts.addWidget(self.resize_check)

        roi_opts.addWidget(QLabel("Target size:"))
        self.target_combo = QComboBox()
        self.target_combo.addItems(["64", "128", "256", "512"])
        self.target_combo.setCurrentText(str(self.pupil.input_size))
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        roi_opts.addWidget(self.target_combo)
        roi_opts.addStretch()
        layout.addLayout(roi_opts)

        self.roi_label = QLabel("ROI: not set")
        self.roi_label.setObjectName("info")
        layout.addWidget(self.roi_label)

        # contrast enhancement controls
        contrast_row = QHBoxLayout()
        self.enhance_check = QCheckBox("Enhance contrast (CLAHE)")
        self.enhance_check.toggled.connect(self._on_enhance_toggled)
        contrast_row.addWidget(self.enhance_check)

        contrast_row.addWidget(QLabel("Strength:"))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 100)  # 1.0 – 10.0 mapped
        self.contrast_slider.setValue(20)  # default 2.0
        self.contrast_slider.setFixedWidth(180)
        self.contrast_slider.setEnabled(False)
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        contrast_row.addWidget(self.contrast_slider)

        self.contrast_value_label = QLabel("2.0")
        self.contrast_value_label.setFixedWidth(40)
        contrast_row.addWidget(self.contrast_value_label)
        contrast_row.addStretch()
        layout.addLayout(contrast_row)

        # side-by-side: full frame with ROI box | cropped
        img_row = QHBoxLayout()
        self.full_viewer = ImageViewer()
        #self.full_viewer.set_max_display(1200, 1200)  # limit full frame display size
        img_row.addWidget(self.full_viewer, 2)
        self.crop_viewer = ImageViewer()
        #self.crop_viewer.set_max_display(512, 512)  # cropped ROI is always square
        img_row.addWidget(self.crop_viewer, 1)
        layout.addLayout(img_row, 1)

    @property
    def enhance_enabled(self) -> bool:
        return self._enhance_enabled

    @property
    def clip_limit(self) -> float:
        return self._clip_limit

    def apply_enhance(self, img: np.ndarray) -> np.ndarray:
        """Apply contrast enhancement if enabled. Works on gray or BGR."""
        if self._enhance_enabled:
            return enhance_contrast(img, self._clip_limit)
        return img

    def _on_enhance_toggled(self, checked: bool):
        self._enhance_enabled = checked
        self.contrast_slider.setEnabled(checked)
        self.contrast_changed.emit(self._enhance_enabled, self._clip_limit)
        self._update_display()

    def _on_contrast_changed(self, val: int):
        self._clip_limit = val / 10.0
        self.contrast_value_label.setText(f"{self._clip_limit:.1f}")
        self.contrast_changed.emit(self._enhance_enabled, self._clip_limit)
        self._update_display()

    def showEvent(self, event):
        """Show first frame when the page becomes visible."""
        super().showEvent(event)
        if self.pupil._reader is not None:
            self._update_display()

    def _update_display(self):
        try:
            frame = self.pupil.read_frame(0)
        except RuntimeError:
            return

        display_frame = self.apply_enhance(frame)
        roi = self.pupil.roi

        if roi:
            self.roi_label.setText(
                f"ROI: x={roi['x']}, y={roi['y']}, w={roi['w']}, h={roi['h']}"
            )
            display = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(
                display,
                (roi["x"], roi["y"]),
                (roi["x"] + roi["w"], roi["y"] + roi["h"]),
                (0, 255, 0),
                2,
            )
            self.full_viewer.set_image(display)
            cropped = self.pupil.crop_frame(frame)
            self.crop_viewer.set_image(self.apply_enhance(cropped))
        else:
            # no ROI yet — just show the full frame
            self.full_viewer.set_image(display_frame)

    def _on_target_changed(self, text: str):
        size = int(text)
        self.pupil.input_size = size
        self.pupil.config.input_size = size
        logger.info("Target size changed to %d", size)

    def _auto(self):
        try:
            self.pupil.auto_roi(
                resize=self.resize_check.isChecked(),
                method=self.auto_method_combo.currentText(),
            )
            self._update_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _manual(self):
        try:
            frame = self.pupil.read_frame(0)
            frame = self.apply_enhance(frame)
            self.pupil.manual_roi(
                resize=self.resize_check.isChecked(), frame=frame
            )
            self._update_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def reset(self):
        """Reset ROI page state for a new video."""
        self.pupil.roi = None
        self.pupil.config.roi = None
        self._enhance_enabled = False
        self._clip_limit = 2.0
        self.enhance_check.setChecked(False)
        self.contrast_slider.setValue(20)
        self.contrast_slider.setEnabled(False)
        self.contrast_value_label.setText("2.0")
        self.resize_check.setChecked(False)
        self.auto_method_combo.setCurrentIndex(0)
        self.target_combo.setCurrentText(str(self.pupil.input_size))
        self.roi_label.setText("ROI: not set")
        self.full_viewer.clear_image()
        self.crop_viewer.clear_image()


class SingleDetectionPage(QWidget):
    """Generic detection page for a single method (unet / intdiff / starburst)."""

    # overlay colours per method
    _COLORS = {
        "unet": (255, 100, 0),
        "intdiff": (0, 255, 0),
        "starburst": (0, 100, 255),
    }

    def __init__(self, pupil: Pupil, method: str, parent=None):
        super().__init__(parent)
        self.pupil = pupil
        self.method = method
        self._enhance_fn = None  # set by MainWindow
        self._worker = None
        self._display_indices: list[int] = []  # frame indices only, loaded on demand
        layout = QVBoxLayout(self)

        title = QLabel(f"Step 3: {method.capitalize()} Detection")
        title.setObjectName("title")
        layout.addWidget(title)

        # controls row 1: frame selection
        ctrl = QHBoxLayout()
        self.all_frames_check = QCheckBox("All frames |")
        self.all_frames_check.setChecked(True)
        self.all_frames_check.toggled.connect(self._on_all_frames_toggled)
        ctrl.addWidget(self.all_frames_check)

        ctrl.addWidget(QLabel("N frames:"))
        self.n_frames_spin = QSpinBox()
        self.n_frames_spin.setRange(5, 99999)
        self.n_frames_spin.setValue(30)
        self.n_frames_spin.setEnabled(False)
        ctrl.addWidget(self.n_frames_spin)

        self.run_btn = QPushButton(f"Run {method.capitalize()}")
        self.run_btn.clicked.connect(self._run)
        ctrl.addWidget(self.run_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # controls row 2: model path (only for unet)
        if method == "unet":
            model_row = QHBoxLayout()
            model_row.addWidget(QLabel("Model:"))
            self.model_edit = QLineEdit()
            self.model_edit.setPlaceholderText("Path to .pth model file...")
            model_row.addWidget(self.model_edit, 1)
            model_browse_btn = QPushButton("Browse")
            model_browse_btn.clicked.connect(self._browse_model)
            model_row.addWidget(model_browse_btn)
            layout.addLayout(model_row)

        self.status_label = QLabel("")
        self.status_label.setObjectName("info")
        layout.addWidget(self.status_label)

        # frame browser
        slider_row = QHBoxLayout()
        self.play_btn = QPushButton(">>")
        self.play_btn.setFixedWidth(80)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._toggle_play)
        slider_row.addWidget(self.play_btn)
        slider_row.addWidget(QLabel("Frame:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._show_result)
        slider_row.addWidget(self.slider, 1)
        self.frame_info = QLabel("")
        self.frame_info.setFixedWidth(120)
        slider_row.addWidget(self.frame_info)
        layout.addLayout(slider_row)

        # mask correction controls above the viewer (unet only)
        self._editing_mask = False
        if method == "unet":
            corr_row = QHBoxLayout()
            self.edit_mask_btn = QPushButton("Edit Mask")
            self.edit_mask_btn.setFixedWidth(200)
            self.edit_mask_btn.setEnabled(False)
            self.edit_mask_btn.clicked.connect(self._toggle_edit_mask)
            corr_row.addWidget(self.edit_mask_btn)

            corr_row.addWidget(QLabel("Brush:"))
            self.corr_brush_slider = QSlider(Qt.Horizontal)
            self.corr_brush_slider.setRange(1, 100)
            self.corr_brush_slider.setValue(20)
            self.corr_brush_slider.setFixedWidth(150)
            self.corr_brush_slider.valueChanged.connect(self._corr_brush_changed)
            corr_row.addWidget(self.corr_brush_slider)
            self.corr_brush_label = QLabel("20")
            self.corr_brush_label.setFixedWidth(40)
            corr_row.addWidget(self.corr_brush_label)

            self.apply_mask_btn = QPushButton("Apply")
            self.apply_mask_btn.setFixedWidth(150)
            self.apply_mask_btn.setEnabled(False)
            self.apply_mask_btn.clicked.connect(self._apply_corrected_mask)
            corr_row.addWidget(self.apply_mask_btn)
            corr_row.addStretch()
            layout.addLayout(corr_row)

        # viewer / canvas in a stacked widget (same position)
        self.viewer = ImageViewer()
        if method == "unet":
            self._view_stack = QStackedWidget()
            self._view_stack.addWidget(self.viewer)       # index 0 = review
            self.corr_canvas = AnnotationCanvas()
            self.corr_canvas.brush_changed.connect(self._on_corr_canvas_brush)
            self._view_stack.addWidget(self.corr_canvas)  # index 1 = edit
            layout.addWidget(self._view_stack, 1)
        else:
            layout.addWidget(self.viewer, 1)

        # play timer
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)  # ~30 fps
        self._play_timer.timeout.connect(self._play_step)
        self.setFocusPolicy(Qt.StrongFocus)

    def _on_all_frames_toggled(self, checked):
        self.n_frames_spin.setEnabled(not checked)

    def _toggle_play(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_btn.setText(">>")
        else:
            self._play_timer.start()
            self.play_btn.setText("<>")

    def _play_step(self):
        val = self.slider.value()
        if val < self.slider.maximum():
            self.slider.setValue(val + 1)
        else:
            self._play_timer.stop()
            self.play_btn.setText(">>")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and self.slider.isEnabled():
            self._toggle_play()
        else:
            super().keyPressEvent(event)

    # ── Mask correction (unet only) ──

    def _load_display_frame(self, frame_idx: int) -> np.ndarray | None:
        """Load, crop, and optionally enhance a single frame on demand."""
        try:
            frame = self.pupil.read_frame(frame_idx)
            return self.pupil.crop_frame(frame)
        except Exception:
            return None

    def _toggle_edit_mask(self):
        if not self._display_indices:
            return
        self._editing_mask = not self._editing_mask
        if self._editing_mask:
            slider_val = self.slider.value()
            idx = self._display_indices[slider_val]
            frame = self._load_display_frame(idx)
            if frame is None:
                return
            display_frame = self._enhance_fn(frame) if self._enhance_fn else frame
            mask = self.pupil.masks.get(self.method, {}).get(idx)
            if mask is None:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            self.corr_canvas.set_data(display_frame, mask.copy())
            self._view_stack.setCurrentIndex(1)
            self.edit_mask_btn.setText("Cancel")
            self.apply_mask_btn.setEnabled(True)
        else:
            self._view_stack.setCurrentIndex(0)
            self.edit_mask_btn.setText("Edit Mask")
            self.apply_mask_btn.setEnabled(False)

    def _apply_corrected_mask(self):
        if not self._display_indices or not self._editing_mask:
            return
        slider_val = self.slider.value()
        idx = self._display_indices[slider_val]
        corrected = self.corr_canvas.mask
        if corrected is not None:
            if self.method not in self.pupil.masks:
                self.pupil.masks[self.method] = {}
            self.pupil.masks[self.method][idx] = corrected.copy()
            if int(idx) not in self.pupil.config.corrected_frames:
                self.pupil.config.corrected_frames.append(int(idx))
            logger.info("Mask corrected for frame %d", idx)
        self._editing_mask = False
        self._view_stack.setCurrentIndex(0)
        self.edit_mask_btn.setText("Edit Mask")
        self.apply_mask_btn.setEnabled(False)
        self._show_result(slider_val)

    def _corr_brush_changed(self, val):
        self.corr_brush_label.setText(str(val))
        if hasattr(self, "corr_canvas"):
            self.corr_canvas.brush_radius = val

    def _on_corr_canvas_brush(self, val: int):
        if hasattr(self, "corr_brush_slider"):
            self.corr_brush_slider.blockSignals(True)
            self.corr_brush_slider.setValue(val)
            self.corr_brush_slider.blockSignals(False)
            self.corr_brush_label.setText(str(val))

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "", "PyTorch Model (*.pth);;All (*)"
        )
        if path:
            self.model_edit.setText(path)
            self.pupil.model_path = Path(path)

    def _run(self):
        # Set model path from the edit field if this is unet
        if self.method == "unet" and hasattr(self, "model_edit"):
            model_text = self.model_edit.text().strip()
            if model_text:
                self.pupil.model_path = Path(model_text)

        try:
            if self.all_frames_check.isChecked():
                indices = self.pupil.get_all_indices()
            else:
                n = self.n_frames_spin.value()
                indices = self.pupil.get_sample_indices(n)
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self.status_label.setText(f"Running {self.method} on {len(indices)} frames...")
        self.run_btn.setEnabled(False)

        self._worker = Worker(self.pupil.detect, self.method, indices)
        self._worker.finished.connect(lambda _: self._on_done(indices))
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, indices):
        self.run_btn.setEnabled(True)

        masks = self.pupil.masks.get(self.method, {})
        n_ok = sum(1 for m in masks.values() if m.sum() > 0)
        self.status_label.setText(f"{self.method}: {n_ok}/{len(masks)} detected")

        # store sorted frame indices only (frames loaded on demand)
        self._display_indices = sorted(masks.keys())

        if self._display_indices:
            self.slider.setEnabled(True)
            self.play_btn.setEnabled(True)
            if hasattr(self, "edit_mask_btn"):
                self.edit_mask_btn.setEnabled(True)
            self.slider.setRange(0, len(self._display_indices) - 1)
            self.slider.setValue(0)
            self._show_result(0)

    def _on_error(self, tb: str):
        self.run_btn.setEnabled(True)
        self.status_label.setText("Error!")
        logger.error(tb)
        QMessageBox.critical(self, "Detection Error", tb[:500])

    def _show_result(self, slider_val):
        if not self._display_indices:
            return
        idx = self._display_indices[slider_val]
        self.frame_info.setText(f"#{idx}")

        frame = self._load_display_frame(idx)
        if frame is None:
            return
        display_frame = self._enhance_fn(frame) if self._enhance_fn else frame
        mask = self.pupil.masks.get(self.method, {}).get(idx)
        color = self._COLORS.get(self.method, (0, 255, 0))
        self.viewer.set_image(overlay_mask(display_frame, mask, color))

    def reset(self):
        """Reset detection state for a new video."""
        self.pupil.masks.pop(self.method, None)
        self._display_indices.clear()
        self.slider.setEnabled(False)
        self.play_btn.setEnabled(False)
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_btn.setText(">>")
        self._editing_mask = False
        if hasattr(self, "edit_mask_btn"):
            self.edit_mask_btn.setEnabled(False)
            self.edit_mask_btn.setText("Edit Mask")
            self.apply_mask_btn.setEnabled(False)
            self._view_stack.setCurrentIndex(0)
        self.status_label.setText("")
        self.frame_info.setText("")
        self.viewer.clear_image()


class AnnotationPage(QWidget):
    """Step 5 — U-Net label: paint training masks with brush tool."""

    def __init__(self, pupil: Pupil, parent=None):
        super().__init__(parent)
        self.pupil = pupil
        self._enhance_fn = None  # set by MainWindow
        self._current = 0
        layout = QHBoxLayout(self)

        # left panel: controls
        left = QVBoxLayout()
        title = QLabel("Step 5:\nU-Net Label")
        title.setObjectName("title")
        left.addWidget(title)

        left.addWidget(QLabel("Frames to label:"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(3, 100)
        self.n_spin.setValue(10)
        left.addWidget(self.n_spin)

        self.prepare_btn = QPushButton("Prepare Frames")
        self.prepare_btn.setMinimumHeight(36)
        self.prepare_btn.clicked.connect(self._prepare)
        left.addWidget(self.prepare_btn)

        left.addSpacing(12)
        left.addWidget(QLabel("Brush size:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 100)
        self.brush_slider.setValue(20)
        self.brush_slider.valueChanged.connect(self._brush_changed)
        left.addWidget(self.brush_slider)
        self.brush_label = QLabel("20")
        left.addWidget(self.brush_label)

        left.addSpacing(12)
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.setMinimumHeight(36)
        self.prev_btn.clicked.connect(self._prev)
        nav.addWidget(self.prev_btn)
        self.next_btn = QPushButton("Next >")
        self.next_btn.setMinimumHeight(36)
        self.next_btn.clicked.connect(self._next)
        nav.addWidget(self.next_btn)
        left.addLayout(nav)

        self.frame_info = QLabel("")
        self.frame_info.setObjectName("info")
        left.addWidget(self.frame_info)

        left.addSpacing(12)
        self.save_btn = QPushButton("Save All Masks")
        self.save_btn.setMinimumHeight(36)
        self.save_btn.clicked.connect(self._save_all)
        left.addWidget(self.save_btn)

        self.save_status = QLabel("")
        self.save_status.setObjectName("status_ok")
        left.addWidget(self.save_status)
        left.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(350)
        layout.addWidget(left_widget)

        # right: annotation canvas
        self.canvas = AnnotationCanvas()
        self.canvas.brush_changed.connect(self._on_canvas_brush_changed)
        layout.addWidget(self.canvas, 1)

        # enable keyboard focus for arrow key navigation
        self.setFocusPolicy(Qt.StrongFocus)

        # per-frame mask storage
        self._masks: dict[int, np.ndarray] = {}

    def _prepare(self):
        try:
            self.pupil.prepare_label_frames(self.n_spin.value())
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        # init blank masks for every frame
        self._masks = {}
        for idx, frame in self.pupil.label_frames:
            self._masks[idx] = np.zeros(frame.shape[:2], dtype=np.uint8)

        self._current = 0
        self._show_current()

    def _show_current(self):
        """Load current frame + its mask into the canvas (no syncing here)."""
        frames = self.pupil.label_frames
        if not frames:
            return
        idx, frame = frames[self._current]
        self.frame_info.setText(f"Frame {idx}  [{self._current + 1}/{len(frames)}]")
        display_frame = self._enhance_fn(frame) if self._enhance_fn else frame
        self.canvas.set_data(display_frame, self._masks.get(idx))

    def _sync_mask_at(self, list_index: int):
        """Save the canvas mask back to our dict for the given list index."""
        frames = self.pupil.label_frames
        if not frames or self.canvas.mask is None:
            return
        idx, _ = frames[list_index]
        self._masks[idx] = self.canvas.mask.copy()

    def _prev(self):
        if self._current > 0:
            self._sync_mask_at(self._current)  # save current before leaving
            self._current -= 1
            self._show_current()

    def _next(self):
        if self._current < len(self.pupil.label_frames) - 1:
            self._sync_mask_at(self._current)  # save current before leaving
            self._current += 1
            self._show_current()

    def _brush_changed(self, val):
        self.brush_label.setText(str(val))
        self.canvas.brush_radius = val

    def _on_canvas_brush_changed(self, val: int):
        """Sync slider/label when canvas brush size changes via mouse wheel."""
        self.brush_slider.blockSignals(True)
        self.brush_slider.setValue(val)
        self.brush_slider.blockSignals(False)
        self.brush_label.setText(str(val))

    def keyPressEvent(self, event):
        """Arrow keys navigate frames."""
        if event.key() == Qt.Key_Left:
            self._prev()
        elif event.key() == Qt.Key_Right:
            self._next()
        else:
            super().keyPressEvent(event)

    def _save_all(self):
        self._sync_mask_at(self._current)
        img_dir = self.pupil.img_dir
        mask_dir = self.pupil.mask_dir
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        stem = self.pupil.video_stem

        # Build lookup: frame_idx → cropped image
        frame_lookup = {idx: frame for idx, frame in self.pupil.label_frames}

        # Only save image + mask pairs where the mask is non-empty
        labeled_stems: set[str] = set()
        count = 0
        for idx, mask in self._masks.items():
            if mask.sum() > 0:
                fname = f"{stem}_{idx:06d}.png"
                cv2.imwrite(str(mask_dir / fname), mask)
                if idx in frame_lookup:
                    cv2.imwrite(str(img_dir / fname), frame_lookup[idx])
                labeled_stems.add(fname)
                count += 1

        # Remove orphan images from this video that have no matching mask
        for img_path in img_dir.glob(f"{stem}_*.png"):
            if img_path.name not in labeled_stems:
                img_path.unlink()
                logger.debug("Removed orphan image: %s", img_path.name)

        n_unlabeled = len(self._masks) - count
        msg = f"Saved {count} image+mask pairs"
        if n_unlabeled > 0:
            msg += f"  ({n_unlabeled} unlabeled frames skipped)"
        self.save_status.setText(msg)
        logger.info("Saved %d image+mask pairs to %s", count, mask_dir)

    def reset(self):
        """Reset annotation state for a new video."""
        self.pupil.label_frames.clear()
        self._masks.clear()
        self._current = 0
        self.frame_info.setText("")
        self.save_status.setText("")
        self.canvas.set_data(
            np.zeros((128, 128), dtype=np.uint8),
            np.zeros((128, 128), dtype=np.uint8),
        )


class PostprocessPage(QWidget):
    """Step 4 — Post-processing, preview, save results."""

    def __init__(self, pupil: Pupil, parent=None):
        super().__init__(parent)
        self.pupil = pupil
        self._vlines = []  # vertical line artists for frame indicator
        layout = QVBoxLayout(self)

        title = QLabel("Step 4: Post-processing & Preview")
        title.setObjectName("title")
        layout.addWidget(title)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["unet", "intdiff", "starburst"])
        ctrl.addWidget(self.method_combo)

        self.process_btn = QPushButton("Post-process")
        self.process_btn.clicked.connect(self._process)
        ctrl.addWidget(self.process_btn)

        self.preview_btn = QPushButton("Preview (cv2)")
        self.preview_btn.clicked.connect(self._preview)
        ctrl.addWidget(self.preview_btn)

        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self._save_results)
        ctrl.addWidget(self.save_btn)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.status_label = QLabel("")
        self.status_label.setObjectName("info")
        layout.addWidget(self.status_label)

        # embedded matplotlib
        try:
            import matplotlib

            matplotlib.use("Qt5Agg")
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=(8, 4), dpi=100)
            self.figure.patch.set_facecolor("#1e1e1e")
            self.mpl_canvas = FigureCanvasQTAgg(self.figure)
            layout.addWidget(self.mpl_canvas, 1)
            self._has_mpl = True
        except Exception:
            self._has_mpl = False
            layout.addWidget(QLabel("matplotlib Qt backend not available"), 1)

    def _method(self):
        return self.method_combo.currentText()

    def _process(self):
        method = self._method()
        if method not in self.pupil.masks:
            QMessageBox.warning(self, "Warning", f"No masks for '{method}'. Run detection first.")
            return
        try:
            raw, smoothed = self.pupil.postprocess(method)
            n_valid = np.sum(~np.isnan(smoothed[:, 2]))
            self.status_label.setText(
                f"{method}: {smoothed.shape[0]} frames, {int(n_valid)} valid"
            )
            if self._has_mpl:
                self._draw_plot(smoothed)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _draw_plot(self, data: np.ndarray):
        self.figure.clear()
        self._axes = self.figure.subplots(3, 1, sharex=True)
        frames = data[:, 0]
        for ax in self._axes:
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#888")
            ax.spines["bottom"].set_color("#444")
            ax.spines["left"].set_color("#444")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.label.set_color("#ccc")
        self._axes[0].plot(frames, data[:, 4], color="#4ec9b0", linewidth=0.8)
        self._axes[0].set_ylabel("Area")
        self._axes[1].plot(frames, data[:, 2], color="#569cd6", linewidth=0.8)
        self._axes[1].set_ylabel("Center X")
        self._axes[2].plot(frames, data[:, 3], color="#ce9178", linewidth=0.8)
        self._axes[2].set_ylabel("Center Y")
        self._axes[2].set_xlabel("Frame")
        self._axes[2].xaxis.label.set_color("#ccc")
        self._vlines = []
        self.figure.tight_layout()
        self.mpl_canvas.draw()

    def _update_frame_indicator(self, frame_idx: int):
        """Move vertical line on the time course plot to indicate current frame."""
        if not self._has_mpl or not hasattr(self, "_axes"):
            return
        # Remove old lines
        for line in self._vlines:
            line.remove()
        self._vlines = []
        # Draw new lines
        for ax in self._axes:
            vl = ax.axvline(x=frame_idx, color="#ffff00", linewidth=0.8, alpha=0.7)
            self._vlines.append(vl)
        self.mpl_canvas.draw_idle()

    def _save_results(self):
        method = self._method()
        if method not in self.pupil.results:
            QMessageBox.warning(self, "Warning", "Run post-processing first.")
            return
        try:
            out_dir = self.pupil.save_results(method)
            self.status_label.setText(f"Results saved to {out_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _preview(self):
        method = self._method()
        if method not in self.pupil.results:
            QMessageBox.warning(self, "Warning", "Run post-processing first.")
            return
        from .visualize import preview

        _, smoothed = self.pupil.results[method]
        masks = self.pupil.masks.get(method)

        # Run preview in a polling loop so we can update the frame indicator
        from .video_io import VideoReader
        reader = VideoReader(self.pupil.reader.path)
        current = 0
        paused = False
        roi = self.pupil.roi
        input_size = self.pupil.input_size

        preview_size = 256
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview", preview_size, preview_size)

        while current < reader.n_frames:
            frame = reader.read_frame(current)
            if frame is None:
                break
            if roi is not None:
                x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
                frame = frame[y : y + h, x : x + w]
            frame = cv2.resize(frame, (input_size, input_size))
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Fitted ellipse overlay
            if masks is not None and current in masks:
                from .visualize import _draw_ellipse_overlay
                _draw_ellipse_overlay(display, masks[current])

            # Centroid
            if current < len(smoothed) and not np.isnan(smoothed[current, 2]):
                cx = int(round(smoothed[current, 2]))
                cy = int(round(smoothed[current, 3]))
                cv2.circle(display, (cx, cy), 2, (0, 255, 0), -1)

            cv2.putText(display, f"F{current}", (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            cv2.imshow("Preview", display)

            # Update frame indicator on the embedded plot
            self._update_frame_indicator(current)

            wait_time = 1 if paused else 30
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == 81 or key == 2:
                current = max(current - 1, 0)
                continue
            elif key == 83 or key == 3:
                current = min(current + 1, reader.n_frames - 1)
                continue

            # Check if window was closed via X button (after waitKey processes events)
            try:
                if cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if not paused:
                current += 1

        cv2.destroyAllWindows()
        reader.close()

    def reset(self):
        """Reset postprocess state for a new video."""
        self.pupil.results.clear()
        self.status_label.setText("")
        self._vlines = []
        if self._has_mpl:
            self.figure.clear()
            self.mpl_canvas.draw()


class TrainingPage(QWidget):
    """Step 6 — U-Net training."""

    def __init__(self, pupil: Pupil, parent=None):
        super().__init__(parent)
        self.pupil = pupil
        self._worker = None
        layout = QVBoxLayout(self)

        title = QLabel("Step 6: U-Net Training")
        title.setObjectName("title")
        layout.addWidget(title)

        # --- Training data selection ---
        data_grp = QGroupBox("Training Data")
        data_layout = QVBoxLayout(data_grp)

        data_layout.addWidget(QLabel(
            "Each folder must contain images/ and masks/ subdirectories."
        ))

        self.data_list = QListWidget()
        self.data_list.setMinimumHeight(80)
        self.data_list.setMaximumHeight(150)
        data_layout.addWidget(self.data_list)

        self.data_info_label = QLabel("")
        self.data_info_label.setObjectName("info")
        data_layout.addWidget(self.data_info_label)

        data_btn_row = QHBoxLayout()
        add_current_btn = QPushButton("Add Current Output")
        add_current_btn.setToolTip("Add the current video's output directory")
        add_current_btn.clicked.connect(self._add_current_dir)
        data_btn_row.addWidget(add_current_btn)

        add_btn = QPushButton("Add Folder...")
        add_btn.clicked.connect(self._add_folder)
        data_btn_row.addWidget(add_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_folder)
        data_btn_row.addWidget(remove_btn)
        data_btn_row.addStretch()

        data_layout.addLayout(data_btn_row)
        layout.addWidget(data_grp)

        # --- Training parameters ---
        form = QFormLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        form.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(4)
        form.addRow("Batch size:", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        form.addRow("Learning rate:", self.lr_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(10)
        form.addRow("Patience:", self.patience_spin)

        self.amp_check = QCheckBox("Mixed precision (AMP)")
        form.addRow(self.amp_check)

        self.freeze_check = QCheckBox("Freeze encoder")
        form.addRow(self.freeze_check)

        self.snapshot_spin = QSpinBox()
        self.snapshot_spin.setRange(0, 50)
        self.snapshot_spin.setValue(3)
        self.snapshot_spin.setToolTip("Number of previous best_model snapshots to keep")
        form.addRow("Snapshots to keep:", self.snapshot_spin)

        # Checkpoint save directory
        ckpt_row = QHBoxLayout()
        self.ckpt_edit = QLineEdit()
        self.ckpt_edit.setPlaceholderText("(default: same folder as training data)")
        ckpt_row.addWidget(self.ckpt_edit)
        ckpt_browse = QPushButton("Browse...")
        ckpt_browse.clicked.connect(self._browse_checkpoint_dir)
        ckpt_row.addWidget(ckpt_browse)
        form.addRow("Save model to:", ckpt_row)

        param_grp = QGroupBox("Training Parameters")
        param_grp.setLayout(form)
        layout.addWidget(param_grp)

        # --- Action buttons + status ---
        btn_row = QHBoxLayout()
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setMinimumHeight(36)
        self.train_btn.clicked.connect(self._start)
        btn_row.addWidget(self.train_btn)

        self.test_btn = QPushButton("Test U-Net Inference")
        self.test_btn.setMinimumHeight(36)
        self.test_btn.setEnabled(False)
        self.test_btn.clicked.connect(self._test_infer)
        btn_row.addWidget(self.test_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.status_label = QLabel("")
        self.status_label.setObjectName("info")
        layout.addWidget(self.status_label)

        # --- Result viewer (takes remaining space) ---
        self.viewer = ImageViewer()
        layout.addWidget(self.viewer, 1)

    # ── Data folder management ──

    def _get_data_dirs(self) -> list[str]:
        """Return all folder paths currently in the list."""
        return [self.data_list.item(i).text()
                for i in range(self.data_list.count())]

    def _validate_data_dir(self, folder: str) -> tuple[int, int]:
        """Count matched image+mask pairs in a folder."""
        from pathlib import Path
        img_dir = Path(folder) / "images"
        mask_dir = Path(folder) / "masks"
        if not img_dir.exists() or not mask_dir.exists():
            return 0, 0
        img_stems = {p.stem for p in img_dir.glob("*.png")}
        mask_stems = {p.stem for p in mask_dir.glob("*.png")}
        matched = img_stems & mask_stems
        return len(matched), len(img_stems)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select data folder")
        if not folder:
            return
        self._add_dir(folder)

    def _add_current_dir(self):
        self._add_dir(str(self.pupil.output_dir))

    def _add_dir(self, folder: str):
        # Avoid duplicates
        if folder in self._get_data_dirs():
            return
        n_matched, n_img = self._validate_data_dir(folder)
        if n_matched == 0:
            QMessageBox.warning(
                self, "Warning",
                f"No matched image+mask pairs found in:\n{folder}\n\n"
                "Expected images/ and masks/ subdirectories with matching .png files.",
            )
            return
        from pathlib import Path
        label = folder
        self.data_list.addItem(label)
        self._refresh_data_info()

    def _browse_checkpoint_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select checkpoint save directory")
        if folder:
            self.ckpt_edit.setText(folder)

    def _remove_folder(self):
        row = self.data_list.currentRow()
        if row >= 0:
            self.data_list.takeItem(row)
            self._refresh_data_info()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_data_info()

    def _refresh_data_info(self):
        from pathlib import Path
        dirs = self._get_data_dirs()
        total_matched = 0
        for d in dirs:
            n_matched, _ = self._validate_data_dir(d)
            total_matched += n_matched
        if dirs:
            ckpt_dir = self.ckpt_edit.text().strip()
            if not ckpt_dir:
                ckpt_dir = str(Path(dirs[0]) / "checkpoints")
            self.data_info_label.setText(
                f"{len(dirs)} folder(s)  |  {total_matched} matched pairs  |  "
                f"Checkpoints: {ckpt_dir}"
            )
        else:
            self.data_info_label.setText("No training data folders selected.")

    def _start(self):
        self._refresh_data_info()
        dirs = self._get_data_dirs()
        if not dirs:
            QMessageBox.warning(
                self, "Warning",
                "No training data folders selected.\n"
                "Add folders containing images/ and masks/ subdirectories.",
            )
            return

        # Collect image/mask directory pairs
        from pathlib import Path
        data_dirs = []
        for d in dirs:
            n_matched, _ = self._validate_data_dir(d)
            if n_matched > 0:
                data_dirs.append((str(Path(d) / "images"), str(Path(d) / "masks")))
        if not data_dirs:
            QMessageBox.warning(self, "Warning", "No matched image+mask pairs in selected folders.")
            return

        self.train_btn.setEnabled(False)
        self.status_label.setText("Training...")

        ckpt_dir = self.ckpt_edit.text().strip() or None
        self._worker = Worker(
            self.pupil.train,
            data_dirs=data_dirs,
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            lr=self.lr_spin.value(),
            patience=self.patience_spin.value(),
            amp=self.amp_check.isChecked(),
            freeze_encoder=self.freeze_check.isChecked(),
            max_snapshots=self.snapshot_spin.value(),
            checkpoint_dir=ckpt_dir,
        )
        self._worker.finished.connect(self._on_train_done)
        self._worker.error.connect(self._on_train_error)
        self._worker.start()

    def _on_train_done(self, best_path):
        self.train_btn.setEnabled(True)
        self.test_btn.setEnabled(True)
        self.status_label.setText(f"Training complete. Model: {best_path}")
        self._refresh_data_info()

    def _on_train_error(self, tb):
        self.train_btn.setEnabled(True)
        self.status_label.setText("Training failed!")
        logger.error(tb)
        QMessageBox.critical(self, "Training Error", tb[:500])

    def _test_infer(self):
        try:
            indices = self.pupil.get_sample_indices(10)
            masks = self.pupil.detect("unet", indices)
            n_ok = sum(1 for m in masks.values() if m.sum() > 0)
            self.status_label.setText(f"U-Net: {n_ok}/{len(masks)} detected")
            # show first detection
            for idx in sorted(masks.keys()):
                frame = self.pupil.read_frame(idx)
                cropped = self.pupil.crop_frame(frame)
                display = overlay_mask(cropped, masks[idx], (255, 100, 0))
                self.viewer.set_image(display)
                break
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════════


class MainWindow(QMainWindow):
    def __init__(self, video_path: str | None = None, output_dir: str = "./output"):
        super().__init__()
        self.setWindowTitle("Pupil Track GUI")
        self.resize(1800, 1500)

        self.pupil = Pupil(output_dir=output_dir)

        # central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        # top: step list + page stack
        top = QSplitter(Qt.Horizontal)
        splitter.addWidget(top)

        # step list
        self.step_list = QListWidget()
        self.step_list.setFixedWidth(300)
        steps = [
            "1. Video I/O",
            "2. ROI Selection",
            "3. U-Net Detection",
            "(Intdiff)",
            "(Starburst)",
            "4. Post-processing",
            "5. U-Net Label",
            "6. U-Net Training",
        ]
        for s in steps:
            self.step_list.addItem(s)
        self.step_list.currentRowChanged.connect(self._switch_page)
        top.addWidget(self.step_list)

        # page stack (order matches step list)
        self.stack = QStackedWidget()
        self.video_page = VideoPage(self.pupil)
        self.roi_page = ROIPage(self.pupil)
        self.unet_page = SingleDetectionPage(self.pupil, "unet")
        self.intdiff_page = SingleDetectionPage(self.pupil, "intdiff")
        self.starburst_page = SingleDetectionPage(self.pupil, "starburst")
        self.postprocess_page = PostprocessPage(self.pupil)
        self.annotation_page = AnnotationPage(self.pupil)
        self.training_page = TrainingPage(self.pupil)
        self.stack.addWidget(self.video_page)       # 0
        self.stack.addWidget(self.roi_page)          # 1
        self.stack.addWidget(self.unet_page)         # 2
        self.stack.addWidget(self.intdiff_page)      # 3
        self.stack.addWidget(self.starburst_page)    # 4
        self.stack.addWidget(self.postprocess_page)  # 5
        self.stack.addWidget(self.annotation_page)   # 6
        self.stack.addWidget(self.training_page)     # 7
        top.addWidget(self.stack)
        top.setStretchFactor(1, 1)

        # wire contrast enhancement from ROIPage to all pages
        self.video_page._enhance_fn = self.roi_page.apply_enhance
        self.unet_page._enhance_fn = self.roi_page.apply_enhance
        self.intdiff_page._enhance_fn = self.roi_page.apply_enhance
        self.starburst_page._enhance_fn = self.roi_page.apply_enhance
        self.annotation_page._enhance_fn = self.roi_page.apply_enhance
        self.roi_page.contrast_changed.connect(self._on_contrast_changed)

        # wire video_loaded signal → enable steps + reset state
        self.video_page.video_loaded.connect(self._on_video_loaded)

        # bottom: log
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(140)
        splitter.addWidget(self.log_widget)
        splitter.setStretchFactor(0, 1)

        # wire up logging
        handler = QTextEditLogHandler(self.log_widget)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%H:%M:%S"))
        logging.getLogger("pupil_track").addHandler(handler)
        logging.getLogger("pupil_track").setLevel(logging.INFO)

        # status bar
        self.statusBar().showMessage("Ready")

        # disable steps 2–5 (indices 1–6) until a video is loaded
        for i in range(1, 7):
            item = self.step_list.item(i)
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)

        # pre-fill video path if provided
        if video_path:
            self.video_page.path_edit.setText(video_path)

        self.step_list.setCurrentRow(0)

    def _switch_page(self, row):
        self.stack.setCurrentIndex(row)

    def _on_video_loaded(self):
        """Enable steps 2–5 and reset/restore downstream state."""
        # enable step list items (indices 1–6)
        for i in range(1, 7):
            item = self.step_list.item(i)
            item.setFlags(item.flags() | Qt.ItemIsEnabled)

        # reset all downstream pages (clears UI + pupil state)
        self.roi_page.reset()
        self.unet_page.reset()
        self.intdiff_page.reset()
        self.starburst_page.reset()
        self.postprocess_page.reset()
        self.annotation_page.reset()

        # clear log console
        self.log_widget.clear()

        # Restore UI from config loaded in VideoPage._load()
        # (load_config sets pupil.roi, pupil.masks, pupil.model_path
        #  but reset() cleared them — re-apply from config)
        self._restore_from_config()

        self.statusBar().showMessage("Video loaded — ready")

    def _restore_from_config(self):
        """Re-load config for the current video and restore UI state.

        Called after reset() has cleared all pages.  If no config file
        exists for this video, nothing happens and the UI stays blank.
        """
        config_path = self.pupil.output_dir / f"{self.pupil.video_stem}_config.json"
        if not config_path.exists():
            return

        try:
            self.pupil.load_config(config_path)
        except Exception as e:
            logger.warning("Failed to load config: %s", e)
            return

        cfg = self.pupil.config

        # Restore ROI
        if self.pupil.roi is not None:
            self.roi_page.roi_label.setText(
                f"ROI: x={self.pupil.roi['x']}, y={self.pupil.roi['y']}, "
                f"w={self.pupil.roi['w']}, h={self.pupil.roi['h']}"
            )
            self.roi_page.target_combo.setCurrentText(str(self.pupil.input_size))
            logger.info("Restored ROI from config")

        # Restore model path
        if self.pupil.model_path and hasattr(self.unet_page, "model_edit"):
            self.unet_page.model_edit.setText(str(self.pupil.model_path))

        # Restore masks from .npy → enable slider, buttons, show first frame
        method = cfg.method
        if method in self.pupil.masks and self.pupil.masks[method]:
            page = {"unet": self.unet_page, "intdiff": self.intdiff_page,
                    "starburst": self.starburst_page}.get(method)
            if page is not None:
                masks = self.pupil.masks[method]
                page._display_indices = sorted(masks.keys())
                page.slider.setEnabled(True)
                page.slider.setRange(0, len(page._display_indices) - 1)
                page.slider.setValue(0)
                page.play_btn.setEnabled(True)
                if hasattr(page, "edit_mask_btn"):
                    page.edit_mask_btn.setEnabled(True)
                page._show_result(0)
                n_ok = sum(1 for m in masks.values() if m.sum() > 0)
                page.status_label.setText(
                    f"{method}: {n_ok}/{len(masks)} detected (loaded)")
                logger.info("Restored %d detection masks for %s", len(masks), method)

    def _on_contrast_changed(self, enabled: bool, clip_limit: float):
        """Refresh the currently visible page when contrast settings change."""
        current = self.stack.currentWidget()
        # re-display VideoPage if visible
        if current is self.video_page and self.video_page.slider.isEnabled():
            self.video_page._show_frame(self.video_page.slider.value())
        # re-display any detection page if visible
        elif isinstance(current, SingleDetectionPage) and current.slider.isEnabled():
            current._show_result(current.slider.value())
        # re-display AnnotationPage if visible
        elif current is self.annotation_page and self.annotation_page.pupil.label_frames:
            self.annotation_page._show_current()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def launch_gui(video_path: str | None = None, output_dir: str = "./output"):
    """Launch the PupilTrack GUI application."""
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow(video_path=video_path, output_dir=output_dir)
    window.show()
    sys.exit(app.exec_())
