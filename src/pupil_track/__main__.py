"""CLI entry point: python -m pupil_track <command> [options]

Commands:
    roi       - Select region of interest (auto or manual)
    annotate  - Launch annotation GUI for labeling masks
    train     - Train shallow U-Net on labeled data
    infer     - Run pupil detection (unet/intdiff/starburst)
    preview   - Interactive playback with overlays
    export    - Export annotated video and/or time-series plot
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np


def cmd_roi(args):
    """ROI selection command."""
    from .config import Config
    from .roi import select_roi
    from .video_io import VideoReader

    config = Config.load(args.config) if args.config and Path(args.config).exists() else Config()

    with VideoReader(args.video) as reader:
        frame = reader.read_frame(args.frame or 0)

    roi = select_roi(frame, mode="auto" if args.auto else "manual", target_size=config.input_size)
    config.roi = roi
    config.save(args.config or "config.json")
    print(f"ROI saved: {roi}")


def cmd_annotate(args):
    """Annotation command."""
    import cv2

    from .config import Config
    from .annotate import annotate_frames
    from .video_io import VideoReader

    config = Config.load(args.config) if args.config else Config()

    with VideoReader(args.video) as reader:
        indices = reader.sample_indices(args.n_frames, mode="uniform")
        frames = []
        for idx, frame in reader.frames(indices):
            if config.roi:
                x, y, w, h = config.roi["x"], config.roi["y"], config.roi["w"], config.roi["h"]
                frame = frame[y : y + h, x : x + w]
            frame = cv2.resize(frame, (config.input_size, config.input_size))
            frames.append((idx, frame))

    annotate_frames(frames, args.output_dir)


def cmd_train(args):
    """Training command."""
    from .train import run_training
    run_training(args.train_args)


def cmd_infer(args):
    """Inference command."""
    from .config import Config
    from .infer import detect
    from .postprocess import postprocess, export_csv, export_npy
    from .video_io import VideoReader

    config = Config.load(args.config) if args.config else Config()
    config.method = args.method or config.method
    if args.model:
        config.model_path = args.model

    with VideoReader(args.video) as reader:
        fps = reader.fps

    masks = detect(args.video, config)
    raw, smoothed = postprocess(masks, fps=fps, smooth_window=config.smooth_window)

    output = Path(args.output or "results.csv")
    if output.suffix == ".npy":
        export_npy(smoothed, output)
    else:
        export_csv(smoothed, output)

    # Also save raw results
    raw_path = output.with_stem(output.stem + "_raw")
    if raw_path.suffix == ".npy":
        export_npy(raw, raw_path)
    else:
        export_csv(raw, raw_path)


def cmd_preview(args):
    """Preview command."""
    import csv

    from .config import Config
    from .visualize import preview

    config = Config.load(args.config) if args.config else Config()

    # Load results
    results_path = Path(args.results)
    if results_path.suffix == ".npy":
        results = np.load(str(results_path))
    else:
        with open(results_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = []
            for row in reader:
                rows.append([float(v) if v else np.nan for v in row])
            results = np.array(rows)

    preview(
        args.video,
        results,
        roi=config.roi,
        input_size=config.input_size,
        start_frame=args.start or 0,
    )


def cmd_export(args):
    """Export command."""
    import csv

    from .config import Config
    from .visualize import export_video, plot_timeseries

    config = Config.load(args.config) if args.config else Config()

    # Load results
    results_path = Path(args.results)
    if results_path.suffix == ".npy":
        results = np.load(str(results_path))
    else:
        with open(results_path) as f:
            reader = csv.reader(f)
            next(reader)
            rows = []
            for row in reader:
                rows.append([float(v) if v else np.nan for v in row])
            results = np.array(rows)

    if args.video_out:
        export_video(
            args.video,
            args.video_out,
            results,
            roi=config.roi,
            input_size=config.input_size,
            fps=config.export_fps,
        )

    if args.plot_out:
        plot_timeseries(results, args.plot_out)


def main():
    parser = argparse.ArgumentParser(
        prog="pupil-track",
        description="Standalone Python pipeline for pupil tracking",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- roi ---
    p_roi = subparsers.add_parser("roi", help="Select region of interest")
    p_roi.add_argument("video", help="Video file path")
    p_roi.add_argument("--auto", action="store_true", help="Auto-detect ROI")
    p_roi.add_argument("--config", "-c", default="config.json", help="Config file path")
    p_roi.add_argument("--frame", type=int, default=0, help="Frame to use for ROI selection")

    # --- annotate ---
    p_ann = subparsers.add_parser("annotate", help="Launch annotation GUI")
    p_ann.add_argument("video", help="Video file path")
    p_ann.add_argument("--config", "-c", default="config.json", help="Config file path")
    p_ann.add_argument("--n-frames", type=int, default=20, help="Number of frames to label")
    p_ann.add_argument("--output-dir", "-o", default="./training_data/masks", help="Mask output dir")

    # --- train ---
    p_train = subparsers.add_parser("train", help="Train shallow U-Net")
    p_train.add_argument("train_args", nargs="*", help="Arguments passed to training (use -- prefix)")

    # --- infer ---
    p_infer = subparsers.add_parser("infer", help="Run pupil detection")
    p_infer.add_argument("video", help="Video file path")
    p_infer.add_argument("--config", "-c", default="config.json", help="Config file path")
    p_infer.add_argument("--method", "-m", choices=["unet", "intdiff", "starburst"], help="Detection method")
    p_infer.add_argument("--model", help="Model .pth path (for unet)")
    p_infer.add_argument("--output", "-o", default="results.csv", help="Output file (.csv or .npy)")

    # --- preview ---
    p_prev = subparsers.add_parser("preview", help="Interactive playback")
    p_prev.add_argument("video", help="Video file path")
    p_prev.add_argument("--results", "-r", required=True, help="Results file (.csv or .npy)")
    p_prev.add_argument("--config", "-c", default="config.json", help="Config file path")
    p_prev.add_argument("--start", type=int, default=0, help="Start frame")

    # --- export ---
    p_exp = subparsers.add_parser("export", help="Export annotated video / plots")
    p_exp.add_argument("video", help="Video file path")
    p_exp.add_argument("--results", "-r", required=True, help="Results file (.csv or .npy)")
    p_exp.add_argument("--config", "-c", default="config.json", help="Config file path")
    p_exp.add_argument("--video-out", help="Output annotated video path")
    p_exp.add_argument("--plot-out", help="Output time-series plot path")

    # --- gui ---
    p_gui = subparsers.add_parser("gui", help="Launch the PyQt5 GUI")
    p_gui.add_argument("--video", "-v", default=None, help="Pre-fill video path")
    p_gui.add_argument("--output", "-o", default="./output", help="Output directory")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "gui":
        from .gui import launch_gui
        launch_gui(video_path=args.video, output_dir=args.output)
        return

    commands = {
        "roi": cmd_roi,
        "annotate": cmd_annotate,
        "train": cmd_train,
        "infer": cmd_infer,
        "preview": cmd_preview,
        "export": cmd_export,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
