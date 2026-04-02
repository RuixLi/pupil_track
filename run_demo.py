"""Launch the PupilTrack GUI.

Usage:
    cd pupil_track
    pip install -e ".[gui]"
    python run_demo.py
    python run_demo.py --video path/to/video.avi --output ./my_output
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="PupilTrack Pipeline GUI")
    parser.add_argument("--video", "-v", default=None, help="Pre-fill video path")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    args = parser.parse_args()

    try:
        from pupil_track.gui import launch_gui
    except ImportError as e:
        print(f"Error: {e}")
        print()
        print("Install GUI dependencies with:")
        print('    pip install -e ".[gui]"')
        sys.exit(1)

    launch_gui(video_path=args.video, output_dir=args.output)


if __name__ == "__main__":
    main()
