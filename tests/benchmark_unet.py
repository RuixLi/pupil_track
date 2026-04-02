"""
Benchmark script for U-Net pupil detection pipeline.

Profiles each phase of the detection pipeline to identify bottlenecks:
  Phase 1: Frame loading (video decode + ROI crop + resize)
  Phase 2: Preprocessing (normalize + stack into tensor)
  Phase 3: Model forward pass (U-Net inference)
  Phase 4: Postprocessing (threshold + resize masks)

Uses the same data paths as script_pupil_track.py.
Run:  python benchmark_unet.py
"""

import time
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PARAMETERS — copy from script_pupil_track.py                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

VIDEO_PATH = r"D:\OneDrive - 同志社大学\Data\pupil_video\pupil_video250123_153514_WT5_5rpm_d1t2.mp4"
OUTPUT_DIR = r"D:\OneDrive - 同志社大学\Data\pupil_video"
MODEL_PATH = r"D:\OneDrive - 同志社大学\Code\pupilTrack\pupil_track\output\checkpoints\best_epoch039_dice_0p8586.pth"
INPUT_SIZE = 128
THRESHOLD = 0.5
BATCH_SIZE = 32

# Number of frames to benchmark (None = all frames)
N_FRAMES = None  # e.g. 1000 for a quick test

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Setup                                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

from pupil_track import Pupil
from pupil_track.video_io import VideoReader
from pupil_track.detectors.unet_detect import load_model, normalize_frame

p = Pupil(
    video_path=VIDEO_PATH,
    output_dir=OUTPUT_DIR or str(Path(VIDEO_PATH).parent),
    input_size=INPUT_SIZE,
)
p.load_config()

if p.roi is None:
    p.manual_roi()

if MODEL_PATH:
    p.model_path = Path(MODEL_PATH)

n_total = p.reader.n_frames
n_frames = min(N_FRAMES or n_total, n_total)
indices = np.arange(n_frames)

print(f"\n{'='*60}")
print(f"Benchmark: {n_frames} frames from {Path(VIDEO_PATH).name}")
print(f"ROI: {p.roi}")
print(f"Input size: {INPUT_SIZE}x{INPUT_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"{'='*60}\n")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Phase 1: Frame loading (video decode + crop + resize)             ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("Phase 1: Frame loading (decode + crop + resize)...")
t0 = time.perf_counter()

frames = []
for idx in indices:
    frame = p.reader.read_frame(idx)
    if frame is not None:
        cropped = p.crop_frame(frame)
        frames.append((idx, cropped))

t_load = time.perf_counter() - t0
fps_load = len(frames) / t_load
print(f"  {len(frames)} frames in {t_load:.2f}s  ({fps_load:.0f} fps)")

# Break down: raw decode vs crop+resize
print("  Breakdown:")
t0 = time.perf_counter()
for idx in indices[:500]:
    p.reader.read_frame(idx)
t_decode = time.perf_counter() - t0

t0 = time.perf_counter()
sample_frame = frames[0][1]
for _ in range(500):
    cv2.resize(sample_frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
t_resize = time.perf_counter() - t0

print(f"    decode: {t_decode/500*1000:.2f} ms/frame")
print(f"    resize: {t_resize/500*1000:.2f} ms/frame")

# Sequential vs seek read comparison
print("  Sequential read (no seek) vs random seek:")
reader2 = VideoReader(VIDEO_PATH)
n_seq = min(500, n_frames)

t0 = time.perf_counter()
for idx in range(n_seq):
    reader2.read_frame(idx)
t_seq = time.perf_counter() - t0

t0 = time.perf_counter()
random_idx = np.sort(np.random.choice(n_total, n_seq, replace=False))
for idx in random_idx:
    reader2.read_frame(idx)
t_rand = time.perf_counter() - t0
reader2.close()

print(f"    sequential: {t_seq/n_seq*1000:.2f} ms/frame")
print(f"    random seek: {t_rand/n_seq*1000:.2f} ms/frame")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Phase 2: Preprocessing (normalize + stack tensor)                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\nPhase 2: Preprocessing (normalize + build tensor)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

frame_data = [f for _, f in frames]

t0 = time.perf_counter()
batches = []
for i in range(0, len(frame_data), BATCH_SIZE):
    batch_frames = frame_data[i : i + BATCH_SIZE]
    batch = []
    for f in batch_frames:
        normed = normalize_frame(f)
        batch.append(normed[np.newaxis, ...])
    tensor = torch.from_numpy(np.stack(batch, axis=0)).to(device=device, dtype=torch.float32)
    batches.append(tensor)

t_preprocess = time.perf_counter() - t0
print(f"  {len(frame_data)} frames in {t_preprocess:.2f}s  ({len(frame_data)/t_preprocess:.0f} fps)")
print(f"  {len(batches)} batches of size {BATCH_SIZE}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Phase 3: Model forward pass                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\nPhase 3: Model forward pass...")
model = load_model(str(p.model_path), device)

# Warmup (first run is slower due to memory allocation, kernel compilation)
with torch.inference_mode():
    _ = model(batches[0])
if device.type == "cuda":
    torch.cuda.synchronize()

t0 = time.perf_counter()
logits_list = []
with torch.inference_mode():
    for tensor in batches:
        logits = model(tensor)
        logits_list.append(logits)
if device.type == "cuda":
    torch.cuda.synchronize()

t_forward = time.perf_counter() - t0
print(f"  {len(frame_data)} frames in {t_forward:.2f}s  ({len(frame_data)/t_forward:.0f} fps)")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Phase 4: Postprocessing (sigmoid + threshold + to numpy)          ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\nPhase 4: Postprocessing (sigmoid + threshold + to numpy)...")
t0 = time.perf_counter()

masks = []
for logits in logits_list:
    probs = torch.sigmoid(logits.squeeze(1))
    batch_masks = (probs > THRESHOLD).cpu().numpy().astype(np.uint8)
    for m in batch_masks:
        masks.append(m)

t_post = time.perf_counter() - t0
print(f"  {len(masks)} masks in {t_post:.2f}s  ({len(masks)/t_post:.0f} fps)")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Summary                                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

t_total = t_load + t_preprocess + t_forward + t_post
print(f"\n{'='*60}")
print(f"SUMMARY ({n_frames} frames)")
print(f"{'='*60}")
print(f"  Phase 1 — Frame loading:    {t_load:7.2f}s  ({t_load/t_total*100:5.1f}%)")
print(f"  Phase 2 — Preprocessing:    {t_preprocess:7.2f}s  ({t_preprocess/t_total*100:5.1f}%)")
print(f"  Phase 3 — Model forward:    {t_forward:7.2f}s  ({t_forward/t_total*100:5.1f}%)")
print(f"  Phase 4 — Postprocessing:   {t_post:7.2f}s  ({t_post/t_total*100:5.1f}%)")
print(f"  {'─'*40}")
print(f"  Total:                      {t_total:7.2f}s")
print(f"  Overall throughput:         {n_frames/t_total:7.0f} fps")
print(f"{'='*60}")

# Memory info
if device.type == "cuda":
    print(f"\nGPU memory: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB peak")
print(f"Batch tensor shape: {batches[0].shape}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

p.close()
