"""Standalone frame timestamps — get every frame's PTS without decoding."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import vex

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures", "formats")

# --- Pick a few files across different container families --------------------

samples = [
    ("h264_mp4.mp4", "MP4 (sample table)"),
    ("h264_mkv.mkv", "MKV (block timestamps)"),
    ("h264_ts.ts", "MPEG-TS (PES timestamps)"),
    ("h264_flv.flv", "FLV (tag timestamps)"),
    ("h264_avi.avi", "AVI (fixed rate)"),
    ("h264_nut.nut", "NUT (generic PTS)"),
]

for filename, label in samples:
    path = os.path.join(FIXTURES, filename)
    if not os.path.isfile(path):
        continue

    ft = vex.get_frame_times(path)

    print(f"\n{label}  ({filename})")
    print(f"  strategy:  {ft.strategy}")
    print(f"  container: {ft.container}")
    print(f"  codec:     {ft.codec}")
    print(f"  frames:    {ft.frame_count}")
    print(f"  fps:       {ft.fps:.2f}")
    print(f"  duration:  {ft.duration_sec:.3f}s")
    print(f"  first PTS: {ft.times[0]:.4f}s")
    print(f"  last PTS:  {ft.times[-1]:.4f}s")

    if ft.frame_count > 1:
        intervals = np.diff(ft.times)
        print(f"  avg interval:    {intervals.mean():.4f}s")
        print(f"  max interval:    {intervals.max():.4f}s")
        print(f"  jitter (stddev): {intervals.std():.6f}s")

# --- Cross-validate against actual decode ------------------------------------

print("\n--- Cross-validation: packet scan vs full decode ---")

path = os.path.join(FIXTURES, "h264_mp4.mp4")
ft = vex.get_frame_times(path)
result = vex.batch_decode([path], keyframes_only=False)

print(f"  Packet scan frame count: {ft.frame_count}")
print(f"  Decoded frame count:     {result.metrics.keyframes_decoded}")
print(f"  Match: {ft.frame_count == result.metrics.keyframes_decoded}")
