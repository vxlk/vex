"""Every-frame decode with frame_skip — extract thumbnails for all frames."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from vex import batch_decode, LevelConfig

VIDEO = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'formats', 'h264_mp4.mp4')

# --- Every frame (default) ---------------------------------------------------

LEVEL = [LevelConfig(width=192, height=192, quality=85)]

result = batch_decode(
    paths=[VIDEO],
    levels=LEVEL,
)

stream = result.jpeg_stream()
n_all = len(stream.offsets[0])
print(f"All frames:  {n_all} frames in {result.metrics.total_wall_us / 1000:.1f} ms")
print(f"  Pipeline: {result.metrics.pipeline_fps:.0f} fps")

# --- Every 5th frame ----------------------------------------------------------

result_skip = batch_decode(
    paths=[VIDEO],
    levels=LEVEL,
    frame_skip=5,
)

stream_skip = result_skip.jpeg_stream()
n_skip = len(stream_skip.offsets[0])
print(f"\nframe_skip=5: {n_skip} frames in {result_skip.metrics.total_wall_us / 1000:.1f} ms")
print(f"  Pipeline: {result_skip.metrics.pipeline_fps:.0f} fps")

# --- Keyframes only -----------------------------------------------------------

result_kf = batch_decode(
    paths=[VIDEO],
    levels=LEVEL,
    keyframes_only=True,
)

stream_kf = result_kf.jpeg_stream()
n_kf = len(stream_kf.offsets[0])
print(f"\nKeyframes:   {n_kf} frames in {result_kf.metrics.total_wall_us / 1000:.1f} ms")
print(f"  Pipeline: {result_kf.metrics.pipeline_fps:.0f} fps")

print(f"\nRatio: all/keyframes = {n_all}/{n_kf} = {n_all/max(n_kf,1):.1f}x more frames")
