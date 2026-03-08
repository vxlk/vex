"""Basic single-file decode — extract keyframe thumbnails as JPEGs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vex import LevelConfig, batch_decode

# Output goes to output/01_basic_decode/
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "example_output", "01_basic_decode"
)
os.makedirs(OUT_DIR, exist_ok=True)

# Decode keyframe thumbnails from a single video file
VIDEO = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "formats", "h264_mp4.mp4"
)

result = batch_decode(
    paths=[VIDEO],
    levels=[LevelConfig()],  # defaults: native resolution, quality 100
)

# Typed accessor — returns JpegStreamResult directly (no isinstance needed)
stream = result.jpeg_stream()

print(
    f"Decoded {result.metrics.keyframes_decoded} keyframes in {result.metrics.total_wall_us / 1000:.1f} ms"
)
print(f"Pipeline throughput: {result.metrics.pipeline_fps:.0f} frames/sec")
print(f"Threads used: {result.metrics.threads_used}")
print()

# Access individual JPEG frames
for i in range(min(3, len(stream.offsets[0]))):
    jpeg = stream.jpeg_bytes(file_index=0, frame_index=i)
    print(f"  Frame {i}: {len(jpeg):,} bytes")

# Save first frame to disk
if len(stream.offsets[0]) > 0:
    out_path = os.path.join(OUT_DIR, "frame_0.jpg")
    with open(out_path, "wb") as f:
        f.write(stream.jpeg_bytes(0, 0))
    print(f"\nSaved first keyframe to {out_path}")
