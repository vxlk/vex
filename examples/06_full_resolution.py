"""Extract keyframes at the video's native resolution with zero scaling.

Confirms that the default LevelConfig (NATIVE width/height, quality 100)
bypasses the scaler entirely, so the output is a lossless JPEG re-encode of
the raw decoder output.  Useful for archival-quality thumbnail extraction
where you want pixel-perfect fidelity.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vex import LevelConfig, batch_decode

# Output goes to output/06_full_resolution/
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "example_output", "06_full_resolution"
)
os.makedirs(OUT_DIR, exist_ok=True)

VIDEO = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "resolutions", "1280x720.mp4"
)

print("Extracting full-resolution keyframes...")

# Default LevelConfig: native resolution, quality 100, no scaling.
results, metrics = batch_decode(
    paths=[VIDEO],
    levels=[LevelConfig()],
)

stream = results[0]
print(
    f"Decoded {metrics.keyframes_decoded} keyframes in {metrics.total_wall_us / 1000:.1f} ms"
)

if metrics.levels:
    lm = metrics.levels[0]
    print(f"Scale time:  {lm['scale_us']} us (should be ~0 when resolution matches)")
    print(f"Encode time: {lm['encode_us']} us")
    print(f"Avg JPEG:    {lm['avg_jpeg_bytes']:.0f} bytes")
print()

# Save all keyframes
n = len(stream.offsets[0])
for i in range(n):
    jpeg = stream.jpeg_bytes(0, i)
    path = os.path.join(OUT_DIR, f"frame_{i:04d}.jpg")
    with open(path, "wb") as f:
        f.write(jpeg)

print(f"Saved {n} full-res keyframes to {OUT_DIR}")
