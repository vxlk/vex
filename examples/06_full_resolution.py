"""Full-resolution keyframe extraction — get I-frames at source quality.

Uses width=NATIVE, height=NATIVE so the output matches whatever the
source resolution is, with no scaling step at all.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from vex import batch_decode, LevelConfig, NATIVE

VIDEO = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'resolutions', '1280x720.mp4')

print("Extracting full-resolution keyframes...")

# NATIVE (0) tells vex to use the video's own width/height — no scaling at all.
results, metrics = batch_decode(
    paths=[VIDEO],
    levels=[LevelConfig(width=NATIVE, height=NATIVE, quality=95)],
)

stream = results[0]
print(f"Decoded {metrics.keyframes_decoded} keyframes in {metrics.total_wall_us / 1000:.1f} ms")

if metrics.levels:
    lm = metrics.levels[0]
    print(f"Scale time:  {lm['scale_us']} us (should be ~0 when resolution matches)")
    print(f"Encode time: {lm['encode_us']} us")
    print(f"Avg JPEG:    {lm['avg_jpeg_bytes']:.0f} bytes")
print()

# Save all keyframes
os.makedirs("keyframes", exist_ok=True)
n = len(stream.offsets[0])
for i in range(n):
    jpeg = stream.jpeg_bytes(0, i)
    path = f"keyframes/frame_{i:04d}.jpg"
    with open(path, "wb") as f:
        f.write(jpeg)

print(f"Saved {n} full-res keyframes to keyframes/")
