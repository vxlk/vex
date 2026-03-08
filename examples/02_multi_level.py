"""Multi-level decode — produce a sprite atlas and a JPEG stream in one pass."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from vex import batch_decode, LevelConfig

FIXTURES = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'formats')
videos = [os.path.join(FIXTURES, f) for f in os.listdir(FIXTURES) if f.endswith(('.mp4', '.mkv', '.avi'))]

print(f"Processing {len(videos)} video files...")

result = batch_decode(
    paths=videos,
    levels=[
        # Level 0: small sprite atlas for timeline overview
        LevelConfig(width=48, height=48, quality=30, output="sprite_atlas"),
        # Level 1: medium JPEG stream for detail previews
        LevelConfig(width=192, height=192, quality=85, output="jpeg_stream"),
    ],
)

print(f"Wall time: {result.metrics.total_wall_us / 1000:.1f} ms")
print(f"Keyframes decoded: {result.metrics.keyframes_decoded}")
print()

# Level 0 — typed accessor returns SpriteAtlasResult directly
atlas = result.sprite_atlas(0)
print(f"Sprite atlas: {atlas}")
print(f"  Grid: {atlas.grid_w}x{atlas.grid_h} ({atlas.thumb_w}x{atlas.thumb_h} px per thumb)")
n_atlases = len(atlas.offsets) - 1
print(f"  Time steps: {n_atlases}")

# Save the first atlas as a JPEG
if n_atlases > 0:
    with open("atlas_0.jpg", "wb") as f:
        f.write(atlas.atlas_jpeg(0))
    print(f"  Saved first atlas to atlas_0.jpg")
print()

# Level 1 — typed accessor returns JpegStreamResult directly
stream = result.jpeg_stream(1)
print(f"JPEG stream: {stream}")
for file_idx in range(len(stream.blobs)):
    n_frames = len(stream.offsets[file_idx])
    blob_size = len(stream.blobs[file_idx])
    print(f"  File {file_idx}: {n_frames} frames, {blob_size:,} bytes total")
