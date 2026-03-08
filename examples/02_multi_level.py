"""Multi-level decode — produce a sprite atlas and a JPEG stream in one pass."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from vex import batch_decode, LevelConfig

# Output goes to output/02_multi_level/
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'example_output', '02_multi_level')
os.makedirs(OUT_DIR, exist_ok=True)

FIXTURES = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'formats')
videos = [os.path.join(FIXTURES, f) for f in os.listdir(FIXTURES) if f.endswith(('.mp4', '.mkv', '.avi'))]

print(f"Processing {len(videos)} video files...")

result = batch_decode(
    paths=videos,
    levels=[
        # Level 0: small sprite atlas for timeline overview
        LevelConfig(width=48, height=48, quality=30, output="sprite_atlas"),
        # Level 1: native-res JPEG stream for detail previews (default)
        LevelConfig(),
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
    out_path = os.path.join(OUT_DIR, "atlas_0.jpg")
    with open(out_path, "wb") as f:
        f.write(atlas.atlas_jpeg(0))
    print(f"  Saved first atlas to {out_path}")
print()

# Level 1 — typed accessor returns JpegStreamResult directly
stream = result.jpeg_stream(1)
print(f"JPEG stream: {stream}")
for file_idx in range(len(stream.blobs)):
    n_frames = len(stream.offsets[file_idx])
    blob_size = len(stream.blobs[file_idx])
    print(f"  File {file_idx}: {n_frames} frames, {blob_size:,} bytes total")
