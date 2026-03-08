"""Disk cache — decode to disk and reload via memory-mapped CachedLevel."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from vex import batch_decode, LevelConfig, CachedLevel

VIDEO = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'formats', 'h264_mp4.mp4')

# Create a temp path for the cache file
cache_path = os.path.join(tempfile.gettempdir(), "vex_example_cache.bin")

print(f"Decoding to disk: {cache_path}")

results, metrics = batch_decode(
    paths=[VIDEO],
    levels=[LevelConfig(
        output="jpeg_stream",
        in_memory=False,
        cache_path=cache_path,
    )],
)

disk = results[0]
print(f"Result: {disk}")
print(f"  Frames: {disk.frame_count}")
print(f"  Total bytes on disk: {disk.total_bytes:,}")
print()

# Reload from disk using CachedLevel (memory-mapped, fast random access)
print("Reloading from disk cache...")
with CachedLevel(cache_path) as cache:
    print(f"CachedLevel: {cache}")
    print(f"  Resolution: {cache.width}x{cache.height}")
    print(f"  Frames: {len(cache)}")

    # Access individual frames by index
    for i in range(min(3, len(cache))):
        jpeg = cache[i]
        print(f"  Frame {i}: {len(jpeg):,} bytes")

    # Save last frame
    if len(cache) > 0:
        with open("cached_frame.jpg", "wb") as f:
            f.write(cache[-1])
        print(f"\n  Saved last frame to cached_frame.jpg")

# Clean up
os.unlink(cache_path)
print(f"Cleaned up {cache_path}")
