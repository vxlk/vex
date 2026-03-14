"""Throw every fixture format at vex in one batch and inspect the per-file
breakdown.

Validates that the pipeline handles a mixed bag of containers (MP4, MKV,
AVI, TS, FLV, ...) in a single call and surfaces per-file diagnostics like
which index strategy was chosen, how many keyframes were found, and the
codec used.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vex import LevelConfig, batch_decode

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")

# Collect all video files across fixture directories
videos = []
for root, dirs, files in os.walk(FIXTURES):
    for f in sorted(files):
        if f.endswith((".mp4", ".mkv", ".avi", ".ts", ".flv")):
            videos.append(os.path.join(root, f))

print(f"Found {len(videos)} video files across fixtures/")
print()

results, metrics = batch_decode(
    paths=videos,
    levels=[LevelConfig(width=96, height=96, quality=60)],
    max_threads=4,
)

# Index strategy names
STRATEGY_NAMES = {
    0: "CONTAINER_INDEX",
    1: "PACKET_SCAN",
    2: "DECODE_SCAN",
    3: "FORCED_INTERVAL",
    4: "SKIPPED",
}

# Print per-file breakdown
print(f"{'File':<45} {'Strategy':<18} {'Keyframes':>9} {'Size':>10}")
print("-" * 85)

for stat in sorted(metrics.file_stats, key=lambda s: s["file_index"]):
    idx = stat["file_index"]
    path = os.path.relpath(videos[idx], FIXTURES)
    strategy = STRATEGY_NAMES.get(stat["index_strategy"], "?")
    kf = stat["keyframe_count"]
    size = stat["file_size_bytes"]
    print(f"{path:<45} {strategy:<18} {kf:>9} {size:>9,}")

print("-" * 85)
print(f"{'Total':<45} {'':18} {metrics.keyframes_decoded:>9} {'':>10}")
print()

# Summary
print(f"Wall time:      {metrics.total_wall_us / 1000:.1f} ms")
print(f"Pipeline FPS:   {metrics.pipeline_fps:.0f}")
print(f"Threads used:   {metrics.threads_used}")
print(f"Output bytes:   {metrics.total_output_bytes:,}")
