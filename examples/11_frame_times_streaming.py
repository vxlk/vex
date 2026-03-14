"""Streaming frame timestamps — collect PTS as a side-effect of decode."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import vex

VIDEO = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "formats", "h264_mp4.mp4"
)

# --- Synchronous: collect_frame_times with batch_decode ----------------------

print("=== Synchronous streaming ===\n")

result = vex.batch_decode(
    [VIDEO],
    keyframes_only=False,
    collect_frame_times=True,
)

times = result.metrics.frame_times(0)
print(f"Decoded {result.metrics.keyframes_decoded} frames")
print(f"Collected {len(times)} timestamps")
print(f"  first: {times[0]:.4f}s")
print(f"  last:  {times[-1]:.4f}s")

# Compare against standalone packet scan
standalone = vex.get_frame_times(VIDEO)
diff = np.abs(times[: len(standalone.times)] - standalone.times[: len(times)])
print(f"\nStandalone vs streaming max delta: {diff.max():.6f}s")

# --- Verify disabled by default ----------------------------------------------

result_no_ft = vex.batch_decode([VIDEO], keyframes_only=False)
assert result_no_ft.metrics.frame_times(0) is None
print("\nCollect disabled by default: confirmed (frame_times is None)")

# --- Async: peek at timestamps during decode ---------------------------------

print("\n=== Async streaming with peek ===\n")

handle = vex.batch_decode_async(
    [VIDEO],
    keyframes_only=False,
    collect_frame_times=True,
)

snapshots = []
while not handle.done:
    ft = handle.peek_frame_times(0)
    if ft is not None and len(ft) > 0:
        snapshots.append(len(ft))
    time.sleep(0.005)

# One final peek after done
ft = handle.peek_frame_times(0)
if ft is not None:
    snapshots.append(len(ft))

result = handle.result()
final_times = result.metrics.frame_times(0)

print(f"Peek snapshots taken: {len(snapshots)}")
if snapshots:
    print(f"  growth: {snapshots[0]} -> {snapshots[-1]} timestamps")
print(f"Final count: {len(final_times)}")

# --- frame_skip interaction --------------------------------------------------

print("\n=== frame_skip=3 with collect_frame_times ===\n")

result_skip = vex.batch_decode(
    [VIDEO],
    keyframes_only=False,
    frame_skip=3,
    collect_frame_times=True,
)

skip_times = result_skip.metrics.frame_times(0)
print(f"Decoded {result_skip.metrics.keyframes_decoded} frames (every 3rd)")
print(f"Collected {len(skip_times)} timestamps")
print(
    f"  Matches decoded count: {len(skip_times) == result_skip.metrics.keyframes_decoded}"
)
