"""Start decoding in the background and consume frames as they arrive.

Demonstrates the event-driven workflow: launch an async decode, poll for
FrameEvents, and process each JPEG the instant it's ready rather than
waiting for the entire batch to finish.  This is the pattern for building
responsive UIs or real-time pipelines on top of vex.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vex import LevelConfig, batch_decode_async

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures", "formats")
videos = [
    os.path.join(FIXTURES, f)
    for f in os.listdir(FIXTURES)
    if f.endswith((".mp4", ".mkv", ".avi", ".ts", ".flv"))
]

print(f"Starting async decode of {len(videos)} files...")

handle = batch_decode_async(
    paths=videos,
    levels=[LevelConfig(width=96, height=96, quality=60)],
)

# Poll for results as they stream in
total_events = 0
while not handle.done:
    events = handle.drain_events()
    if events:
        total_events += len(events)
        for evt in events:
            jpeg_data = handle.peek_jpeg(evt)
            # In a real app, you'd paint this to a canvas
            print(
                f"  [stream] file={evt['file_index']}, frame={evt['frame_index']}, "
                f"size={len(jpeg_data)} bytes"
            )
    else:
        time.sleep(0.001)  # brief sleep to avoid busy-wait

    p = handle.progress
    print(
        f"  Progress: {p['files_completed']}/{p['total_files']} files, "
        f"{p['keyframes_decoded']} keyframes",
        end="\r",
    )

# Finalize
results, metrics = handle.result()
print(f"\nDone! {total_events} events streamed during decode.")
print(f"Wall time: {metrics.total_wall_us / 1000:.1f} ms")
print(f"Keyframes decoded: {metrics.keyframes_decoded}")
