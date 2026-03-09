"""Multi-file async streaming pipeline — decode N videos concurrently,
use zero-copy views into vex's VirtualBlob buffers, and have a client
poll for new segments.

Architecture:
  - One batch_decode_async handle per video, all running concurrently.
  - A background poll thread drains FrameEvents from every handle and
    appends them to a per-stream event list.  Each event carries
    blob_offset and jpeg_size — the index into the stable VirtualBlob.
  - A client loop polls each stream's event index, compares against its
    own read cursor, and reads the new segment via peek_stream — a
    zero-copy memoryview into the C++ blob (no copy until the client
    needs to materialize bytes, e.g. for file I/O).

Usage:
    python examples/09_multi_stream_pipeline.py [video_dir]

If no directory is given, the default fixtures/formats/ folder is used.
"""

import os
import sys
import time
import threading
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vex import DecodeHandle, LevelConfig, batch_decode_async


# ---------------------------------------------------------------------------
# StreamHub — multiplexes N async decode handles
# ---------------------------------------------------------------------------


class StreamHub:
    """Manages multiple concurrent video decode streams.

    Each video is decoded independently via batch_decode_async.  A
    background poll thread drains FrameEvents from all handles and
    appends them to per-stream event lists.  The events themselves
    are the index — each one carries blob_offset and jpeg_size
    pointing into vex's VirtualBlob (stable pointer, never moves).

    Clients call read_new(stream_id, cursor) to get new events,
    then peek_stream(stream_id) to get a zero-copy memoryview into
    the C++ buffer.  Slicing the memoryview with the event offsets
    yields each JPEG — no copy until the client materializes bytes.
    """

    def __init__(
        self,
        paths: List[str],
        level: LevelConfig = None,
        keyframes_only: bool = True,
    ):
        if level is None:
            level = LevelConfig(width=320, height=180, quality=70)

        self._paths = list(paths)
        self._handles: Dict[int, DecodeHandle] = {}
        self._events: Dict[int, List[dict]] = {}
        self._done_flags: Dict[int, bool] = {}

        for i, path in enumerate(paths):
            self._handles[i] = batch_decode_async(
                paths=[path],
                levels=[level],
                keyframes_only=keyframes_only,
            )
            self._events[i] = []
            self._done_flags[i] = False

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def stream_ids(self) -> List[int]:
        return list(self._handles.keys())

    def stream_path(self, stream_id: int) -> str:
        return self._paths[stream_id]

    def frame_count(self, stream_id: int) -> int:
        with self._lock:
            return len(self._events[stream_id])

    def is_done(self, stream_id: int) -> bool:
        with self._lock:
            return self._done_flags.get(stream_id, True)

    @property
    def all_done(self) -> bool:
        with self._lock:
            return all(self._done_flags.values())

    def read_new(self, stream_id: int, cursor: int) -> Tuple[List[dict], int]:
        """Return new events from *cursor* onward for a stream.

        Returns (events, new_cursor).  Each event dict contains
        blob_offset, jpeg_size, pts_ms — the native index into
        vex's VirtualBlob.  Empty list if no new data.
        """
        with self._lock:
            evts = self._events[stream_id]
            if cursor >= len(evts):
                return [], cursor
            return evts[cursor:], len(evts)

    def peek_stream(self, stream_id: int) -> Optional[memoryview]:
        """Zero-copy memoryview into the C++ JPEG buffer.

        The memoryview is backed by stable virtual memory (VirtualAlloc/
        mmap).  It grows between calls as more frames are decoded.
        Slice it with event offsets to get individual JPEGs — no copy
        until you materialize bytes (e.g. write to file).
        """
        return self._handles[stream_id].peek_stream(file_index=0, level_index=0)

    # -- background poll loop ------------------------------------------------

    def _poll_once(self) -> int:
        new_frames = 0
        for sid in self.stream_ids:
            with self._lock:
                if self._done_flags[sid]:
                    continue

            handle = self._handles[sid]
            events = handle.drain_events()
            if events:
                with self._lock:
                    self._events[sid].extend(events)
                new_frames += len(events)

            if handle.done:
                with self._lock:
                    self._done_flags[sid] = True

        return new_frames

    def _run(self):
        while not self._stop.is_set() and not self.all_done:
            n = self._poll_once()
            if n == 0:
                time.sleep(0.001)
        # Final drain
        self._poll_once()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def join(self, timeout: float = None):
        if self._thread:
            self._thread.join(timeout=timeout)

    def stop(self):
        self._stop.set()

    def finalize(self) -> dict:
        results = {}
        for sid in self.stream_ids:
            batch = self._handles[sid].result()
            results[sid] = {
                "path": self._paths[sid],
                "metrics": batch.metrics,
                "frame_count": len(self._events[sid]),
            }
        return results


# ---------------------------------------------------------------------------
# Demo: client polls stream indices and saves every frame via zero-copy view
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) > 1:
        video_dir = sys.argv[1]
    else:
        video_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures", "formats")

    extensions = (".mp4", ".mkv", ".avi", ".ts", ".flv", ".webm")
    videos = sorted(
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.lower().endswith(extensions)
    )

    if not videos:
        print(f"No video files found in {video_dir}")
        sys.exit(1)

    print(f"Streaming {len(videos)} videos concurrently:\n")
    for i, v in enumerate(videos):
        print(f"  [{i}] {os.path.basename(v)}")
    print()

    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "example_output", "09_multi_stream"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Create per-stream output directories
    stream_dirs = {}
    for i, v in enumerate(videos):
        name = os.path.splitext(os.path.basename(v))[0]
        d = os.path.join(out_dir, f"{i:02d}_{name}")
        os.makedirs(d, exist_ok=True)
        stream_dirs[i] = d

    hub = StreamHub(
        videos,
        level=LevelConfig(width=320, height=180, quality=70),
        keyframes_only=True,
    )

    hub.start()

    # -- Client: per-stream cursors into the event index ---------------------
    cursors: Dict[int, int] = {sid: 0 for sid in hub.stream_ids}
    total_saved = 0
    poll_count = 0
    t0 = time.perf_counter()

    while True:
        new_this_tick = 0

        for sid in hub.stream_ids:
            events, new_cursor = hub.read_new(sid, cursors[sid])
            if not events:
                continue

            # Get the zero-copy view into the VirtualBlob
            buf = hub.peek_stream(sid)
            if buf is None:
                continue

            # Slice the view with event offsets — no copy until write()
            for j, evt in enumerate(events):
                frame_idx = cursors[sid] + j
                off = evt["blob_offset"]
                sz = evt["jpeg_size"]
                pts_ms = evt["pts_ms"]
                jpeg_slice = buf[off : off + sz]  # zero-copy slice

                path = os.path.join(
                    stream_dirs[sid], f"frame_{frame_idx:04d}_{pts_ms}ms.jpg"
                )
                with open(path, "wb") as f:
                    f.write(jpeg_slice)  # only copy happens here (kernel I/O)
                new_this_tick += 1

            cursors[sid] = new_cursor

        total_saved += new_this_tick
        poll_count += 1

        # Status line
        elapsed = time.perf_counter() - t0
        parts = []
        for sid in hub.stream_ids:
            count = hub.frame_count(sid)
            done = hub.is_done(sid)
            tag = f"{count}{'*' if done else ''}"
            parts.append(f"{cursors[sid]}/{tag}")
        print(
            f"  {elapsed:5.2f}s  poll#{poll_count:3d}  "
            f"saved={total_saved:4d}  "
            f"streams: {' '.join(parts)}",
            end="\r",
        )

        if hub.all_done and all(
            cursors[sid] >= hub.frame_count(sid) for sid in hub.stream_ids
        ):
            break

        if new_this_tick == 0:
            time.sleep(0.005)

    hub.join()
    elapsed = time.perf_counter() - t0

    print(f"\n\nDone in {elapsed:.2f}s  |  {poll_count} polls  |  {total_saved} frames saved\n")

    # Per-stream summary
    final = hub.finalize()
    for sid, info in sorted(final.items()):
        m = info["metrics"]
        name = os.path.basename(info["path"])
        print(
            f"  [{sid:2d}] {name:30s}  "
            f"frames={info['frame_count']:3d}  "
            f"wall={m.total_wall_us / 1000:.1f}ms  "
            f"fps={m.pipeline_fps:.0f}"
        )

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
