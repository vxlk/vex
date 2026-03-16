"""Regression tests for concurrent batch_decode_async calls.

The thread pool originally used shared member variables (func_, total_,
completed_) for tracking a single batch.  When multiple async decodes
ran simultaneously — each on its own manager thread calling
ThreadPool::run() — the batches corrupted each other's state, causing
a deadlock (example 09 would hang forever).

The fix (queue-based pool with per-Batch completion tracking) makes
concurrent run() calls safe.  These tests verify that:

  1. Multiple batch_decode_async handles can be created in a tight loop
     and all complete without hanging.
  2. Results from each handle are independent and correct.
  3. The streaming APIs (peek_stream, drain_events) work while multiple
     decodes are in flight.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vex import LevelConfig, batch_decode_async

HAS_NATIVE = False
try:
    from vex import _load_native

    _load_native()
    HAS_NATIVE = True
except (ImportError, Exception):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "fixtures"


def _fixture_path(*parts: str) -> str:
    return str(FIXTURES_DIR.joinpath(*parts))


def _has_fixture(*parts: str) -> bool:
    return FIXTURES_DIR.joinpath(*parts).is_file()


needs_native = pytest.mark.skipif(
    not HAS_NATIVE, reason="native module (_vex_core) not built"
)

needs_fixtures = pytest.mark.skipif(
    not _has_fixture("formats", "h264_mp4.mp4"),
    reason="test fixtures not generated (run tools/generate_fixtures.py)",
)


def _pick_fixtures(n: int) -> list[str]:
    """Return up to n distinct fixture paths from formats/."""
    formats_dir = FIXTURES_DIR / "formats"
    # Prefer a variety of codecs to stress different decoder paths.
    preferred = [
        "h264_mp4.mp4",
        "h264_mkv.mkv",
        "h264_avi.avi",
        "mpeg4_avi.avi",
        "flv1_flv.flv",
        "mjpeg_avi.avi",
        "h265_mp4.mp4",
        "vp9_webm.webm",
        "mpeg2_ts.ts",
        "msmpeg4_avi.avi",
    ]
    paths = []
    for name in preferred:
        if len(paths) >= n:
            break
        p = formats_dir / name
        if p.is_file():
            paths.append(str(p))
    return paths


def _wait_handle(handle, timeout: float = 30.0):
    """Poll handle.done until True or timeout.  Raises on timeout."""
    deadline = time.monotonic() + timeout
    while not handle.done:
        if time.monotonic() > deadline:
            pytest.fail("handle did not complete within timeout — likely deadlock")
        time.sleep(0.005)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

TIMEOUT_SECONDS = 30  # generous; a hang means something is very wrong


@needs_native
@needs_fixtures
class TestConcurrentAsyncDecode:
    """Verify that multiple batch_decode_async calls in rapid succession
    all complete without deadlocking the thread pool."""

    def test_two_concurrent_async_handles(self):
        """Minimum reproduction: two overlapping async decodes.

        This is the simplest case that triggered the original deadlock —
        two manager threads calling ThreadPool::run() concurrently.
        """
        paths = _pick_fixtures(2)
        assert len(paths) >= 2, "need at least 2 fixture files"

        level = LevelConfig(width=192, height=192, quality=70)

        h1 = batch_decode_async([paths[0]], levels=[level], keyframes_only=True)
        h2 = batch_decode_async([paths[1]], levels=[level], keyframes_only=True)

        # Both must complete within the timeout — a hang here means the
        # thread pool is still broken.
        _wait_handle(h1, TIMEOUT_SECONDS)
        _wait_handle(h2, TIMEOUT_SECONDS)

        r1 = h1.result()
        r2 = h2.result()

        assert h1.done
        assert h2.done
        assert r1.metrics.files_processed == 1
        assert r2.metrics.files_processed == 1
        assert r1.metrics.keyframes_decoded > 0
        assert r2.metrics.keyframes_decoded > 0

    def test_many_concurrent_async_handles(self):
        """Stress test: launch N async decodes before waiting on any of
        them.  This mirrors the StreamHub pattern from example 09.

        With the old single-batch thread pool, this would deadlock as
        soon as the second manager thread entered ThreadPool::run()
        while the first was still blocking inside it.
        """
        paths = _pick_fixtures(6)
        assert len(paths) >= 3, "need at least 3 fixture files"

        level = LevelConfig(width=160, height=120, quality=60)

        # Launch all handles up-front — no waiting between launches.
        handles = []
        for p in paths:
            handles.append(batch_decode_async([p], levels=[level], keyframes_only=True))

        # Now collect results.  Timeout ensures we don't hang forever.
        for i, h in enumerate(handles):
            _wait_handle(h, TIMEOUT_SECONDS)
            result = h.result()
            assert h.done, f"handle {i} ({paths[i]}) did not complete"
            assert result.metrics.files_processed == 1
            assert result.metrics.keyframes_decoded > 0

    def test_concurrent_async_results_are_independent(self):
        """Verify that frames decoded by one handle don't leak into
        another — each handle's results must be self-contained."""
        paths = _pick_fixtures(4)
        assert len(paths) >= 4, "need at least 4 fixture files"

        level = LevelConfig(width=192, height=192, quality=75)

        handles = [
            batch_decode_async([p], levels=[level], keyframes_only=True) for p in paths
        ]

        frame_counts = []
        for h in handles:
            _wait_handle(h, TIMEOUT_SECONDS)
            result = h.result()
            assert len(result.levels) == 1
            frame_counts.append(result.metrics.keyframes_decoded)

        # Each video should report a positive frame count.
        assert all(c > 0 for c in frame_counts)

        # Re-run sequentially and verify counts match (proving no
        # cross-contamination from concurrent execution).
        from vex import batch_decode

        for i, p in enumerate(paths):
            result = batch_decode([p], levels=[level], keyframes_only=True)
            assert result.metrics.keyframes_decoded == frame_counts[i], (
                f"file {os.path.basename(p)}: concurrent got "
                f"{frame_counts[i]} frames, sequential got "
                f"{result.metrics.keyframes_decoded}"
            )

    def test_concurrent_async_with_streaming(self):
        """Launch multiple async decodes and poll peek_stream / drain_events
        on all of them concurrently, similar to example 09's StreamHub."""
        paths = _pick_fixtures(4)
        assert len(paths) >= 3, "need at least 3 fixture files"

        level = LevelConfig(width=160, height=120, quality=60)

        handles = [
            batch_decode_async([p], levels=[level], keyframes_only=True) for p in paths
        ]

        # Poll all handles until all are done, collecting streaming data.
        total_events = [0] * len(handles)
        deadline = time.monotonic() + TIMEOUT_SECONDS

        while time.monotonic() < deadline:
            all_done = True
            for i, h in enumerate(handles):
                events = h.drain_events()
                total_events[i] += len(events)

                # peek_stream should not raise even while decode is in flight
                h.peek_stream(file_index=0, level_index=0)

                if not h.done:
                    all_done = False

            if all_done:
                # Final drain
                for i, h in enumerate(handles):
                    events = h.drain_events()
                    total_events[i] += len(events)
                break

            time.sleep(0.002)
        else:
            pytest.fail("timed out waiting for concurrent async handles")

        # Every handle must have produced at least one frame event.
        for i, count in enumerate(total_events):
            assert count > 0, (
                f"handle {i} ({os.path.basename(paths[i])}) "
                f"produced 0 events via drain_events"
            )

    def test_concurrent_async_from_multiple_python_threads(self):
        """Launch batch_decode_async from several Python threads at once.

        This is a stronger version of the sequential-launch test: the
        GIL is released during the C++ call, so the manager threads may
        truly race into ThreadPool::run() simultaneously.
        """
        paths = _pick_fixtures(6)
        assert len(paths) >= 3, "need at least 3 fixture files"

        level = LevelConfig(width=128, height=96, quality=50)

        handles = [None] * len(paths)
        errors = [None] * len(paths)
        barrier = threading.Barrier(len(paths))

        def launch(idx: int):
            try:
                barrier.wait(timeout=5)  # sync all threads to start together
                handles[idx] = batch_decode_async(
                    [paths[idx]], levels=[level], keyframes_only=True
                )
            except Exception as exc:
                errors[idx] = exc

        threads = [
            threading.Thread(target=launch, args=(i,)) for i in range(len(paths))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=TIMEOUT_SECONDS)

        # Check no launch errors.
        for i, err in enumerate(errors):
            assert err is None, f"thread {i} failed: {err}"

        # All handles must complete.
        for i, h in enumerate(handles):
            assert h is not None, f"handle {i} was never created"
            _wait_handle(h, TIMEOUT_SECONDS)
            result = h.result()
            assert h.done
            assert result.metrics.keyframes_decoded > 0
