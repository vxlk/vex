"""Stress-test VirtualBlob's ability to grow under memory pressure.

Forces the blob to exceed its initial reservation by using a tiny
blob_reservation, then confirms that the output is still correct --
no stale pointers, no data corruption, no crashes.  This is the
safety net for the virtual-memory growth path that most real workloads
never hit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
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

needs_native = pytest.mark.skipif(
    not HAS_NATIVE, reason="native module (_vex_core) not built"
)
needs_fixtures = pytest.mark.skipif(
    not (FIXTURES_DIR / "formats" / "h264_mp4.mp4").is_file(),
    reason="test fixtures not generated (run tools/generate_fixtures.py)",
)


def _fixture_path(*parts: str) -> str:
    return str(FIXTURES_DIR.joinpath(*parts))


def _jpeg_dimensions(data: bytes) -> tuple:
    """Extract (width, height) from a JPEG's SOF marker."""
    i = 0
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xD8:
            i += 2
            continue
        if marker == 0x00 or (0xD0 <= marker <= 0xD7):
            i += 2
            continue
        if i + 3 >= len(data):
            break
        seg_len = (data[i + 2] << 8) | data[i + 3]
        if 0xC0 <= marker <= 0xC3:
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            return w, h
        i += 2 + seg_len
    raise ValueError("SOF marker not found in JPEG data")


# Use 64 KB — far too small for any real video.  Forces multiple grows.
TINY_RESERVATION = 64 * 1024


def _decode_with_tiny_blob(path, *, keyframes_only=False, width=192, height=192):
    """Decode a file using a tiny blob_reservation to force growth."""
    handle = batch_decode_async(
        [path],
        levels=[LevelConfig(width=width, height=height, quality=85)],
        keyframes_only=keyframes_only,
        blob_reservation=TINY_RESERVATION,
    )
    results, metrics = handle.result()
    return results, metrics


def _decode_with_default_blob(path, *, keyframes_only=False, width=192, height=192):
    """Decode a file using the default (256 MB) blob reservation."""
    handle = batch_decode_async(
        [path],
        levels=[LevelConfig(width=width, height=height, quality=85)],
        keyframes_only=keyframes_only,
        blob_reservation=0,
    )
    results, metrics = handle.result()
    return results, metrics


@needs_native
@needs_fixtures
class TestBlobGrowth:
    """Verify VirtualBlob growth produces identical results to a large reservation."""

    def test_grow_produces_frames(self):
        """A tiny reservation should still decode all frames (no silent drops)."""
        path = _fixture_path("formats", "h264_mp4.mp4")
        results_tiny, metrics_tiny = _decode_with_tiny_blob(path)
        results_default, metrics_default = _decode_with_default_blob(path)

        assert metrics_tiny.keyframes_decoded == metrics_default.keyframes_decoded
        assert metrics_tiny.keyframes_decoded > 0

    def test_grow_data_matches_default(self):
        """JPEG data from a grown blob should be byte-identical to default."""
        path = _fixture_path("formats", "h264_mp4.mp4")
        results_tiny, _ = _decode_with_tiny_blob(path)
        results_default, _ = _decode_with_default_blob(path)

        blob_tiny = results_tiny[0].blobs[0]
        blob_default = results_default[0].blobs[0]

        assert len(blob_tiny) == len(blob_default), (
            f"blob sizes differ: tiny={len(blob_tiny)}, default={len(blob_default)}"
        )
        assert bytes(blob_tiny) == bytes(blob_default), "blob contents differ"

    def test_grow_offsets_match_default(self):
        """Offset tables from a grown blob should match default."""
        path = _fixture_path("formats", "h264_mp4.mp4")
        results_tiny, _ = _decode_with_tiny_blob(path)
        results_default, _ = _decode_with_default_blob(path)

        offsets_tiny = results_tiny[0].offsets[0]
        offsets_default = results_default[0].offsets[0]

        np.testing.assert_array_equal(offsets_tiny, offsets_default)

    def test_grow_jpegs_are_valid(self):
        """Each JPEG extracted from a grown blob should have valid dimensions."""
        path = _fixture_path("formats", "h264_mp4.mp4")
        results, _ = _decode_with_tiny_blob(path, width=192, height=192)
        stream = results[0]

        offsets = stream.offsets[0]
        assert len(offsets) > 0

        for i in range(len(offsets)):
            jpeg = stream.jpeg_bytes(0, i)
            assert len(jpeg) > 0, f"frame {i} is empty"
            w, h = _jpeg_dimensions(jpeg)
            assert w == 192, f"frame {i}: expected width 192, got {w}"
            assert h == 192, f"frame {i}: expected height 192, got {h}"

    def test_grow_sequential_decode(self):
        """Sequential (every-frame) decode with tiny blob should match default."""
        path = _fixture_path("formats", "h264_mp4.mp4")
        results_tiny, metrics_tiny = _decode_with_tiny_blob(path, keyframes_only=False)
        results_default, metrics_default = _decode_with_default_blob(
            path, keyframes_only=False
        )

        assert metrics_tiny.keyframes_decoded == metrics_default.keyframes_decoded
        assert metrics_tiny.keyframes_decoded > 0

        blob_tiny = results_tiny[0].blobs[0]
        blob_default = results_default[0].blobs[0]

        assert len(blob_tiny) == len(blob_default), (
            f"sequential blob sizes differ: tiny={len(blob_tiny)}, "
            f"default={len(blob_default)}"
        )
        assert bytes(blob_tiny) == bytes(blob_default), (
            "sequential blob contents differ"
        )

    def test_grow_multi_file(self):
        """Multiple files with tiny reservation should all decode fully."""
        paths = [
            _fixture_path("formats", "h264_mp4.mp4"),
            _fixture_path("formats", "h264_mkv.mkv"),
        ]
        handle_tiny = batch_decode_async(
            paths,
            levels=[LevelConfig(width=192, height=192, quality=85)],
            keyframes_only=True,
            blob_reservation=TINY_RESERVATION,
        )
        results_tiny, metrics_tiny = handle_tiny.result()

        handle_default = batch_decode_async(
            paths,
            levels=[LevelConfig(width=192, height=192, quality=85)],
            keyframes_only=True,
            blob_reservation=0,
        )
        results_default, metrics_default = handle_default.result()

        assert metrics_tiny.keyframes_decoded == metrics_default.keyframes_decoded
        assert metrics_tiny.files_processed == 2

        for fi in range(2):
            blob_tiny = results_tiny[0].blobs[fi]
            blob_default = results_default[0].blobs[fi]
            assert len(blob_tiny) == len(blob_default), f"file {fi}: blob sizes differ"
            assert bytes(blob_tiny) == bytes(blob_default), (
                f"file {fi}: blob contents differ"
            )


@needs_native
@needs_fixtures
class TestBlobGrowStreaming:
    """Verify streaming APIs work correctly when the blob grows."""

    def test_peek_stream_valid_during_grow(self):
        """peek_stream should return valid, growing data even when the blob
        reallocates underneath."""
        path = _fixture_path("formats", "h264_mp4.mp4")
        handle = batch_decode_async(
            [path],
            levels=[LevelConfig(width=320, height=240, quality=90)],
            keyframes_only=False,
            blob_reservation=TINY_RESERVATION,
        )

        import time

        sizes = []
        for _ in range(500):
            mv = handle.peek_stream(file_index=0, level_index=0)
            if mv is not None and len(mv) > 0:
                sizes.append(len(mv))
            if handle.done:
                mv = handle.peek_stream(file_index=0, level_index=0)
                if mv is not None and len(mv) > 0:
                    sizes.append(len(mv))
                break
            time.sleep(0.001)

        assert len(sizes) >= 1, "never got a valid peek_stream"
        # Sizes should be monotonically non-decreasing
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1], (
                f"peek_stream size decreased: {sizes[i - 1]} -> {sizes[i]}"
            )

    def test_drain_events_valid_after_grow(self):
        """Events drained during/after blob growth should have valid JPEG data."""
        path = _fixture_path("formats", "h264_mp4.mp4")
        handle = batch_decode_async(
            [path],
            levels=[LevelConfig(width=192, height=192, quality=85)],
            keyframes_only=False,
            blob_reservation=TINY_RESERVATION,
        )

        import time

        all_events = []
        while not handle.done:
            evts = handle.drain_events()
            all_events.extend(evts)
            time.sleep(0.001)
        # Final drain
        all_events.extend(handle.drain_events())

        assert len(all_events) > 0, "no events drained"

        # Each event should produce valid JPEG bytes via peek_jpeg
        for evt in all_events:
            jpeg = handle.peek_jpeg(evt)
            assert len(jpeg) > 0, f"empty JPEG for frame {evt['frame_index']}"
            w, h = _jpeg_dimensions(jpeg)
            assert w == 192
            assert h == 192

    def test_result_valid_after_streaming_with_grow(self):
        """Calling result() after streaming with a tiny blob should produce
        the same data as a non-streaming decode."""
        path = _fixture_path("formats", "h264_mp4.mp4")

        # Streaming path with tiny blob
        handle = batch_decode_async(
            [path],
            levels=[LevelConfig(width=192, height=192, quality=85)],
            keyframes_only=True,
            blob_reservation=TINY_RESERVATION,
        )

        import time

        # Simulate a streaming consumer
        while not handle.done:
            handle.drain_events()
            handle.peek_stream(file_index=0, level_index=0)
            time.sleep(0.001)

        results_stream, metrics_stream = handle.result()

        # Non-streaming path with default blob (same keyframes_only setting)
        results_default, metrics_default = _decode_with_default_blob(
            path, keyframes_only=True
        )

        assert metrics_stream.keyframes_decoded == metrics_default.keyframes_decoded

        blob_stream = results_stream[0].blobs[0]
        blob_default = results_default[0].blobs[0]
        assert len(blob_stream) == len(blob_default)
        assert bytes(blob_stream) == bytes(blob_default)
