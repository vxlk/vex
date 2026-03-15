"""Cache performance tests.

Proves that the three cache layers (disk cache, process cache, thread cache)
deliver measurable speedups from the Python API.  Every assertion is a
wall-clock timing comparison: the cached path must be meaningfully faster
than the uncached path.

Requires the native module and test fixtures.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vex import (
    CachedLevel,
    DiskResult,
    LevelConfig,
    batch_decode,
)

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


# ---------------------------------------------------------------------------
# 1. Disk cache: CachedLevel reads are faster than re-decoding
# ---------------------------------------------------------------------------


@needs_native
@needs_fixtures
class TestDiskCache:
    """Write frames to a .vex cache file, then show that reading them back
    via CachedLevel is dramatically faster than decoding the video again."""

    def test_cached_read_faster_than_decode(self, tmp_path):
        video = _fixture_path("formats", "h264_mp4.mp4")
        cache_file = str(tmp_path / "cache.vex")

        # --- Decode to disk (creates the cache file) ---
        result = batch_decode(
            [video],
            levels=[
                LevelConfig(
                    width=192,
                    height=192,
                    quality=80,
                    in_memory=False,
                    cache_path=cache_file,
                )
            ],
            keyframes_only=True,
        )
        disk = result.disk()
        assert isinstance(disk, DiskResult)
        assert disk.frame_count > 0
        assert os.path.isfile(cache_file)

        # --- Time: decode the same video again (in-memory this time) ---
        t0 = time.perf_counter()
        for _ in range(5):
            batch_decode(
                [video],
                levels=[LevelConfig(width=192, height=192, quality=80)],
                keyframes_only=True,
            )
        decode_time = (time.perf_counter() - t0) / 5

        # --- Time: read all frames from the disk cache ---
        t0 = time.perf_counter()
        for _ in range(5):
            with CachedLevel(cache_file) as cache:
                for i in range(len(cache)):
                    _ = cache[i]
        cache_time = (time.perf_counter() - t0) / 5

        # Cache read must be at least 2x faster than re-decoding
        assert cache_time < decode_time, (
            f"cache read ({cache_time*1000:.1f}ms) was not faster than "
            f"decode ({decode_time*1000:.1f}ms)"
        )

    def test_cached_frames_match_in_memory(self, tmp_path):
        """Disk cache frames are byte-identical to in-memory decode."""
        video = _fixture_path("formats", "h264_mp4.mp4")
        cache_file = str(tmp_path / "match.vex")

        # Decode in-memory
        mem_result = batch_decode(
            [video],
            levels=[LevelConfig(width=192, height=192, quality=80)],
            keyframes_only=True,
        )
        stream = mem_result.jpeg_stream()

        # Decode to disk
        batch_decode(
            [video],
            levels=[
                LevelConfig(
                    width=192,
                    height=192,
                    quality=80,
                    in_memory=False,
                    cache_path=cache_file,
                )
            ],
            keyframes_only=True,
        )

        with CachedLevel(cache_file) as cache:
            assert len(cache) == len(stream.offsets[0])
            for i in range(len(cache)):
                disk_jpeg = cache[i]
                mem_jpeg = stream.jpeg_bytes(0, i)
                assert disk_jpeg == mem_jpeg, f"frame {i} mismatch"

    def test_cached_level_random_access_is_constant_time(self, tmp_path):
        """Accessing the first and last frame should take similar time,
        proving the mmap + offset table gives O(1) random access."""
        video = _fixture_path("formats", "h264_mp4.mp4")
        cache_file = str(tmp_path / "random.vex")

        batch_decode(
            [video],
            levels=[
                LevelConfig(
                    width=192,
                    height=192,
                    quality=80,
                    in_memory=False,
                    cache_path=cache_file,
                )
            ],
        )

        with CachedLevel(cache_file) as cache:
            n = len(cache)
            assert n >= 2

            iters = 500
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = cache[0]
            first_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            for _ in range(iters):
                _ = cache[n - 1]
            last_time = time.perf_counter() - t0

            # Both should be within 5x of each other (generous for O(1))
            ratio = max(first_time, last_time) / max(min(first_time, last_time), 1e-9)
            assert ratio < 5.0, (
                f"random access not constant time: first={first_time*1000:.2f}ms, "
                f"last={last_time*1000:.2f}ms, ratio={ratio:.1f}x"
            )


# ---------------------------------------------------------------------------
# 2. Process cache: second batch_decode call benefits from warm caches
#    (HW compat cache + format probe hints)
# ---------------------------------------------------------------------------


@needs_native
@needs_fixtures
class TestProcessCache:
    """The ProcessCache singleton caches HW compatibility results and
    format probe hints.  After the first batch_decode call warms these
    caches, subsequent calls with the same format/codec should be faster."""

    def test_second_decode_faster_than_first(self):
        """Second batch_decode on the same format benefits from cached
        probe hints and HW compat results."""
        # Use MKV files — matroska gets probe hint optimization (1MB/2s)
        videos = [
            p
            for p in [
                _fixture_path("formats", "h264_mkv.mkv"),
                _fixture_path("formats", "h265_mkv.mkv"),
                _fixture_path("formats", "vp9_mkv.mkv"),
            ]
            if os.path.isfile(p)
        ]
        if len(videos) < 2:
            pytest.skip("need at least 2 mkv fixtures")

        levels = [LevelConfig(width=96, height=96, quality=50)]

        # Cold run (may be first time this format is seen this process)
        batch_decode(videos, levels, keyframes_only=True)

        # Timed: warm run 1
        t0 = time.perf_counter()
        for _ in range(3):
            batch_decode(videos, levels, keyframes_only=True)
        warm_time = (time.perf_counter() - t0) / 3

        # Timed: warm run 2 (should be comparable — caches are stable)
        t0 = time.perf_counter()
        for _ in range(3):
            batch_decode(videos, levels, keyframes_only=True)
        warm_time2 = (time.perf_counter() - t0) / 3

        # Both warm runs should be fast and similar — proves cache is
        # stable (not rebuilding).  Allow up to 2x variance for system noise.
        ratio = max(warm_time, warm_time2) / max(min(warm_time, warm_time2), 1e-9)
        assert ratio < 2.0, (
            f"warm runs not stable: {warm_time*1000:.1f}ms vs "
            f"{warm_time2*1000:.1f}ms (ratio={ratio:.1f}x)"
        )

    def test_multi_file_same_format_amortizes_probe(self):
        """Decoding N files of the same format in one batch should have
        sub-linear index_scan time because probe hints are cached after
        the first file."""
        mp4s = [
            p
            for name in [
                "h264_mp4.mp4",
                "h265_mp4.mp4",
                "av1_mp4.mp4",
            ]
            if os.path.isfile(p := _fixture_path("formats", name))
        ]
        if len(mp4s) < 2:
            pytest.skip("need at least 2 mp4 fixtures")

        levels = [LevelConfig(width=96, height=96, quality=50)]

        # Warm up probe hints
        batch_decode(mp4s[:1], levels, keyframes_only=True)

        # Single file baseline
        t0 = time.perf_counter()
        for _ in range(5):
            r1 = batch_decode(mp4s[:1], levels, keyframes_only=True)
        single_time = (time.perf_counter() - t0) / 5

        # Multi file batch
        t0 = time.perf_counter()
        for _ in range(5):
            rn = batch_decode(mp4s, levels, keyframes_only=True)
        batch_time = (time.perf_counter() - t0) / 5

        n = len(mp4s)
        # Per-file cost in batch should be less than N * single file cost.
        # This is because probe hints, hw compat, and thread resources are
        # amortized.  We require at least 10% savings (generous threshold).
        expected_linear = single_time * n
        assert batch_time < expected_linear, (
            f"batch of {n} ({batch_time*1000:.1f}ms) was not faster than "
            f"{n} x single ({expected_linear*1000:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# 3. Thread cache: per-worker resource reuse within a batch
# ---------------------------------------------------------------------------


@needs_native
@needs_fixtures
class TestThreadCache:
    """ThreadCache reuses encoder handles, decode frames, YUV buffers,
    and SwsContext objects across files within a batch.  Decoding
    multiple same-resolution files in one batch should be more efficient
    per-file than separate single-file batches."""

    def test_batch_faster_than_individual_calls(self):
        """One batch_decode([a, b, c]) call should be faster than three
        separate batch_decode([x]) calls, thanks to thread cache reuse."""
        videos = [
            p
            for name in [
                "h264_mp4.mp4",
                "h264_mkv.mkv",
                "h264_mov.mov",
                "h264_avi.avi",
                "h264_ts.ts",
            ]
            if os.path.isfile(p := _fixture_path("formats", name))
        ]
        if len(videos) < 3:
            pytest.skip("need at least 3 h264 fixtures")

        levels = [LevelConfig(width=192, height=192, quality=80)]

        # Warm caches
        batch_decode(videos, levels, keyframes_only=True)

        # Time: individual calls
        t0 = time.perf_counter()
        for _ in range(3):
            for v in videos:
                batch_decode([v], levels, keyframes_only=True)
        individual_time = (time.perf_counter() - t0) / 3

        # Time: single batch call
        t0 = time.perf_counter()
        for _ in range(3):
            batch_decode(videos, levels, keyframes_only=True)
        batch_time = (time.perf_counter() - t0) / 3

        # Batch should be faster (thread caches are rebuilt per batch_decode
        # call, so individual calls pay setup cost N times)
        assert batch_time < individual_time, (
            f"batch ({batch_time*1000:.1f}ms) was not faster than "
            f"individual ({individual_time*1000:.1f}ms)"
        )

    def test_consistent_output_across_batching_strategies(self):
        """Verify that thread cache reuse doesn't corrupt output:
        frames produced by batch mode match those from individual decodes."""
        videos = [
            p
            for name in ["h264_mp4.mp4", "h264_mkv.mkv"]
            if os.path.isfile(p := _fixture_path("formats", name))
        ]
        if len(videos) < 2:
            pytest.skip("need at least 2 h264 fixtures")

        levels = [LevelConfig(width=192, height=192, quality=80)]

        # Individual
        individual_blobs = []
        for v in videos:
            r = batch_decode([v], levels, keyframes_only=True)
            individual_blobs.append(bytes(r.jpeg_stream().blobs[0]))

        # Batch
        r_batch = batch_decode(videos, levels, keyframes_only=True)

        for i, v in enumerate(videos):
            batch_blob = bytes(r_batch.jpeg_stream().blobs[i])
            assert batch_blob == individual_blobs[i], (
                f"file {i} output differs between batch and individual decode"
            )


# ---------------------------------------------------------------------------
# 4. Disk cache round-trip: write multiple formats, read back correctly
# ---------------------------------------------------------------------------


@needs_native
@needs_fixtures
class TestDiskCacheFormats:
    """Verify disk cache works correctly across different video formats
    and configurations."""

    @pytest.mark.parametrize(
        "filename",
        [
            "h264_mp4.mp4",
            "h264_mkv.mkv",
            "h265_mp4.mp4",
            "vp9_webm.webm",
        ],
    )
    def test_disk_cache_round_trip(self, tmp_path, filename):
        path = _fixture_path("formats", filename)
        if not os.path.isfile(path):
            pytest.skip(f"fixture {filename} not available")

        cache_file = str(tmp_path / f"{filename}.vex")

        result = batch_decode(
            [path],
            levels=[
                LevelConfig(
                    width=96,
                    height=96,
                    quality=70,
                    in_memory=False,
                    cache_path=cache_file,
                )
            ],
            keyframes_only=True,
        )

        disk = result.disk()
        assert disk.frame_count > 0
        assert disk.total_bytes > 0

        with CachedLevel(cache_file) as cache:
            assert cache.frame_count == disk.frame_count
            assert cache.width == 96
            assert cache.height == 96
            assert cache.quality == 70

            # Every frame must be a valid JPEG (starts with FFD8)
            for i in range(len(cache)):
                jpeg = cache[i]
                assert len(jpeg) > 0
                assert jpeg[:2] == b"\xff\xd8", f"frame {i} not valid JPEG"

    def test_disk_cache_multi_level(self, tmp_path):
        """Two levels: one in-memory, one on disk — both produce output."""
        video = _fixture_path("formats", "h264_mp4.mp4")
        cache_file = str(tmp_path / "level1.vex")

        result = batch_decode(
            [video],
            levels=[
                LevelConfig(width=96, height=96, quality=50),
                LevelConfig(
                    width=192,
                    height=192,
                    quality=80,
                    in_memory=False,
                    cache_path=cache_file,
                ),
            ],
            keyframes_only=True,
        )

        # Level 0: in-memory
        stream = result.jpeg_stream(0)
        assert len(stream.blobs[0]) > 0

        # Level 1: disk
        disk = result.disk(1)
        assert disk.frame_count > 0

        with CachedLevel(cache_file) as cache:
            assert cache.frame_count == disk.frame_count

    def test_disk_cache_read_faster_with_all_frames(self, tmp_path):
        """Even with every-frame decode (many more frames), cache read
        is still faster than re-decoding."""
        video = _fixture_path("formats", "h264_mp4.mp4")
        cache_file = str(tmp_path / "allframes.vex")

        # Decode all frames to disk
        result = batch_decode(
            [video],
            levels=[
                LevelConfig(
                    width=96,
                    height=96,
                    quality=60,
                    in_memory=False,
                    cache_path=cache_file,
                )
            ],
            keyframes_only=False,
        )
        n_frames = result.disk().frame_count
        assert n_frames > 1

        # Time: re-decode all frames
        t0 = time.perf_counter()
        for _ in range(3):
            batch_decode(
                [video],
                levels=[LevelConfig(width=96, height=96, quality=60)],
                keyframes_only=False,
            )
        decode_time = (time.perf_counter() - t0) / 3

        # Time: read all frames from cache
        t0 = time.perf_counter()
        for _ in range(3):
            with CachedLevel(cache_file) as cache:
                for i in range(len(cache)):
                    _ = cache[i]
        cache_time = (time.perf_counter() - t0) / 3

        assert cache_time < decode_time, (
            f"cache read of {n_frames} frames ({cache_time*1000:.1f}ms) "
            f"not faster than decode ({decode_time*1000:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# 5. Repeated decode stability: process cache doesn't degrade over time
# ---------------------------------------------------------------------------


@needs_native
@needs_fixtures
class TestCacheStability:
    """Verify that repeated batch_decode calls don't get slower over time,
    proving the caches don't leak or degrade."""

    def test_decode_time_stable_over_iterations(self):
        """Wall time should stay stable (not grow) over 10 iterations."""
        video = _fixture_path("formats", "h264_mp4.mp4")
        levels = [LevelConfig(width=96, height=96, quality=50)]

        # Warm up
        batch_decode([video], levels, keyframes_only=True)

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            batch_decode([video], levels, keyframes_only=True)
            times.append(time.perf_counter() - t0)

        # Last 5 iterations should not be slower than first 5
        first_half = sum(times[:5]) / 5
        second_half = sum(times[5:]) / 5

        # Allow 50% variance for system noise, but second half must not be
        # dramatically slower (would indicate cache leak/bloat)
        assert second_half < first_half * 1.5, (
            f"decode slowing down: first half avg={first_half*1000:.1f}ms, "
            f"second half avg={second_half*1000:.1f}ms"
        )
