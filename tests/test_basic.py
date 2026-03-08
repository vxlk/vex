"""Basic tests for the vex Python layer.

Tests are split into two groups:
  - Pure-Python tests that run without the C++ extension module.
  - Integration tests that require _vex_core to be built (auto-skipped otherwise).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure the python/vex package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vex import (
    NATIVE, LevelConfig, BatchResult, CachedLevel,
    batch_decode, batch_decode_async,
)

# Check whether the native module is available
HAS_NATIVE = False
try:
    from vex import _load_native
    _load_native()
    HAS_NATIVE = True
except (ImportError, Exception):
    pass

# Project root for locating fixture files
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "fixtures"


# ---------------------------------------------------------------------------
# Pure-Python tests (no native module required)
# ---------------------------------------------------------------------------

class TestLevelConfig:
    """Tests for LevelConfig dataclass."""

    def test_level_config_defaults(self):
        cfg = LevelConfig()
        assert cfg.width == NATIVE
        assert cfg.height == NATIVE
        assert cfg.quality == 100
        assert cfg.output == "jpeg_stream"
        assert cfg.in_memory is True
        assert cfg.cache_path is None
        assert cfg.atlas_columns == 10

    def test_level_config_custom(self):
        cfg = LevelConfig(
            width=640,
            height=480,
            quality=70,
            output="sprite_atlas",
            in_memory=False,
            cache_path="/tmp/test.vex",
            atlas_columns=5,
        )
        assert cfg.width == 640
        assert cfg.height == 480
        assert cfg.quality == 70
        assert cfg.output == "sprite_atlas"
        assert cfg.in_memory is False
        assert cfg.cache_path == "/tmp/test.vex"
        assert cfg.atlas_columns == 5


class TestCachedLevel:
    """Tests for CachedLevel cache reader."""

    def test_cached_level_missing_file(self):
        with pytest.raises(FileNotFoundError):
            CachedLevel("/nonexistent/path/does_not_exist.vex")

    def test_cached_level_too_small(self, tmp_path):
        tiny = tmp_path / "tiny.vex"
        tiny.write_bytes(b"\x00" * 10)
        with pytest.raises(ValueError, match="too small"):
            CachedLevel(str(tiny))

    def test_cached_level_bad_magic(self, tmp_path):
        bad = tmp_path / "bad_magic.vex"
        # Write 64 bytes of zeros (wrong magic)
        bad.write_bytes(b"\x00" * 64)
        with pytest.raises(ValueError, match="Invalid cache magic"):
            CachedLevel(str(bad))


class TestOutputFormatValidation:
    """Tests for output format validation."""

    def test_valid_formats(self):
        # These should not raise
        cfg_jpeg = LevelConfig(output="jpeg_stream")
        assert cfg_jpeg.output == "jpeg_stream"

        cfg_atlas = LevelConfig(output="sprite_atlas")
        assert cfg_atlas.output == "sprite_atlas"

    def test_format_used_in_config_conversion(self):
        from vex import _VALID_OUTPUTS
        assert "jpeg_stream" in _VALID_OUTPUTS
        assert "sprite_atlas" in _VALID_OUTPUTS


class TestDecodeMetrics:
    """Tests for DecodeMetrics wrapper."""

    def test_default_metrics(self):
        from vex import DecodeMetrics
        m = DecodeMetrics()
        assert m.total_wall_us == 0
        assert m.decode_fps == 0.0
        assert m.pipeline_fps == 0.0

    def test_metrics_fps(self):
        from vex import DecodeMetrics
        m = DecodeMetrics(
            total_wall_us=1_000_000,  # 1 second
            decode_us=500_000,        # 0.5 seconds
            keyframes_decoded=100,
        )
        assert m.pipeline_fps == pytest.approx(100.0)
        assert m.decode_fps == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Integration tests (require native module + fixtures)
# ---------------------------------------------------------------------------

def _fixture_path(*parts: str) -> str:
    """Return absolute path to a fixture file."""
    return str(FIXTURES_DIR.joinpath(*parts))


def _has_fixture(*parts: str) -> bool:
    """Check if a fixture file exists."""
    return FIXTURES_DIR.joinpath(*parts).is_file()


needs_native = pytest.mark.skipif(
    not HAS_NATIVE, reason="native module (_vex_core) not built"
)

needs_fixtures = pytest.mark.skipif(
    not _has_fixture("formats", "h264_mp4.mp4"),
    reason="test fixtures not generated (run tools/generate_fixtures.py)",
)


def _discover_format_fixtures():
    """Discover all fixture files in the formats/ directory."""
    formats_dir = FIXTURES_DIR / "formats"
    if not formats_dir.is_dir():
        return []
    return sorted(p.name for p in formats_dir.iterdir() if p.is_file())


_FORMAT_FIXTURES = _discover_format_fixtures()


@needs_native
@pytest.mark.skipif(not _FORMAT_FIXTURES, reason="no format fixtures found")
class TestKeyframeDecode:
    """Verify keyframe-only decode works for every container format."""

    @pytest.mark.parametrize("filename", _FORMAT_FIXTURES)
    def test_keyframe_decode(self, filename):
        path = _fixture_path("formats", filename)
        results, metrics = batch_decode(
            [path],
            levels=[LevelConfig(width=192, height=192, quality=85)],
            keyframes_only=True,
        )

        assert metrics.files_processed == 1
        assert metrics.keyframes_decoded > 0
        assert len(results) == 1

        from vex import JpegStreamResult
        assert isinstance(results[0], JpegStreamResult)
        assert len(results[0].blobs) == 1
        assert len(results[0].blobs[0]) > 0


@needs_native
@pytest.mark.skipif(not _FORMAT_FIXTURES, reason="no format fixtures found")
class TestSequentialDecode:
    """Verify sequential (every-frame) decode works for every container format."""

    @pytest.mark.parametrize("filename", _FORMAT_FIXTURES)
    def test_sequential_decode(self, filename):
        path = _fixture_path("formats", filename)
        results, metrics = batch_decode(
            [path],
            levels=[LevelConfig(width=192, height=192, quality=85)],
            keyframes_only=False,
        )

        assert metrics.files_processed == 1
        assert metrics.keyframes_decoded > 0
        assert len(results) == 1

        from vex import JpegStreamResult
        assert isinstance(results[0], JpegStreamResult)
        assert len(results[0].blobs) == 1
        assert len(results[0].blobs[0]) > 0

    @pytest.mark.parametrize("filename", _FORMAT_FIXTURES)
    def test_sequential_decodes_more_than_keyframes(self, filename):
        """Sequential decode should produce >= as many frames as keyframe-only."""
        path = _fixture_path("formats", filename)
        levels = [LevelConfig(width=192, height=192, quality=85)]
        _, kf_metrics = batch_decode([path], levels=levels, keyframes_only=True)
        _, seq_metrics = batch_decode([path], levels=levels, keyframes_only=False)

        assert seq_metrics.keyframes_decoded >= kf_metrics.keyframes_decoded

    @pytest.mark.parametrize("filename", _FORMAT_FIXTURES)
    def test_frame_skip(self, filename):
        """frame_skip=2 should produce roughly half the frames of skip=1."""
        path = _fixture_path("formats", filename)
        levels = [LevelConfig(width=192, height=192, quality=85)]
        _, all_metrics = batch_decode([path], levels=levels, keyframes_only=False, frame_skip=1)
        _, skip_metrics = batch_decode([path], levels=levels, keyframes_only=False, frame_skip=2)

        assert skip_metrics.keyframes_decoded <= all_metrics.keyframes_decoded
        assert skip_metrics.keyframes_decoded > 0


@needs_native
@needs_fixtures
class TestBatchDecode:
    """Integration tests for synchronous batch_decode."""

    def test_batch_decode_single_file(self):
        path = _fixture_path("formats", "h264_mp4.mp4")
        results, metrics = batch_decode(
            [path],
            levels=[LevelConfig(width=192, height=192, quality=85)],
        )

        # Should return one level result
        assert len(results) == 1
        result = results[0]

        # Default level produces JpegStreamResult
        from vex import JpegStreamResult
        assert isinstance(result, JpegStreamResult)

        # Should have decoded at least one keyframe
        assert metrics.keyframes_decoded > 0
        assert metrics.files_processed == 1
        assert metrics.total_wall_us > 0

        # Blobs list should have one entry (one source file)
        assert len(result.blobs) == 1
        assert len(result.blobs[0]) > 0  # non-empty JPEG data

    def test_batch_decode_multi_level(self):
        path = _fixture_path("formats", "h264_mp4.mp4")
        levels = [
            LevelConfig(width=96, height=96, quality=60),
            LevelConfig(width=320, height=240, quality=90),
        ]
        results, metrics = batch_decode([path], levels)

        # Should return two level results
        assert len(results) == 2
        assert metrics.keyframes_decoded > 0

        # Both levels should have produced output
        for result in results:
            from vex import JpegStreamResult
            assert isinstance(result, JpegStreamResult)
            assert len(result.blobs) == 1
            assert len(result.blobs[0]) > 0


@needs_native
@needs_fixtures
class TestBatchResult:
    """Tests for BatchResult typed accessors."""

    def test_returns_batch_result(self):
        path = _fixture_path("formats", "h264_mp4.mp4")
        result = batch_decode([path], levels=[LevelConfig(width=192, height=192, quality=85)])
        assert isinstance(result, BatchResult)
        assert result.metrics.keyframes_decoded > 0

    def test_typed_jpeg_stream_accessor(self):
        path = _fixture_path("formats", "h264_mp4.mp4")
        result = batch_decode([path], levels=[LevelConfig(width=192, height=192, quality=85)])
        from vex import JpegStreamResult
        stream = result.jpeg_stream()  # level 0 (default)
        assert isinstance(stream, JpegStreamResult)
        assert len(stream.blobs) == 1
        assert len(stream.blobs[0]) > 0

    def test_typed_sprite_atlas_accessor(self):
        path = _fixture_path("formats", "h264_mp4.mp4")
        result = batch_decode(
            [path],
            levels=[LevelConfig(width=48, height=48, quality=30,
                                output="sprite_atlas")],
        )
        from vex import SpriteAtlasResult
        atlas = result.sprite_atlas()
        assert isinstance(atlas, SpriteAtlasResult)
        assert atlas.thumb_w == 48
        assert atlas.thumb_h == 48

    def test_typed_accessor_wrong_type_raises(self):
        path = _fixture_path("formats", "h264_mp4.mp4")
        result = batch_decode([path], levels=[LevelConfig(width=192, height=192, quality=85)])
        with pytest.raises(TypeError, match="not SpriteAtlasResult"):
            result.sprite_atlas(0)
        with pytest.raises(TypeError, match="not DiskResult"):
            result.disk(0)

    def test_typed_accessor_index_error(self):
        path = _fixture_path("formats", "h264_mp4.mp4")
        result = batch_decode([path], levels=[LevelConfig(width=192, height=192, quality=85)])
        with pytest.raises(IndexError):
            result.jpeg_stream(99)

    def test_tuple_unpacking_still_works(self):
        path = _fixture_path("formats", "h264_mp4.mp4")
        results, metrics = batch_decode([path], levels=[LevelConfig(width=192, height=192, quality=85)])
        assert isinstance(results, list)
        assert metrics.keyframes_decoded > 0
        from vex import JpegStreamResult
        assert isinstance(results[0], JpegStreamResult)

    def test_index_access(self):
        path = _fixture_path("formats", "h264_mp4.mp4")
        result = batch_decode(
            [path],
            levels=[
                LevelConfig(width=96, height=96, quality=60),
                LevelConfig(width=192, height=192, quality=85),
            ],
        )
        from vex import JpegStreamResult
        assert isinstance(result[0], JpegStreamResult)
        assert isinstance(result[1], JpegStreamResult)

    def test_multi_level_typed_access(self):
        path = _fixture_path("formats", "h264_mp4.mp4")
        result = batch_decode(
            [path],
            levels=[
                LevelConfig(width=48, height=48, quality=30,
                            output="sprite_atlas"),
                LevelConfig(width=192, height=192, quality=85),
            ],
        )
        from vex import SpriteAtlasResult, JpegStreamResult
        atlas = result.sprite_atlas(0)
        stream = result.jpeg_stream(1)
        assert isinstance(atlas, SpriteAtlasResult)
        assert isinstance(stream, JpegStreamResult)


@needs_native
@needs_fixtures
class TestNativeResolution:
    """Tests for NATIVE (0) width/height sentinel."""

    def test_native_resolution_matches_source(self):
        """width=NATIVE, height=NATIVE should produce full source-res JPEGs."""
        path = _fixture_path("resolutions", "640x480.mp4")
        results, metrics = batch_decode(
            [path],
            levels=[LevelConfig(width=NATIVE, height=NATIVE, quality=85)],
        )
        assert metrics.keyframes_decoded > 0
        result = results[0]
        from vex import JpegStreamResult
        assert isinstance(result, JpegStreamResult)

        # Verify the JPEG is actually 640x480 by reading the SOF0 marker
        jpeg = result.jpeg_bytes(0, 0)
        w, h = _jpeg_dimensions(jpeg)
        assert w == 640
        assert h == 480

    def test_native_mixed_resolutions(self):
        """NATIVE with files of different resolutions should respect each."""
        path_small = _fixture_path("resolutions", "192x192.mp4")
        path_large = _fixture_path("resolutions", "640x480.mp4")
        results, metrics = batch_decode(
            [path_small, path_large],
            levels=[LevelConfig(width=NATIVE, height=NATIVE, quality=80)],
        )
        assert metrics.keyframes_decoded > 0
        stream = results[0]

        jpeg_small = stream.jpeg_bytes(0, 0)
        w_s, h_s = _jpeg_dimensions(jpeg_small)
        assert w_s == 192
        assert h_s == 192

        jpeg_large = stream.jpeg_bytes(1, 0)
        w_l, h_l = _jpeg_dimensions(jpeg_large)
        assert w_l == 640
        assert h_l == 480

    def test_native_with_explicit_level(self):
        """NATIVE and explicit levels can coexist in the same decode pass."""
        path = _fixture_path("resolutions", "640x480.mp4")
        results, metrics = batch_decode(
            [path],
            levels=[
                LevelConfig(width=NATIVE, height=NATIVE, quality=90),
                LevelConfig(width=96, height=96, quality=50),
            ],
        )
        assert len(results) == 2
        # Level 0: native → 640x480
        w0, h0 = _jpeg_dimensions(results[0].jpeg_bytes(0, 0))
        assert (w0, h0) == (640, 480)
        # Level 1: explicit → 96x96
        w1, h1 = _jpeg_dimensions(results[1].jpeg_bytes(0, 0))
        assert (w1, h1) == (96, 96)

    def test_native_rejected_for_sprite_atlas(self):
        """sprite_atlas output must not accept width=NATIVE."""
        with pytest.raises(ValueError, match="sprite_atlas requires explicit"):
            batch_decode(
                [_fixture_path("formats", "h264_mp4.mp4")],
                levels=[LevelConfig(width=NATIVE, height=NATIVE,
                                    output="sprite_atlas")],
            )


def _jpeg_dimensions(data: bytes) -> tuple:
    """Extract (width, height) from a JPEG's SOF marker."""
    i = 0
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xD8:  # SOI
            i += 2
            continue
        if marker == 0x00 or (0xD0 <= marker <= 0xD7):  # stuffed byte or RST
            i += 2
            continue
        if i + 3 >= len(data):
            break
        seg_len = (data[i + 2] << 8) | data[i + 3]
        # SOF0..SOF3 carry the dimensions
        if 0xC0 <= marker <= 0xC3:
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            return w, h
        i += 2 + seg_len
    raise ValueError("SOF marker not found in JPEG data")


@needs_native
@needs_fixtures
class TestBatchDecodeAsync:
    """Integration tests for asynchronous batch_decode_async."""

    def test_batch_decode_async(self):
        path = _fixture_path("formats", "h264_mp4.mp4")
        handle = batch_decode_async([path], levels=[LevelConfig(width=192, height=192, quality=85)])

        # Handle should exist and eventually complete
        assert handle is not None

        # Get results (blocks until done)
        results, metrics = handle.result()

        # Should be done after result() returns
        assert handle.done is True

        # Same checks as synchronous
        assert len(results) == 1
        assert metrics.keyframes_decoded > 0
        assert metrics.files_processed == 1
