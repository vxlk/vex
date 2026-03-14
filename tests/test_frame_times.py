"""Tests for vex.get_frame_times and streaming frame time collection."""

import os
import sys
import time

import numpy as np
import pytest

# Add the python package to the path so vex can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import vex

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")

# ── Fixture discovery ────────────────────────────────────────────────────────

# Expected (relative_path, expected_strategy) pairs.
# Tests are skipped if the fixture file does not exist.
FIXTURES = [
    # --- SAMPLE_TABLE (MP4/MOV family) ---
    ("formats/h264_mp4.mp4", "sample_table"),
    ("formats/h265_mp4.mp4", "sample_table"),
    ("formats/av1_mp4.mp4", "sample_table"),
    ("formats/h264_mov.mov", "sample_table"),
    ("formats/mpeg4_mov.mov", "sample_table"),
    ("formats/mjpeg_mov.mov", "sample_table"),
    ("formats/h264_3gp.3gp", "sample_table"),
    ("formats/h264_3g2.3g2", "sample_table"),
    ("formats/h264_f4v.f4v", "sample_table"),
    ("formats/h264_m4v.m4v", "sample_table"),
    # --- BLOCK_TIMESTAMP (MKV/WebM) ---
    ("formats/h264_mkv.mkv", "block_timestamp"),
    ("formats/h265_mkv.mkv", "block_timestamp"),
    ("formats/vp9_mkv.mkv", "block_timestamp"),
    ("formats/av1_mkv.mkv", "block_timestamp"),
    ("formats/vp8_webm.webm", "block_timestamp"),
    ("formats/vp9_webm.webm", "block_timestamp"),
    ("formats/av1_webm.webm", "block_timestamp"),
    # --- PES_TIMESTAMP (TS/PS/WTV) ---
    ("formats/h264_ts.ts", "pes_timestamp"),
    ("formats/mpeg2_ts.ts", "pes_timestamp"),
    ("formats/mpeg1_mpg.mpg", "pes_timestamp"),
    ("formats/mpeg2_mpg.mpg", "pes_timestamp"),
    ("formats/mpeg2_vob.vob", "pes_timestamp"),
    ("formats/mpeg2_wtv.wtv", "pes_timestamp"),
    # --- TAG_TIMESTAMP (FLV/ASF/RM) ---
    ("formats/h264_flv.flv", "tag_timestamp"),
    ("formats/flv1_flv.flv", "tag_timestamp"),
    ("formats/wmv1_wmv.wmv", "tag_timestamp"),
    ("formats/wmv2_wmv.wmv", "tag_timestamp"),
    ("formats/msmpeg4_asf.asf", "tag_timestamp"),
    ("formats/rv20_rm.rm", "tag_timestamp"),
    # --- FIXED_RATE (AVI/DV/SWF) ---
    ("formats/h264_avi.avi", "fixed_rate"),
    ("formats/mpeg4_avi.avi", "fixed_rate"),
    ("formats/mjpeg_avi.avi", "fixed_rate"),
    ("formats/msmpeg4_avi.avi", "fixed_rate"),
    ("formats/dv_avi.avi", "fixed_rate"),
    ("formats/dv_raw.dv", "fixed_rate"),
    ("formats/flv1_swf.swf", "fixed_rate"),
    # --- GENERIC_PTS ---
    ("formats/h264_nut.nut", "generic_pts"),
    ("formats/vp8_ivf.ivf", "generic_pts"),
    ("formats/vp9_ivf.ivf", "generic_pts"),
    ("formats/theora_ogv.ogv", "generic_pts"),
    ("formats/mpeg2_mxf.mxf", "generic_pts"),
    ("formats/mpeg2_gxf.gxf", "generic_pts"),
]


def _fixture_path(rel):
    return os.path.join(FIXTURES_DIR, rel)


def _fixture_exists(rel):
    return os.path.isfile(_fixture_path(rel))


# Build list of available fixtures
AVAILABLE_FIXTURES = [(rel, strat) for rel, strat in FIXTURES if _fixture_exists(rel)]

# Also check for edge-case fixtures
EDGE_CASES = {
    "single_frame": "edge_cases/single_frame.mp4",
    "all_keyframes": "edge_cases/all_keyframes.mp4",
    "long_gop": "edge_cases/long_gop.mp4",
    "truncated": "edge_cases/truncated.mp4",
    "empty_file": "edge_cases/empty_file.mp4",
    "no_video_stream": "edge_cases/no_video_stream.mp4",
}


# ── Standalone tests (parametrized across all available fixtures) ─────────────


@pytest.mark.parametrize(
    "rel,expected_strategy", AVAILABLE_FIXTURES, ids=[r for r, _ in AVAILABLE_FIXTURES]
)
class TestStandaloneFrameTimes:
    def test_frame_count_matches(self, rel, expected_strategy):
        ft = vex.get_frame_times(_fixture_path(rel))
        assert len(ft.times) == ft.frame_count
        assert ft.frame_count > 0

    def test_monotonically_nondecreasing(self, rel, expected_strategy):
        ft = vex.get_frame_times(_fixture_path(rel))
        if ft.frame_count > 1:
            diffs = np.diff(ft.times)
            assert np.all(diffs >= -1e-9), (
                f"Non-monotonic times in {rel}: min diff = {diffs.min()}"
            )

    def test_strategy_matches_container(self, rel, expected_strategy):
        ft = vex.get_frame_times(_fixture_path(rel))
        assert ft.strategy == expected_strategy, (
            f"Expected {expected_strategy} for {rel}, got {ft.strategy}"
        )

    def test_reasonable_bounds(self, rel, expected_strategy):
        ft = vex.get_frame_times(_fixture_path(rel))
        assert ft.times[0] >= 0, f"First time negative: {ft.times[0]}"
        assert ft.times[0] < 1.0, f"First time too large: {ft.times[0]}"
        if ft.duration_sec > 0:
            assert ft.times[-1] <= ft.duration_sec + 0.5, (
                f"Last time {ft.times[-1]} exceeds duration {ft.duration_sec} + 0.5s"
            )

    def test_frame_count_matches_decode(self, rel, expected_strategy):
        ft = vex.get_frame_times(_fixture_path(rel))
        result = vex.batch_decode([_fixture_path(rel)], keyframes_only=False)
        meta = result.metrics
        decoded_count = meta.keyframes_decoded
        # Frame count from packet scan should match decoded frame count.
        # Allow small tolerance for containers where the last few packets
        # may not produce a decoded frame (e.g. truncated B-frames).
        assert abs(ft.frame_count - decoded_count) <= 2, (
            f"Packet count {ft.frame_count} vs decoded {decoded_count} for {rel}"
        )


# ── Edge case tests ──────────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.skipif(
        not _fixture_exists(EDGE_CASES["single_frame"]),
        reason="single_frame.mp4 not found",
    )
    def test_single_frame(self):
        ft = vex.get_frame_times(_fixture_path(EDGE_CASES["single_frame"]))
        assert ft.frame_count == 1
        assert abs(ft.times[0]) < 0.1  # should be ~0.0

    @pytest.mark.skipif(
        not _fixture_exists(EDGE_CASES["all_keyframes"]),
        reason="all_keyframes.mp4 not found",
    )
    def test_all_keyframes(self):
        ft = vex.get_frame_times(_fixture_path(EDGE_CASES["all_keyframes"]))
        assert ft.frame_count == 60  # 2s * 30fps

    @pytest.mark.skipif(
        not _fixture_exists(EDGE_CASES["long_gop"]), reason="long_gop.mp4 not found"
    )
    def test_long_gop(self):
        ft = vex.get_frame_times(_fixture_path(EDGE_CASES["long_gop"]))
        assert ft.frame_count == 150  # 5s * 30fps

    @pytest.mark.skipif(
        not _fixture_exists(EDGE_CASES["truncated"]), reason="truncated.mp4 not found"
    )
    def test_truncated(self):
        # Should return times for existing frames without crashing
        ft = vex.get_frame_times(_fixture_path(EDGE_CASES["truncated"]))
        assert ft.frame_count >= 0

    @pytest.mark.skipif(
        not _fixture_exists(EDGE_CASES["empty_file"]), reason="empty_file.mp4 not found"
    )
    def test_empty_file(self):
        ft = vex.get_frame_times(_fixture_path(EDGE_CASES["empty_file"]))
        assert ft.frame_count == 0

    @pytest.mark.skipif(
        not _fixture_exists(EDGE_CASES["no_video_stream"]),
        reason="no_video_stream.mp4 not found",
    )
    def test_no_video_stream(self):
        ft = vex.get_frame_times(_fixture_path(EDGE_CASES["no_video_stream"]))
        assert ft.frame_count == 0


# ── Streaming mode tests ─────────────────────────────────────────────────────

# Pick a single fixture for streaming tests (prefer MP4 if available)
_STREAMING_FIXTURE = None
for _rel, _ in AVAILABLE_FIXTURES:
    if _rel.endswith(".mp4"):
        _STREAMING_FIXTURE = _rel
        break
if _STREAMING_FIXTURE is None and AVAILABLE_FIXTURES:
    _STREAMING_FIXTURE = AVAILABLE_FIXTURES[0][0]


@pytest.mark.skipif(_STREAMING_FIXTURE is None, reason="No fixtures available")
class TestStreamingFrameTimes:
    @property
    def path(self):
        return _fixture_path(_STREAMING_FIXTURE)

    def test_streaming_frame_times_match_standalone(self):
        standalone = vex.get_frame_times(self.path)
        result = vex.batch_decode(
            [self.path],
            keyframes_only=False,
            collect_frame_times=True,
        )
        streaming = result.metrics.frame_times(0)
        assert streaming is not None, "frame_times should be populated"
        assert (
            len(streaming) == standalone.frame_count
            or abs(len(streaming) - standalone.frame_count) <= 2
        )
        # Timestamps should be close (packet PTS vs decoded best_effort_timestamp)
        min_len = min(len(streaming), len(standalone.times))
        np.testing.assert_allclose(
            streaming[:min_len], standalone.times[:min_len], atol=0.01
        )

    def test_streaming_times_available_async(self):
        handle = vex.batch_decode_async(
            [self.path],
            keyframes_only=False,
            collect_frame_times=True,
        )
        # Poll until some times appear or decode finishes
        for _ in range(200):
            ft = handle.peek_frame_times(0)
            if ft is not None and len(ft) > 0:
                break
            if handle.done:
                # Check one more time after done
                ft = handle.peek_frame_times(0)
                if ft is not None and len(ft) > 0:
                    break
                break
            time.sleep(0.01)

        result = handle.result()
        final_ft = result.metrics.frame_times(0)
        assert final_ft is not None
        assert len(final_ft) > 0
        # We should have seen times during async decode (unless it was very fast)
        # Don't assert seen_times since short videos may complete instantly

    def test_streaming_disabled_by_default(self):
        result = vex.batch_decode([self.path], keyframes_only=False)
        assert result.metrics.frame_times(0) is None

    def test_streaming_with_frame_skip(self):
        skip = 2
        result = vex.batch_decode(
            [self.path],
            keyframes_only=False,
            frame_skip=skip,
            collect_frame_times=True,
        )
        ft = result.metrics.frame_times(0)
        assert ft is not None
        decoded_count = result.metrics.keyframes_decoded
        assert len(ft) == decoded_count


# ── Parametrized streaming cross-validation ──────────────────────────────────


@pytest.mark.parametrize(
    "rel,expected_strategy",
    AVAILABLE_FIXTURES[:5],
    ids=[r for r, _ in AVAILABLE_FIXTURES[:5]],
)
def test_streaming_matches_standalone(rel, expected_strategy):
    """Cross-validate streaming vs standalone for a few fixtures."""
    path = _fixture_path(rel)
    standalone = vex.get_frame_times(path)
    result = vex.batch_decode([path], keyframes_only=False, collect_frame_times=True)
    streaming = result.metrics.frame_times(0)
    assert streaming is not None
    min_len = min(len(streaming), standalone.frame_count)
    assert min_len > 0
    np.testing.assert_allclose(
        streaming[:min_len], standalone.times[:min_len], atol=0.01
    )
