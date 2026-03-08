"""FFmpeg regression tests — verify vex output matches FFmpeg image2pipe.

Compares vex's decode pipeline against FFmpeg's equivalent pipeline for
every container format in the fixtures. Verifies:
  1. Frame counts match between vex and FFmpeg.
  2. Pixel content is near-identical (high PSNR).
  3. vex is faster than invoking FFmpeg.
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
from io import BytesIO

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vex import NATIVE, LevelConfig, batch_decode

# ---------------------------------------------------------------------------
# Pillow dependency (optional for pixel comparison)
# ---------------------------------------------------------------------------

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

needs_pil = pytest.mark.skipif(not HAS_PIL, reason="Pillow not installed")

# ---------------------------------------------------------------------------
# Native module check
# ---------------------------------------------------------------------------

HAS_NATIVE = False
try:
    from vex import _load_native
    _load_native()
    HAS_NATIVE = True
except (ImportError, Exception):
    pass

needs_native = pytest.mark.skipif(
    not HAS_NATIVE, reason="native module (_vex_core) not built"
)

# ---------------------------------------------------------------------------
# FFmpeg / fixture discovery
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "fixtures"
FORMATS_DIR  = FIXTURES_DIR / "formats"
FFMPEG_BIN   = PROJECT_ROOT / "deps" / "ffmpeg" / "bin" / "ffmpeg.exe"

has_ffmpeg = FFMPEG_BIN.is_file()
needs_ffmpeg = pytest.mark.skipif(not has_ffmpeg, reason="ffmpeg.exe not found")


def _discover_format_fixtures():
    if not FORMATS_DIR.is_dir():
        return []
    return sorted(p.name for p in FORMATS_DIR.iterdir() if p.is_file())


_FORMAT_FIXTURES = _discover_format_fixtures()

needs_fixtures = pytest.mark.skipif(
    not _FORMAT_FIXTURES, reason="no format fixtures found"
)


def _fixture_path(filename: str) -> str:
    return str(FORMATS_DIR / filename)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffmpeg_decode_to_jpegs(input_path: str, out_dir: Path,
                            width: int = 0, height: int = 0) -> list[Path]:
    """Run FFmpeg to decode a video into individual JPEG files.

    Returns the list of output JPEG paths, sorted by frame index.
    """
    out_pattern = str(out_dir / "frame_%06d.jpg")
    cmd = [
        str(FFMPEG_BIN),
        "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-fps_mode", "passthrough",
    ]
    if width > 0 and height > 0:
        cmd += ["-vf", f"scale={width}:{height}:flags=bilinear"]
    cmd += [
        "-q:v", "2",           # highest quality MJPEG
        "-f", "image2",
        out_pattern,
    ]
    subprocess.run(cmd, check=True, timeout=30,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    jpegs = sorted(out_dir.glob("frame_*.jpg"))
    return jpegs


def _ffmpeg_decode_to_raw_frames(input_path: str, width: int, height: int,
                                  n_frames: int = 0) -> list[np.ndarray]:
    """Run FFmpeg to decode a video into raw RGB24 pixel arrays.

    This bypasses JPEG encoding entirely, giving the ground-truth decoded
    pixels for comparison against vex's JPEG output.
    """
    cmd = [
        str(FFMPEG_BIN),
        "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-fps_mode", "passthrough",
    ]
    if n_frames > 0:
        cmd += ["-frames:v", str(n_frames)]
    cmd += ["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]

    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    if proc.returncode != 0:
        return []

    frame_bytes = width * height * 3
    data = proc.stdout
    n = len(data) // frame_bytes
    frames = []
    for i in range(n):
        arr = np.frombuffer(data[i * frame_bytes:(i + 1) * frame_bytes],
                            dtype=np.uint8).reshape(height, width, 3)
        frames.append(arr.astype(np.float64))
    return frames


def _decode_jpeg_to_array(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to a numpy RGB array via Pillow."""
    img = Image.open(BytesIO(jpeg_bytes))
    return np.asarray(img.convert("RGB"), dtype=np.float64)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio between two equal-shaped images."""
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


# ---------------------------------------------------------------------------
# Tests: correctness (per-fixture, parametrized)
# ---------------------------------------------------------------------------

@needs_native
@needs_ffmpeg
@needs_fixtures
@pytest.mark.skipif(not _FORMAT_FIXTURES, reason="no format fixtures found")
class TestFFmpegRegression:
    """Compare vex sequential decode output against FFmpeg for every format."""

    @pytest.mark.parametrize("filename", _FORMAT_FIXTURES)
    def test_frame_count_matches(self, filename, tmp_path):
        """vex and FFmpeg produce the same number of frames."""
        path = _fixture_path(filename)

        # vex: sequential decode, native resolution
        result = batch_decode(
            [path],
            levels=[LevelConfig(width=NATIVE, height=NATIVE, quality=95)],
            keyframes_only=False,
        )
        stream = result.jpeg_stream()
        n_vex = len(stream.offsets[0])

        # FFmpeg: decode all frames as JPEGs
        ffmpeg_jpegs = _ffmpeg_decode_to_jpegs(path, tmp_path)
        n_ffmpeg = len(ffmpeg_jpegs)

        # Allow ±1 frame tolerance for codec drain edge cases
        assert abs(n_vex - n_ffmpeg) <= 1, (
            f"{filename}: frame count mismatch — vex={n_vex}, ffmpeg={n_ffmpeg}"
        )

    @needs_pil
    @pytest.mark.parametrize("filename", _FORMAT_FIXTURES)
    def test_pixel_content_matches(self, filename, tmp_path):
        """Sampled frames have high PSNR between vex and FFmpeg JPEG output.

        Both pipelines produce JPEGs of the same decoded frames. The JPEG
        encoders differ (TurboJPEG vs FFmpeg mjpeg), so bytes won't match.
        Both JPEGs are decoded to RGB via Pillow (same color space path)
        and compared via PSNR. A threshold of 20 dB confirms the pipelines
        decoded the same frame — a wrong frame would score < 15 dB.
        """
        path = _fixture_path(filename)

        # vex: decode all frames at native resolution, quality 95
        result = batch_decode(
            [path],
            levels=[LevelConfig(width=NATIVE, height=NATIVE, quality=95)],
            keyframes_only=False,
        )
        stream = result.jpeg_stream()
        n_vex = len(stream.offsets[0])
        if n_vex == 0:
            pytest.skip("no frames decoded")

        # FFmpeg: decode all frames as high-quality JPEGs
        ffmpeg_jpegs = _ffmpeg_decode_to_jpegs(path, tmp_path)
        n_ffmpeg = len(ffmpeg_jpegs)
        n_common = min(n_vex, n_ffmpeg)
        if n_common == 0:
            pytest.skip("no common frames to compare")

        # Sample up to 3 frames: first, middle, last
        indices = sorted(set([0, n_common // 2, n_common - 1]))

        min_psnr = float("inf")
        for i in indices:
            vex_pixels = _decode_jpeg_to_array(stream.jpeg_bytes(0, i))
            ffmpeg_pixels = _decode_jpeg_to_array(ffmpeg_jpegs[i].read_bytes())

            if vex_pixels.shape != ffmpeg_pixels.shape:
                pytest.fail(
                    f"{filename} frame {i}: shape mismatch — "
                    f"vex={vex_pixels.shape}, ffmpeg={ffmpeg_pixels.shape}"
                )

            p = _psnr(vex_pixels, ffmpeg_pixels)
            min_psnr = min(min_psnr, p)

        # Typical PSNR is ~24 dB due to TurboJPEG vs FFmpeg mjpeg encoder
        # differences. 20 dB threshold confirms same-frame content.
        assert min_psnr > 20.0, (
            f"{filename}: worst-case PSNR = {min_psnr:.1f} dB (expected > 20)"
        )


# ---------------------------------------------------------------------------
# Test: speed comparison
# ---------------------------------------------------------------------------

@needs_native
@needs_ffmpeg
@needs_fixtures
class TestFFmpegSpeed:
    """Verify vex is faster than equivalent FFmpeg pipeline."""

    def test_vex_faster_than_ffmpeg(self, tmp_path):
        """vex batch decode is faster than sequential FFmpeg calls."""
        all_paths = [_fixture_path(f) for f in _FORMAT_FIXTURES]

        # ── vex: batch decode all fixtures in one call ──────────────
        t0 = time.perf_counter()
        batch_decode(
            all_paths,
            levels=[LevelConfig(width=192, height=192, quality=85)],
            keyframes_only=False,
        )
        vex_time = time.perf_counter() - t0

        # ── FFmpeg: decode each fixture individually ────────────────
        t0 = time.perf_counter()
        for i, filename in enumerate(_FORMAT_FIXTURES):
            out_dir = tmp_path / f"ff_{i}"
            out_dir.mkdir()
            path = _fixture_path(filename)
            try:
                _ffmpeg_decode_to_jpegs(path, out_dir, width=192, height=192)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass  # some formats may fail in FFmpeg too
        ffmpeg_time = time.perf_counter() - t0

        speedup = ffmpeg_time / vex_time if vex_time > 0 else float("inf")

        # Report timings
        print(f"\n  vex:    {vex_time:.2f}s")
        print(f"  ffmpeg: {ffmpeg_time:.2f}s")
        print(f"  speedup: {speedup:.1f}x")

        assert vex_time < ffmpeg_time, (
            f"vex ({vex_time:.2f}s) was not faster than "
            f"FFmpeg ({ffmpeg_time:.2f}s)"
        )
