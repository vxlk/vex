#!/usr/bin/env python3
"""Test vex.probe() decode_threads field across codecs and edge cases."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import vex  # noqa: E402

FIXTURES = PROJECT_ROOT / "fixtures"

VIDEO_EXTENSIONS = (
    ".mp4", ".mkv", ".avi", ".ts", ".flv", ".mov", ".webm", ".ogv",
    ".mpg", ".vob", ".wmv", ".asf", ".3gp", ".3g2", ".nut", ".ivf",
    ".f4v", ".m4v", ".swf", ".rm", ".wtv", ".mxf", ".gxf", ".dv",
)

# Codecs known to support multi-threaded decoding (slice or frame threading).
# FFmpeg will assign thread_count > 1 for these when auto-detecting.
MULTITHREADED_CODECS = {"h264", "h265", "vp9", "mpeg2", "vp8", "theora", "mpeg1", "dv"}

# Codecs that are typically single-threaded in FFmpeg.
# probe().decode_threads returns 0 for these (active_thread_type == 0).
SINGLETHREADED_CODECS = {"mjpeg", "flv1", "rv20", "wmv1", "wmv2", "msmpeg4"}

# Codecs that may not be available in every FFmpeg build.
OPTIONAL_CODECS = {"av1"}


def _probe_threads(path: str) -> int:
    """Return decode_threads from vex.probe(), or 0 on failure.

    Returns 0 for files that FFmpeg can demux but aren't real video
    (e.g. image pipe demuxers with width=0, height=0).
    """
    try:
        p = vex.probe(path)
        if p.width == 0 and p.height == 0:
            return 0
        return p.decode_threads
    except Exception:
        return 0


def test_valid_files_return_nonnegative():
    """Every valid video fixture must return decode_threads >= 0."""
    failures = []
    count = 0
    for path in sorted(FIXTURES.rglob("*")):
        if not path.is_file() or path.suffix not in VIDEO_EXTENSIONS:
            continue
        if path.name in ("empty_file.mp4", "no_video_stream.mp4", "truncated.mp4"):
            continue
        codec = path.stem.lower().split("_")[0]
        if codec in OPTIONAL_CODECS:
            continue
        threads = _probe_threads(str(path))
        if threads < 0:
            failures.append(f"  {path.name}: got {threads}, expected >= 0")
        count += 1
    assert count > 0, "No fixture files found!"
    assert not failures, "Files returned negative thread count:\n" + "\n".join(failures)
    print(f"  PASS: {count} valid files all returned decode_threads >= 0")


def test_invalid_file_returns_zero():
    """Non-existent file must return 0."""
    assert _probe_threads("nonexistent_video.mp4") == 0
    print("  PASS: non-existent file returns 0")


def test_empty_file_returns_zero():
    """Empty file must return 0."""
    empty = FIXTURES / "edge_cases" / "empty_file.mp4"
    if empty.exists():
        assert _probe_threads(str(empty)) == 0
        print("  PASS: empty file returns 0")
    else:
        print("  SKIP: empty_file.mp4 not found")


def test_multithreaded_codecs():
    """Codecs known to support threading should return > 1 threads (with auto-detect)."""
    fmt_dir = FIXTURES / "formats"
    if not fmt_dir.exists():
        print("  SKIP: formats/ directory not found")
        return
    checked = 0
    for path in sorted(fmt_dir.iterdir()):
        if not path.is_file():
            continue
        codec = path.stem.lower().split("_")[0]
        if codec in OPTIONAL_CODECS:
            continue
        if codec in MULTITHREADED_CODECS:
            threads = _probe_threads(str(path))
            assert threads > 1, (
                f"{path.name} (codec={codec}) expected > 1 thread, got {threads}"
            )
            checked += 1
    assert checked > 0, "No multi-threaded codec fixtures found!"
    print(f"  PASS: {checked} multi-threaded codecs all use > 1 thread")


def test_singlethreaded_codecs():
    """Codecs without threading support should return 0 (active_thread_type == 0)."""
    fmt_dir = FIXTURES / "formats"
    if not fmt_dir.exists():
        print("  SKIP: formats/ directory not found")
        return
    checked = 0
    for path in sorted(fmt_dir.iterdir()):
        if not path.is_file():
            continue
        codec = path.stem.lower().split("_")[0]
        if codec not in SINGLETHREADED_CODECS:
            continue
        threads = _probe_threads(str(path))
        assert threads == 0, (
            f"{path.name} (codec={codec}) expected 0 (no threading), got {threads}"
        )
        checked += 1
    if checked > 0:
        print(f"  PASS: {checked} single-threaded codecs all return 0")
    else:
        print("  SKIP: no single-threaded codec fixtures found")


def test_probe_returns_all_fields():
    """probe() must populate codec_id, width, height, file_size, decode_threads."""
    fixture = FIXTURES / "formats" / "h264_mp4.mp4"
    if not fixture.exists():
        print("  SKIP: h264_mp4.mp4 not found")
        return
    p = vex.probe(str(fixture))
    assert p.codec_id > 0, f"codec_id should be > 0, got {p.codec_id}"
    assert p.width > 0, f"width should be > 0, got {p.width}"
    assert p.height > 0, f"height should be > 0, got {p.height}"
    assert p.file_size > 0, f"file_size should be > 0, got {p.file_size}"
    assert p.decode_threads > 1, f"h264 should use > 1 thread, got {p.decode_threads}"
    assert p.frame_count > 0, f"frame_count should be > 0, got {p.frame_count}"
    assert p.fps > 0, f"fps should be > 0, got {p.fps}"
    assert len(p.codec) > 0, f"codec name should be non-empty"
    print(
        f"  PASS: h264_mp4.mp4 -> codec_id={p.codec_id}, {p.width}x{p.height}, "
        f"file_size={p.file_size}, threads={p.decode_threads}"
    )


def test_non_video_returns_zero():
    """Non-video files (text, images, etc.) should return 0."""
    import tempfile

    cases = {
        "test.txt": b"hello world",
        "test.json": b'{"key": "value"}',
        "test.xml": b"<root><child/></root>",
        "test.csv": b"a,b,c\n1,2,3\n",
        "test.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,
        "test.bmp": b"BM" + b"\x00" * 62,
        "test.jpg": b"\xff\xd8\xff\xe0" + b"\x00" * 60,
        "test.webp": b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 52,
    }
    with tempfile.TemporaryDirectory() as td:
        for name, content in cases.items():
            fp = os.path.join(td, name)
            with open(fp, "wb") as f:
                f.write(content)
            result = _probe_threads(fp)
            assert result == 0, f"{name}: expected 0, got {result}"
    print(f"  PASS: {len(cases)} non-video files all return 0")


def test_thread_count_table():
    """Print a summary table of all fixtures and their thread counts."""
    print("\n  %-35s  %s" % ("Fixture", "Threads"))
    print("  " + "-" * 45)
    for path in sorted(FIXTURES.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(FIXTURES)
        threads = _probe_threads(str(path))
        print(f"  {str(rel):<35s}  {threads}")


if __name__ == "__main__":
    tests = [
        ("Valid files return >= 0", test_valid_files_return_nonnegative),
        ("Invalid file returns 0", test_invalid_file_returns_zero),
        ("Empty file returns 0", test_empty_file_returns_zero),
        ("Non-video files return 0", test_non_video_returns_zero),
        ("Multi-threaded codecs use > 1", test_multithreaded_codecs),
        ("Single-threaded codecs return 0", test_singlethreaded_codecs),
        ("probe() returns all fields", test_probe_returns_all_fields),
    ]

    print("vex.probe() decode_threads tests")
    print("=" * 50)
    failed = []
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
        except Exception as e:
            print(f"  FAIL: {e}")
            failed.append(name)

    test_thread_count_table()

    print("\n" + "=" * 50)
    if failed:
        print(f"FAILED ({len(failed)}/{len(tests)}): {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"ALL PASSED ({len(tests)}/{len(tests)})")
