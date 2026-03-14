#!/usr/bin/env python3
"""Test vex.probe_decode_threads and vex.probe_batch_threads."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from vex import probe_batch_threads, probe_decode_threads  # noqa: E402

FIXTURES = PROJECT_ROOT / "fixtures"

# Codecs known to support multi-threaded decoding (slice or frame threading).
# FFmpeg will assign thread_count > 1 for these when auto-detecting.
MULTITHREADED_CODECS = {"h264", "h265", "vp9", "mpeg2", "vp8", "theora", "mpeg1", "dv"}

# Codecs that are typically single-threaded in FFmpeg.
SINGLETHREADED_CODECS = {"mjpeg", "flv1", "rv20", "wmv1", "wmv2", "msmpeg4"}

# Codecs that may not be available in every FFmpeg build (returns 0 = codec not found).
OPTIONAL_CODECS = {"av1"}


def test_valid_files_return_positive():
    """Every valid video fixture must return thread count >= 1."""
    failures = []
    count = 0
    for path in sorted(FIXTURES.rglob("*")):
        if (
            not path.is_file()
            or path.suffix == ".mp4"
            and path.name == "empty_file.mp4"
        ):
            continue
        if path.suffix not in (
            ".mp4",
            ".mkv",
            ".avi",
            ".ts",
            ".flv",
            ".mov",
            ".webm",
            ".ogv",
            ".mpg",
            ".vob",
            ".wmv",
            ".asf",
            ".3gp",
            ".3g2",
            ".nut",
            ".ivf",
            ".f4v",
            ".m4v",
            ".swf",
            ".rm",
            ".wtv",
            ".mxf",
            ".gxf",
            ".dv",
        ):
            continue
        if path.name in ("empty_file.mp4", "no_video_stream.mp4", "truncated.mp4"):
            continue
        # Skip codecs that may not be in this FFmpeg build
        codec = path.stem.lower().split("_")[0]
        if codec in OPTIONAL_CODECS:
            continue
        threads = probe_decode_threads(str(path))
        if threads < 1:
            failures.append(f"  {path.name}: got {threads}, expected >= 1")
        count += 1
    assert count > 0, "No fixture files found!"
    assert not failures, "Files returned thread count < 1:\n" + "\n".join(failures)
    print(f"  PASS: {count} valid files all returned thread count >= 1")


def test_invalid_file_returns_zero():
    """Non-existent file must return 0."""
    assert probe_decode_threads("nonexistent_video.mp4") == 0
    print("  PASS: non-existent file returns 0")


def test_empty_file_returns_zero():
    """Empty file must return 0."""
    empty = FIXTURES / "edge_cases" / "empty_file.mp4"
    if empty.exists():
        assert probe_decode_threads(str(empty)) == 0
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
    results = []
    for path in sorted(fmt_dir.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem.lower()
        codec = stem.split("_")[0]
        threads = probe_decode_threads(str(path))
        results.append((path.name, codec, threads))
        if codec in OPTIONAL_CODECS:
            continue
        if codec in MULTITHREADED_CODECS:
            assert threads > 1, (
                f"{path.name} (codec={codec}) expected > 1 thread, got {threads}"
            )
            checked += 1
    assert checked > 0, "No multi-threaded codec fixtures found!"
    print(f"  PASS: {checked} multi-threaded codecs all use > 1 thread")


def test_singlethreaded_codecs():
    """Codecs known to be single-threaded should return exactly 1."""
    fmt_dir = FIXTURES / "formats"
    if not fmt_dir.exists():
        print("  SKIP: formats/ directory not found")
        return
    checked = 0
    for path in sorted(fmt_dir.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem.lower()
        codec = stem.split("_")[0]
        if codec not in SINGLETHREADED_CODECS:
            continue
        threads = probe_decode_threads(str(path))
        assert threads == 1, (
            f"{path.name} (codec={codec}) expected 1 thread, got {threads}"
        )
        checked += 1
    if checked > 0:
        print(f"  PASS: {checked} single-threaded codecs all use exactly 1 thread")
    else:
        print("  SKIP: no single-threaded codec fixtures found")


def test_explicit_thread_cap():
    """Passing decode_threads=1 should force single-threaded decode."""
    fixture = FIXTURES / "formats" / "h264_mp4.mp4"
    if not fixture.exists():
        print("  SKIP: h264_mp4.mp4 not found")
        return
    auto_threads = probe_decode_threads(str(fixture))
    capped = probe_decode_threads(str(fixture), decode_threads=1)
    assert auto_threads > 1, f"h264 auto should be > 1, got {auto_threads}"
    assert capped == 1, f"decode_threads=1 should give 1, got {capped}"
    print(f"  PASS: h264 auto={auto_threads}, capped to 1 with decode_threads=1")


def test_thread_count_table():
    """Print a summary table of all fixtures and their thread counts."""
    print("\n  %-35s  %s" % ("Fixture", "Threads"))
    print("  " + "-" * 45)
    for path in sorted(FIXTURES.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(FIXTURES)
        threads = probe_decode_threads(str(path))
        print(f"  {str(rel):<35s}  {threads}")


def test_batch_directory():
    """probe_batch_threads with a directory should sum threads, capped to max."""
    fmt_dir = FIXTURES / "formats"
    if not fmt_dir.exists():
        print("  SKIP: formats/ directory not found")
        return
    result = probe_batch_threads(str(fmt_dir))
    # Many multi-threaded codecs in there, sum will exceed 8
    assert result == 8, f"expected 8 (default cap), got {result}"
    # With a low cap
    result_low = probe_batch_threads(str(fmt_dir), max_threads=2)
    assert result_low == 2, f"expected 2, got {result_low}"
    print(f"  PASS: directory default={result}, max_threads=2 -> {result_low}")


def test_batch_file_list():
    """probe_batch_threads with a list of files."""
    fmt_dir = FIXTURES / "formats"
    if not fmt_dir.exists():
        print("  SKIP: formats/ directory not found")
        return
    # Single single-threaded file -> 1
    single = [str(fmt_dir / "mjpeg_avi.avi")]
    assert probe_batch_threads(single) == 1, "single mjpeg should be 1"
    # Two single-threaded files -> 2
    two = [str(fmt_dir / "mjpeg_avi.avi"), str(fmt_dir / "flv1_flv.flv")]
    assert probe_batch_threads(two) == 2, "two single-threaded should be 2"
    # One multi-threaded file exceeding cap
    multi = [str(fmt_dir / "h264_mp4.mp4")]
    result = probe_batch_threads(multi, max_threads=4)
    assert result == 4, f"single h264 capped to 4, got {result}"
    print(f"  PASS: list [1]={1}, [1,1]={2}, h264 capped={result}")


def test_batch_single_file_string():
    """probe_batch_threads with a single file path string (not a list)."""
    fixture = FIXTURES / "formats" / "mjpeg_avi.avi"
    if not fixture.exists():
        print("  SKIP: mjpeg_avi.avi not found")
        return
    result = probe_batch_threads(str(fixture))
    assert result == 1, f"single mjpeg file should be 1, got {result}"
    print(f"  PASS: single file string -> {result}")


def test_batch_empty_dir(tmp_path=None):
    """probe_batch_threads with an empty directory should return 0."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        result = probe_batch_threads(td)
        assert result == 0, f"empty dir should be 0, got {result}"
    print("  PASS: empty directory -> 0")


def test_non_video_returns_zero():
    """Non-video files (text, images, etc.) should return 0."""
    import tempfile

    cases = {
        "test.txt": b"hello world",
        "test.json": b'{"key": "value"}',
        "test.xml": b"<root><child/></root>",
        "test.csv": b"a,b,c\n1,2,3\n",
        # Real image headers that FFmpeg would accept via pipe demuxers
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
            result = probe_decode_threads(fp)
            assert result == 0, f"{name}: expected 0, got {result}"
        # batch should also be 0 for a dir of non-video files
        batch = probe_batch_threads(td)
        assert batch == 0, f"batch of non-video files: expected 0, got {batch}"
    print(f"  PASS: {len(cases)} non-video files all return 0 (single and batch)")


if __name__ == "__main__":
    tests = [
        ("Valid files return >= 1", test_valid_files_return_positive),
        ("Invalid file returns 0", test_invalid_file_returns_zero),
        ("Empty file returns 0", test_empty_file_returns_zero),
        ("Non-video files return 0", test_non_video_returns_zero),
        ("Multi-threaded codecs use > 1", test_multithreaded_codecs),
        ("Single-threaded codecs use 1", test_singlethreaded_codecs),
        ("Explicit thread cap", test_explicit_thread_cap),
        ("Batch: directory", test_batch_directory),
        ("Batch: file list", test_batch_file_list),
        ("Batch: single file string", test_batch_single_file_string),
        ("Batch: empty directory", test_batch_empty_dir),
    ]

    print("probe_decode_threads tests")
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
