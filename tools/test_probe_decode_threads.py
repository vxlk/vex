#!/usr/bin/env python3
"""Test vex.probe_decode_threads against various fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from vex import probe_decode_threads

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
        if not path.is_file() or path.suffix == ".mp4" and path.name == "empty_file.mp4":
            continue
        if path.suffix not in (
            ".mp4", ".mkv", ".avi", ".ts", ".flv", ".mov",
            ".webm", ".ogv", ".mpg", ".vob", ".wmv", ".asf",
            ".3gp", ".3g2", ".nut", ".ivf", ".f4v", ".m4v",
            ".swf", ".rm", ".wtv", ".mxf", ".gxf", ".dv",
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


if __name__ == "__main__":
    tests = [
        ("Valid files return >= 1", test_valid_files_return_positive),
        ("Invalid file returns 0", test_invalid_file_returns_zero),
        ("Empty file returns 0", test_empty_file_returns_zero),
        ("Multi-threaded codecs use > 1", test_multithreaded_codecs),
        ("Single-threaded codecs use 1", test_singlethreaded_codecs),
        ("Explicit thread cap", test_explicit_thread_cap),
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
