#!/usr/bin/env python3
"""Generate test video fixtures for the vex test suite.

Usage:
    python tools/generate_fixtures.py [--ffmpeg path/to/ffmpeg.exe] [--output fixtures/] [--force]

All generated files are short (1-5 seconds) and small resolution (320x240) to keep the
repository and test times compact.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def find_ffmpeg(hint: Optional[str] = None) -> str:
    """Locate the ffmpeg executable.

    Search order:
    1. Explicit --ffmpeg argument
    2. deps/ffmpeg/bin/ffmpeg.exe in project root
    3. ffmpeg on PATH
    """
    if hint and os.path.isfile(hint):
        return hint

    # Project root = two levels up from this script (tools/ -> project root)
    project_root = Path(__file__).resolve().parent.parent
    local = project_root / "deps" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if local.is_file():
        return str(local)

    # Try PATH
    found = shutil.which("ffmpeg")
    if found:
        return found

    print("ERROR: ffmpeg not found.", file=sys.stderr)
    print(
        "  Provide --ffmpeg <path> or place ffmpeg.exe in deps/ffmpeg/bin/",
        file=sys.stderr,
    )
    sys.exit(1)


def run_ffmpeg(ffmpeg: str, args: List[str], label: str) -> bool:
    """Run an ffmpeg command. Returns True on success."""
    cmd = [ffmpeg] + args
    print(f"  [{label}] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
        )
        if result.returncode != 0:
            print(f"    FAILED (exit {result.returncode})", file=sys.stderr)
            stderr_text = result.stderr.decode("utf-8", errors="replace")
            # Print last few lines of stderr for diagnosis
            for line in stderr_text.strip().splitlines()[-5:]:
                print(f"    {line}", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print("    TIMEOUT", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"    ffmpeg not found: {ffmpeg}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Fixture definitions
# ---------------------------------------------------------------------------


def formats_fixtures(ffmpeg: str, out_dir: Path, force: bool) -> int:
    """Generate format/container test files.

    Covers every video container format that FFmpeg can mux, using an
    appropriate codec for each container.
    """
    d = out_dir / "formats"
    d.mkdir(parents=True, exist_ok=True)

    # Standard 320x240 @ 30fps test source
    si = "-f lavfi -i testsrc2=duration=3:size=320x240:rate=30"
    bv = "-pix_fmt yuv420p -g 15"

    # Broadcast-resolution source for containers that require standard sizes
    # PAL DV / MXF / GXF expect 720x576 @ 25fps
    bi = "-f lavfi -i testsrc2=duration=3:size=720x576:rate=25"

    fixtures = [
        # ── MP4 (MPEG-4 Part 14) ──
        ("h264_mp4.mp4", f"{si} -c:v libx264 {bv}"),
        ("h265_mp4.mp4", f"{si} -c:v libx265 {bv}"),
        ("av1_mp4.mp4", f"{si} -c:v libsvtav1 -preset 8 -pix_fmt yuv420p"),
        # ── Matroska (MKV) ──
        ("h264_mkv.mkv", f"{si} -c:v libx264 {bv}"),
        ("h265_mkv.mkv", f"{si} -c:v libx265 {bv}"),
        ("vp9_mkv.mkv", f"{si} -c:v libvpx-vp9 -b:v 1M {bv}"),
        ("av1_mkv.mkv", f"{si} -c:v libsvtav1 -preset 8 -pix_fmt yuv420p"),
        # ── AVI ──
        ("h264_avi.avi", f"{si} -c:v libx264 {bv}"),
        ("mpeg4_avi.avi", f"{si} -c:v mpeg4 -b:v 2M {bv}"),
        ("mjpeg_avi.avi", f"{si} -c:v mjpeg -q:v 3 -pix_fmt yuvj420p"),
        ("msmpeg4_avi.avi", f"{si} -c:v msmpeg4v2 -b:v 2M -pix_fmt yuv420p"),
        # ── MPEG-TS (Transport Stream) ──
        ("h264_ts.ts", f"{si} -c:v libx264 {bv}"),
        ("mpeg2_ts.ts", f"{si} -c:v mpeg2video -b:v 2M {bv}"),
        # ── FLV (Flash Video) ──
        ("h264_flv.flv", f"{si} -c:v libx264 {bv}"),
        ("flv1_flv.flv", f"{si} -c:v flv -b:v 2M -pix_fmt yuv420p"),
        # ── QuickTime / MOV ──
        ("h264_mov.mov", f"{si} -c:v libx264 {bv}"),
        ("mpeg4_mov.mov", f"{si} -c:v mpeg4 -b:v 2M {bv}"),
        ("mjpeg_mov.mov", f"{si} -c:v mjpeg -q:v 3 -pix_fmt yuvj420p"),
        # ── WebM ──
        ("vp8_webm.webm", f"{si} -c:v libvpx -b:v 1M {bv}"),
        ("vp9_webm.webm", f"{si} -c:v libvpx-vp9 -b:v 1M {bv}"),
        ("av1_webm.webm", f"{si} -c:v libsvtav1 -preset 8 -pix_fmt yuv420p"),
        # ── Ogg / OGV ──
        ("theora_ogv.ogv", f"{si} -c:v libtheora -q:v 5 -pix_fmt yuv420p"),
        # ── MPEG Program Stream (.mpg) ──
        ("mpeg1_mpg.mpg", f"{si} -c:v mpeg1video -b:v 2M -pix_fmt yuv420p"),
        ("mpeg2_mpg.mpg", f"{si} -c:v mpeg2video -b:v 2M {bv}"),
        # ── DVD VOB ──
        ("mpeg2_vob.vob", f"{si} -c:v mpeg2video -b:v 2M {bv}"),
        # ── Windows Media Video (.wmv via ASF muxer) ──
        ("wmv1_wmv.wmv", f"{si} -c:v wmv1 -b:v 2M -pix_fmt yuv420p"),
        ("wmv2_wmv.wmv", f"{si} -c:v wmv2 -b:v 2M -pix_fmt yuv420p"),
        # ── ASF (Advanced Streaming Format) ──
        ("msmpeg4_asf.asf", f"{si} -c:v msmpeg4v2 -b:v 2M -pix_fmt yuv420p"),
        # ── 3GPP / 3GPP2 ──
        ("h264_3gp.3gp", f"{si} -c:v libx264 -profile:v baseline -level 3.0 {bv}"),
        ("h264_3g2.3g2", f"{si} -c:v libx264 -profile:v baseline -level 3.0 {bv}"),
        # ── NUT (FFmpeg native container) ──
        ("h264_nut.nut", f"{si} -c:v libx264 {bv}"),
        # ── IVF (VP8/VP9 elementary container) ──
        ("vp8_ivf.ivf", f"{si} -c:v libvpx -b:v 1M {bv}"),
        ("vp9_ivf.ivf", f"{si} -c:v libvpx-vp9 -b:v 1M {bv}"),
        # ── F4V (Flash MP4 variant) ──
        ("h264_f4v.f4v", f"{si} -c:v libx264 {bv}"),
        # ── M4V (iTunes MP4 variant) ──
        ("h264_m4v.m4v", f"{si} -c:v libx264 {bv}"),
        # ── SWF (Shockwave Flash) ──
        ("flv1_swf.swf", f"{si} -c:v flv -b:v 2M -pix_fmt yuv420p"),
        # ── RealMedia ──
        ("rv20_rm.rm", f"{si} -c:v rv20 -b:v 1M -pix_fmt yuv420p"),
        # ── WTV (Windows Recorded TV) ──
        ("mpeg2_wtv.wtv", f"{si} -c:v mpeg2video -b:v 2M {bv}"),
        # ── MXF (broadcast — requires standard resolution) ──
        ("mpeg2_mxf.mxf", f"{bi} -c:v mpeg2video -b:v 5M -pix_fmt yuv420p -g 15"),
        # ── GXF (broadcast — requires standard resolution) ──
        ("mpeg2_gxf.gxf", f"{bi} -c:v mpeg2video -b:v 5M -pix_fmt yuv420p -g 15"),
        # ── DV (requires 720x576 PAL or 720x480 NTSC) ──
        ("dv_raw.dv", f"{bi} -c:v dvvideo -pix_fmt yuv420p"),
        ("dv_avi.avi", f"{bi} -c:v dvvideo -pix_fmt yuv420p"),
    ]

    count = 0
    for filename, args_str in fixtures:
        path = d / filename
        if path.exists() and not force:
            print(f"  [skip] {path} (already exists)")
            continue
        args = ["-y"] + args_str.split() + [str(path)]
        if run_ffmpeg(ffmpeg, args, filename):
            count += 1
    return count


def resolution_fixtures(ffmpeg: str, out_dir: Path, force: bool) -> int:
    """Generate resolution test files."""
    d = out_dir / "resolutions"
    d.mkdir(parents=True, exist_ok=True)

    resolutions = [
        ("192x192.mp4", "192x192"),
        ("640x480.mp4", "640x480"),
        ("1280x720.mp4", "1280x720"),
        ("1920x1080.mp4", "1920x1080"),
    ]

    count = 0
    for filename, size in resolutions:
        path = d / filename
        if path.exists() and not force:
            print(f"  [skip] {path} (already exists)")
            continue
        args = [
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=duration=3:size={size}:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "15",
            str(path),
        ]
        if run_ffmpeg(ffmpeg, args, filename):
            count += 1
    return count


def edge_case_fixtures(ffmpeg: str, out_dir: Path, force: bool) -> int:
    """Generate edge-case test files."""
    d = out_dir / "edge_cases"
    d.mkdir(parents=True, exist_ok=True)

    count = 0

    # single_frame.mp4: 1 frame only
    path = d / "single_frame.mp4"
    if not path.exists() or force:
        args = [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=1:size=320x240:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-frames:v",
            "1",
            str(path),
        ]
        if run_ffmpeg(ffmpeg, args, "single_frame.mp4"):
            count += 1
    else:
        print(f"  [skip] {path} (already exists)")

    # all_keyframes.mp4: GOP=1, every frame is I-frame, 2 seconds
    path = d / "all_keyframes.mp4"
    if not path.exists() or force:
        args = [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=2:size=320x240:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "1",
            str(path),
        ]
        if run_ffmpeg(ffmpeg, args, "all_keyframes.mp4"):
            count += 1
    else:
        print(f"  [skip] {path} (already exists)")

    # long_gop.mp4: GOP=250, 5 seconds (very few keyframes)
    path = d / "long_gop.mp4"
    if not path.exists() or force:
        args = [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=5:size=320x240:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "250",
            str(path),
        ]
        if run_ffmpeg(ffmpeg, args, "long_gop.mp4"):
            count += 1
    else:
        print(f"  [skip] {path} (already exists)")

    # truncated.mp4: normal video truncated to 70% of its size
    path = d / "truncated.mp4"
    if not path.exists() or force:
        # First generate a normal file
        temp_path = d / "_truncated_source.mp4"
        args = [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=3:size=320x240:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "15",
            str(temp_path),
        ]
        if run_ffmpeg(ffmpeg, args, "truncated.mp4 (source)"):
            # Truncate to 70%
            try:
                full_size = temp_path.stat().st_size
                truncated_size = int(full_size * 0.7)
                with open(temp_path, "rb") as src:
                    data = src.read(truncated_size)
                with open(path, "wb") as dst:
                    dst.write(data)
                print(f"    Truncated {full_size} -> {truncated_size} bytes")
                count += 1
            except Exception as e:
                print(f"    Failed to truncate: {e}", file=sys.stderr)
            finally:
                temp_path.unlink(missing_ok=True)
    else:
        print(f"  [skip] {path} (already exists)")

    # no_video_stream.mp4: audio only
    path = d / "no_video_stream.mp4"
    if not path.exists() or force:
        args = [
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=3",
            "-vn",
            "-c:a",
            "aac",
            str(path),
        ]
        if run_ffmpeg(ffmpeg, args, "no_video_stream.mp4"):
            count += 1
    else:
        print(f"  [skip] {path} (already exists)")

    # empty_file.mp4: literally 0 bytes
    path = d / "empty_file.mp4"
    if not path.exists() or force:
        path.write_bytes(b"")
        print("  [empty_file.mp4] Created 0-byte file")
        count += 1
    else:
        print(f"  [skip] {path} (already exists)")

    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate test video fixtures for vex")
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default=None,
        help="Path to ffmpeg executable (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: fixtures/ in project root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate files that already exist",
    )
    args = parser.parse_args()

    ffmpeg = find_ffmpeg(args.ffmpeg)
    print(f"Using ffmpeg: {ffmpeg}")

    # Determine output directory
    if args.output:
        out_dir = Path(args.output)
    else:
        project_root = Path(__file__).resolve().parent.parent
        out_dir = project_root / "fixtures"

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print()

    total = 0

    print("=== Format/Container fixtures ===")
    total += formats_fixtures(ffmpeg, out_dir, args.force)
    print()

    print("=== Resolution fixtures ===")
    total += resolution_fixtures(ffmpeg, out_dir, args.force)
    print()

    print("=== Edge-case fixtures ===")
    total += edge_case_fixtures(ffmpeg, out_dir, args.force)
    print()

    print(f"Done. Generated {total} fixture(s) in {out_dir}")


if __name__ == "__main__":
    main()
