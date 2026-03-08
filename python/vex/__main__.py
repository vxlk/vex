"""vex CLI — extract video thumbnails from the command line.

Usage:
    python -m vex [OPTIONS] VIDEO [VIDEO ...]

Examples:
    python -m vex video.mp4
    python -m vex -o thumbs/ --width 320 --height 240 -q 80 *.mp4
    python -m vex --keyframes --atlas --atlas-cols 8 -o atlas.jpg video.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vex",
        description="High-performance video thumbnail extraction.",
    )
    p.add_argument(
        "videos", nargs="*", metavar="VIDEO",
        help="Video file paths to process.",
    )

    # Output
    p.add_argument(
        "-o", "--output", default=".",
        help="Output directory (default: current directory). "
             "For atlas mode with a single video, can be a file path.",
    )
    p.add_argument(
        "--format", choices=["jpeg", "atlas"], default="jpeg",
        help="Output format: individual jpegs or sprite atlas (default: jpeg).",
    )

    # Resolution / quality
    p.add_argument(
        "-W", "--width", type=int, default=0,
        help="Target width in pixels (0 = native, default: 0).",
    )
    p.add_argument(
        "-H", "--height", type=int, default=0,
        help="Target height in pixels (0 = native, default: 0).",
    )
    p.add_argument(
        "-q", "--quality", type=int, default=85,
        help="JPEG quality 1-100 (default: 85).",
    )

    # Decode options
    p.add_argument(
        "-k", "--keyframes", action="store_true",
        help="Decode only keyframes (I-frames).",
    )
    p.add_argument(
        "--frame-skip", type=int, default=1,
        help="Decode every N-th frame (default: 1, ignored with --keyframes).",
    )
    p.add_argument(
        "-t", "--threads", type=int, default=8,
        help="Max worker threads (default: 8).",
    )
    p.add_argument(
        "--no-hw-accel", action="store_true",
        help="Disable hardware-accelerated decoding.",
    )

    # Atlas options
    p.add_argument(
        "--atlas-cols", type=int, default=10,
        help="Columns in sprite atlas grid (default: 10).",
    )

    # Misc
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print decode metrics and per-file stats.",
    )
    p.add_argument(
        "--version", action="version",
        version=f"%(prog)s {_get_version()}",
    )
    return p


def _get_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.videos:
        parser.print_help()
        return 0

    # Validate inputs exist
    missing = [v for v in args.videos if not os.path.isfile(v)]
    if missing:
        print(f"error: files not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    # Validate atlas requires explicit dimensions
    if args.format == "atlas" and (args.width <= 0 or args.height <= 0):
        print("error: --format atlas requires explicit --width and --height",
              file=sys.stderr)
        return 1

    from . import LevelConfig, batch_decode

    # Build level config
    if args.format == "atlas":
        level = LevelConfig(
            width=args.width,
            height=args.height,
            quality=args.quality,
            output="sprite_atlas",
            atlas_columns=args.atlas_cols,
        )
    else:
        level = LevelConfig(
            width=args.width,
            height=args.height,
            quality=args.quality,
            output="jpeg_stream",
        )

    # Decode
    t0 = time.perf_counter()
    result = batch_decode(
        args.videos,
        levels=[level],
        max_threads=args.threads,
        keyframes_only=args.keyframes,
        frame_skip=args.frame_skip,
        use_hw_accel=not args.no_hw_accel,
    )
    elapsed = time.perf_counter() - t0

    # Write output
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    written = 0
    if args.format == "atlas":
        atlas = result.sprite_atlas()
        n_atlases = len(atlas.offsets) - 1 if len(atlas.offsets) > 1 else 0
        for i in range(n_atlases):
            jpeg = atlas.atlas_jpeg(i)
            path = os.path.join(out_dir, f"atlas_{i:04d}.jpg")
            with open(path, "wb") as f:
                f.write(jpeg)
            written += 1
    else:
        stream = result.jpeg_stream()
        for fi in range(len(stream.blobs)):
            video_name = os.path.splitext(os.path.basename(args.videos[fi]))[0]
            n_frames = len(stream.offsets[fi])
            for frame_idx in range(n_frames):
                jpeg = stream.jpeg_bytes(fi, frame_idx)
                path = os.path.join(out_dir, f"{video_name}_{frame_idx:06d}.jpg")
                with open(path, "wb") as f:
                    f.write(jpeg)
                written += 1

    # Summary
    m = result.metrics
    print(f"Extracted {written} frames from {len(args.videos)} file(s) "
          f"in {elapsed:.2f}s")

    if args.verbose:
        print()
        print(m.log_summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
