#!/usr/bin/env python3
"""Benchmark vex vs FFmpeg subprocess pipeline.

Decodes every fixture in fixtures/formats/ at 192x192 quality-85 using both:
  1. vex.batch_decode  — in-process, batched C++ pipeline (one call for all files)
  2. FFmpeg subprocess — one ffmpeg invocation per file, MJPEG piped to stdout,
     JPEGs split from the byte stream and stored in a Python list

This mirrors the real-world comparison: a Python application that needs
thumbnails from many videos can either shell out to ffmpeg per file or
call vex once.

Produces a horizontal bar chart saved to assets/benchmark.png.

Usage:
    python tools/benchmark.py              # run benchmark + generate chart
    python tools/benchmark.py --runs 3     # best-of-3
"""
from __future__ import annotations

import argparse
import sys
import subprocess
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure vex is importable
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from vex import batch_decode, LevelConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = PROJECT_ROOT / "fixtures" / "formats"
FFMPEG_BIN = PROJECT_ROOT / "deps" / "ffmpeg" / "bin" / "ffmpeg.exe"
ASSETS_DIR = PROJECT_ROOT / "assets"

# ---------------------------------------------------------------------------
# FFmpeg pipe decode — the "naive" approach vex replaces
# ---------------------------------------------------------------------------

def ffmpeg_pipe_decode(input_path: str, width: int = 192, height: int = 192
                       ) -> tuple[float, int]:
    """Spawn FFmpeg, pipe MJPEG to stdout, scan markers, build index.

    The full pipeline a caller would need:
      1. Spawn ffmpeg subprocess
      2. Read piped MJPEG byte stream
      3. Scan SOI/EOI markers to find frame boundaries
      4. Slice into individual JPEG buffers (indexed array)

    All four steps are timed — vex returns a pre-indexed blob with an
    offset array, so the FFmpeg path must include the equivalent work.

    Returns (elapsed_seconds, frame_count).
    """
    cmd = [
        str(FFMPEG_BIN),
        "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-fps_mode", "passthrough",
        "-vf", f"scale={width}:{height}:flags=bilinear",
        "-q:v", "2",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, timeout=60)

    if proc.returncode != 0:
        elapsed = time.perf_counter() - t0
        return elapsed, 0

    # Scan JPEG markers and build indexed frame array — equivalent to
    # the offset table vex produces during encode.
    frames = _split_jpegs(proc.stdout)
    elapsed = time.perf_counter() - t0
    return elapsed, len(frames)


def _split_jpegs(data: bytes) -> list[bytes]:
    """Split a concatenated MJPEG stream into individual JPEG buffers."""
    jpegs: list[bytes] = []
    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"
    pos = 0
    while pos < len(data):
        start = data.find(SOI, pos)
        if start < 0:
            break
        end = data.find(EOI, start + 2)
        if end < 0:
            break
        end += 2
        jpegs.append(data[start:end])
        pos = end
    return jpegs


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(fixtures: list[tuple[str, str]], width: int, height: int,
                  quality: int, runs: int) -> tuple[dict, list[dict]]:
    """Run the full benchmark.

    Returns (aggregate_dict, per_format_list).
    """
    all_paths = [path for _, path in fixtures]
    n_files = len(fixtures)

    # -- Warmup (one quick vex call to JIT any lazy init) --------------------
    batch_decode(all_paths[:1],
                 levels=[LevelConfig(width=width, height=height,
                                     quality=quality)],
                 keyframes_only=False)

    # -- Batch vex vs sequential FFmpeg, best-of-N runs ----------------------
    best_vex = float("inf")
    best_ff = float("inf")
    vex_frames = 0

    # Also collect per-file FFmpeg times from the best FFmpeg run
    best_ff_per_file: list[tuple[float, int]] = []

    for run_i in range(runs):
        # vex: one batch call
        t0 = time.perf_counter()
        result = batch_decode(
            all_paths,
            levels=[LevelConfig(width=width, height=height, quality=quality)],
            keyframes_only=False,
        )
        vt = time.perf_counter() - t0
        if vt < best_vex:
            best_vex = vt
            stream = result.jpeg_stream()
            vex_frames = sum(len(stream.offsets[i]) for i in range(n_files))

        # FFmpeg: one subprocess per file, sequential
        ff_per_file: list[tuple[float, int]] = []
        ff_total = 0.0
        for _, path in fixtures:
            ft_i, fc_i = ffmpeg_pipe_decode(path, width, height)
            ff_per_file.append((ft_i, fc_i))
            ff_total += ft_i
        if ff_total < best_ff:
            best_ff = ff_total
            best_ff_per_file = ff_per_file

        if runs > 1:
            print(f"  run {run_i+1}/{runs}:  "
                  f"vex={vt:.3f}s  ffmpeg={ff_total:.3f}s")

    speedup = best_ff / best_vex if best_vex > 0 else float("inf")
    vex_amortized = best_vex / n_files

    aggregate = {
        "vex_time": best_vex,
        "ffmpeg_time": best_ff,
        "total_frames": vex_frames,
        "speedup": speedup,
        "n_files": n_files,
        "vex_amortized": vex_amortized,
    }

    # Build per-format rows using FFmpeg per-file times and vex amortized time
    rows = []
    for i, (name, _) in enumerate(fixtures):
        ff_time_i, ff_frames_i = best_ff_per_file[i]
        rows.append({
            "name": name,
            "ffmpeg_time": ff_time_i,
            "ffmpeg_frames": ff_frames_i,
            "vex_amortized": vex_amortized,
        })

    return aggregate, rows


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def generate_chart(rows: list[dict], aggregate: dict, out_path: Path):
    """Generate the benchmark chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("ERROR: matplotlib required.  pip install matplotlib")
        sys.exit(1)

    # Sort by FFmpeg time descending (slowest at top)
    rows = sorted(rows, key=lambda r: r["ffmpeg_time"])
    names = [r["name"] for r in rows]
    ff_ms = [r["ffmpeg_time"] * 1000 for r in rows]
    vex_ms = aggregate["vex_amortized"] * 1000

    n = len(names)
    fig_h = max(6, n * 0.32 + 2.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y_pos = list(range(n))
    bar_h = 0.6

    # FFmpeg bars
    ax.barh(y_pos, ff_ms, bar_h, color="#e74c3c", alpha=0.85,
            label="FFmpeg subprocess (per file)", zorder=2)

    # vex amortized line
    ax.axvline(vex_ms, color="#2ecc71", linewidth=2.5, linestyle="--",
               label=f"vex batch amortized ({vex_ms:.1f} ms/file)", zorder=3)

    # Shade the vex region
    ax.axvspan(0, vex_ms, alpha=0.08, color="#2ecc71", zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7.5, family="monospace")
    ax.set_xlabel("Time per file (ms)", fontsize=10)

    spd = aggregate["speedup"]
    total_frames = aggregate["total_frames"]
    n_files = aggregate["n_files"]
    ax.set_title(
        f"vex vs FFmpeg image2pipe  —  "
        f"{aggregate['vex_time']*1000:.0f} ms vs "
        f"{aggregate['ffmpeg_time']*1000:.0f} ms  "
        f"({spd:.1f}x faster)\n"
        f"{n_files} formats, {total_frames} total frames, "
        f"192x192 q85, sequential decode",
        fontsize=10, pad=14,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.set_xlim(left=0)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vex batch decode vs FFmpeg subprocess pipeline")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs; reports best-of-N (default: 1)")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--output", type=str,
                        default=str(ASSETS_DIR / "benchmark.png"))
    args = parser.parse_args()

    if not FIXTURES_DIR.is_dir():
        print(f"ERROR: Fixtures not found at {FIXTURES_DIR}")
        print("       Run: ./dev.sh fixtures")
        sys.exit(1)

    if not FFMPEG_BIN.is_file():
        print(f"ERROR: ffmpeg.exe not found at {FFMPEG_BIN}")
        sys.exit(1)

    fixtures = sorted(
        (p.name, str(p))
        for p in FIXTURES_DIR.iterdir()
        if p.is_file()
    )
    if not fixtures:
        print("ERROR: No fixture files found.")
        sys.exit(1)

    print(f"Benchmarking {len(fixtures)} formats at "
          f"{args.width}x{args.height} q{args.quality} "
          f"({'best of ' + str(args.runs) if args.runs > 1 else '1 run'})...\n")

    aggregate, rows = run_benchmark(
        fixtures, args.width, args.height, args.quality, args.runs)

    # -- Print summary -------------------------------------------------------
    print(f"\n{'='*64}")
    print(f"  vex batch_decode ({aggregate['n_files']} files, "
          f"{aggregate['total_frames']} frames)")
    print(f"    total:     {aggregate['vex_time']*1000:>8.1f} ms")
    print(f"    per file:  {aggregate['vex_amortized']*1000:>8.1f} ms")
    print(f"  FFmpeg image2pipe (sequential, {aggregate['n_files']} files)")
    print(f"    total:     {aggregate['ffmpeg_time']*1000:>8.1f} ms")
    print(f"    per file:  {aggregate['ffmpeg_time']/aggregate['n_files']*1000:>8.1f} ms")
    print(f"  Speedup: {aggregate['speedup']:.1f}x")
    print(f"{'='*64}\n")

    print(f"  {'Format':<28} {'ffmpeg':>9} {'vex (amort)':>11}")
    print(f"  {'-'*28} {'-'*9} {'-'*11}")
    for r in sorted(rows, key=lambda r: -r["ffmpeg_time"]):
        print(f"  {r['name']:<28} "
              f"{r['ffmpeg_time']*1000:>8.1f}ms "
              f"{r['vex_amortized']*1000:>10.1f}ms")

    # -- Generate chart ------------------------------------------------------
    generate_chart(rows, aggregate, Path(args.output))


if __name__ == "__main__":
    main()
