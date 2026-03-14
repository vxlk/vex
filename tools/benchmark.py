#!/usr/bin/env python3
"""Benchmark vex vs FFmpeg subprocess pipeline.

Four benchmark sections:
  1. Batched thumbnail  — 192x192 q85, one batch_decode call for all files
  2. Batched native     — full resolution q100, one batch_decode call
  3. Per-file thumbnail — 192x192 q85, one batch_decode call per file
  4. Per-file native    — full resolution q100, one batch_decode call per file

Sections 1-2 show batching/throughput advantage (amortized average).
Sections 3-4 show true per-format decode performance (apples-to-apples).

Produces charts saved to assets/.

Usage:
    python tools/benchmark.py                  # run all four sections
    python tools/benchmark.py --runs 3         # best-of-3
    python tools/benchmark.py --section batch-thumb
    python tools/benchmark.py --section perfile-native
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure vex is importable
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from vex import LevelConfig, batch_decode  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = PROJECT_ROOT / "fixtures" / "formats"
FFMPEG_BIN = PROJECT_ROOT / "deps" / "ffmpeg" / "bin" / "ffmpeg.exe"
ASSETS_DIR = PROJECT_ROOT / "assets"

# ---------------------------------------------------------------------------
# FFmpeg pipe decode
# ---------------------------------------------------------------------------


def _quality_to_ffmpeg_qv(quality: int) -> int:
    """Map TurboJPEG quality (1-100) to FFmpeg MJPEG ``-q:v`` (1-31).

    Empirically calibrated at 720x576 MPEG-2 content by matching output
    file sizes between TurboJPEG and FFmpeg's MJPEG encoder.

    FFmpeg's MJPEG encoder maxes out at ``-q:v 1`` which produces output
    roughly equivalent to TurboJPEG quality 91.  For higher TJ qualities
    we still use ``-q:v 1`` (FFmpeg's best).
    """
    if quality >= 91:
        return 1
    return max(2, round((100 - quality) / 5))


def ffmpeg_pipe_decode(
    input_path: str,
    width: int | None = None,
    height: int | None = None,
    quality: int = 85,
) -> tuple[float, int]:
    """Spawn FFmpeg, pipe MJPEG to stdout, scan markers, build index.

    Returns (elapsed_seconds, frame_count).
    """
    qv = _quality_to_ffmpeg_qv(quality)

    cmd = [
        str(FFMPEG_BIN),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-fps_mode",
        "passthrough",
    ]

    if width is not None and height is not None:
        cmd += ["-vf", f"scale={width}:{height}:flags=bilinear"]

    cmd += [
        "-q:v",
        str(qv),
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, timeout=120)

    if proc.returncode != 0:
        elapsed = time.perf_counter() - t0
        return elapsed, 0

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
# vex single-file decode (for per-file benchmark)
# ---------------------------------------------------------------------------


def vex_single_decode(
    path: str, width: int | None, height: int | None, quality: int
) -> tuple[float, int]:
    """Decode a single file with vex. Returns (elapsed_seconds, frame_count)."""
    lc_w = width if width is not None else 0
    lc_h = height if height is not None else 0

    t0 = time.perf_counter()
    result = batch_decode(
        [path],
        levels=[LevelConfig(width=lc_w, height=lc_h, quality=quality)],
        keyframes_only=False,
    )
    elapsed = time.perf_counter() - t0

    stream = result.jpeg_stream()
    n_frames = len(stream.offsets[0])
    return elapsed, n_frames


# ---------------------------------------------------------------------------
# Batch benchmark runner (existing logic, cleaned up)
# ---------------------------------------------------------------------------


def run_batch_benchmark(
    fixtures: list[tuple[str, str]],
    width: int | None,
    height: int | None,
    quality: int,
    runs: int,
) -> tuple[dict, list[dict]]:
    """Batch vex (one call) vs sequential FFmpeg (one subprocess per file).

    Returns (aggregate_dict, per_format_list).
    """
    all_paths = [path for _, path in fixtures]
    n_files = len(fixtures)

    lc_w = width if width is not None else 0
    lc_h = height if height is not None else 0

    # Warmup
    batch_decode(
        all_paths[:1],
        levels=[LevelConfig(width=lc_w, height=lc_h, quality=quality)],
        keyframes_only=False,
    )

    best_vex = float("inf")
    best_ff = float("inf")
    vex_frames = 0
    best_ff_per_file: list[tuple[float, int]] = []

    for run_i in range(runs):
        # vex: one batch call for all files
        t0 = time.perf_counter()
        result = batch_decode(
            all_paths,
            levels=[LevelConfig(width=lc_w, height=lc_h, quality=quality)],
            keyframes_only=False,
        )
        vt = time.perf_counter() - t0
        if vt < best_vex:
            best_vex = vt
            stream = result.jpeg_stream()
            vex_frames = sum(len(stream.offsets[i]) for i in range(n_files))

        # FFmpeg: concurrent subprocesses (matches vex's internal threading)
        n_workers = min(os.cpu_count() or 4, n_files)
        ff_t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(ffmpeg_pipe_decode, path, width, height, quality)
                for _, path in fixtures
            ]
            ff_per_file = [f.result() for f in futures]
        ff_wall = time.perf_counter() - ff_t0
        if ff_wall < best_ff:
            best_ff = ff_wall
            best_ff_per_file = ff_per_file

        if runs > 1:
            print(f"  run {run_i + 1}/{runs}:  vex={vt:.3f}s  ffmpeg={ff_wall:.3f}s")

    speedup = best_ff / best_vex if best_vex > 0 else float("inf")
    vex_amortized = best_vex / n_files
    res_label = f"{width}x{height}" if width is not None else "native"

    ffmpeg_amortized = best_ff / n_files

    aggregate = {
        "vex_time": best_vex,
        "ffmpeg_time": best_ff,
        "total_frames": vex_frames,
        "speedup": speedup,
        "n_files": n_files,
        "vex_amortized": vex_amortized,
        "ffmpeg_amortized": ffmpeg_amortized,
        "res_label": res_label,
        "quality": quality,
    }

    return aggregate


# ---------------------------------------------------------------------------
# Per-file benchmark runner (new — true per-format comparison)
# ---------------------------------------------------------------------------


def run_perfile_benchmark(
    fixtures: list[tuple[str, str]],
    width: int | None,
    height: int | None,
    quality: int,
    runs: int,
) -> tuple[dict, list[dict]]:
    """Per-file vex vs per-file FFmpeg (apples-to-apples).

    Returns (aggregate_dict, per_format_list).
    """
    n_files = len(fixtures)

    lc_w = width if width is not None else 0
    lc_h = height if height is not None else 0

    # Warmup
    batch_decode(
        [fixtures[0][1]],
        levels=[LevelConfig(width=lc_w, height=lc_h, quality=quality)],
        keyframes_only=False,
    )

    # Best-of-N per file
    best_vex_per_file: list[tuple[float, int]] = [(float("inf"), 0)] * n_files
    best_ff_per_file: list[tuple[float, int]] = [(float("inf"), 0)] * n_files

    for run_i in range(runs):
        vex_total = 0.0
        ff_total = 0.0

        for i, (_, path) in enumerate(fixtures):
            vt_i, vf_i = vex_single_decode(path, width, height, quality)
            ft_i, fc_i = ffmpeg_pipe_decode(path, width, height, quality)

            vex_total += vt_i
            ff_total += ft_i

            if vt_i < best_vex_per_file[i][0]:
                best_vex_per_file[i] = (vt_i, vf_i)
            if ft_i < best_ff_per_file[i][0]:
                best_ff_per_file[i] = (ft_i, fc_i)

        if runs > 1:
            print(
                f"  run {run_i + 1}/{runs}:  "
                f"vex={vex_total:.3f}s  ffmpeg={ff_total:.3f}s"
            )

    total_vex = sum(t for t, _ in best_vex_per_file)
    total_ff = sum(t for t, _ in best_ff_per_file)
    total_frames = sum(f for _, f in best_vex_per_file)
    speedup = total_ff / total_vex if total_vex > 0 else float("inf")
    res_label = f"{width}x{height}" if width is not None else "native"

    aggregate = {
        "vex_time": total_vex,
        "ffmpeg_time": total_ff,
        "total_frames": total_frames,
        "speedup": speedup,
        "n_files": n_files,
        "res_label": res_label,
        "quality": quality,
    }

    rows = []
    for i, (name, _) in enumerate(fixtures):
        vex_t, vex_f = best_vex_per_file[i]
        ff_t, ff_f = best_ff_per_file[i]
        rows.append(
            {
                "name": name,
                "vex_time": vex_t,
                "vex_frames": vex_f,
                "ffmpeg_time": ff_t,
                "ffmpeg_frames": ff_f,
            }
        )

    return aggregate, rows


# ---------------------------------------------------------------------------
# Print summaries
# ---------------------------------------------------------------------------


def print_batch_summary(aggregate: dict, label: str):
    print(f"\n{'=' * 64}")
    print(f"  {label}  [BATCHED]")
    print(
        f"  vex batch_decode ({aggregate['n_files']} files, "
        f"{aggregate['total_frames']} frames, single call)"
    )
    print(f"    total:     {aggregate['vex_time'] * 1000:>8.1f} ms")
    print(
        f"    amortized: {aggregate['vex_amortized'] * 1000:>8.1f} ms  "
        f"(= {aggregate['vex_time'] * 1000:.0f} / {aggregate['n_files']})"
    )
    print(
        f"  FFmpeg image2pipe (threaded, {aggregate['n_files']} subprocesses)"
    )
    print(f"    total:     {aggregate['ffmpeg_time'] * 1000:>8.1f} ms")
    print(
        f"    amortized: {aggregate['ffmpeg_amortized'] * 1000:>8.1f} ms  "
        f"(= {aggregate['ffmpeg_time'] * 1000:.0f} / {aggregate['n_files']})"
    )
    print(f"  Speedup: {aggregate['speedup']:.1f}x")
    print(f"{'=' * 64}")


def print_perfile_summary(aggregate: dict, rows: list[dict], label: str):
    print(f"\n{'=' * 64}")
    print(f"  {label}  [PER-FILE]")
    print(
        f"  vex single-file decode ({aggregate['n_files']} files, "
        f"{aggregate['total_frames']} frames)"
    )
    print(f"    total:     {aggregate['vex_time'] * 1000:>8.1f} ms")
    print(
        f"    avg:       {aggregate['vex_time'] / aggregate['n_files'] * 1000:>8.1f} ms"
    )
    print(f"  FFmpeg image2pipe ({aggregate['n_files']} subprocesses)")
    print(f"    total:     {aggregate['ffmpeg_time'] * 1000:>8.1f} ms")
    print(
        f"    avg:       "
        f"{aggregate['ffmpeg_time'] / aggregate['n_files'] * 1000:>8.1f} ms"
    )
    print(f"  Speedup: {aggregate['speedup']:.1f}x")
    print(f"{'=' * 64}\n")

    print(f"  {'Format':<28} {'ffmpeg':>9} {'vex':>9} {'speedup':>8}")
    print(f"  {'-' * 28} {'-' * 9} {'-' * 9} {'-' * 8}")
    for r in sorted(rows, key=lambda r: -r["ffmpeg_time"]):
        spd = r["ffmpeg_time"] / r["vex_time"] if r["vex_time"] > 0 else 0
        print(
            f"  {r['name']:<28} "
            f"{r['ffmpeg_time'] * 1000:>8.1f}ms "
            f"{r['vex_time'] * 1000:>8.1f}ms "
            f"{spd:>7.1f}x"
        )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _init_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("ERROR: matplotlib required.  pip install matplotlib")
        sys.exit(1)


def generate_batch_chart(aggregate: dict, out_path: Path):
    """Batched benchmark chart — total wall time comparison."""
    plt = _init_matplotlib()
    import numpy as np

    res = aggregate["res_label"]
    q = aggregate["quality"]
    spd = aggregate["speedup"]
    n_files = aggregate["n_files"]
    total_frames = aggregate["total_frames"]

    vex_ms = aggregate["vex_time"] * 1000
    ff_ms = aggregate["ffmpeg_time"] * 1000

    fig, ax = plt.subplots(figsize=(8, 3.5))

    labels = ["vex (single call)", f"FFmpeg (threaded, {n_files} subprocesses)"]
    times = [vex_ms, ff_ms]
    colors = ["#2ecc71", "#e74c3c"]

    y_pos = np.arange(len(labels))
    bar_h = 0.5

    bars = ax.barh(y_pos, times, bar_h, color=colors, alpha=0.85, zorder=2)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + max(times) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.0f} ms",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Wall time (ms)", fontsize=10)

    ax.set_title(
        f"vex vs FFmpeg batched  —  "
        f"{vex_ms:.0f} ms vs {ff_ms:.0f} ms  "
        f"({spd:.1f}x faster)\n"
        f"{n_files} formats, {total_frames} total frames, {res} q{q}",
        fontsize=10,
        pad=14,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.set_xlim(left=0, right=max(times) * 1.15)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved to {out_path}")


def generate_perfile_chart(rows: list[dict], aggregate: dict, out_path: Path):
    """Per-file benchmark chart — side-by-side bars for vex and FFmpeg."""
    plt = _init_matplotlib()
    import numpy as np

    res = aggregate["res_label"]
    q = aggregate["quality"]

    rows = sorted(rows, key=lambda r: r["ffmpeg_time"])
    names = [r["name"] for r in rows]
    ff_ms = [r["ffmpeg_time"] * 1000 for r in rows]
    vex_ms = [r["vex_time"] * 1000 for r in rows]

    n = len(names)
    fig_h = max(6, n * 0.38 + 2.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y_pos = np.arange(n)
    bar_h = 0.35

    ax.barh(
        y_pos + bar_h / 2,
        ff_ms,
        bar_h,
        color="#e74c3c",
        alpha=0.85,
        label="FFmpeg subprocess",
        zorder=2,
    )
    ax.barh(
        y_pos - bar_h / 2,
        vex_ms,
        bar_h,
        color="#2ecc71",
        alpha=0.85,
        label="vex (single file)",
        zorder=2,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7.5, family="monospace")
    ax.set_xlabel("Time per file (ms)", fontsize=10)

    spd = aggregate["speedup"]
    total_frames = aggregate["total_frames"]
    n_files = aggregate["n_files"]
    ax.set_title(
        f"vex vs FFmpeg per-file  —  "
        f"{aggregate['vex_time'] * 1000:.0f} ms vs "
        f"{aggregate['ffmpeg_time'] * 1000:.0f} ms total  "
        f"({spd:.1f}x faster)\n"
        f"{n_files} formats, {total_frames} total frames, "
        f"{res} q{q}, sequential decode",
        fontsize=9.5,
        pad=14,
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

SECTIONS = ["batch-thumb", "batch-native", "perfile-thumb", "perfile-native"]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vex batch decode vs FFmpeg subprocess pipeline"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs; reports best-of-N (default: 1)",
    )
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument(
        "--section",
        type=str,
        default="all",
        choices=["all"] + SECTIONS,
        help="Which benchmark section to run",
    )
    args = parser.parse_args()

    if not FIXTURES_DIR.is_dir():
        print(f"ERROR: Fixtures not found at {FIXTURES_DIR}")
        print("       Run: ./dev.sh fixtures")
        sys.exit(1)

    if not FFMPEG_BIN.is_file():
        print(f"ERROR: ffmpeg.exe not found at {FFMPEG_BIN}")
        sys.exit(1)

    fixtures = sorted((p.name, str(p)) for p in FIXTURES_DIR.iterdir() if p.is_file())
    if not fixtures:
        print("ERROR: No fixture files found.")
        sys.exit(1)

    run_all = args.section == "all"

    # -- Section 1: Batched thumbnail -----------------------------------------
    if run_all or args.section == "batch-thumb":
        print(
            f"\n[Batched Thumbnail] {len(fixtures)} formats at "
            f"{args.width}x{args.height} q{args.quality} "
            f"({'best of ' + str(args.runs) if args.runs > 1 else '1 run'})...\n"
        )
        agg = run_batch_benchmark(
            fixtures, args.width, args.height, args.quality, args.runs
        )
        print_batch_summary(
            agg, f"Thumbnail ({args.width}x{args.height} q{args.quality})"
        )
        generate_batch_chart(agg, ASSETS_DIR / "benchmark_batch_thumb.png")

    # -- Section 2: Batched native --------------------------------------------
    if run_all or args.section == "batch-native":
        print(
            f"\n{'#' * 64}\n"
            f"[Batched Native] {len(fixtures)} formats at native resolution q100 "
            f"({'best of ' + str(args.runs) if args.runs > 1 else '1 run'})...\n"
        )
        agg = run_batch_benchmark(fixtures, None, None, 100, args.runs)
        print_batch_summary(agg, "Native resolution (q100)")
        generate_batch_chart(agg, ASSETS_DIR / "benchmark_batch_native.png")

    # -- Section 3: Per-file thumbnail ----------------------------------------
    if run_all or args.section == "perfile-thumb":
        print(
            f"\n{'#' * 64}\n"
            f"[Per-File Thumbnail] {len(fixtures)} formats at "
            f"{args.width}x{args.height} q{args.quality} "
            f"({'best of ' + str(args.runs) if args.runs > 1 else '1 run'})...\n"
        )
        agg, rows = run_perfile_benchmark(
            fixtures, args.width, args.height, args.quality, args.runs
        )
        print_perfile_summary(
            agg, rows, f"Thumbnail ({args.width}x{args.height} q{args.quality})"
        )
        generate_perfile_chart(rows, agg, ASSETS_DIR / "benchmark_perfile_thumb.png")

    # -- Section 4: Per-file native -------------------------------------------
    if run_all or args.section == "perfile-native":
        print(
            f"\n{'#' * 64}\n"
            f"[Per-File Native] {len(fixtures)} formats at native resolution q100 "
            f"({'best of ' + str(args.runs) if args.runs > 1 else '1 run'})...\n"
        )
        agg, rows = run_perfile_benchmark(fixtures, None, None, 100, args.runs)
        print_perfile_summary(agg, rows, "Native resolution (q100)")
        generate_perfile_chart(rows, agg, ASSETS_DIR / "benchmark_perfile_native.png")


if __name__ == "__main__":
    main()
