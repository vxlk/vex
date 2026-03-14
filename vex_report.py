"""vex_report.py - Decode a video with vex and report results to stdout."""

import os
import sys
import time

import vex

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / (1024 * 1024):.1f} MB"


class ProgressBar:
    """Reusable \r-overwriting progress bar for async decode loops."""

    def __init__(self, est_total: int, t0: float):
        self.est_total = est_total
        self.t0 = t0

    def update(self, frames_done: int):
        if self.est_total <= 0:
            return
        pct = min(frames_done / self.est_total, 1.0)
        bar_w = 40
        filled = int(bar_w * pct)
        bar = "#" * filled + "-" * (bar_w - filled)
        elapsed_ms = (time.perf_counter() - self.t0) * 1000
        sys.stdout.write(
            f"\r  {bar} {pct:>6.1%}  {frames_done}/{self.est_total}  "
            f"{elapsed_ms:.0f}ms"
        )
        sys.stdout.flush()

    def finish(self, frames_done: int):
        self.update(frames_done)
        sys.stdout.write("\n")


def parse_args(argv: list[str]) -> dict:
    """Parse CLI arguments into a config dict."""
    if len(argv) < 2:
        print(
            f"Usage: python {argv[0]} <video> [options]\n"
            f"\n"
            f"Options:\n"
            f"  --keyframes       decode only keyframes (fast preview)\n"
            f"  --width W         output width  (0 = native)\n"
            f"  --height H        output height (0 = native)\n"
            f"  --quality Q       JPEG quality 1-100 (default 90)\n"
            f"  --frame-skip N    decode every Nth frame (default 1)\n"
        )
        sys.exit(1)

    cfg = {
        "path": argv[1],
        "keyframes_only": "--keyframes" in argv,
        "width": vex.NATIVE,
        "height": vex.NATIVE,
        "quality": 90,
        "frame_skip": 1,
    }

    args = argv[2:]
    for i, arg in enumerate(args):
        if arg == "--width" and i + 1 < len(args):
            cfg["width"] = int(args[i + 1])
        elif arg == "--height" and i + 1 < len(args):
            cfg["height"] = int(args[i + 1])
        elif arg == "--quality" and i + 1 < len(args):
            cfg["quality"] = int(args[i + 1])
        elif arg == "--frame-skip" and i + 1 < len(args):
            cfg["frame_skip"] = int(args[i + 1])

    return cfg


def probe_video(path: str) -> vex.ProbeResult:
    """Probe a video file and print metadata."""
    probe = vex.probe(path)
    num_threads = probe.decode_threads
    print(f"--- probe ---")
    print(f"  file            : {os.path.basename(path)}")
    print(f"  file size       : {fmt_bytes(probe.file_size)}")
    print(f"  container       : {probe.container}")
    print(f"  codec           : {probe.codec}")
    print(f"  resolution      : {probe.width}x{probe.height}")
    print(f"  duration        : {probe.duration_sec:.2f}s")
    print(f"  fps             : {probe.fps:.2f}")
    print(f"  frames          : {probe.frame_count}")
    print(f"  timestamp strat : {probe.strategy}")
    print(f"  decode threads  : {'MAX' if num_threads == 0 else num_threads}")
    return probe


def decode_video(
    cfg: dict, probe: vex.ProbeResult
) -> tuple[vex.BatchResult, float, float, float, int]:
    """Run async decode, drain events, return (result, elapsed, first_frame_ms, last_frame_ms, frames_decoded)."""
    level = vex.LevelConfig(
        width=cfg["width"], height=cfg["height"], quality=cfg["quality"]
    )
    t0 = time.perf_counter()
    handle = vex.batch_decode_async(
        [cfg["path"]],
        [level],
        keyframes_only=cfg["keyframes_only"],
        frame_skip=cfg["frame_skip"],
        collect_frame_times=True,
        max_threads=probe.decode_threads,
        probe_info=[probe],
    )

    frames_decoded = 0
    t_first_frame = None
    t_last_frame = None
    frame_skip = max(1, cfg["frame_skip"])
    est_total = (probe.frame_count + frame_skip - 1) // frame_skip
    progress = ProgressBar(est_total, t0)

    while True:
        events = handle.drain_events()
        for evt in events:
            handle.peek_jpeg(evt)
            frames_decoded += 1
            now = time.perf_counter()
            if t_first_frame is None:
                t_first_frame = now
            t_last_frame = now

        if events:
            progress.update(frames_decoded)

        if not events:
            if handle.done:
                for evt in handle.drain_events():
                    handle.peek_jpeg(evt)
                    frames_decoded += 1
                    now = time.perf_counter()
                    if t_first_frame is None:
                        t_first_frame = now
                    t_last_frame = now
                progress.finish(frames_decoded)
                break
            time.sleep(0.001)

    elapsed = time.perf_counter() - t0
    first_ms = (t_first_frame - t0) * 1000 if t_first_frame else float("nan")
    last_ms = (t_last_frame - t0) * 1000 if t_last_frame else float("nan")
    result = handle.result()
    return result, elapsed, first_ms, last_ms, frames_decoded


def print_report(
    result: vex.BatchResult,
    elapsed: float,
    probe: vex.ProbeResult,
    first_frame_ms: float,
    last_frame_ms: float,
    frames_decoded: int,
):
    """Print decode results, metrics, and size comparison to stdout."""
    m = result.metrics
    output_bytes = m.total_output_bytes

    print(f"\n--- decode ---")
    print(f"  wall time       : {elapsed * 1000:.1f} ms")
    print(f"  first frame     : {first_frame_ms:.1f} ms")
    print(f"  last frame      : {last_frame_ms:.1f} ms")
    print(f"  frames decoded  : {frames_decoded}")
    print(f"  pipeline fps    : {m.pipeline_fps:.1f}")
    print(f"  decode fps      : {m.decode_fps:.1f}")
    print(f"  threads used    : {m.threads_used}")
    hw = m.hw_accel_backend or "none"
    print(f"  hw accel        : {hw}")
    print(f"  encoder         : {m.encoder}")

    print(f"\n--- timing breakdown ---")
    print(f"  index scan      : {m.index_scan_us / 1000:>8.1f} ms")
    print(f"  seek            : {m.seek_us / 1000:>8.1f} ms")
    print(f"  decode          : {m.decode_us / 1000:>8.1f} ms")
    print(f"  i/o             : {m.io_us / 1000:>8.1f} ms")
    print(f"  total (C++)     : {m.total_wall_us / 1000:>8.1f} ms")

    print(f"\n--- memory ---")
    print(f"  peak decode     : {fmt_bytes(m.peak_decode_memory)}")
    print(f"  atlas scratch   : {fmt_bytes(m.atlas_scratch_bytes)}")

    for i, lm in enumerate(m.levels):
        print(f"\n--- level[{i}] ---")
        print(f"  output res      : {lm.get('width', 0)}x{lm.get('height', 0)}")
        print(f"  quality         : {lm.get('quality', 0)}")
        print(f"  format          : {lm.get('output_format', '?')}")
        print(f"  frames          : {lm.get('frame_count', 0)}")
        print(f"  output size     : {fmt_bytes(lm.get('output_bytes', 0))}")

    for fs in m.file_stats:
        idx = fs.get("file_index", 0)
        strat_names = {
            0: "container_index",
            1: "packet_scan",
            2: "decode_scan",
            4: "skipped",
        }
        strat = strat_names.get(fs.get("index_strategy", 4), "unknown")
        hw_flag = "hw" if fs.get("hw_accel_used") else "sw"
        print(f"\n--- file[{idx}] ---")
        print(f"  codec           : {fs.get('codec_name', '?')}")
        print(
            f"  source res      : {fs.get('source_width', 0)}x{fs.get('source_height', 0)}"
        )
        print(f"  decode mode     : {hw_flag}")
        print(f"  index strategy  : {strat}")
        print(f"  keyframes       : {fs.get('keyframe_count', 0)}")
        print(f"  decode time     : {fs.get('decode_us', 0) / 1000:.1f} ms")

    print(f"\n--- size comparison ---")
    print(f"  source file     : {fmt_bytes(probe.file_size)}")
    print(f"  vex output      : {fmt_bytes(output_bytes)}")
    if probe.file_size > 0:
        ratio = output_bytes / probe.file_size
        print(f"  ratio           : {ratio:.2f}x ({ratio * 100:.1f}%)")


def main():
    cfg = parse_args(sys.argv)
    probe = probe_video(cfg["path"])
    result, elapsed, first_ms, last_ms, frames = decode_video(cfg, probe)
    print_report(result, elapsed, probe, first_ms, last_ms, frames)


if __name__ == "__main__":
    main()
