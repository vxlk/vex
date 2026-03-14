"""vex_play.py - Decode a video with vex and stream it to ffplay."""

import os
import subprocess
import sys
import time

import vex

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_ffplay():
    local = os.path.join(SCRIPT_DIR, "deps", "ffmpeg", "bin", "ffplay.exe")
    if os.path.isfile(local):
        return local
    return "ffplay"


def main():
    if len(sys.argv) < 2:
        print(
            f"Usage: python {sys.argv[0]} <video> [options]\n"
            f"\n"
            f"Options:\n"
            f"  --keyframes       decode only keyframes (fast preview)\n"
            f"  --width W         output width  (0 = native)\n"
            f"  --height H        output height (0 = native)\n"
            f"  --quality Q       JPEG quality 1-100 (default 90)\n"
            f"  --frame-skip N    decode every Nth frame (default 1)\n"
        )
        sys.exit(1)

    path = sys.argv[1]
    keyframes_only = "--keyframes" in sys.argv
    width = height = vex.NATIVE
    quality = 90
    frame_skip = 1

    args = sys.argv[2:]
    for i, arg in enumerate(args):
        if arg == "--width" and i + 1 < len(args):
            width = int(args[i + 1])
        elif arg == "--height" and i + 1 < len(args):
            height = int(args[i + 1])
        elif arg == "--quality" and i + 1 < len(args):
            quality = int(args[i + 1])
        elif arg == "--frame-skip" and i + 1 < len(args):
            frame_skip = int(args[i + 1])

    # --- probe video (single file open) --------------------------------------
    probe = vex.probe(path)
    num_threads = probe.decode_threads
    print(
        f"{probe.frame_count} frames, {probe.width}x{probe.height}, "
        f"{probe.duration_sec:.2f}s, {probe.fps:.2f} fps, "
        f"codec={probe.codec}, strategy={probe.strategy}, "
        f"{'MAX' if num_threads == 0 else num_threads} threads to be used in decode."
    )

    # --- start async decode -------------------------------------------------
    level = vex.LevelConfig(width=width, height=height, quality=quality)
    t_decode_start = time.perf_counter()
    handle = vex.batch_decode_async(
        [path],
        [level],
        keyframes_only=keyframes_only,
        frame_skip=frame_skip,
        collect_frame_times=True,
        max_threads=num_threads,
        probe_info=[probe],
    )

    # --- launch ffplay ------------------------------------------------------
    if keyframes_only:
        fps_hint = 5
    else:
        fps_hint = probe.fps / frame_skip

    ffplay = find_ffplay()
    proc = subprocess.Popen(
        [
            ffplay,
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-framerate", f"{fps_hint:.4f}",
            "-window_title", f"vex: {os.path.basename(path)}",
            "-i", "-",
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    # --- stream frames to ffplay --------------------------------------------
    frames_sent = 0
    t_first_frame = None
    t_last_frame = None

    try:
        while True:
            events = handle.drain_events()
            for evt in events:
                jpeg = handle.peek_jpeg(evt)
                proc.stdin.write(jpeg)
                proc.stdin.flush()
                frames_sent += 1
                if t_first_frame is None:
                    t_first_frame = time.perf_counter()
                t_last_frame = time.perf_counter()

            if not events:
                if handle.done:
                    # final drain
                    for evt in handle.drain_events():
                        proc.stdin.write(handle.peek_jpeg(evt))
                        proc.stdin.flush()
                        frames_sent += 1
                        if t_first_frame is None:
                            t_first_frame = time.perf_counter()
                        t_last_frame = time.perf_counter()
                    break
                time.sleep(0.001)

    except (KeyboardInterrupt, BrokenPipeError, OSError):
        pass
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        proc.wait()

    # --- report -------------------------------------------------------------
    t_total = time.perf_counter() - t_decode_start
    result = handle.result()

    first_ms = (t_first_frame - t_decode_start) * 1000 if t_first_frame else float("nan")
    last_ms = (t_last_frame - t_decode_start) * 1000 if t_last_frame else float("nan")

    print(f"\n--- vex timing ---")
    print(f"  first frame ready : {first_ms:>8.1f} ms")
    print(f"  last frame ready  : {last_ms:>8.1f} ms")
    print(f"  total (decode)    : {t_total*1000:>8.1f} ms")
    print(f"  frames decoded    : {frames_sent}")
    print(result.metrics.log_summary())


if __name__ == "__main__":
    main()
