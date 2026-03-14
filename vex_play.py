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

    # --- probe video --------------------------------------------------------
    ft = vex.get_frame_times(path)
    num_threads = vex.probe_decode_threads(path)
    print(
        f"{ft.frame_count} frames, {ft.duration_sec:.2f}s, "
        f"{ft.fps:.2f} fps, codec={ft.codec}, strategy={ft.strategy}"
        f"{num_threads} threads to be used in decode."
    )

    # --- start async decode -------------------------------------------------
    level = vex.LevelConfig(width=width, height=height, quality=quality)
    handle = vex.batch_decode_async(
        [path],
        [level],
        keyframes_only=keyframes_only,
        frame_skip=frame_skip,
        collect_frame_times=True,
    )

    # --- launch ffplay ------------------------------------------------------
    if keyframes_only:
        fps_hint = 5
    else:
        fps_hint = ft.fps / frame_skip

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
    t0 = time.perf_counter()

    try:
        while True:
            events = handle.drain_events()
            for evt in events:
                jpeg = handle.peek_jpeg(evt)
                proc.stdin.write(jpeg)
                proc.stdin.flush()
                frames_sent += 1

            if not events:
                if handle.done:
                    # final drain
                    for evt in handle.drain_events():
                        proc.stdin.write(handle.peek_jpeg(evt))
                        proc.stdin.flush()
                        frames_sent += 1
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
    elapsed = time.perf_counter() - t0
    result = handle.result()
    print(f"\n{frames_sent} frames piped in {elapsed:.2f}s")
    print(result.metrics.log_summary())


if __name__ == "__main__":
    main()
