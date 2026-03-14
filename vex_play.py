"""vex_play.py - Decode a video with vex and stream it to ffplay."""

import os
import subprocess
import sys
import time

import vex

from vex_report import parse_args, print_report, probe_video

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_ffplay():
    local = os.path.join(SCRIPT_DIR, "deps", "ffmpeg", "bin", "ffplay.exe")
    if os.path.isfile(local):
        return local
    return "ffplay"


def main():
    cfg = parse_args(sys.argv)
    path = cfg["path"]

    probe = probe_video(path)

    # --- start async decode -------------------------------------------------
    level = vex.LevelConfig(
        width=cfg["width"], height=cfg["height"], quality=cfg["quality"]
    )
    t_decode_start = time.perf_counter()
    handle = vex.batch_decode_async(
        [path],
        [level],
        keyframes_only=cfg["keyframes_only"],
        frame_skip=cfg["frame_skip"],
        collect_frame_times=True,
        max_threads=probe.decode_threads,
        probe_info=[probe],
    )

    # --- launch ffplay ------------------------------------------------------
    if cfg["keyframes_only"]:
        fps_hint = 5
    else:
        fps_hint = probe.fps / cfg["frame_skip"]

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
                now = time.perf_counter()
                if t_first_frame is None:
                    t_first_frame = now
                t_last_frame = now

            if not events:
                if handle.done:
                    for evt in handle.drain_events():
                        proc.stdin.write(handle.peek_jpeg(evt))
                        proc.stdin.flush()
                        frames_sent += 1
                        now = time.perf_counter()
                        if t_first_frame is None:
                            t_first_frame = now
                        t_last_frame = now
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
    elapsed = time.perf_counter() - t_decode_start
    result = handle.result()

    first_ms = (t_first_frame - t_decode_start) * 1000 if t_first_frame else float("nan")
    last_ms = (t_last_frame - t_decode_start) * 1000 if t_last_frame else float("nan")

    print(f"\n--- streaming ---")
    print(f"  frames streamed   : {frames_sent}")

    print_report(result, elapsed, probe, first_ms, last_ms, frames_sent)


if __name__ == "__main__":
    main()
