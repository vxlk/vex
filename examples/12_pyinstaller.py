"""Prove that vex works inside a PyInstaller frozen bundle.

This script does two things depending on how it's invoked:

  1. **Normal Python** (``python examples/12_pyinstaller.py``):
     Builds a one-folder frozen app with PyInstaller, runs it, and
     verifies the output.

  2. **Inside the frozen bundle** (``dist/vex_frozen/vex_frozen.exe``):
     Decodes a video with vex and prints a machine-readable result line
     so the outer script can validate it.

Usage:
    python examples/12_pyinstaller.py [video_path]

If no video is given, the default fixture is used.
"""

import os
import sys

# ── Frozen-bundle entry point ───────────────────────────────────────────────
# When PyInstaller runs this script, sys.frozen is set.

if getattr(sys, "frozen", False):
    from vex import LevelConfig, batch_decode

    video = sys.argv[1] if len(sys.argv) > 1 else None
    if video is None:
        print("FAIL: no video path provided", file=sys.stderr)
        sys.exit(1)

    result = batch_decode(
        paths=[video],
        levels=[LevelConfig(width=64, height=64, quality=50)],
        keyframes_only=True,
    )
    n = result.metrics.keyframes_decoded
    fps = result.metrics.pipeline_fps
    print(f"OK frames={n} fps={fps:.0f}")
    sys.exit(0)


# ── Builder entry point (normal Python) ─────────────────────────────────────

import shutil
import subprocess
import tempfile


def main():
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    hook_dir = os.path.join(project_root, "pyinstaller")
    this_script = os.path.abspath(__file__)

    # Pick a video fixture
    if len(sys.argv) > 1:
        video = os.path.abspath(sys.argv[1])
    else:
        video = os.path.join(
            project_root, "fixtures", "formats", "h264_mp4.mp4"
        )
    if not os.path.isfile(video):
        print(f"Video not found: {video}", file=sys.stderr)
        sys.exit(1)

    # Ensure vex package is importable for PyInstaller analysis
    python_dir = os.path.join(project_root, "python")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Building frozen app with PyInstaller ...")
        build_cmd = [
            sys.executable, "-m", "PyInstaller",
            "--noconfirm",
            "--log-level", "WARN",
            "--distpath", os.path.join(tmpdir, "dist"),
            "--workpath", os.path.join(tmpdir, "build"),
            "--specpath", tmpdir,
            "--additional-hooks-dir", hook_dir,
            "--paths", python_dir,
            "--name", "vex_frozen",
            this_script,
        ]
        subprocess.check_call(build_cmd)

        exe = os.path.join(tmpdir, "dist", "vex_frozen", "vex_frozen.exe")
        if not os.path.isfile(exe):
            # Unix
            exe = os.path.join(tmpdir, "dist", "vex_frozen", "vex_frozen")
        assert os.path.isfile(exe), f"Frozen executable not found: {exe}"

        print(f"Running frozen app: {exe}")
        proc = subprocess.run(
            [exe, video],
            capture_output=True, text=True, timeout=30,
        )

        print(f"stdout: {proc.stdout.strip()}")
        if proc.stderr.strip():
            print(f"stderr: {proc.stderr.strip()}")

        if proc.returncode != 0:
            print(f"FAIL: frozen app exited with code {proc.returncode}")
            sys.exit(1)

        if not proc.stdout.strip().startswith("OK"):
            print(f"FAIL: unexpected output: {proc.stdout.strip()}")
            sys.exit(1)

        # Parse result
        parts = proc.stdout.strip().split()
        frames = int(parts[1].split("=")[1])
        assert frames > 0, f"Expected >0 frames, got {frames}"
        print(f"\nPyInstaller bundle works: decoded {frames} keyframes")


if __name__ == "__main__":
    main()
