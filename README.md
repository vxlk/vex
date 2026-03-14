# vex

Zero-copy, in-process video image extraction built for maximum decode speed.

vex links directly against FFmpeg's `libavcodec`/`libavformat`/`libswscale` and TurboJPEG to decode, scale, and JPEG-encode keyframes entirely in-process — no subprocess spawning, no shell pipes, no intermediate files. Decoded frames flow through a single pipeline where each stage operates on raw pointers, eliminating serialization overhead and unnecessary copies.

## Benchmark

### Batched throughput

vex batch-decodes all frames from 42 container formats in a single `batch_decode` call. The dashed line is the **amortized average** (total wall time / N files) — not per-format timing. FFmpeg bars show actual per-file wall time (one subprocess each).

**Thumbnail** — 192x192 q85:

![vex batched thumbnail benchmark](assets/benchmark_batch_thumb.png)

**Native** — full source resolution, q100:

![vex batched native benchmark](assets/benchmark_batch_native.png)

### Per-file comparison

True per-format decode: both vex and FFmpeg process one file at a time.

**Thumbnail** — 192x192 q85:

![vex per-file thumbnail benchmark](assets/benchmark_perfile_thumb.png)

**Native** — full source resolution, q100:

![vex per-file native benchmark](assets/benchmark_perfile_native.png)

**Note on MXF/GXF native results:** At q100, vex appears slower than FFmpeg on MXF and GXF because the two tools produce different quality output. FFmpeg's built-in MJPEG encoder has a quality ceiling around TurboJPEG q91 — its `-q:v 1` and `-q:v 2` produce identical files. vex at q100 uses TurboJPEG's accurate DCT path and produces ~2.2x larger (higher fidelity) JPEGs, so the extra time is spent encoding more data, not decoding slower. At thumbnail sizes (192x192 q85) where encoding cost is negligible, vex is faster across all formats.

Regenerate with `./dev.sh bench` (or `./dev.sh bench --runs 3` for best-of-3).

## Features

- **Multi-level output** — decode once, produce multiple resolutions (thumbnails, scrub bars, previews) in a single pass
- **Sprite atlas** — composite frames into grid JPEGs for timeline hover previews
- **Threaded pipeline** — parallel file decode with configurable thread count
- **Disk cache** — binary cache format with offset tables for random-access reads via `mmap`
- **Frame timestamps** — extract per-frame PTS from any container without decoding, or collect timestamps as a side-effect of decode
- **Async events** — per-frame callbacks for streaming results during decode
- **Python bindings** — pybind11 module exposing the full pipeline to Python/numpy

## Architecture

```
video file → demux → decode → scale → encode → blob/atlas/disk
               ↑ libavformat   ↑ libswscale  ↑ TurboJPEG
```

11 C++ components in `src/`:

| Component | Role |
|---|---|
| `index_scanner` | Keyframe discovery (container index, packet scan, decode scan) |
| `decoder` | Frame-accurate seeking + decode via libavcodec |
| `scaler` | Colorspace conversion + resize via libswscale |
| `encoder` | JPEG compression via TurboJPEG |
| `atlas` | Sprite sheet compositing |
| `disk_writer` | Binary cache with offset table + mmap support |
| `frame_timer` | Per-frame PTS extraction (packet scan, no decode) |
| `async_handle` | Per-frame event dispatch |
| `orchestrator` | Thread pool coordination, owns the full pipeline |
| `bindings` | pybind11 → Python `_vex_core` module |

## Quick start

```bash
# deps: FFmpeg shared build + TurboJPEG in deps/
# see deps/README.md

mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

```python
import vex

results = vex.batch_decode(
    ["video.mp4"],
    levels=[
        vex.LevelConfig(width=192, height=192, quality=75),
        vex.LevelConfig(width=640, height=360, quality=90),
    ],
    max_threads=8,
)
```

## Frame timestamps

Get the presentation timestamp of every frame without decoding pixel data:

```python
ft = vex.get_frame_times("video.mp4")

print(ft.strategy)      # "sample_table", "block_timestamp", "pes_timestamp", etc.
print(ft.frame_count)   # number of frames
print(ft.fps)           # detected frame rate
print(ft.times)         # float64 numpy array — one PTS per frame, in seconds
```

vex reads timestamps directly from container metadata (MP4 sample tables, MKV block timecodes, PES headers, FLV tags, etc.) using a packet-only scan. No frames are decoded, so this is fast regardless of codec or resolution.

Supported container strategies:

| Strategy | Containers |
|---|---|
| `sample_table` | MP4, MOV, M4V, F4V, 3GP, 3G2 |
| `block_timestamp` | MKV, WebM |
| `pes_timestamp` | MPEG-TS, MPEG-PS (MPG/VOB), WTV |
| `tag_timestamp` | FLV, ASF/WMV, RealMedia |
| `fixed_rate` | AVI, DV, SWF |
| `generic_pts` | NUT, IVF, OGG, MXF, GXF |

Timestamps can also be collected as a side-effect of decoding, with no extra I/O:

```python
# Synchronous
result = vex.batch_decode(["video.mp4"], keyframes_only=False, collect_frame_times=True)
times = result.metrics.frame_times(0)  # numpy array for file 0

# Async — peek at timestamps while decode is still running
handle = vex.batch_decode_async(["video.mp4"], keyframes_only=False, collect_frame_times=True)
while not handle.done:
    partial = handle.peek_frame_times(0)  # growing array
result = handle.result()
```

See `examples/10_frame_times_standalone.py` and `examples/11_frame_times_streaming.py`.

## Install

```bash
# Full setup — creates venv, builds C++ extension, installs vex + test deps:
./dev.sh install

# Then activate the venv:
source .venv/Scripts/activate   # Windows
source .venv/bin/activate       # Linux/Mac

# Install additional extras as needed:
pip install -e .[examples]      # + Pillow (for examples/)
pip install -e .[bench]         # + matplotlib (for benchmarks)
pip install -e .[examples,test,bench]  # everything
```

## Requirements

- Windows 10+, MSVC 2022, C++17
- FFmpeg 6+ shared libs (`deps/ffmpeg/`)
- libjpeg-turbo (`deps/turbojpeg/`)
- Python 3.10+ with pybind11 (fetched by CMake)
