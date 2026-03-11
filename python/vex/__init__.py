"""vex -- High-Performance Video Thumbnail Extraction"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NATIVE = 0
"""Sentinel for :class:`LevelConfig` ``width`` / ``height``.

Pass ``width=NATIVE`` (or ``width=0``) to decode at the video's native
resolution on that axis.  When both width and height are ``NATIVE``, the
frame is encoded directly from the decoder output with no scaling at all.
"""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LevelConfig:
    """Configuration for one output level in a batch decode.

    Each level describes a target resolution, JPEG quality, and output
    format.  Multiple levels can be passed to :func:`batch_decode` to
    produce several resolutions in a single decode pass.

    Args:
        width:   Target width in pixels.  Use ``0`` (or :data:`NATIVE`) to
                 keep the video's native width.  Defaults to ``NATIVE`` (0).
        height:  Target height in pixels.  Use ``0`` (or :data:`NATIVE`) to
                 keep the video's native height.  Defaults to ``NATIVE`` (0).
        quality: JPEG quality, 1 -- 100.  Passed directly to libjpeg-turbo.
                 Rough guide:

                 * **95 -- 100** : near-lossless, large files.
                 * **80 -- 94**  : high quality — good default for detail
                   previews.
                 * **50 -- 79**  : medium — good for thumbnails / scrub bars.
                 * **20 -- 49**  : low — aggressive compression, visible
                   artifacts.
                 * **1 -- 19**   : very low — only useful for tiny sprites.

                 Defaults to ``100``.
        output:  ``"jpeg_stream"`` — one concatenated JPEG blob per source
                 file (random-access via offset table).
                 ``"sprite_atlas"`` — all files composited into a single
                 grid JPEG per keyframe time-step.
                 Defaults to ``"jpeg_stream"``.
        in_memory:  If ``True`` (default), JPEG data is returned as numpy
                    arrays.  If ``False``, output is written to
                    ``cache_path`` on disk.
        cache_path: File path for on-disk output.  Required when
                    ``in_memory=False``.
        atlas_columns: Number of columns in the sprite-atlas grid.
                       Only used when ``output="sprite_atlas"``.
                       Defaults to ``10``.
    """

    width: int = NATIVE
    height: int = NATIVE
    quality: int = 100
    output: str = "jpeg_stream"
    in_memory: bool = True
    cache_path: Optional[str] = None
    atlas_columns: int = 10


# ---------------------------------------------------------------------------
# Result wrapper classes
# ---------------------------------------------------------------------------


class JpegStreamResult:
    """In-memory JPEG stream result for one level.

    Attributes:
        blobs:    List of numpy byte arrays, one per source file.
                  Each array contains all JPEG frames concatenated.
        offsets:  List of numpy int64 arrays (one per file) with byte offsets
                  for each JPEG within the blob.
        metadata: numpy structured array with columns
                  [file_index, frame_number, pts_ms].
    """

    def __init__(
        self, blobs: List[np.ndarray], offsets: List[np.ndarray], metadata: np.ndarray
    ):
        self.blobs = blobs
        self.offsets = offsets
        self.metadata = metadata

    def jpeg_bytes(self, file_index: int, frame_index: int) -> bytes:
        """Extract a single JPEG as Python bytes."""
        off = self.offsets[file_index]
        start = int(off[frame_index])
        end = (
            int(off[frame_index + 1])
            if frame_index + 1 < len(off)
            else len(self.blobs[file_index])
        )
        return bytes(self.blobs[file_index][start:end])

    def __repr__(self) -> str:
        n_files = len(self.blobs)
        total_frames = len(self.metadata) if self.metadata is not None else 0
        return f"JpegStreamResult(files={n_files}, frames={total_frames})"


class SpriteAtlasResult:
    """Sprite atlas result for one level.

    Attributes:
        blob:       numpy byte array of all atlas JPEGs concatenated.
        offsets:    numpy int64 array with T+1 entries (byte boundaries).
        grid_w:     number of columns in each atlas grid.
        grid_h:     number of rows in each atlas grid.
        thumb_w:    width of each thumbnail in pixels.
        thumb_h:    height of each thumbnail in pixels.
        file_count: number of source files.
    """

    def __init__(
        self,
        blob: np.ndarray,
        offsets: np.ndarray,
        grid_w: int,
        grid_h: int,
        thumb_w: int,
        thumb_h: int,
        file_count: int,
    ):
        self.blob = blob
        self.offsets = offsets
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.thumb_w = thumb_w
        self.thumb_h = thumb_h
        self.file_count = file_count

    def atlas_jpeg(self, time_step: int) -> bytes:
        """Extract a single atlas JPEG as Python bytes."""
        start = int(self.offsets[time_step])
        end = int(self.offsets[time_step + 1])
        return bytes(self.blob[start:end])

    def __repr__(self) -> str:
        n_atlases = len(self.offsets) - 1 if len(self.offsets) > 0 else 0
        return (
            f"SpriteAtlasResult(atlases={n_atlases}, "
            f"grid={self.grid_w}x{self.grid_h}, "
            f"thumb={self.thumb_w}x{self.thumb_h})"
        )


class DiskResult:
    """Result when output was written to disk.

    Attributes:
        cache_path:  path to the cache file.
        frame_count: number of frames written.
        total_bytes: total bytes written to disk.
        metadata:    numpy structured array [file_index, frame_number, pts_ms].
    """

    def __init__(
        self, cache_path: str, frame_count: int, total_bytes: int, metadata: np.ndarray
    ):
        self.cache_path = cache_path
        self.frame_count = frame_count
        self.total_bytes = total_bytes
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"DiskResult(path={self.cache_path!r}, "
            f"frames={self.frame_count}, bytes={self.total_bytes})"
        )


class BatchResult:
    """Result of a :func:`batch_decode` or :meth:`DecodeHandle.result` call.

    Provides **typed accessors** so callers get the exact result class
    without down-casting or ``isinstance`` checks::

        result = batch_decode(["video.mp4"])
        stream = result.jpeg_stream(0)   # -> JpegStreamResult (typed)
        print(result.metrics)

    Also supports **tuple unpacking** for backward compatibility::

        results, metrics = batch_decode(["video.mp4"])
        stream = results[0]              # -> Union[...] (untyped)

    Attributes:
        levels:  List of per-level result objects.
        metrics: :class:`DecodeMetrics` with timing / throughput data.
    """

    __slots__ = ("levels", "metrics")

    def __init__(
        self,
        levels: List[Union[JpegStreamResult, SpriteAtlasResult, DiskResult]],
        metrics: DecodeMetrics,
    ):
        self.levels = levels
        self.metrics = metrics

    # -- tuple unpacking: results, metrics = batch_decode(...) ----------------

    def __iter__(self):
        return iter((self.levels, self.metrics))

    # -- index access: result[0] ----------------------------------------------

    def __getitem__(
        self, index: int
    ) -> Union[JpegStreamResult, SpriteAtlasResult, DiskResult]:
        return self.levels[index]

    # -- typed accessors ------------------------------------------------------

    def jpeg_stream(self, level: int = 0) -> JpegStreamResult:
        """Return the :class:`JpegStreamResult` for *level*.

        Raises:
            TypeError: If *level* produced a different result type.
            IndexError: If *level* is out of range.
        """
        r = self.levels[level]
        if not isinstance(r, JpegStreamResult):
            raise TypeError(
                f"Level {level} is {type(r).__name__}, not JpegStreamResult"
            )
        return r

    def sprite_atlas(self, level: int = 0) -> SpriteAtlasResult:
        """Return the :class:`SpriteAtlasResult` for *level*.

        Raises:
            TypeError: If *level* produced a different result type.
            IndexError: If *level* is out of range.
        """
        r = self.levels[level]
        if not isinstance(r, SpriteAtlasResult):
            raise TypeError(
                f"Level {level} is {type(r).__name__}, not SpriteAtlasResult"
            )
        return r

    def disk(self, level: int = 0) -> DiskResult:
        """Return the :class:`DiskResult` for *level*.

        Raises:
            TypeError: If *level* produced a different result type.
            IndexError: If *level* is out of range.
        """
        r = self.levels[level]
        if not isinstance(r, DiskResult):
            raise TypeError(f"Level {level} is {type(r).__name__}, not DiskResult")
        return r

    def __repr__(self) -> str:
        types = [type(lv).__name__ for lv in self.levels]
        return f"BatchResult(levels={types}, {self.metrics!r})"


class DecodeMetrics:
    """Performance metrics from a batch decode operation.

    Attributes:
        total_wall_us:       total wall-clock time in microseconds.
        index_scan_us:       time spent scanning keyframe indices.
        seek_us:             time spent seeking.
        decode_us:           total CPU decode time across all threads.
        io_us:               time spent on I/O.
        files_processed:     number of files decoded.
        keyframes_decoded:   total keyframes decoded.
        keyframes_skipped:   keyframes skipped.
        threads_used:        number of worker threads.
        peak_decode_memory:  peak memory during decode (bytes).
        total_output_bytes:  total output bytes produced.
        atlas_scratch_bytes: scratch memory for atlas composition.
        hw_accel_backend:    HW accel device used (``"d3d11va"``,
                             ``"cuda"``, ``"qsv"``, ``"vaapi"``, or
                             ``""`` when software-only).
        encoder:             JPEG encoder library (e.g. ``"libturbojpeg"``).
        levels:              per-level metrics (list of dicts).
        file_stats:          per-file stats (list of dicts).  Each dict
                             includes ``codec_name``, ``source_width``,
                             ``source_height``, and ``hw_accel_used``.
    """

    def __init__(self, **kwargs):
        self.total_wall_us: int = kwargs.get("total_wall_us", 0)
        self.index_scan_us: int = kwargs.get("index_scan_us", 0)
        self.seek_us: int = kwargs.get("seek_us", 0)
        self.decode_us: int = kwargs.get("decode_us", 0)
        self.io_us: int = kwargs.get("io_us", 0)
        self.files_processed: int = kwargs.get("files_processed", 0)
        self.keyframes_decoded: int = kwargs.get("keyframes_decoded", 0)
        self.keyframes_skipped: int = kwargs.get("keyframes_skipped", 0)
        self.threads_used: int = kwargs.get("threads_used", 0)
        self.peak_decode_memory: int = kwargs.get("peak_decode_memory", 0)
        self.total_output_bytes: int = kwargs.get("total_output_bytes", 0)
        self.atlas_scratch_bytes: int = kwargs.get("atlas_scratch_bytes", 0)
        self.hw_accel_backend: str = kwargs.get("hw_accel_backend", "")
        self.encoder: str = kwargs.get("encoder", "")
        self.levels: list = kwargs.get("levels", [])
        self.file_stats: list = kwargs.get("file_stats", [])

    @property
    def decode_fps(self) -> float:
        if self.decode_us > 0:
            return self.keyframes_decoded / (self.decode_us / 1e6)
        return 0.0

    @property
    def pipeline_fps(self) -> float:
        if self.total_wall_us > 0:
            return self.keyframes_decoded / (self.total_wall_us / 1e6)
        return 0.0

    _STRATEGY_NAMES = {
        0: "container_index",
        1: "packet_scan",
        2: "decode_scan",
        3: "forced_interval",
        4: "skipped",
    }

    def log_summary(self) -> str:
        """Return a multi-line string summarising the decode pipeline.

        Intended for logging — includes wall time, decoder/encoder info,
        HW backend, per-file codec details, and per-level output stats.

        Example::

            import logging
            result = batch_decode(["video.mp4"])
            logging.info(result.metrics.log_summary())
        """
        lines = []
        wall_ms = self.total_wall_us / 1000
        lines.append(
            f"vex decode  wall={wall_ms:.1f}ms  "
            f"frames={self.keyframes_decoded}  "
            f"fps={self.pipeline_fps:.1f}"
        )
        hw = self.hw_accel_backend or "none"
        lines.append(
            f"  hw_accel={hw}  encoder={self.encoder}  threads={self.threads_used}"
        )

        for fs in self.file_stats:
            strat = self._STRATEGY_NAMES.get(fs.get("index_strategy", 4), "unknown")
            hw_flag = "hw" if fs.get("hw_accel_used") else "sw"
            lines.append(
                f"  file[{fs['file_index']}]  "
                f"codec={fs.get('codec_name', '?')}  "
                f"{fs.get('source_width', 0)}x{fs.get('source_height', 0)}  "
                f"decode={hw_flag}  strategy={strat}  "
                f"frames={fs.get('keyframe_count', 0)}  "
                f"time={fs.get('decode_us', 0) / 1000:.1f}ms"
            )

        for i, lm in enumerate(self.levels):
            lines.append(
                f"  level[{i}]  {lm.get('width', 0)}x{lm.get('height', 0)}  "
                f"q={lm.get('quality', 0)}  fmt={lm.get('output_format', '?')}  "
                f"frames={lm.get('frame_count', 0)}  "
                f"output={lm.get('output_bytes', 0)} bytes"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        wall_ms = self.total_wall_us / 1000
        return (
            f"DecodeMetrics(wall={wall_ms:.1f}ms, "
            f"files={self.files_processed}, "
            f"keyframes={self.keyframes_decoded}, "
            f"pipeline_fps={self.pipeline_fps:.1f})"
        )


# ---------------------------------------------------------------------------
# Disk cache reader
# ---------------------------------------------------------------------------

# Cache file format (from types.h):
#   [JPEG frames concatenated] [offset table: int64[frame_count+1]] [CacheHeader: 64 bytes]
# CacheHeader is at the END of the file (last 64 bytes).

_CACHE_MAGIC = 0x00584556  # 'VEX\0'
_CACHE_VERSION = 1
_HEADER_SIZE = 64
_HEADER_FMT = "<IIIIqqQiiII8s"  # matches CacheHeader layout


class CachedLevel:
    """Read-only accessor for a vex disk cache file.

    Opens the cache file, validates the header, memory-maps the data region,
    and provides indexed access to individual JPEG frames.

    Usage::

        with CachedLevel("thumbnails.vex") as cache:
            for i in range(len(cache)):
                jpeg_bytes = cache[i]
    """

    def __init__(self, path: str):
        self._path = path
        self._mm = None
        self._file = None

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cache file not found: {path}")

        self._file = open(path, "rb")
        file_size = os.path.getsize(path)

        if file_size < _HEADER_SIZE:
            self._file.close()
            raise ValueError(f"Cache file too small ({file_size} bytes): {path}")

        # Read the 64-byte header from the end of the file
        self._file.seek(file_size - _HEADER_SIZE)
        header_bytes = self._file.read(_HEADER_SIZE)

        parsed = struct.unpack(_HEADER_FMT, header_bytes)
        magic = parsed[0]
        version = parsed[1]
        frame_count = parsed[2]
        # reserved1 = parsed[3]
        offset_table_pos = parsed[4]
        # metadata_pos = parsed[5]
        # source_hash = parsed[6]
        self._width = parsed[7]
        self._height = parsed[8]
        self._quality = parsed[9]
        # output_format = parsed[10]
        # padding = parsed[11]

        if magic != _CACHE_MAGIC:
            self._file.close()
            raise ValueError(
                f"Invalid cache magic: 0x{magic:08X} (expected 0x{_CACHE_MAGIC:08X})"
            )
        if version != _CACHE_VERSION:
            self._file.close()
            raise ValueError(
                f"Unsupported cache version: {version} (expected {_CACHE_VERSION})"
            )

        self._frame_count = frame_count

        # Read the offset table: frame_count + 1 int64 entries
        self._file.seek(offset_table_pos)
        offset_bytes = self._file.read((frame_count + 1) * 8)
        self._offsets = struct.unpack(f"<{frame_count + 1}q", offset_bytes)

        # Memory-map the file for fast random access to JPEG data
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self._frame_count

    def __getitem__(self, idx: int) -> bytes:
        """Return the JPEG bytes for frame at index *idx*."""
        if idx < 0:
            idx += self._frame_count
        if idx < 0 or idx >= self._frame_count:
            raise IndexError(f"Frame index {idx} out of range [0, {self._frame_count})")
        start = self._offsets[idx]
        end = self._offsets[idx + 1]
        return bytes(self._mm[start:end])

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def quality(self) -> int:
        return self._quality

    @property
    def path(self) -> str:
        return self._path

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        return (
            f"CachedLevel(path={self._path!r}, frames={self._frame_count}, "
            f"size={self._width}x{self._height})"
        )


# ---------------------------------------------------------------------------
# Native module import helper
# ---------------------------------------------------------------------------


def _load_native():
    """Import the C++ _vex_core extension module.

    Raises ImportError with a helpful message if the module is not built.
    """
    # On Windows, add the package directory to the DLL search path so that
    # FFmpeg and TurboJPEG DLLs sitting next to _vex_core.pyd can be found.
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_pkg_dir)

    try:
        from . import _vex_core

        return _vex_core
    except ImportError:
        pass

    try:
        import _vex_core

        return _vex_core
    except ImportError:
        pass

    raise ImportError(
        "vex native module (_vex_core) is not built.\n"
        "Build it with:\n"
        "  mkdir build && cd build\n"
        '  cmake .. -G "Visual Studio 17 2022" -A x64\n'
        "  cmake --build . --config Release\n"
        "See deps/README.md for dependency setup instructions."
    )


# ---------------------------------------------------------------------------
# Helper: convert Python LevelConfig to native C++ LevelConfig
# ---------------------------------------------------------------------------

_VALID_OUTPUTS = {"jpeg_stream", "sprite_atlas"}


def _to_native_levels(_vex_core, levels: List[LevelConfig]):
    """Convert Python LevelConfig list to native _vex_core.LevelConfig list."""
    native = []
    for i, lv in enumerate(levels):
        fmt = lv.output.lower().strip()
        if fmt not in _VALID_OUTPUTS:
            raise ValueError(
                f"Invalid output format {lv.output!r}. Must be one of: {sorted(_VALID_OUTPUTS)}"
            )
        if fmt == "sprite_atlas" and (lv.width <= 0 or lv.height <= 0):
            raise ValueError(
                f"Level {i}: sprite_atlas requires explicit width and height "
                f"(got {lv.width}x{lv.height}).  NATIVE (0) is only "
                f"supported with jpeg_stream output."
            )
        native.append(
            _vex_core.LevelConfig(
                width=lv.width,
                height=lv.height,
                quality=lv.quality,
                output=fmt,
                in_memory=lv.in_memory,
                cache_path=lv.cache_path or "",
                atlas_columns=lv.atlas_columns,
            )
        )
    return native


# ---------------------------------------------------------------------------
# Helper: wrap raw C++ results into Python classes
# ---------------------------------------------------------------------------


def _wrap_level_result(raw) -> Union[JpegStreamResult, SpriteAtlasResult, DiskResult]:
    """Convert a raw C++ level result dict into the appropriate Python wrapper."""
    fmt = raw.get("format", "jpeg_stream")

    if "blobs" in raw:
        # JPEG stream in-memory result
        return JpegStreamResult(
            blobs=raw.get("blobs", []),
            offsets=raw.get("offsets", []),
            metadata=raw.get("metadata", np.empty((0, 3), dtype=np.int64)),
        )
    elif "atlas" in raw:
        # Sprite atlas result — data is in the nested "atlas" dict
        atlas = raw["atlas"]
        return SpriteAtlasResult(
            blob=atlas.get("blob", np.array([], dtype=np.uint8)),
            offsets=atlas.get("offsets", np.array([], dtype=np.int64)),
            grid_w=atlas.get("grid_w", 0),
            grid_h=atlas.get("grid_h", 0),
            thumb_w=atlas.get("thumb_w", 0),
            thumb_h=atlas.get("thumb_h", 0),
            file_count=atlas.get("file_count", 0),
        )
    elif "disk" in raw:
        # Disk output result — data is in the nested "disk" dict
        disk = raw["disk"]
        return DiskResult(
            cache_path=disk.get("cache_path", ""),
            frame_count=disk.get("frame_count", 0),
            total_bytes=disk.get("total_bytes", 0),
            metadata=disk.get("metadata", np.empty((0, 3), dtype=np.int64)),
        )
    else:
        # Fallback: empty result based on format
        if fmt == "sprite_atlas":
            return SpriteAtlasResult(
                blob=np.array([], dtype=np.uint8),
                offsets=np.array([], dtype=np.int64),
                grid_w=0,
                grid_h=0,
                thumb_w=0,
                thumb_h=0,
                file_count=0,
            )
        elif not raw.get("in_memory", True):
            return DiskResult(
                cache_path="",
                frame_count=0,
                total_bytes=0,
                metadata=np.empty((0, 3), dtype=np.int64),
            )
        else:
            return JpegStreamResult(
                blobs=[], offsets=[], metadata=np.empty((0, 3), dtype=np.int64)
            )


def _wrap_metrics(raw) -> DecodeMetrics:
    """Convert a raw C++ metrics dict into a DecodeMetrics instance."""
    if isinstance(raw, DecodeMetrics):
        return raw
    return DecodeMetrics(**raw)


# ---------------------------------------------------------------------------
# Synchronous batch decode
# ---------------------------------------------------------------------------


def batch_decode(
    paths: List[str],
    levels: Optional[List[LevelConfig]] = None,
    *,
    max_threads: int = 8,
    keyframes_only: bool = False,
    frame_skip: int = 1,
    use_hw_accel: bool = True,
) -> BatchResult:
    """Decode video frames from one or more video files.

    By default, decodes every frame sequentially.  Set
    ``keyframes_only=True`` to decode only I-frames (faster, fewer
    thumbnails).

    The entire pipeline runs in C++ with the GIL released.

    Args:
        paths:  List of video file paths (mp4, mkv, avi, ts, flv, ...).
        levels: One or more :class:`LevelConfig` describing the output
                resolutions / formats.  Defaults to a single
                native-resolution JPEG stream at quality 100.  Multiple
                levels produce multiple output resolutions from a single
                decode pass.
        max_threads:    Maximum worker threads.  Clamped to
                        ``min(max_threads, len(paths), cpu_count)``.
        keyframes_only: If ``True``, only I-frames are decoded.
        frame_skip:     Decode every *N*-th frame.  Only used when
                        ``keyframes_only=False``.
        use_hw_accel:   If ``False``, skip GPU-accelerated decoding and
                        use software (CPU) decode only.  Useful when HW
                        decode produces incorrect output for certain
                        videos.

    Returns:
        A :class:`BatchResult`.  Access typed results via
        :meth:`~BatchResult.jpeg_stream`, :meth:`~BatchResult.sprite_atlas`,
        or :meth:`~BatchResult.disk`.  Tuple unpacking is also supported
        for convenience.

    Example — typed access (preferred)::

        result = batch_decode(["video.mp4"])
        stream = result.jpeg_stream()        # -> JpegStreamResult
        jpeg   = stream.jpeg_bytes(0, 0)     # first file, first keyframe
        print(result.metrics)

    Example — tuple unpacking (backward-compatible)::

        results, metrics = batch_decode(["video.mp4"])
        stream = results[0]                  # -> Union[...] (untyped)

    Example — native resolution at max quality (the default)::

        result = batch_decode(["video.mp4"])

    Example — two levels in one pass::

        result = batch_decode(
            ["video.mp4"],
            levels=[
                LevelConfig(width=48, height=48, quality=30,
                            output="sprite_atlas"),
                LevelConfig(width=320, height=240, quality=90),
            ],
        )
        atlas  = result.sprite_atlas(0)   # -> SpriteAtlasResult
        stream = result.jpeg_stream(1)    # -> JpegStreamResult

    Raises:
        ImportError:  If the native ``_vex_core`` extension is not built.
        ValueError:   If a :class:`LevelConfig` has invalid settings
                      (e.g. ``sprite_atlas`` with ``width=NATIVE``).
    """
    _vex_core = _load_native()

    if levels is None:
        levels = [LevelConfig()]

    native_levels = _to_native_levels(_vex_core, levels)
    raw_results, raw_metrics = _vex_core.batch_decode(
        list(paths),
        native_levels,
        max_threads,
        keyframes_only,
        frame_skip,
        use_hw_accel,
    )

    level_results = [_wrap_level_result(r) for r in raw_results]
    metrics = _wrap_metrics(raw_metrics)

    return BatchResult(level_results, metrics)


# ---------------------------------------------------------------------------
# Asynchronous batch decode
# ---------------------------------------------------------------------------


class DecodeHandle:
    """Handle for an in-progress asynchronous decode operation.

    Provides non-blocking progress inspection, event draining, and
    JPEG peeking while decode threads are still running.
    """

    def __init__(self, native_handle):
        self._handle = native_handle

    @property
    def done(self) -> bool:
        """True when all decode work has completed."""
        return self._handle.done

    def drain_events(self) -> list:
        """Drain all pending FrameEvent objects from worker threads.

        Returns a list of event dicts with keys:
        file_index, frame_index, level_index, pts_ms, blob_offset, jpeg_size.
        """
        return self._handle.drain_events()

    def peek_jpeg(self, evt) -> bytes:
        """Read JPEG bytes for a drained event without copying the full blob.

        Args:
            evt: An event dict from drain_events().

        Returns:
            The raw JPEG bytes for this frame.
        """
        return bytes(self._handle.peek_jpeg(evt))

    @property
    def progress(self) -> dict:
        """Current progress as a dict with keys:
        keyframes_decoded, files_completed, total_files.
        """
        return self._handle.progress

    def peek_stream(
        self, file_index: int = 0, level_index: int = 0
    ) -> Optional[memoryview]:
        """Zero-copy view into the JPEG stream buffer for a file/level.

        Returns a memoryview backed by stable virtual memory that never
        moves.  The view's length reflects the current number of bytes
        written; it may be larger on subsequent calls as more frames are
        decoded.

        Use ``drain_events()`` to discover the offset and size of each
        JPEG within this buffer (``blob_offset`` and ``jpeg_size``).

        The memoryview is valid while this DecodeHandle is alive.
        After calling ``result()``, the underlying memory is released.

        Returns:
            A read-only memoryview, or None if unavailable.
        """
        return self._handle.peek_stream(file_index, level_index)

    def peek_stream_numpy(
        self, file_index: int = 0, level_index: int = 0
    ) -> Optional[np.ndarray]:
        """Zero-copy numpy view into the JPEG stream buffer.

        Like ``peek_stream()`` but returns a numpy uint8 array instead
        of a memoryview.  The array shares memory with the underlying
        buffer (no copy).

        Returns:
            A numpy uint8 array, or None if unavailable.
        """
        mv = self.peek_stream(file_index, level_index)
        if mv is None:
            return None
        return np.frombuffer(mv, dtype=np.uint8)

    def result(self) -> BatchResult:
        """Block until decode completes and return a :class:`BatchResult`.

        Same return type as :func:`batch_decode`.  After this call,
        streaming views from ``peek_stream()`` are invalidated and
        VirtualBlob reservations are released.
        """
        raw_results, raw_metrics = self._handle.result()
        level_results = [_wrap_level_result(r) for r in raw_results]
        metrics = _wrap_metrics(raw_metrics)
        return BatchResult(level_results, metrics)

    def __repr__(self) -> str:
        state = "done" if self.done else "running"
        return f"DecodeHandle({state})"


def batch_decode_async(
    paths: List[str],
    levels: Optional[List[LevelConfig]] = None,
    *,
    max_threads: int = 8,
    keyframes_only: bool = False,
    frame_skip: int = 1,
    use_hw_accel: bool = True,
    blob_reservation: int = 0,
) -> DecodeHandle:
    """Start an asynchronous batch decode operation.

    Same parameters as :func:`batch_decode`, but returns immediately with
    a :class:`DecodeHandle` that can be polled for progress and
    streamed events while decode threads are still running.

    Args:
        blob_reservation: Virtual memory reservation per file per level
                          in bytes.  0 (default) = 256 MB.  Only affects
                          in-memory JPEG stream output.  The reservation
                          is address space only — physical memory is
                          committed on demand as frames are written.

    Returns:
        A :class:`DecodeHandle`.  Call ``handle.result()`` to block until
        completion and retrieve the same ``(results, metrics)`` tuple
        returned by :func:`batch_decode`.
    """
    _vex_core = _load_native()

    if levels is None:
        levels = [LevelConfig()]

    native_levels = _to_native_levels(_vex_core, levels)
    native_handle = _vex_core.batch_decode_async(
        list(paths),
        native_levels,
        max_threads,
        keyframes_only,
        frame_skip,
        use_hw_accel,
        blob_reservation,
    )

    return DecodeHandle(native_handle)


# ---------------------------------------------------------------------------
# Probe decoder threading
# ---------------------------------------------------------------------------


def probe_decode_threads(path: str, decode_threads: int = 0) -> int:
    """Return how many threads FFmpeg will use to decode a video file.

    Opens the file, initialises the codec, and queries the actual thread
    count chosen by FFmpeg.  A return value of ``1`` means the codec does
    not support multi-threaded decoding.

    Args:
        path:           Path to a video file.
        decode_threads: Thread count hint.  ``0`` (default) lets FFmpeg
                        auto-detect based on CPU cores and codec
                        capabilities.  A positive value requests at most
                        that many threads (FFmpeg may use fewer if the
                        codec doesn't support it).

    Returns:
        The actual decoder thread count (``>= 1``), or ``0`` if the file
        could not be opened / decoded.
    """
    _vex_core = _load_native()
    return _vex_core.probe_decode_threads(path, decode_threads)


def probe_batch_threads(
    paths: Union[str, List[str]],
    *,
    max_threads: int = 8,
) -> int:
    """Return the number of threads a batch decode job will use.

    Accepts a directory path or a list of file paths. Sums per-file
    decoder thread counts capped to *max_threads*.
    """
    if isinstance(paths, str):
        p = os.path.abspath(paths)
        if os.path.isdir(p):
            file_list = [
                os.path.join(p, f)
                for f in sorted(os.listdir(p))
                if os.path.isfile(os.path.join(p, f))
            ]
        else:
            file_list = [p]
    else:
        file_list = list(paths)

    total = sum(probe_decode_threads(fp) for fp in file_list)
    return min(total, max_threads)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "NATIVE",
    "LevelConfig",
    "BatchResult",
    "JpegStreamResult",
    "SpriteAtlasResult",
    "DiskResult",
    "DecodeMetrics",
    "CachedLevel",
    "DecodeHandle",
    "batch_decode",
    "batch_decode_async",
    "probe_decode_threads",
    "probe_batch_threads",
]

__version__ = "0.1.0"
