"""Integrate vex with numpy and Pillow without copying the JPEG data.

Demonstrates that vex's output blobs are zero-copy numpy arrays backed by
C++ memory: you slice them with the offset table, hand the slice to Pillow
for pixel decoding, and build a (N, H, W, 3) pixel tensor -- all without
ever duplicating the compressed JPEG bytes on the Python side.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import io

import numpy as np
from PIL import Image
from vex import LevelConfig, batch_decode

VIDEO = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "formats", "h264_mp4.mp4"
)

result = batch_decode(
    paths=[VIDEO],
    levels=[LevelConfig(width=320, height=240, quality=85)],
)

stream = result.jpeg_stream()

# ---------------------------------------------------------------------------
# JPEG blob — uint8, zero-copy from C++
# ---------------------------------------------------------------------------
# stream.blobs[file_index] is a contiguous np.ndarray(dtype=uint8) backed
# directly by the C++ malloc'd buffer via pybind11 capsule.  No Python-side
# copy occurs — numpy owns a view into the same memory.

jpeg_data: np.ndarray = stream.blobs[0]  # shape (N,), dtype uint8
assert jpeg_data.dtype == np.uint8
assert jpeg_data.base is not None, "expected zero-copy view into C++ memory"
print(
    f"JPEG blob: {jpeg_data.shape[0]:,} bytes, dtype {jpeg_data.dtype}, "
    f"owns_data={jpeg_data.flags.owndata}"
)

# ---------------------------------------------------------------------------
# Offset indices — int64 from C++ (zero-copy), int32 requires one cast-copy
# ---------------------------------------------------------------------------
# stream.offsets[file_index] holds byte offsets into the blob.  The C++ layer
# produces int64; these are zero-copy.  Casting to int32 is safe when the
# total blob is < 2 GB and requires exactly one small copy for the index array.

offsets_i64: np.ndarray = stream.offsets[0]  # zero-copy int64 from C++
assert offsets_i64.dtype == np.int64

# Cast to int32 (one copy of just the index array, not the blob data).
offsets_i32: np.ndarray = offsets_i64.astype(np.int32)
assert offsets_i32.dtype == np.int32

n_frames = len(offsets_i32)
print(f"Frames: {n_frames}, offsets dtype {offsets_i32.dtype}")

# ---------------------------------------------------------------------------
# Extract and decode individual JPEGs using the index array
# ---------------------------------------------------------------------------
for i in range(n_frames):
    # Slice the blob using the int32 offset array — this is a numpy view,
    # no copy.  The JPEG bytes stay in the original C++ buffer.
    start = int(offsets_i32[i])
    end = int(offsets_i32[i + 1]) if i + 1 < n_frames else jpeg_data.shape[0]
    frame_view: np.ndarray = jpeg_data[start:end]

    # Verify the slice is a view (shares memory), not a copy
    assert frame_view.base is jpeg_data or frame_view.base is jpeg_data.base, (
        "slice should be a zero-copy view"
    )

    # Decode JPEG with Pillow — Pillow reads from the buffer and decompresses.
    # BytesIO copies the bytes into its internal buffer (unavoidable for the
    # Pillow file-like interface), but the source blob itself is never copied.
    img = Image.open(io.BytesIO(frame_view))
    img.load()  # force decode

    print(
        f"  Frame {i}: offset {start:>8,}  size {end - start:>7,} bytes  "
        f"-> {img.size[0]}x{img.size[1]} {img.mode}"
    )

# ---------------------------------------------------------------------------
# Bulk: build a pixels array from all frames
# ---------------------------------------------------------------------------
images = []
for i in range(n_frames):
    start = int(offsets_i32[i])
    end = int(offsets_i32[i + 1]) if i + 1 < n_frames else jpeg_data.shape[0]
    img = Image.open(io.BytesIO(jpeg_data[start:end]))
    images.append(np.asarray(img))  # HWC uint8 pixel array

if images:
    pixels = np.stack(images)  # shape (N, H, W, 3), dtype uint8
    print(
        f"\nPixel array: shape {pixels.shape}, dtype {pixels.dtype}, {pixels.nbytes / 1024:.1f} KB"
    )

print("\nDone — all assertions passed, zero-copy blob access verified.")
