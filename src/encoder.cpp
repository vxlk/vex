#include "encoder.h"

#include <stdexcept>
#include <turbojpeg.h>

namespace vex {

// ── Constructor ─────────────────────────────────────────────────────────────

JpegEncoder::JpegEncoder() {
    tj_ = tjInitCompress();
    if (!tj_) {
        throw std::runtime_error("tjInitCompress failed");
    }
}

// ── Destructor ──────────────────────────────────────────────────────────────

JpegEncoder::~JpegEncoder() {
    if (tj_) {
        tjDestroy(tj_);
    }
}

// ── max_jpeg_size ───────────────────────────────────────────────────────────

size_t JpegEncoder::max_jpeg_size(int width, int height) {
    return static_cast<size_t>(tjBufSize(width, height, TJSAMP_420));
}

// ── encode ──────────────────────────────────────────────────────────────────

size_t JpegEncoder::encode(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                           int y_stride, int uv_stride,
                           int width, int height, int quality,
                           uint8_t* output, size_t capacity)
{
    const unsigned char* planes[3] = { y, u, v };
    int strides[3] = { y_stride, uv_stride, uv_stride };

    unsigned char* jpeg_buf  = output;
    unsigned long  jpeg_size = static_cast<unsigned long>(capacity);

    // Use fast DCT for quality <= 90 where rounding differences are
    // absorbed by quantization and produce no visible artifacts.
    int flags = TJFLAG_NOREALLOC;
    if (quality <= 90) {
        flags |= TJFLAG_FASTDCT;
    }

    int ret = tjCompressFromYUVPlanes(
        tj_,
        planes,
        width,
        strides,
        height,
        TJSAMP_420,
        &jpeg_buf,
        &jpeg_size,
        quality,
        flags);

    if (ret != 0) {
        return 0;
    }

    return static_cast<size_t>(jpeg_size);
}

} // namespace vex
