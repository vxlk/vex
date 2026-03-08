#pragma once

#include <cstdint>
#include <cstddef>
#include <turbojpeg.h>

namespace vex {

class JpegEncoder {
public:
    JpegEncoder();
    ~JpegEncoder();

    JpegEncoder(const JpegEncoder&) = delete;
    JpegEncoder& operator=(const JpegEncoder&) = delete;

    // Encode YUV420P planes directly into output buffer using TJFLAG_NOREALLOC.
    // Returns actual JPEG size in bytes. Returns 0 on failure.
    size_t encode(const uint8_t* y, const uint8_t* u, const uint8_t* v, int y_stride, int uv_stride,
                  int width, int height, int quality, uint8_t* output, size_t capacity);

    // Upper bound on JPEG size for given dimensions (YUV420P).
    static size_t max_jpeg_size(int width, int height);

private:
    tjhandle tj_ = nullptr;
};

}  // namespace vex
