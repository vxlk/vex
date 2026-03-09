#pragma once

extern "C" {
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

#include <cstdint>

namespace vex {

class FrameScaler {
public:
    FrameScaler(int src_w, int src_h, int src_fmt, int dst_w, int dst_h);
    ~FrameScaler();

    FrameScaler(const FrameScaler&) = delete;
    FrameScaler& operator=(const FrameScaler&) = delete;

    bool needs_scaling() const { return !identity_; }
    int dst_width() const { return dst_w_; }
    int dst_height() const { return dst_h_; }

    // Scale + convert src frame into YUV420P destination plane buffers.
    // Handles pixel format conversion and scaling in a single pass.
    // If identity (src == dst dimensions and already YUV420P), copies directly.
    void scale(const AVFrame* src, uint8_t* dst_y, uint8_t* dst_u, uint8_t* dst_v, int dst_y_stride,
               int dst_uv_stride);

private:
    SwsContext* sws_ctx_ = nullptr;
    bool identity_;
    int src_w_, src_h_, dst_w_, dst_h_;
};

}  // namespace vex
