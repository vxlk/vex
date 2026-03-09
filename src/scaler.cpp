#include "scaler.h"

extern "C" {
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
#include <libavutil/pixfmt.h>
}

#include <cstring>
#include <algorithm>

namespace vex {

// ── Constructor ─────────────────────────────────────────────────────────────

FrameScaler::FrameScaler(int src_w, int src_h, int src_fmt, int dst_w, int dst_h)
    : identity_(src_w == dst_w && src_h == dst_h &&
                (src_fmt == AV_PIX_FMT_YUV420P || src_fmt == AV_PIX_FMT_YUVJ420P)),
      src_w_(src_w),
      src_h_(src_h),
      dst_w_(dst_w),
      dst_h_(dst_h) {
    if (!identity_) {
        sws_ctx_ = sws_getContext(src_w, src_h, static_cast<AVPixelFormat>(src_fmt), dst_w, dst_h,
                                  AV_PIX_FMT_YUV420P, SWS_BILINEAR, nullptr, nullptr, nullptr);
    }
}

// ── Destructor ──────────────────────────────────────────────────────────────

FrameScaler::~FrameScaler() {
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
    }
}

// ── scale ───────────────────────────────────────────────────────────────────

void FrameScaler::scale(const AVFrame* src, uint8_t* dst_y, uint8_t* dst_u, uint8_t* dst_v,
                        int dst_y_stride, int dst_uv_stride) {
    if (identity_) {
        // Y plane: src_h_ lines, each src_w_ bytes wide
        for (int row = 0; row < src_h_; ++row) {
            std::memcpy(dst_y + row * dst_y_stride, src->data[0] + row * src->linesize[0],
                        static_cast<size_t>(src_w_));
        }

        // U plane: src_h_/2 lines, each src_w_/2 bytes wide
        int chroma_h = src_h_ / 2;
        int chroma_w = src_w_ / 2;
        for (int row = 0; row < chroma_h; ++row) {
            std::memcpy(dst_u + row * dst_uv_stride, src->data[1] + row * src->linesize[1],
                        static_cast<size_t>(chroma_w));
        }

        // V plane: src_h_/2 lines, each src_w_/2 bytes wide
        for (int row = 0; row < chroma_h; ++row) {
            std::memcpy(dst_v + row * dst_uv_stride, src->data[2] + row * src->linesize[2],
                        static_cast<size_t>(chroma_w));
        }
    } else {
        uint8_t* dst_planes[3] = {dst_y, dst_u, dst_v};
        int dst_strides[3] = {dst_y_stride, dst_uv_stride, dst_uv_stride};

        sws_scale(sws_ctx_, src->data, src->linesize, 0, src_h_, dst_planes, dst_strides);
    }
}

}  // namespace vex
