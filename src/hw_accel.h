#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

#include <optional>

namespace vex {

struct HWAccelContext {
    AVBufferRef*   device_ctx  = nullptr;
    AVHWDeviceType device_type = AV_HWDEVICE_TYPE_NONE;
    AVPixelFormat  hw_pix_fmt  = AV_PIX_FMT_NONE;
};

// Try NVDEC → QSV → D3D11VA → VAAPI in order. Returns nullopt if all fail.
std::optional<HWAccelContext> probe_hw_accel(const AVCodec* codec);

// Release the HW device context.
void release_hw_accel(HWAccelContext& ctx);

// Copy a hardware-surface frame to system memory (YUV420P).
bool transfer_hw_frame(const AVFrame* hw_frame, AVFrame* sw_frame);

// AVCodecContext::get_format callback for HW decode.
AVPixelFormat hw_get_format(AVCodecContext* ctx, const AVPixelFormat* pix_fmts);

} // namespace vex
