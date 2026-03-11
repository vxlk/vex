#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

#include <optional>
#include <mutex>

namespace vex {

struct HWAccelContext {
    AVBufferRef* device_ctx = nullptr;
    AVHWDeviceType device_type = AV_HWDEVICE_TYPE_NONE;
    AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
};

// Try NVDEC → QSV → D3D11VA → VAAPI in order. Returns nullopt if all fail.
std::optional<HWAccelContext> probe_hw_accel(const AVCodec* codec);

// Get a cached HW context. Probes once on first call; subsequent calls return
// a reference to the same device context (the AVBufferRef is ref-counted so
// each codec context that uses it gets its own reference via av_buffer_ref).
// Thread-safe via std::call_once.
std::optional<HWAccelContext> get_cached_hw_accel();

// Release the HW device context.
void release_hw_accel(HWAccelContext& ctx);

// Copy a hardware-surface frame to system memory (YUV420P).
bool transfer_hw_frame(const AVFrame* hw_frame, AVFrame* sw_frame);

// AVCodecContext::get_format callback for HW decode.
AVPixelFormat hw_get_format(AVCodecContext* ctx, const AVPixelFormat* pix_fmts);

// Check whether a given codec can be HW-decoded on the given device type.
// Queries FFmpeg's codec hw_config list at runtime (in-memory, no I/O).
bool can_hw_decode(AVHWDeviceType device_type, AVCodecID codec_id);

// For CUDA devices, return the name of the dedicated cuvid decoder for a
// codec (e.g. "h264_cuvid"), or nullptr if none exists.
// Scans FFmpeg's codec registry at runtime (in-memory, no I/O).
const char* get_cuvid_decoder_name(AVCodecID codec_id);

}  // namespace vex
