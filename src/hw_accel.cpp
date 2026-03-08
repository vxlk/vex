#include "hw_accel.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

#include <mutex>

namespace vex {

// ── Cached HW context (process-global singleton) ────────────────────────────

static std::once_flag g_hw_probe_flag;
static std::optional<HWAccelContext> g_hw_cached;

std::optional<HWAccelContext> get_cached_hw_accel() {
    std::call_once(g_hw_probe_flag, []() {
        const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
        if (codec) {
            g_hw_cached = probe_hw_accel(codec);
        }
    });

    if (!g_hw_cached.has_value()) {
        return std::nullopt;
    }

    // Return a copy with a new reference to the same device context.
    // Each caller (FileDecoder) will av_buffer_ref() this in its own
    // codec context, so the underlying device stays alive as long as
    // any codec context holds a reference.
    HWAccelContext copy{};
    copy.device_ctx  = av_buffer_ref(g_hw_cached->device_ctx);
    copy.device_type = g_hw_cached->device_type;
    copy.hw_pix_fmt  = g_hw_cached->hw_pix_fmt;
    return copy;
}

// ── probe_hw_accel ──────────────────────────────────────────────────────────

std::optional<HWAccelContext> probe_hw_accel(const AVCodec* codec) {
    static const AVHWDeviceType types_to_try[] = {
        AV_HWDEVICE_TYPE_CUDA,
        AV_HWDEVICE_TYPE_QSV,
        AV_HWDEVICE_TYPE_D3D11VA,
        AV_HWDEVICE_TYPE_VAAPI,
    };

    for (auto device_type : types_to_try) {
        AVBufferRef* device_ctx = nullptr;
        int ret = av_hwdevice_ctx_create(&device_ctx, device_type, nullptr, nullptr, 0);
        if (ret < 0) {
            continue;
        }

        // Determine the hardware pixel format for this device type by
        // iterating the codec's supported hw_configs.
        AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
        for (int i = 0; ; ++i) {
            const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
            if (!config) {
                break;
            }
            if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
                config->device_type == device_type) {
                hw_pix_fmt = config->pix_fmt;
                break;
            }
        }

        if (hw_pix_fmt == AV_PIX_FMT_NONE) {
            av_buffer_unref(&device_ctx);
            continue;
        }

        HWAccelContext ctx{};
        ctx.device_ctx  = device_ctx;
        ctx.device_type = device_type;
        ctx.hw_pix_fmt  = hw_pix_fmt;
        return ctx;
    }

    return std::nullopt;
}

// ── release_hw_accel ────────────────────────────────────────────────────────

void release_hw_accel(HWAccelContext& ctx) {
    if (ctx.device_ctx) {
        av_buffer_unref(&ctx.device_ctx);
    }
    ctx.device_ctx  = nullptr;
    ctx.device_type = AV_HWDEVICE_TYPE_NONE;
    ctx.hw_pix_fmt  = AV_PIX_FMT_NONE;
}

// ── transfer_hw_frame ───────────────────────────────────────────────────────

bool transfer_hw_frame(const AVFrame* hw_frame, AVFrame* sw_frame) {
    int ret = av_hwframe_transfer_data(sw_frame, hw_frame, 0);
    return ret >= 0;
}

// ── hw_get_format ───────────────────────────────────────────────────────────

AVPixelFormat hw_get_format(AVCodecContext* ctx, const AVPixelFormat* pix_fmts) {
    // The HWAccelContext pointer is stored in codec_ctx->opaque.
    auto* hw_ctx = static_cast<HWAccelContext*>(ctx->opaque);

    if (hw_ctx) {
        for (const AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
            if (*p == hw_ctx->hw_pix_fmt) {
                return *p;
            }
        }
    }

    // Fallback: return the first format in the list.
    return pix_fmts[0];
}

} // namespace vex
