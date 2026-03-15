#pragma once

#include "hw_accel.h"
#include "encoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdint>

namespace vex {

// ── YUV buffer for per-thread scaling ───────────────────────────────────────

struct YUVBuffer {
    std::vector<uint8_t> y, u, v;
    int width = 0;
    int height = 0;

    void ensure(int w, int h) {
        if (w == width && h == height)
            return;
        width = w;
        height = h;
        y.resize(static_cast<size_t>(w) * h);
        u.resize(static_cast<size_t>(w / 2) * (h / 2));
        v.resize(static_cast<size_t>(w / 2) * (h / 2));
    }
};

// ── Process-wide caches (singleton, survives across batch_decode calls) ──
//
// ProcessCache holds state that is valid for the lifetime of the process
// and should NOT be rebuilt between batch_decode calls.  This includes
// one-time probe results (HW compat) and learned heuristics (probe hints).
// Thread-safe: each map has its own mutex.

struct ProcessCache {
    // --- HW codec compatibility map ---
    // Maps AVCodecID (cast to int) → whether the codec can be HW-decoded
    // on the current device.  Populated lazily by workers in orchestrator.cpp.
    // Previously lived as file-scope globals g_hw_compat_cache / g_hw_cache_mutex
    // in orchestrator.cpp; moved here for consolidated ownership.
    std::mutex hw_compat_mutex;
    std::unordered_map<int, bool> hw_compat;

    // --- Format probe hints ---
    // After the first file of each container format (e.g. "mov,mp4,m4a,3gp,3g2,mj2")
    // completes avformat_find_stream_info() successfully, we record optimized
    // probesize / analyzeduration values.  Subsequent files of the same format
    // use these reduced values, cutting I/O during stream probing.
    //
    // probesize: max bytes FFmpeg reads to determine stream params (default 5 MB).
    // analyze_duration: max microseconds FFmpeg analyzes (default 5,000,000 = 5s).
    // A value of 0 means "use FFmpeg's built-in default" (no override).
    struct ProbeHint {
        int64_t probesize = 0;
        int64_t analyze_duration = 0;
    };
    std::mutex probe_hints_mutex;
    std::unordered_map<std::string, ProbeHint> probe_hints;

    static ProcessCache& instance() {
        static ProcessCache s;
        return s;
    }

private:
    ProcessCache() = default;
};

// ── Per-worker caches (one per thread, reused across files within a batch) ──
//
// ThreadCache consolidates all per-worker resources that were previously
// local variables in worker_func (orchestrator.cpp).  Grouping them here
// gives clear ownership semantics and makes it obvious what survives
// across files within a single worker's loop.
//
// Previously each worker created:
//   JpegEncoder encoder;              — TurboJPEG compressor handle
//   AVFrame* decode_frame;            — reusable decode target
//   std::vector<YUVBuffer> scale_bufs; — per-level YUV plane buffers
// All were destroyed and recreated per-batch.  Now they live here.

struct ThreadCache {
    // --- JPEG encoder (TurboJPEG compressor handle) ---
    // One tjhandle per thread — TurboJPEG is thread-safe only when each
    // thread uses its own compressor instance.
    JpegEncoder encoder;

    // --- Reusable decode frame ---
    // Single AVFrame allocation reused for every decoded frame across all
    // files.  av_frame_unref() between frames resets data pointers without
    // freeing the AVFrame struct itself.
    AVFrame* decode_frame = nullptr;

    // --- Per-level YUV scale buffers ---
    // YUVBuffer::ensure() is a no-op when dimensions match, so these
    // effectively become free after the first frame of each level.
    std::vector<YUVBuffer> scale_bufs;

    // --- SwsContext pool ---
    // sws_getContext() allocates internal coefficient tables and SIMD-
    // optimized scaler state.  For a batch of same-resolution files with
    // 2 output levels, this pool turns 2*N sws_getContext + sws_freeContext
    // calls into just 2 (one per unique scaling configuration).
    //
    // Key: (src_w, src_h, src_fmt, dst_w, dst_h).
    // dst_fmt is always AV_PIX_FMT_YUV420P (our pipeline's intermediate
    // format before JPEG encoding), so it's omitted from the key.
    struct SwsKey {
        int src_w, src_h, src_fmt, dst_w, dst_h;
        bool operator==(const SwsKey& o) const {
            return src_w == o.src_w && src_h == o.src_h && src_fmt == o.src_fmt
                && dst_w == o.dst_w && dst_h == o.dst_h;
        }
    };
    struct SwsKeyHash {
        size_t operator()(const SwsKey& k) const {
            // Boost-style hash combine.  0x9e3779b9 is the integer part of
            // the golden ratio (2^32 / phi), chosen because it produces a
            // maximally spread bit pattern — XOR with shifted accumulators
            // ensures each field contributes to different bits of the hash,
            // minimizing collisions for small integer keys like dimensions.
            size_t h = 0;
            h ^= std::hash<int>()(k.src_w) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.src_h) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.src_fmt) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.dst_w) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.dst_h) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    std::unordered_map<SwsKey, SwsContext*, SwsKeyHash> sws_pool;

    // Look up or create an SwsContext for the given scaling parameters.
    // Returned pointer is owned by the pool — callers must NOT free it.
    // FrameScaler uses the "borrowed ctx" constructor to hold a non-owning
    // reference (owns_ctx_ = false).
    SwsContext* get_sws(int src_w, int src_h, int src_fmt, int dst_w, int dst_h) {
        SwsKey key{src_w, src_h, src_fmt, dst_w, dst_h};
        auto it = sws_pool.find(key);
        if (it != sws_pool.end()) return it->second;
        SwsContext* ctx = sws_getContext(
            src_w, src_h, static_cast<AVPixelFormat>(src_fmt),
            dst_w, dst_h, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (ctx) sws_pool[key] = ctx;
        return ctx;
    }

    // --- Lifecycle ---

    ThreadCache() {
        decode_frame = av_frame_alloc();
    }

    ~ThreadCache() {
        if (decode_frame) av_frame_free(&decode_frame);
        for (auto& [key, ctx] : sws_pool) {
            sws_freeContext(ctx);
        }
    }

    void init_levels(int num_levels) {
        scale_bufs.resize(static_cast<size_t>(num_levels));
    }

    // Non-copyable (SwsContext* and AVFrame* have no copy semantics)
    ThreadCache(const ThreadCache&) = delete;
    ThreadCache& operator=(const ThreadCache&) = delete;
};

}  // namespace vex
