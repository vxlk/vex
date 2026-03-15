#pragma once

#include "types.h"
#include "hw_accel.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
}

#include <string>

namespace vex {

// Lightweight file probe that reads codec_id, width, and height from the
// container's stream parameters WITHOUT allocating or opening a codec context.
//
// A full FileDecoder constructor does: avformat_open_input → avformat_find_stream_info
// → avcodec_alloc_context3 → avcodec_parameters_to_context → avcodec_open2
// (which spins up FFmpeg's internal thread pool for slice/frame threading).
//
// light_probe skips the last three steps — it only needs the demuxer to read
// codecpar from the stream header.  This makes it ~10x cheaper than a full
// FileDecoder when the only goal is reading codec_id + resolution for the
// HW compatibility check in the orchestrator's worker loop.
struct LightProbe {
    AVCodecID codec_id = AV_CODEC_ID_NONE;
    int width = 0;
    int height = 0;
    bool valid = false;
};
LightProbe light_probe(const std::string& path);

class FileDecoder {
public:
    // Opens the file and initialises the video decoder.
    // hw may be nullptr for software-only decode.
    // decode_threads: 0 = auto (all cores), >0 = limit FFmpeg's internal
    // slice/frame threading to this many threads.
    FileDecoder(const std::string& path, HWAccelContext* hw = nullptr, int decode_threads = 0);
    ~FileDecoder();

    FileDecoder(const FileDecoder&) = delete;
    FileDecoder& operator=(const FileDecoder&) = delete;

    bool is_valid() const { return valid_; }

    // The codec ID of the video stream (available after construction,
    // even if the codec failed to open — useful for HW compatibility checks).
    AVCodecID codec_id() const { return codec_id_; }

    // Extract keyframe index via the fallback chain.
    KeyframeIndex scan_keyframes();

    // Seek to keyframe and decode one I-frame into out_frame (native pix fmt).
    bool seek_and_decode(const KeyframeInfo& kf, AVFrame* out_frame);

    // ── Sequential decode API ──────────────────────────────────────────────

    // Seek to the beginning of the stream and flush codec state.
    // Call before the first decode_next() call.
    bool seek_to_start();

    // Decode the next video frame sequentially (no seeking).
    // Returns false on EOF or error.
    bool decode_next(AVFrame* out_frame);

    // Estimate total frame count from stream metadata.
    int estimated_frame_count() const;

    // Convert a decoded frame's PTS to milliseconds.
    int64_t frame_pts_ms(const AVFrame* frame) const;

    // ── Codec context reuse ───────────────────────────────────────────────

    // Reopen the decoder with a new file, reusing the existing codec context
    // via avcodec_flush_buffers() instead of destroying and recreating it.
    //
    // Why this matters: avcodec_open2() spins up FFmpeg's internal thread
    // pool (for slice/frame threading), and avcodec_free_context() tears it
    // down.  For a 100-file batch with decode_threads=4, that's 99 full
    // thread pool create/destroy cycles saved per worker.
    //
    // Returns false if the new file's parameters (codec_id, width, height,
    // pixel format) don't match the current context, or if the codec isn't
    // on the flush-reuse safety whitelist.  On false, the caller must
    // destroy this decoder and create a fresh one.
    //
    // Safety constraints — see flush_reuse_safe() and update_extradata()
    // in decoder.cpp for the full rationale on which codecs are safe to
    // flush-reuse and what state must be updated between files.
    bool reopen_file(const std::string& new_path, HWAccelContext* hw, int decode_threads);

    // ── Accessors ──────────────────────────────────────────────────────────

    int video_stream_index() const { return video_idx_; }
    int source_width() const { return width_; }
    int source_height() const { return height_; }
    std::string codec_name() const { return codec_name_; }
    int64_t file_size() const { return file_size_; }
    // Returns the number of FFmpeg-managed decode threads, or 0 when the
    // codec manages its own thread pool (e.g. libdav1d for AV1).
    int decode_thread_count() const {
        if (!codec_ctx_) return 0;
        if (codec_ctx_->active_thread_type == 0) return 0;
        return codec_ctx_->thread_count;
    }

    // True if the demuxer is a video container (not a still-image pipe).
    // Image formats use demuxers like "png_pipe", "image2", etc.
    bool is_video_format() const {
        if (!fmt_ctx_ || !fmt_ctx_->iformat || !fmt_ctx_->iformat->name)
            return false;
        std::string name = fmt_ctx_->iformat->name;
        if (name == "image2" || name == "image2pipe")
            return false;
        if (name.size() > 5 && name.compare(name.size() - 5, 5, "_pipe") == 0)
            return false;
        return true;
    }

private:
    // Shared post-decode processing: HW frame transfer.
    // Pixel format conversion is deferred to FrameScaler.
    bool finalize_frame(AVFrame* decode_target, AVFrame* out_frame);

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVFrame* hw_frame_ = nullptr;  // scratch frame for HW decode
    AVPacket* pkt_ = nullptr;      // reusable packet for decode_next
    HWAccelContext* hw_accel_ = nullptr;
    int video_idx_ = -1;
    int width_ = 0;
    int height_ = 0;
    int64_t file_size_ = 0;
    std::string codec_name_;
    AVCodecID codec_id_ = AV_CODEC_ID_NONE;
    bool valid_ = false;
};

}  // namespace vex
