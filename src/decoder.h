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

    // ── Accessors ──────────────────────────────────────────────────────────

    int video_stream_index() const { return video_idx_; }
    int source_width() const { return width_; }
    int source_height() const { return height_; }
    std::string codec_name() const { return codec_name_; }
    int64_t file_size() const { return file_size_; }
    int decode_thread_count() const { return codec_ctx_ ? codec_ctx_->thread_count : 0; }

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
