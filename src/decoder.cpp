#include "decoder.h"
#include "index_scanner.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

namespace vex {

// ── Constructor ─────────────────────────────────────────────────────────────

FileDecoder::FileDecoder(const std::string& path, HWAccelContext* hw) {
    // Step 1: Open input
    int ret = avformat_open_input(&fmt_ctx_, path.c_str(), nullptr, nullptr);
    if (ret < 0) {
        valid_ = false;
        return;
    }

    // Step 2: Find stream info
    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        avformat_close_input(&fmt_ctx_);
        valid_ = false;
        return;
    }

    // Step 3: Find best video stream
    const AVCodec* codec = nullptr;
    video_idx_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (video_idx_ < 0 || !codec) {
        avformat_close_input(&fmt_ctx_);
        valid_ = false;
        return;
    }

    // Step 4: Get stream parameters
    AVStream* stream = fmt_ctx_->streams[video_idx_];
    width_  = stream->codecpar->width;
    height_ = stream->codecpar->height;

    // Step 5: Determine file size
    if (fmt_ctx_->pb) {
        int64_t sz = avio_size(fmt_ctx_->pb);
        file_size_ = (sz > 0) ? sz : 0;
    } else {
        file_size_ = 0;
    }

    // Step 6: Allocate and configure codec context
    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        avformat_close_input(&fmt_ctx_);
        valid_ = false;
        return;
    }

    ret = avcodec_parameters_to_context(codec_ctx_, stream->codecpar);
    if (ret < 0) {
        avcodec_free_context(&codec_ctx_);
        avformat_close_input(&fmt_ctx_);
        valid_ = false;
        return;
    }

    // Step 7: HW acceleration setup
    if (hw && hw->device_ctx) {
        codec_ctx_->hw_device_ctx = av_buffer_ref(hw->device_ctx);
        codec_ctx_->opaque        = hw;
        codec_ctx_->get_format    = hw_get_format;
        hw_frame_ = av_frame_alloc();
        hw_accel_ = hw;
    }

    // Step 8: Open codec
    ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        if (hw_frame_) {
            av_frame_free(&hw_frame_);
        }
        hw_accel_ = nullptr;
        avcodec_free_context(&codec_ctx_);
        avformat_close_input(&fmt_ctx_);
        valid_ = false;
        return;
    }

    // Step 9: Store codec name, mark valid
    codec_name_ = codec->name;
    valid_ = true;
}

// ── Destructor ──────────────────────────────────────────────────────────────

FileDecoder::~FileDecoder() {
    if (hw_frame_) {
        av_frame_free(&hw_frame_);
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
    }
    hw_accel_ = nullptr;
}

// ── scan_keyframes ──────────────────────────────────────────────────────────

KeyframeIndex FileDecoder::scan_keyframes() {
    return vex::scan_keyframes(fmt_ctx_, video_idx_);
}

// ── seek_and_decode ─────────────────────────────────────────────────────────

bool FileDecoder::seek_and_decode(const KeyframeInfo& kf, AVFrame* out_frame) {
    // Step 1: Seek — prefer PTS-based seeking (more compatible with HW decode),
    // fall back to byte-offset seeking if PTS is not available.
    int seek_ret;
    if (kf.pts != AV_NOPTS_VALUE && kf.pts >= 0) {
        seek_ret = av_seek_frame(fmt_ctx_, video_idx_, kf.pts,
                                 AVSEEK_FLAG_BACKWARD);
    } else if (kf.byte_offset > 0) {
        seek_ret = av_seek_frame(fmt_ctx_, -1, kf.byte_offset,
                                 AVSEEK_FLAG_BYTE | AVSEEK_FLAG_BACKWARD);
    } else {
        seek_ret = av_seek_frame(fmt_ctx_, video_idx_, 0,
                                 AVSEEK_FLAG_BACKWARD);
    }
    if (seek_ret < 0) {
        return false;
    }

    // Step 2: Flush codec buffers
    avcodec_flush_buffers(codec_ctx_);

    // Step 3: Read + decode loop
    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        return false;
    }

    // Decide which frame receives the decoded output:
    // If HW accel is active, decode into hw_frame_ then transfer.
    // Otherwise, decode directly into out_frame.
    AVFrame* decode_target = (hw_accel_ && hw_frame_) ? hw_frame_ : out_frame;

    bool got_frame = false;
    while (av_read_frame(fmt_ctx_, pkt) >= 0) {
        if (pkt->stream_index != video_idx_) {
            av_packet_unref(pkt);
            continue;
        }

        int ret = avcodec_send_packet(codec_ctx_, pkt);
        av_packet_unref(pkt);

        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            continue;
        }

        ret = avcodec_receive_frame(codec_ctx_, decode_target);
        if (ret == 0) {
            got_frame = true;
            break;
        }
        // EAGAIN: need more packets; other errors: keep trying
    }

    av_packet_free(&pkt);

    if (!got_frame) {
        return false;
    }

    // Step 4: HW transfer if needed
    if (hw_accel_ && hw_frame_) {
        if (!transfer_hw_frame(hw_frame_, out_frame)) {
            av_frame_unref(hw_frame_);
            return false;
        }
        av_frame_unref(hw_frame_);
    } else if (decode_target != out_frame) {
        // Shouldn't happen, but handle defensively
        av_frame_move_ref(out_frame, decode_target);
    }

    // Step 5: Convert non-YUV420P frames (e.g. NV12 from HW decode)
    if (out_frame->format != AV_PIX_FMT_YUV420P &&
        out_frame->format != AV_PIX_FMT_YUVJ420P) {
        SwsContext* conv = sws_getContext(
            out_frame->width, out_frame->height,
            static_cast<AVPixelFormat>(out_frame->format),
            out_frame->width, out_frame->height,
            AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!conv) {
            av_frame_unref(out_frame);
            return false;
        }
        AVFrame* tmp = av_frame_alloc();
        tmp->format = AV_PIX_FMT_YUV420P;
        tmp->width  = out_frame->width;
        tmp->height = out_frame->height;
        av_frame_get_buffer(tmp, 0);
        sws_scale(conv, out_frame->data, out_frame->linesize,
                  0, out_frame->height, tmp->data, tmp->linesize);
        sws_freeContext(conv);
        av_frame_unref(out_frame);
        av_frame_move_ref(out_frame, tmp);
        av_frame_free(&tmp);
    }

    return true;
}

} // namespace vex
