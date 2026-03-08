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

    // Step 9: Allocate reusable packet for decode_next
    pkt_ = av_packet_alloc();

    // Step 10: Store codec name, mark valid
    codec_name_ = codec->name;
    valid_ = true;
}

// ── Destructor ──────────────────────────────────────────────────────────────

FileDecoder::~FileDecoder() {
    if (pkt_) {
        av_packet_free(&pkt_);
    }
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

// ── finalize_frame (shared HW transfer + pixel format conversion) ───────────

bool FileDecoder::finalize_frame(AVFrame* decode_target, AVFrame* out_frame) {
    // HW transfer if needed. Some codecs fall back to software even when
    // HW context is set, so check hw_frames_ctx to confirm it's truly a HW frame.
    if (hw_accel_ && hw_frame_ && decode_target == hw_frame_) {
        if (decode_target->hw_frames_ctx) {
            if (!transfer_hw_frame(hw_frame_, out_frame)) {
                av_frame_unref(hw_frame_);
                return false;
            }
        } else {
            // Codec produced a SW frame despite HW setup — just move it
            av_frame_move_ref(out_frame, hw_frame_);
        }
        av_frame_unref(hw_frame_);
    } else if (decode_target != out_frame) {
        av_frame_move_ref(out_frame, decode_target);
    }

    // Convert non-YUV420P frames (e.g. NV12 from HW decode)
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
    }

    av_packet_free(&pkt);

    if (!got_frame) {
        return false;
    }

    return finalize_frame(decode_target, out_frame);
}

// ── Sequential decode API ───────────────────────────────────────────────────

bool FileDecoder::seek_to_start() {
    if (!valid_ || !fmt_ctx_ || !codec_ctx_) return false;

    // Try multiple seek strategies. Some containers (SWF, GXF) don't
    // support standard seeking.
    int ret = avformat_seek_file(fmt_ctx_, video_idx_,
                                  INT64_MIN, 0, 0, 0);
    if (ret < 0) {
        ret = av_seek_frame(fmt_ctx_, video_idx_, 0, AVSEEK_FLAG_BACKWARD);
    }
    if (ret < 0) {
        // Byte-level seek as last resort — some containers need this
        ret = av_seek_frame(fmt_ctx_, -1, 0,
                            AVSEEK_FLAG_BYTE | AVSEEK_FLAG_BACKWARD);
    }

    avcodec_flush_buffers(codec_ctx_);
    // Return true even if seeking failed — decode_next() will try to read
    // from the current position (works for forward-only containers).
    return true;
}

bool FileDecoder::decode_next(AVFrame* out_frame) {
    if (!valid_ || !fmt_ctx_ || !codec_ctx_ || !pkt_) return false;

    AVFrame* decode_target = (hw_accel_ && hw_frame_) ? hw_frame_ : out_frame;

    // Try to receive a frame from already-sent packets first
    int ret = avcodec_receive_frame(codec_ctx_, decode_target);
    if (ret == 0) {
        return finalize_frame(decode_target, out_frame);
    }

    // Feed packets until we get a frame or hit EOF
    while (av_read_frame(fmt_ctx_, pkt_) >= 0) {
        if (pkt_->stream_index != video_idx_) {
            av_packet_unref(pkt_);
            continue;
        }

        ret = avcodec_send_packet(codec_ctx_, pkt_);
        av_packet_unref(pkt_);

        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            continue;
        }

        ret = avcodec_receive_frame(codec_ctx_, decode_target);
        if (ret == 0) {
            return finalize_frame(decode_target, out_frame);
        }
        // EAGAIN: need more packets
    }

    // EOF: drain the decoder
    avcodec_send_packet(codec_ctx_, nullptr);
    ret = avcodec_receive_frame(codec_ctx_, decode_target);
    if (ret == 0) {
        return finalize_frame(decode_target, out_frame);
    }

    return false;
}

int FileDecoder::estimated_frame_count() const {
    if (!valid_ || !fmt_ctx_) return 0;

    AVStream* stream = fmt_ctx_->streams[video_idx_];

    // Try nb_frames first
    if (stream->nb_frames > 0) {
        return static_cast<int>(stream->nb_frames);
    }

    // Fallback: duration * avg_frame_rate
    if (stream->duration > 0 && stream->avg_frame_rate.den > 0) {
        double dur_sec = stream->duration * av_q2d(stream->time_base);
        double fps = av_q2d(stream->avg_frame_rate);
        if (fps > 0) {
            return static_cast<int>(dur_sec * fps);
        }
    }

    // Container-level duration fallback
    if (fmt_ctx_->duration > 0 && stream->avg_frame_rate.den > 0) {
        double dur_sec = fmt_ctx_->duration / static_cast<double>(AV_TIME_BASE);
        double fps = av_q2d(stream->avg_frame_rate);
        if (fps > 0) {
            return static_cast<int>(dur_sec * fps);
        }
    }

    return 300; // conservative fallback
}

int64_t FileDecoder::frame_pts_ms(const AVFrame* frame) const {
    if (!valid_ || !fmt_ctx_ || !frame) return 0;

    AVStream* stream = fmt_ctx_->streams[video_idx_];
    int64_t pts = frame->pts;
    if (pts == AV_NOPTS_VALUE) pts = frame->best_effort_timestamp;
    if (pts == AV_NOPTS_VALUE) return 0;

    return av_rescale_q(pts, stream->time_base, {1, 1000});
}

} // namespace vex
