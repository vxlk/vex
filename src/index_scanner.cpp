#include "index_scanner.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

#include <algorithm>
#include <chrono>

namespace vex {

// ── Helpers ─────────────────────────────────────────────────────────────────

static int64_t to_ms(int64_t timestamp, AVRational time_base) {
    if (timestamp == AV_NOPTS_VALUE) return 0;
    return av_rescale_q(timestamp, time_base, {1, 1000});
}

// ── Strategy 1: Container index ─────────────────────────────────────────────

static std::vector<KeyframeInfo> scan_container_index(AVFormatContext* fmt_ctx, int stream_index) {
    std::vector<KeyframeInfo> keyframes;
    AVStream* stream = fmt_ctx->streams[stream_index];

    int nb_entries = avformat_index_get_entries_count(stream);
    if (nb_entries <= 0) {
        return keyframes;
    }

    for (int i = 0; i < nb_entries; ++i) {
        const AVIndexEntry* entry = avformat_index_get_entry(stream, i);
        if (!entry) continue;
        if (!(entry->flags & AVINDEX_KEYFRAME)) continue;

        KeyframeInfo kf{};
        kf.byte_offset  = entry->pos;
        kf.pts          = entry->timestamp;
        kf.pts_ms       = to_ms(entry->timestamp, stream->time_base);
        kf.size         = entry->size > 0 ? entry->size : -1;
        keyframes.push_back(kf);
    }

    return keyframes;
}

// ── Strategy 2: Packet scan ─────────────────────────────────────────────────

static std::vector<KeyframeInfo> scan_packets(AVFormatContext* fmt_ctx, int stream_index) {
    std::vector<KeyframeInfo> keyframes;
    AVStream* stream = fmt_ctx->streams[stream_index];

    // Seek back to the beginning
    av_seek_frame(fmt_ctx, stream_index, 0, AVSEEK_FLAG_BACKWARD);

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) return keyframes;

    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == stream_index) {
            if (pkt->flags & AV_PKT_FLAG_KEY) {
                KeyframeInfo kf{};
                kf.byte_offset = pkt->pos;
                kf.pts         = pkt->pts;
                kf.pts_ms      = to_ms(pkt->pts, stream->time_base);
                kf.size        = pkt->size > 0 ? pkt->size : -1;
                keyframes.push_back(kf);
            }
        }
        av_packet_unref(pkt);
    }

    av_packet_free(&pkt);
    return keyframes;
}

// ── Strategy 3: Decode scan ─────────────────────────────────────────────────

static std::vector<KeyframeInfo> scan_decode(AVFormatContext* fmt_ctx, int stream_index) {
    std::vector<KeyframeInfo> keyframes;
    AVStream* stream = fmt_ctx->streams[stream_index];

    // Create a temporary codec context for decoding
    const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) return keyframes;

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) return keyframes;

    if (avcodec_parameters_to_context(codec_ctx, stream->codecpar) < 0) {
        avcodec_free_context(&codec_ctx);
        return keyframes;
    }

    // Skip non-reference frames — we only need I-frames, so avoid
    // fully decoding B-frames (often 60-70% of a typical GOP).
    codec_ctx->skip_frame = AVDISCARD_NONREF;

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        avcodec_free_context(&codec_ctx);
        return keyframes;
    }

    // Seek back to the beginning
    av_seek_frame(fmt_ctx, stream_index, 0, AVSEEK_FLAG_BACKWARD);

    AVPacket* pkt   = av_packet_alloc();
    AVFrame*  frame = av_frame_alloc();
    if (!pkt || !frame) {
        av_packet_free(&pkt);
        av_frame_free(&frame);
        avcodec_free_context(&codec_ctx);
        return keyframes;
    }

    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index != stream_index) {
            av_packet_unref(pkt);
            continue;
        }

        int ret = avcodec_send_packet(codec_ctx, pkt);
        if (ret < 0) {
            av_packet_unref(pkt);
            continue;
        }

        while (avcodec_receive_frame(codec_ctx, frame) >= 0) {
            if (frame->pict_type == AV_PICTURE_TYPE_I) {
                KeyframeInfo kf{};
                kf.byte_offset = pkt->pos;
                kf.pts         = frame->pts;
                kf.pts_ms      = to_ms(frame->pts, stream->time_base);
                kf.size        = pkt->size > 0 ? pkt->size : -1;
                keyframes.push_back(kf);
            }
            av_frame_unref(frame);
        }

        av_packet_unref(pkt);
    }

    // Flush the decoder
    avcodec_send_packet(codec_ctx, nullptr);
    while (avcodec_receive_frame(codec_ctx, frame) >= 0) {
        if (frame->pict_type == AV_PICTURE_TYPE_I) {
            KeyframeInfo kf{};
            kf.byte_offset = 0;
            kf.pts         = frame->pts;
            kf.pts_ms      = to_ms(frame->pts, stream->time_base);
            kf.size        = -1;
            keyframes.push_back(kf);
        }
        av_frame_unref(frame);
    }

    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&codec_ctx);
    return keyframes;
}

// ── Strategy 4: Forced interval ─────────────────────────────────────────────

static std::vector<KeyframeInfo> scan_forced_interval(AVFormatContext* fmt_ctx, int stream_index) {
    std::vector<KeyframeInfo> keyframes;
    AVStream* stream = fmt_ctx->streams[stream_index];

    // Seek back to the beginning
    av_seek_frame(fmt_ctx, stream_index, 0, AVSEEK_FLAG_BACKWARD);

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) return keyframes;

    int packet_count = 0;
    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == stream_index) {
            if (packet_count % 30 == 0) {
                KeyframeInfo kf{};
                kf.byte_offset = pkt->pos;
                kf.pts         = pkt->pts;
                kf.pts_ms      = to_ms(pkt->pts, stream->time_base);
                kf.size        = pkt->size > 0 ? pkt->size : -1;
                keyframes.push_back(kf);
            }
            ++packet_count;
        }
        av_packet_unref(pkt);
    }

    av_packet_free(&pkt);
    return keyframes;
}

// ── Finalize keyframes (sort + number) ──────────────────────────────────────

static void finalize_keyframes(std::vector<KeyframeInfo>& keyframes) {
    std::sort(keyframes.begin(), keyframes.end(),
              [](const KeyframeInfo& a, const KeyframeInfo& b) {
                  return a.byte_offset < b.byte_offset;
              });

    for (int i = 0; i < static_cast<int>(keyframes.size()); ++i) {
        keyframes[i].frame_number = i;
    }
}

// ── scan_keyframes (public) ─────────────────────────────────────────────────

KeyframeIndex scan_keyframes(AVFormatContext* fmt_ctx, int stream_index) {
    KeyframeIndex result{};

    auto start = std::chrono::high_resolution_clock::now();

    // Strategy 1: Container index
    auto keyframes = scan_container_index(fmt_ctx, stream_index);
    if (!keyframes.empty()) {
        finalize_keyframes(keyframes);
        result.keyframes = std::move(keyframes);
        result.strategy  = IndexStrategy::CONTAINER_INDEX;
        auto end = std::chrono::high_resolution_clock::now();
        result.scan_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return result;
    }

    // Strategy 2: Packet scan
    keyframes = scan_packets(fmt_ctx, stream_index);
    if (!keyframes.empty()) {
        finalize_keyframes(keyframes);
        result.keyframes = std::move(keyframes);
        result.strategy  = IndexStrategy::PACKET_SCAN;
        auto end = std::chrono::high_resolution_clock::now();
        result.scan_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return result;
    }

    // Strategy 3: Decode scan
    keyframes = scan_decode(fmt_ctx, stream_index);
    if (!keyframes.empty()) {
        finalize_keyframes(keyframes);
        result.keyframes = std::move(keyframes);
        result.strategy  = IndexStrategy::DECODE_SCAN;
        auto end = std::chrono::high_resolution_clock::now();
        result.scan_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return result;
    }

    // Strategy 4: Forced interval
    keyframes = scan_forced_interval(fmt_ctx, stream_index);
    if (!keyframes.empty()) {
        finalize_keyframes(keyframes);
        result.keyframes = std::move(keyframes);
        result.strategy  = IndexStrategy::FORCED_INTERVAL;
        auto end = std::chrono::high_resolution_clock::now();
        result.scan_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return result;
    }

    // Strategy 5: Skip
    result.strategy = IndexStrategy::SKIPPED;
    auto end = std::chrono::high_resolution_clock::now();
    result.scan_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return result;
}

} // namespace vex
