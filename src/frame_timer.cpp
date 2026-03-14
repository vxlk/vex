#include "frame_timer.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace vex {

// ── Container classification ────────────────────────────────────────────────

static TimestampStrategy classify_container(const std::string& name) {
    if (name.find("mov") != std::string::npos || name.find("mp4") != std::string::npos ||
        name.find("3gp") != std::string::npos || name.find("3g2") != std::string::npos ||
        name.find("mj2") != std::string::npos)
        return TimestampStrategy::SAMPLE_TABLE;

    if (name.find("matroska") != std::string::npos || name.find("webm") != std::string::npos)
        return TimestampStrategy::BLOCK_TIMESTAMP;

    if (name == "mpegts" || name == "mpeg" || name == "wtv")
        return TimestampStrategy::PES_TIMESTAMP;

    if (name == "flv" || name == "asf" || name == "rm")
        return TimestampStrategy::TAG_TIMESTAMP;

    if (name == "avi" || name == "dv" || name == "swf")
        return TimestampStrategy::FIXED_RATE;

    return TimestampStrategy::GENERIC_PTS;
}

// ── get_frame_times ─────────────────────────────────────────────────────────

FrameTimesResult get_frame_times(const std::string& path) {
    FrameTimesResult result;

    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, path.c_str(), nullptr, nullptr) < 0)
        return result;

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        avformat_close_input(&fmt_ctx);
        return result;
    }

    int video_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_idx < 0) {
        avformat_close_input(&fmt_ctx);
        return result;
    }

    AVStream* stream = fmt_ctx->streams[video_idx];

    // ── FPS ─────────────────────────────────────────────────────────────
    double fps = 0.0;
    if (stream->avg_frame_rate.num > 0 && stream->avg_frame_rate.den > 0)
        fps = av_q2d(stream->avg_frame_rate);
    else if (stream->r_frame_rate.num > 0 && stream->r_frame_rate.den > 0)
        fps = av_q2d(stream->r_frame_rate);

    // ── Duration ────────────────────────────────────────────────────────
    double duration = 0.0;
    if (stream->duration != AV_NOPTS_VALUE && stream->duration > 0)
        duration = stream->duration * av_q2d(stream->time_base);
    else if (fmt_ctx->duration > 0)
        duration = fmt_ctx->duration / static_cast<double>(AV_TIME_BASE);

    // ── Container classification ────────────────────────────────────────
    std::string iformat_name;
    if (fmt_ctx->iformat && fmt_ctx->iformat->name)
        iformat_name = fmt_ctx->iformat->name;

    TimestampStrategy strategy = classify_container(iformat_name);

    // ── Codec name ──────────────────────────────────────────────────────
    std::string codec_name;
    const AVCodecDescriptor* desc = avcodec_descriptor_get(stream->codecpar->codec_id);
    if (desc && desc->name)
        codec_name = desc->name;

    result.container = iformat_name;
    result.codec = codec_name;
    result.fps = fps;
    result.duration_sec = duration;

    double tb = av_q2d(stream->time_base);
    AVPacket* pkt = av_packet_alloc();

    if (strategy == TimestampStrategy::FIXED_RATE) {
        // ── FIXED_RATE: count packets, synthesize timestamps ────────────
        int count = 0;
        while (av_read_frame(fmt_ctx, pkt) >= 0) {
            if (pkt->stream_index == video_idx)
                count++;
            av_packet_unref(pkt);
        }

        result.frame_count = count;
        result.times_sec.resize(static_cast<size_t>(count));
        if (fps > 0) {
            for (int i = 0; i < count; i++)
                result.times_sec[static_cast<size_t>(i)] = i / fps;
        } else if (count > 1 && duration > 0) {
            for (int i = 0; i < count; i++)
                result.times_sec[static_cast<size_t>(i)] = i * duration / (count - 1);
        }
        result.strategy = TimestampStrategy::FIXED_RATE;

    } else {
        // ── PTS-based path ──────────────────────────────────────────────
        std::vector<int64_t> raw_pts;
        raw_pts.reserve(4096);

        while (av_read_frame(fmt_ctx, pkt) >= 0) {
            if (pkt->stream_index == video_idx) {
                int64_t pts = pkt->pts;
                if (pts == AV_NOPTS_VALUE)
                    pts = pkt->dts;
                raw_pts.push_back(pts);
            }
            av_packet_unref(pkt);
        }

        int total = static_cast<int>(raw_pts.size());

        // Count valid PTS
        int valid_count = 0;
        for (auto p : raw_pts) {
            if (p != AV_NOPTS_VALUE)
                valid_count++;
        }

        if (valid_count == 0) {
            // ── LINEAR_FALLBACK ─────────────────────────────────────────
            int frame_count = total;
            if (frame_count == 0 && fps > 0 && duration > 0)
                frame_count = static_cast<int>(duration * fps + 0.5);

            result.frame_count = frame_count;
            result.times_sec.resize(static_cast<size_t>(frame_count));
            for (int i = 0; i < frame_count; i++) {
                result.times_sec[static_cast<size_t>(i)] =
                    frame_count > 1 ? i * duration / (frame_count - 1) : 0.0;
            }
            result.strategy = TimestampStrategy::LINEAR_FALLBACK;

        } else {
            // Convert valid PTS to seconds, mark invalid as -1
            std::vector<double> times(static_cast<size_t>(total));
            for (int i = 0; i < total; i++) {
                if (raw_pts[static_cast<size_t>(i)] != AV_NOPTS_VALUE)
                    times[static_cast<size_t>(i)] = raw_pts[static_cast<size_t>(i)] * tb;
                else
                    times[static_cast<size_t>(i)] = -1.0;
            }

            // ── Interpolate gaps ────────────────────────────────────────
            int first_valid = -1, last_valid = -1;
            for (int i = 0; i < total; i++) {
                if (raw_pts[static_cast<size_t>(i)] != AV_NOPTS_VALUE) {
                    if (first_valid < 0)
                        first_valid = i;
                    last_valid = i;
                }
            }

            // Fill leading invalid entries
            if (first_valid > 0) {
                for (int i = 0; i < first_valid; i++)
                    times[static_cast<size_t>(i)] = times[static_cast<size_t>(first_valid)];
            }

            // Interpolate interior gaps
            for (int i = first_valid + 1; i <= last_valid; i++) {
                if (times[static_cast<size_t>(i)] < 0) {
                    int prev = i - 1;
                    int next = i + 1;
                    while (next <= last_valid && times[static_cast<size_t>(next)] < 0)
                        next++;
                    if (next <= last_valid) {
                        double t0 = times[static_cast<size_t>(prev)];
                        double t1 = times[static_cast<size_t>(next)];
                        int gap = next - prev;
                        for (int j = prev + 1; j < next; j++) {
                            times[static_cast<size_t>(j)] = t0 + (t1 - t0) * (j - prev) / gap;
                        }
                        i = next - 1;  // loop will increment
                    }
                }
            }

            // Fill trailing invalid entries
            if (last_valid < total - 1) {
                double last_t = times[static_cast<size_t>(last_valid)];
                double step = fps > 0 ? 1.0 / fps : 0.0;
                for (int i = last_valid + 1; i < total; i++)
                    times[static_cast<size_t>(i)] = last_t + (i - last_valid) * step;
            }

            // Sort for display order (B-frames may arrive out-of-order)
            std::sort(times.begin(), times.end());

            // Some containers (MPEG-TS, MPEG-PS, FLV, etc.) start at an
            // arbitrary PTS rather than zero — e.g. a TS muxed mid-stream
            // may begin at PTS 126000.  Subtract the first timestamp so
            // the returned array is zero-based regardless of origin.
            if (!times.empty() && times[0] > 0.001) {
                double base = times[0];
                for (auto& t : times)
                    t -= base;
            }

            // ── PES discontinuity handling ──────────────────────────────
            if (strategy == TimestampStrategy::PES_TIMESTAMP && total > 1) {
                std::vector<double> diffs;
                diffs.reserve(static_cast<size_t>(total - 1));
                for (int i = 1; i < total; i++) {
                    double d = times[static_cast<size_t>(i)] - times[static_cast<size_t>(i - 1)];
                    if (d > 0)
                        diffs.push_back(d);
                }

                if (!diffs.empty()) {
                    std::sort(diffs.begin(), diffs.end());
                    double median = diffs[diffs.size() / 2];

                    if (median > 0) {
                        double threshold = median * 3.0;
                        double cumulative_offset = 0.0;
                        for (int i = 1; i < total; i++) {
                            double gap = times[static_cast<size_t>(i)] -
                                         times[static_cast<size_t>(i - 1)] + cumulative_offset;
                            // Compare original gap (before offset adjustment)
                            double orig_gap =
                                times[static_cast<size_t>(i)] - times[static_cast<size_t>(i - 1)];
                            if (orig_gap > threshold) {
                                cumulative_offset -= (orig_gap - median);
                            }
                            times[static_cast<size_t>(i)] += cumulative_offset;
                        }
                    }
                }
            }

            result.times_sec = std::move(times);
            result.frame_count = total;
            result.strategy = strategy;
        }
    }

    av_packet_free(&pkt);
    avformat_close_input(&fmt_ctx);

    return result;
}

}  // namespace vex
