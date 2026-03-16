#include "decoder.h"
#include "cache.h"
#include "keyframe_index_scanner.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libavutil/log.h>
}

#include <cstring>

namespace vex {

// ── light_probe ─────────────────────────────────────────────────────────────
//
// Reads codec_id, width, height from a video file using only the demuxer.
// Does NOT allocate a codec context or call avcodec_open2(), avoiding the
// expensive FFmpeg internal thread pool creation.  Used by the orchestrator
// to check HW decode compatibility before committing to a full decode.
//
// Cost breakdown vs full FileDecoder constructor:
//   avformat_open_input        — same (I/O-bound, unavoidable)
//   avformat_find_stream_info  — same (reads initial frames to fill codecpar)
//   avcodec_alloc_context3     — SKIPPED (heap alloc + codec private data)
//   avcodec_parameters_to_context — SKIPPED
//   avcodec_open2              — SKIPPED (thread pool spin-up, codec init)

LightProbe light_probe(const std::string& path) {
    LightProbe result;
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) < 0)
        return result;
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
        avformat_close_input(&fmt);
        return result;
    }
    const AVCodec* codec = nullptr;
    int idx = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (idx >= 0) {
        // Read directly from the stream's codec parameters — these are
        // populated by the demuxer from container headers, no codec
        // context needed.
        result.codec_id = fmt->streams[idx]->codecpar->codec_id;
        result.width = fmt->streams[idx]->codecpar->width;
        result.height = fmt->streams[idx]->codecpar->height;
        result.valid = true;
    }
    avformat_close_input(&fmt);
    return result;
}

// ── Codec context flush-reuse helpers ───────────────────────────────────────
//
// Background: avcodec_flush_buffers() resets a live codec context so it can
// decode a new stream without the overhead of tearing down and recreating
// the context (which includes destroying and respawning FFmpeg's internal
// thread pool).  However, not all codecs implement flush correctly.
//
// What flush preserves (safe to reuse across files):
//   - FFmpeg internal thread pool (threads are parked, not destroyed)
//   - Codec identity and pixel format
//   - HW device/frames contexts
//   - SPS/PPS parameter sets (H.264/HEVC) in codec private data
//
// What flush resets (correct for new-file decode):
//   - Draining state (avci->draining, avci->draining_done)
//   - Buffered frames and packets
//   - PTS correction counters
//   - H.264: DPB (decoded picture buffer — all 16 ref slots), delayed_pic
//     array, context_initialized → 0, error concealment refs, SEI state
//   - HEVC: DPB, max_ra, SEI state, Dolby Vision context
//   - VP9: all 3 frame structures, all 8 reference slots

static bool flush_reuse_safe(AVCodecID id, bool /*is_hw*/) {
    // Conservative whitelist — only codecs whose flush callbacks we have
    // verified clear all inter-file state correctly:
    //
    //   H.264:  flush_dpb() — clears DPB, delayed_pic, sets
    //           context_initialized=0 which forces SPS/PPS re-read from
    //           extradata on next frame.  Stable across FFmpeg 4.x–8.x.
    //
    //   HEVC:   hevc_decode_flush() — clears DPB, resets max_ra, SEI,
    //           and Dolby Vision context.  Same stability story.
    //
    //   VP9:    vp9_decode_flush() — clears all 3 frame structures and
    //           all 8 reference slots.
    //
    // NOT included (known issues):
    //   AV1/dav1d: requires a new sequence header OBU after flush. If the
    //              container stores it only in extradata (not in-band),
    //              decoding stalls silently.
    //   MPEG-1/2, MJPEG, FLV1, etc.: no verified flush callbacks.
    //              AAC (audio) had a known bug (FFmpeg trac #420) where
    //              flush was a no-op.  We take no chances with codecs
    //              that haven't been audited.
    switch (id) {
        case AV_CODEC_ID_H264:
        case AV_CODEC_ID_HEVC:
        case AV_CODEC_ID_VP9:
            return true;
        default:
            return false;
    }
}

// Copy the new file's extradata into the existing codec context.
//
// CRITICAL for H.264/HEVC flush-reuse correctness:
//
// In MP4/MOV containers, SPS (Sequence Parameter Set) and PPS (Picture
// Parameter Set) are stored out-of-band in the "avcC" or "hvcC" box,
// which FFmpeg exposes as stream->codecpar->extradata.  The codec context
// reads this on first decode to configure resolution, profile, ref frames,
// etc.
//
// When we flush and switch to a new file, the codec context still points
// to the OLD file's extradata.  H.264's flush sets context_initialized=0,
// which forces a re-read of extradata on the next frame — but if extradata
// still contains the old file's SPS/PPS, the decoder may misconfigure
// itself (wrong ref count, wrong profile) causing corruption or crashes.
//
// We must replace extradata AFTER flush but BEFORE sending the first
// packet from the new file.
//
// AV_INPUT_BUFFER_PADDING_SIZE (typically 64 bytes) is required by FFmpeg
// for SIMD-safe overreads during bitstream parsing.
static void update_extradata(AVCodecContext* ctx, const AVCodecParameters* par) {
    av_freep(&ctx->extradata);
    ctx->extradata_size = 0;
    if (par->extradata_size > 0) {
        ctx->extradata =
            static_cast<uint8_t*>(av_mallocz(par->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE));
        if (ctx->extradata) {
            memcpy(ctx->extradata, par->extradata, par->extradata_size);
            ctx->extradata_size = par->extradata_size;
        }
    }
}

// ── Constructor ─────────────────────────────────────────────────────────────

FileDecoder::FileDecoder(const std::string& path, HWAccelContext* hw, int decode_threads) {
    // Step 1: Open input
    int ret = avformat_open_input(&fmt_ctx_, path.c_str(), nullptr, nullptr);
    if (ret < 0) {
        valid_ = false;
        return;
    }

    // Step 2: Apply format probe hints to reduce avformat_find_stream_info cost.
    //
    // avformat_find_stream_info() reads initial frames to determine stream
    // parameters (codec, resolution, frame rate).  Its two knobs are:
    //   probesize        — max bytes to read (default 5 MB)
    //   max_analyze_duration — max duration to analyze (default 5 seconds)
    //
    // For well-muxed containers like MP4 (moov atom) and MKV (EBML header),
    // all stream params are in the header — a smaller probe suffices.
    // We cache per-format hints after the first successful file so that
    // subsequent files of the same format skip unnecessary I/O.
    {
        auto& cache = ProcessCache::instance();
        std::lock_guard<std::mutex> lock(cache.probe_hints_mutex);
        if (fmt_ctx_->iformat && fmt_ctx_->iformat->name) {
            auto it = cache.probe_hints.find(fmt_ctx_->iformat->name);
            if (it != cache.probe_hints.end() && it->second.probesize > 0) {
                fmt_ctx_->probesize = it->second.probesize;
                fmt_ctx_->max_analyze_duration = it->second.analyze_duration;
            }
        }
    }

    // Step 2b: Find stream info
    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        avformat_close_input(&fmt_ctx_);
        valid_ = false;
        return;
    }

    // Step 2c: After a successful probe, record hints for this container format
    // so subsequent files skip the full probe.  Only the first file of each
    // format populates the cache entry.
    //
    // MP4/Matroska get reduced hints (1 MB probe, 2s analysis) because their
    // container headers contain complete stream metadata up front.
    // Formats without an explicit hint entry get default (0,0) which means
    // "use FFmpeg defaults" — no regression for TS, FLV, raw streams, etc.
    {
        auto& cache = ProcessCache::instance();
        if (fmt_ctx_->iformat && fmt_ctx_->iformat->name) {
            std::string fmt_name = fmt_ctx_->iformat->name;
            std::lock_guard<std::mutex> lock(cache.probe_hints_mutex);
            if (cache.probe_hints.find(fmt_name) == cache.probe_hints.end()) {
                ProcessCache::ProbeHint hint;
                if (fmt_name.find("mp4") != std::string::npos ||
                    fmt_name.find("matroska") != std::string::npos) {
                    hint.probesize = 1024 * 1024;     // 1MB (vs 5MB default)
                    hint.analyze_duration = 2000000;  // 2s  (vs 5s default)
                }
                cache.probe_hints[fmt_name] = hint;
            }
        }
    }

    // Step 3: Find best video stream
    const AVCodec* codec = nullptr;
    video_idx_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (video_idx_ < 0 || !codec) {
        avformat_close_input(&fmt_ctx_);
        valid_ = false;
        return;
    }

    // Store codec ID early — available even if codec open fails later.
    codec_id_ = codec->id;

    // Step 4: Get stream parameters
    AVStream* stream = fmt_ctx_->streams[video_idx_];
    width_ = stream->codecpar->width;
    height_ = stream->codecpar->height;

    // Step 5: Determine file size
    if (fmt_ctx_->pb) {
        int64_t sz = avio_size(fmt_ctx_->pb);
        file_size_ = (sz > 0) ? sz : 0;
    } else {
        file_size_ = 0;
    }

    // Step 6: Stage 4 — For CUDA devices, try dedicated cuvid decoders first.
    // These bypass FFmpeg's generic HW abstraction and talk directly to NVDEC,
    // giving better performance.  If the cuvid decoder isn't available in this
    // FFmpeg build, fall back to the generic decoder + hwaccel path.
    const AVCodec* actual_codec = codec;
    bool using_cuvid = false;
    if (hw && hw->device_type == AV_HWDEVICE_TYPE_CUDA) {
        const char* cuvid_name = get_cuvid_decoder_name(codec->id);
        if (cuvid_name) {
            const AVCodec* cuvid_codec = avcodec_find_decoder_by_name(cuvid_name);
            if (cuvid_codec) {
                actual_codec = cuvid_codec;
                using_cuvid = true;
            }
        }
    }

    // Step 7: Allocate and configure codec context
    codec_ctx_ = avcodec_alloc_context3(actual_codec);
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

    // Step 8: HW acceleration setup.
    // For cuvid decoders: attach the CUDA device context but skip the
    // get_format callback — cuvid handles format negotiation internally.
    // For generic hwaccel: set up the full callback chain.
    if (hw && hw->device_ctx) {
        codec_ctx_->hw_device_ctx = av_buffer_ref(hw->device_ctx);

        if (!using_cuvid) {
            // Generic hwaccel path (D3D11VA, QSV, VAAPI, or CUDA generic)
            codec_ctx_->opaque = hw;
            codec_ctx_->get_format = hw_get_format;
        }

        hw_frame_ = av_frame_alloc();
        hw_accel_ = hw;
    }

    // Step 9: Enable multi-threaded decode.  Codecs like DV support slice
    // threading (parallel DCT blocks within a frame) which gives a 2-3x
    // decode speedup.  thread_count=0 tells FFmpeg to auto-detect the
    // optimal thread count based on CPU cores and codec capabilities.
    // When the caller specifies a limit (decode_threads > 0), we cap the
    // count to avoid oversubscription when multiple workers each open
    // their own codec context.
    codec_ctx_->thread_count = decode_threads;

    // Step 10: Open codec.  Suppress noisy log output during open — FFmpeg
    // logs warnings for every HW format it tries and rejects.
    int saved_level = av_log_get_level();
    if (hw)
        av_log_set_level(AV_LOG_FATAL);

    ret = avcodec_open2(codec_ctx_, actual_codec, nullptr);

    if (hw)
        av_log_set_level(saved_level);

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

    // Step 10: Allocate reusable packet for decode_next
    pkt_ = av_packet_alloc();
    if (!pkt_) {
        if (hw_frame_) {
            av_frame_free(&hw_frame_);
        }
        hw_accel_ = nullptr;
        avcodec_free_context(&codec_ctx_);
        avformat_close_input(&fmt_ctx_);
        valid_ = false;
        return;
    }

    // Step 11: Store codec name, mark valid
    codec_name_ = codec->name;  // Use original codec name (not cuvid name)
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

// ── reopen_file (codec context flush-reuse) ─────────────────────────────────
//
// Switches this decoder to a new file without destroying and recreating the
// codec context.  The key savings come from preserving FFmpeg's internal
// thread pool (created in avcodec_open2, destroyed in avcodec_free_context).
//
// Approach: open a new format context (demuxer), verify the new file's
// codec parameters match the existing context, drain + flush the decoder,
// update extradata, then swap the format context.  The codec context
// (including its thread pool, HW device refs, and codec private data)
// survives the transition untouched.
//
// Returns false (caller must create fresh decoder) when:
//   - New file has different codec, resolution, or pixel format
//   - Codec is not on the flush-reuse safety whitelist
//   - Demuxer fails to open the new file

bool FileDecoder::reopen_file(const std::string& new_path, HWAccelContext* /*hw*/,
                              int /*decode_threads*/) {
    if (!valid_ || !codec_ctx_)
        return false;

    // Step 1: Open a fresh format context (demuxer) for the new file.
    // This is independent of the codec context — we're just reading
    // container metadata and packet data from a different file.
    AVFormatContext* new_fmt = nullptr;
    if (avformat_open_input(&new_fmt, new_path.c_str(), nullptr, nullptr) < 0)
        return false;

    // Apply cached probe hints (see constructor Step 2 for rationale)
    {
        auto& cache = ProcessCache::instance();
        std::lock_guard<std::mutex> lock(cache.probe_hints_mutex);
        if (new_fmt->iformat && new_fmt->iformat->name) {
            auto it = cache.probe_hints.find(new_fmt->iformat->name);
            if (it != cache.probe_hints.end() && it->second.probesize > 0) {
                new_fmt->probesize = it->second.probesize;
                new_fmt->max_analyze_duration = it->second.analyze_duration;
            }
        }
    }

    if (avformat_find_stream_info(new_fmt, nullptr) < 0) {
        avformat_close_input(&new_fmt);
        return false;
    }
    const AVCodec* codec = nullptr;
    int new_idx = av_find_best_stream(new_fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (new_idx < 0) {
        avformat_close_input(&new_fmt);
        return false;
    }

    AVStream* new_stream = new_fmt->streams[new_idx];
    AVCodecParameters* new_par = new_stream->codecpar;

    // Step 2: Verify the new file's stream parameters match our existing
    // codec context exactly.  Any mismatch means the codec context's
    // internal state (allocated buffers, HW surface pool dimensions,
    // reference frame configuration) would be wrong for the new stream.
    //
    // For HW decode, this is especially critical: AVHWFramesContext is
    // tied to resolution, and mismatches cause real bugs:
    //   - VideoToolbox stalled on SPS change (FFmpeg commit 9519983c)
    //   - QSV lost frames during reinit (Intel issue #48)
    //   - CUDA stale memory on 10-bit HEVC (mpv #4115)
    if (new_par->codec_id != codec_id_ || new_par->width != width_ || new_par->height != height_ ||
        (new_par->format != AV_PIX_FMT_NONE && new_par->format != codec_ctx_->pix_fmt)) {
        avformat_close_input(&new_fmt);
        return false;
    }

    // Step 3: Check that this codec is on the flush-reuse safety whitelist.
    // See flush_reuse_safe() above for which codecs qualify and why.
    if (!flush_reuse_safe(codec_id_, hw_accel_ != nullptr)) {
        avformat_close_input(&new_fmt);
        return false;
    }

    // Step 4: Drain any buffered frames from the previous file's decode.
    // Sending nullptr signals EOF to the decoder, then we pull all
    // remaining frames (B-frame reordering can leave frames buffered).
    // This ensures the DPB is in a clean state before flush.
    avcodec_send_packet(codec_ctx_, nullptr);
    AVFrame* tmp = av_frame_alloc();
    while (avcodec_receive_frame(codec_ctx_, tmp) == 0)
        av_frame_unref(tmp);
    av_frame_free(&tmp);

    // Step 5: Flush the codec context.
    // For threaded decoders, this parks worker threads (ff_thread_flush →
    // park_frame_worker_threads) rather than destroying them — this is
    // where the thread pool reuse savings come from.
    // For H.264, this clears the DPB (all 16 reference slots), resets
    // context_initialized to 0 (forcing SPS/PPS re-read from extradata),
    // and clears error concealment state.
    avcodec_flush_buffers(codec_ctx_);

    // Step 6: Replace extradata with the new file's SPS/PPS.
    // MUST happen after flush but before the first packet from the new
    // file.  See update_extradata() comment for the full safety rationale.
    update_extradata(codec_ctx_, new_par);

    // Step 7: Close the old format context and install the new one.
    // After this point, av_read_frame() will read packets from the new file.
    avformat_close_input(&fmt_ctx_);
    fmt_ctx_ = new_fmt;
    video_idx_ = new_idx;

    // Step 8: Update file metadata for metrics reporting.
    if (fmt_ctx_->pb) {
        int64_t sz = avio_size(fmt_ctx_->pb);
        file_size_ = (sz > 0) ? sz : 0;
    } else {
        file_size_ = 0;
    }

    return true;
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

    // Pixel format conversion (e.g. YUV411P, NV12) is deferred to the
    // FrameScaler which combines it with downscaling in a single sws_scale
    // pass, avoiding a redundant full-resolution intermediate conversion.

    return true;
}

// ── seek_and_decode ─────────────────────────────────────────────────────────

bool FileDecoder::seek_and_decode(const KeyframeInfo& kf, AVFrame* out_frame) {
    // Step 1: Seek — prefer PTS-based seeking (more compatible with HW decode),
    // fall back to byte-offset seeking if PTS is not available.
    int seek_ret;
    if (kf.pts != AV_NOPTS_VALUE && kf.pts >= 0) {
        seek_ret = av_seek_frame(fmt_ctx_, video_idx_, kf.pts, AVSEEK_FLAG_BACKWARD);
    } else if (kf.byte_offset > 0) {
        seek_ret =
            av_seek_frame(fmt_ctx_, -1, kf.byte_offset, AVSEEK_FLAG_BYTE | AVSEEK_FLAG_BACKWARD);
    } else {
        seek_ret = av_seek_frame(fmt_ctx_, video_idx_, 0, AVSEEK_FLAG_BACKWARD);
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
    if (!valid_ || !fmt_ctx_ || !codec_ctx_)
        return false;

    // Try multiple seek strategies. Some containers (SWF, GXF) don't
    // support standard seeking.
    int ret = avformat_seek_file(fmt_ctx_, video_idx_, INT64_MIN, 0, 0, 0);
    if (ret < 0) {
        ret = av_seek_frame(fmt_ctx_, video_idx_, 0, AVSEEK_FLAG_BACKWARD);
    }
    if (ret < 0) {
        // Byte-level seek as last resort — some containers need this
        ret = av_seek_frame(fmt_ctx_, -1, 0, AVSEEK_FLAG_BYTE | AVSEEK_FLAG_BACKWARD);
    }

    avcodec_flush_buffers(codec_ctx_);
    // Return true even if seeking failed — decode_next() will try to read
    // from the current position (works for forward-only containers).
    return true;
}

bool FileDecoder::decode_next(AVFrame* out_frame) {
    if (!valid_ || !fmt_ctx_ || !codec_ctx_ || !pkt_)
        return false;

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
    if (!valid_ || !fmt_ctx_)
        return 0;

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

    return 300;  // conservative fallback
}

int64_t FileDecoder::frame_pts_ms(const AVFrame* frame) const {
    if (!valid_ || !fmt_ctx_ || !frame)
        return 0;

    AVStream* stream = fmt_ctx_->streams[video_idx_];
    int64_t pts = frame->pts;
    if (pts == AV_NOPTS_VALUE)
        pts = frame->best_effort_timestamp;
    if (pts == AV_NOPTS_VALUE)
        return 0;

    return av_rescale_q(pts, stream->time_base, {1, 1000});
}

}  // namespace vex
