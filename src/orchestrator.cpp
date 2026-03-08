#include "orchestrator.h"
#include "decoder.h"
#include "scaler.h"
#include "encoder.h"
#include "atlas.h"
#include "disk_writer.h"
#include "metrics.h"
#include "hw_accel.h"
#include "index_scanner.h"

#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <algorithm>
#include <chrono>
#include <memory>
#include <cstring>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

namespace vex {

// ── Shared work queue ───────────────────────────────────────────────────────

struct WorkQueue {
    std::mutex              mutex;
    std::condition_variable cv;
    std::deque<int>         file_indices;
    bool                    shutdown = false;

    void populate(int num_files) {
        std::lock_guard<std::mutex> lock(mutex);
        for (int i = 0; i < num_files; ++i) {
            file_indices.push_back(i);
        }
    }

    // Returns -1 when no more work.
    int dequeue() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this]() { return !file_indices.empty() || shutdown; });
        if (file_indices.empty()) return -1;
        int idx = file_indices.front();
        file_indices.pop_front();
        return idx;
    }

    void signal_shutdown() {
        std::lock_guard<std::mutex> lock(mutex);
        shutdown = true;
        cv.notify_all();
    }
};

// ── Per-level accumulator for disk output ───────────────────────────────────

struct DiskFrameEntry {
    int      file_index;
    int      frame_index;
    int64_t  pts_ms;
    int      frame_number;
    std::vector<uint8_t> jpeg_data;
};

struct DiskLevelAccum {
    std::mutex mutex;
    std::vector<DiskFrameEntry> entries;
};

// ── Per-level in-memory blobs container ─────────────────────────────────────

struct MemoryLevelAccum {
    std::vector<FileBlob> blobs;          // one per file, pre-sized
    std::vector<std::vector<std::array<int64_t,3>>> metadata; // per-file metadata
};

// ── YUV buffer for per-thread scaling ───────────────────────────────────────

struct YUVBuffer {
    std::vector<uint8_t> y, u, v;
    int width  = 0;
    int height = 0;

    void ensure(int w, int h) {
        if (w == width && h == height) return;
        width  = w;
        height = h;
        y.resize(static_cast<size_t>(w) * h);
        u.resize(static_cast<size_t>(w / 2) * (h / 2));
        v.resize(static_cast<size_t>(w / 2) * (h / 2));
    }
};

// ── Shared context for all threads ──────────────────────────────────────────

struct SharedContext {
    BatchConfig config;
    int num_levels;
    int num_files;

    std::shared_ptr<WorkQueue> work_queue;
    std::shared_ptr<DecodeHandle> handle;

    // HW acceleration (optional)
    std::optional<HWAccelContext> hw_ctx;

    // Per-level accumulators (shared_ptrs for thread-safe sharing)
    std::vector<std::shared_ptr<MemoryLevelAccum>> mem_accums;
    std::vector<std::shared_ptr<AtlasBuilder>> atlas_builders;
    std::vector<std::shared_ptr<DiskLevelAccum>> disk_accums;

    // Per-thread metrics
    std::vector<std::shared_ptr<ThreadMetrics>> all_metrics;

    // Wall-clock start
    std::chrono::high_resolution_clock::time_point wall_start;

    HWAccelContext* hw_ptr() {
        return hw_ctx.has_value() ? &hw_ctx.value() : nullptr;
    }
};

// ── Worker thread function ──────────────────────────────────────────────────

static void worker_func(std::shared_ptr<SharedContext> ctx, int thread_id) {
    ThreadMetrics& tm = *ctx->all_metrics[static_cast<size_t>(thread_id)];
    tm.init_levels(ctx->num_levels);

    // Per-thread resources
    JpegEncoder encoder;
    AVFrame* decode_frame = av_frame_alloc();
    std::vector<YUVBuffer> scale_bufs(static_cast<size_t>(ctx->num_levels));

    HWAccelContext* hw_ptr = ctx->hw_ptr();

    // Worker loop: dequeue files one at a time
    while (true) {
        int file_idx = ctx->work_queue->dequeue();
        if (file_idx < 0) break;

        const std::string& path = ctx->config.paths[static_cast<size_t>(file_idx)];

        // Create decoder for this file
        FileDecoder decoder_file(path, hw_ptr);
        if (!decoder_file.is_valid()) {
            FileStats fs{};
            fs.file_index      = file_idx;
            fs.index_strategy  = IndexStrategy::SKIPPED;
            fs.file_size_bytes = 0;
            tm.file_stats.push_back(fs);
            ctx->handle->increment_files(1);
            continue;
        }

        int src_w = decoder_file.source_width();
        int src_h = decoder_file.source_height();

        // Resolve native-resolution sentinels (width=0 or height=0 → source)
        std::vector<LevelConfig> resolved = ctx->config.levels;
        for (auto& rl : resolved) {
            if (rl.width  <= 0) rl.width  = src_w;
            if (rl.height <= 0) rl.height = src_h;
        }

        // Scan keyframes (timed)
        KeyframeIndex kf_index;
        {
            ScopedTimer timer(tm.index_scan_us);
            kf_index = decoder_file.scan_keyframes();
        }

        const auto& keyframes = kf_index.keyframes;
        int kf_count = static_cast<int>(keyframes.size());

        if (kf_count == 0) {
            FileStats fs{};
            fs.file_index      = file_idx;
            fs.index_strategy  = kf_index.strategy;
            fs.file_size_bytes = decoder_file.file_size();
            fs.keyframe_count  = 0;
            tm.file_stats.push_back(fs);
            ctx->handle->increment_files(1);
            continue;
        }

        // Pre-allocate FileBlobs for in-memory JPEG_STREAM levels
        for (int l = 0; l < ctx->num_levels; ++l) {
            const auto& lc = resolved[static_cast<size_t>(l)];
            if (lc.output == OutputFormat::JPEG_STREAM && lc.in_memory) {
                size_t max_per_frame = JpegEncoder::max_jpeg_size(lc.width, lc.height);
                size_t total_cap = max_per_frame * kf_count;
                auto& accum = ctx->mem_accums[static_cast<size_t>(l)];
                accum->blobs[static_cast<size_t>(file_idx)].allocate(total_cap);
            }
        }

        // Per-file scaler cache (one per level)
        std::vector<std::unique_ptr<FrameScaler>> scalers(
            static_cast<size_t>(ctx->num_levels));

        int64_t file_decode_us = 0;

        // Process each keyframe
        for (int ki = 0; ki < kf_count; ++ki) {
            const KeyframeInfo& kf = keyframes[static_cast<size_t>(ki)];

            // Decode (timed)
            bool decoded = false;
            {
                ScopedTimer dec_timer(tm.decode_us);
                decoded = decoder_file.seek_and_decode(kf, decode_frame);
            }

            if (!decoded) {
                tm.keyframes_skipped++;
                continue;
            }

            tm.keyframes_decoded++;
            ctx->handle->increment_keyframes(1);

            // Process each level
            for (int l = 0; l < ctx->num_levels; ++l) {
                const auto& lc = resolved[static_cast<size_t>(l)];
                auto& lm = tm.levels[static_cast<size_t>(l)];

                // Set level metadata once
                lm.width         = lc.width;
                lm.height        = lc.height;
                lm.quality       = lc.quality;
                lm.output_format = (lc.output == OutputFormat::JPEG_STREAM) ?
                                   "jpeg_stream" : "sprite_atlas";

                // Create/reuse scaler for this level
                if (!scalers[static_cast<size_t>(l)]) {
                    scalers[static_cast<size_t>(l)] =
                        std::make_unique<FrameScaler>(src_w, src_h, lc.width, lc.height);
                }

                auto& scaler = *scalers[static_cast<size_t>(l)];

                // Resolve YUV plane pointers: scale into buffer, or
                // read decoded frame planes directly (zero-copy identity).
                const uint8_t* plane_y;
                const uint8_t* plane_u;
                const uint8_t* plane_v;
                int y_stride, uv_stride;

                if (scaler.needs_scaling()) {
                    auto& buf = scale_bufs[static_cast<size_t>(l)];
                    buf.ensure(lc.width, lc.height);
                    y_stride  = lc.width;
                    uv_stride = lc.width / 2;
                    {
                        ScopedTimer scale_timer(lm.scale_us);
                        scaler.scale(decode_frame,
                                     buf.y.data(), buf.u.data(), buf.v.data(),
                                     y_stride, uv_stride);
                    }
                    plane_y = buf.y.data();
                    plane_u = buf.u.data();
                    plane_v = buf.v.data();
                } else {
                    // Identity: use decoded frame planes directly — no copy
                    plane_y = decode_frame->data[0];
                    plane_u = decode_frame->data[1];
                    plane_v = decode_frame->data[2];
                    y_stride  = decode_frame->linesize[0];
                    uv_stride = decode_frame->linesize[1];
                }

                // Handle output based on format
                if (lc.output == OutputFormat::JPEG_STREAM && lc.in_memory) {
                    auto& accum = ctx->mem_accums[static_cast<size_t>(l)];
                    auto& blob = accum->blobs[static_cast<size_t>(file_idx)];

                    size_t remaining = blob.capacity - blob.size;

                    size_t jpeg_size;
                    {
                        ScopedTimer enc_timer(lm.encode_us);
                        jpeg_size = encoder.encode(
                            plane_y, plane_u, plane_v,
                            y_stride, uv_stride,
                            lc.width, lc.height, lc.quality,
                            blob.data + blob.size,
                            remaining);
                    }

                    if (jpeg_size > 0) {
                        uint32_t blob_offset = static_cast<uint32_t>(blob.size);
                        blob.offsets.push_back(static_cast<int64_t>(blob.size));
                        blob.size += jpeg_size;

                        lm.output_bytes += static_cast<int64_t>(jpeg_size);
                        lm.frame_count++;

                        // Record metadata
                        std::array<int64_t,3> meta = {
                            static_cast<int64_t>(file_idx),
                            static_cast<int64_t>(kf.frame_number),
                            kf.pts_ms
                        };
                        accum->metadata[static_cast<size_t>(file_idx)].push_back(meta);

                        // Publish event for async consumers
                        FrameEvent evt{};
                        evt.file_index  = file_idx;
                        evt.frame_index = ki;
                        evt.level_index = l;
                        evt.pts_ms      = kf.pts_ms;
                        evt.blob_offset = blob_offset;
                        evt.jpeg_size   = static_cast<uint32_t>(jpeg_size);
                        ctx->handle->publish_event(thread_id, evt);
                    }

                } else if (lc.output == OutputFormat::SPRITE_ATLAS) {
                    auto& atlas = ctx->atlas_builders[static_cast<size_t>(l)];
                    atlas->ensure_time_step(ki);

                    {
                        ScopedTimer comp_timer(lm.atlas_composite_us);
                        atlas->add_thumbnail(ki, file_idx,
                            plane_y, plane_u, plane_v,
                            y_stride, uv_stride);
                    }

                    lm.frame_count++;

                } else if (lc.output == OutputFormat::JPEG_STREAM && !lc.in_memory) {
                    // Disk output: encode to temp buffer, store for later
                    size_t max_size = JpegEncoder::max_jpeg_size(lc.width, lc.height);
                    std::vector<uint8_t> jpeg_buf(max_size);

                    size_t jpeg_size;
                    {
                        ScopedTimer enc_timer(lm.encode_us);
                        jpeg_size = encoder.encode(
                            plane_y, plane_u, plane_v,
                            y_stride, uv_stride,
                            lc.width, lc.height, lc.quality,
                            jpeg_buf.data(), max_size);
                    }

                    if (jpeg_size > 0) {
                        jpeg_buf.resize(jpeg_size);
                        lm.output_bytes += static_cast<int64_t>(jpeg_size);
                        lm.frame_count++;

                        DiskFrameEntry entry{};
                        entry.file_index   = file_idx;
                        entry.frame_index  = ki;
                        entry.pts_ms       = kf.pts_ms;
                        entry.frame_number = kf.frame_number;
                        entry.jpeg_data    = std::move(jpeg_buf);

                        auto& disk_accum = ctx->disk_accums[static_cast<size_t>(l)];
                        std::lock_guard<std::mutex> lock(disk_accum->mutex);
                        disk_accum->entries.push_back(std::move(entry));
                    }
                }
            } // per-level

            av_frame_unref(decode_frame);
        } // per-keyframe

        // Shrink in-memory blobs and register with handle
        for (int l = 0; l < ctx->num_levels; ++l) {
            const auto& lc = resolved[static_cast<size_t>(l)];
            if (lc.output == OutputFormat::JPEG_STREAM && lc.in_memory) {
                auto& accum = ctx->mem_accums[static_cast<size_t>(l)];
                auto& blob = accum->blobs[static_cast<size_t>(file_idx)];
                blob.shrink_to_fit();
                ctx->handle->register_blob(file_idx, blob.data);
            }
        }

        // Record file stats
        FileStats fs{};
        fs.file_index      = file_idx;
        fs.decode_us       = file_decode_us;
        fs.keyframe_count  = kf_count;
        fs.file_size_bytes = decoder_file.file_size();
        fs.index_strategy  = kf_index.strategy;
        tm.file_stats.push_back(fs);

        ctx->handle->increment_files(1);
    } // while (dequeue)

    av_frame_free(&decode_frame);
}

// ── Manager thread function (joins workers, finalizes results) ──────────────

static void manager_func(std::shared_ptr<SharedContext> ctx,
                          std::vector<std::thread> workers) {
    // Signal shutdown so workers exit after queue is drained
    ctx->work_queue->signal_shutdown();

    // Join all workers
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    // Wall-clock end
    auto wall_end = std::chrono::high_resolution_clock::now();
    int64_t wall_us = std::chrono::duration_cast<std::chrono::microseconds>(
        wall_end - ctx->wall_start).count();

    int num_threads = static_cast<int>(ctx->all_metrics.size());

    // Collect ThreadMetrics
    std::vector<ThreadMetrics> tm_vec;
    tm_vec.reserve(ctx->all_metrics.size());
    for (auto& sp : ctx->all_metrics) {
        tm_vec.push_back(std::move(*sp));
    }

    // Merge metrics
    DecodeMetrics metrics = merge_thread_metrics(tm_vec, ctx->num_levels, wall_us,
                                                 num_threads);

    // Build LevelResults
    std::vector<LevelResult> results(static_cast<size_t>(ctx->num_levels));

    for (int l = 0; l < ctx->num_levels; ++l) {
        const auto& lc = ctx->config.levels[static_cast<size_t>(l)];
        auto& lr = results[static_cast<size_t>(l)];
        lr.format    = lc.output;
        lr.in_memory = lc.in_memory;

        if (lc.output == OutputFormat::JPEG_STREAM && lc.in_memory) {
            auto& accum = ctx->mem_accums[static_cast<size_t>(l)];
            JpegStreamResult jsr{};
            jsr.blobs = std::move(accum->blobs);

            // Concatenate per-file metadata
            for (int fi = 0; fi < ctx->num_files; ++fi) {
                auto& fm = accum->metadata[static_cast<size_t>(fi)];
                jsr.metadata.insert(jsr.metadata.end(), fm.begin(), fm.end());
            }

            lr.jpeg_stream = std::move(jsr);

        } else if (lc.output == OutputFormat::SPRITE_ATLAS) {
            auto& atlas = ctx->atlas_builders[static_cast<size_t>(l)];

            // Time the atlas encode
            int64_t atlas_encode_us = 0;
            SpriteAtlasResult sar;
            {
                ScopedTimer timer(atlas_encode_us);
                sar = atlas->compose_all();
            }
            if (l < static_cast<int>(metrics.levels.size())) {
                metrics.levels[static_cast<size_t>(l)].atlas_encode_us += atlas_encode_us;
            }

            lr.sprite_atlas = std::move(sar);

        } else if (lc.output == OutputFormat::JPEG_STREAM && !lc.in_memory) {
            auto& disk_accum = ctx->disk_accums[static_cast<size_t>(l)];

            // Sort entries by file_index, then frame_index for sequential writes
            std::sort(disk_accum->entries.begin(), disk_accum->entries.end(),
                [](const DiskFrameEntry& a, const DiskFrameEntry& b) {
                    if (a.file_index != b.file_index) return a.file_index < b.file_index;
                    return a.frame_index < b.frame_index;
                });

            DiskWriter writer(lc.cache_path);
            if (writer.is_valid()) {
                std::vector<std::array<int64_t,3>> metadata;

                int64_t io_us = 0;
                for (auto& entry : disk_accum->entries) {
                    {
                        ScopedTimer timer(io_us);
                        writer.write_frame(entry.jpeg_data.data(), entry.jpeg_data.size());
                    }
                    metadata.push_back({
                        static_cast<int64_t>(entry.file_index),
                        static_cast<int64_t>(entry.frame_number),
                        entry.pts_ms
                    });
                }
                metrics.io_us += io_us;

                DiskResult dr = writer.finalize(metadata, lc, 0);
                lr.disk = std::move(dr);
            }
        }
    }

    // Release HW context
    if (ctx->hw_ctx.has_value()) {
        release_hw_accel(ctx->hw_ctx.value());
    }

    ctx->handle->set_results(std::move(results), std::move(metrics));
    ctx->handle->set_done();
}

// ── batch_decode_async ──────────────────────────────────────────────────────

std::shared_ptr<DecodeHandle>
Orchestrator::batch_decode_async(const BatchConfig& config) {
    const int num_files  = static_cast<int>(config.paths.size());
    const int num_levels = static_cast<int>(config.levels.size());

    // Determine thread count
    int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (hw_threads <= 0) hw_threads = 4;
    int num_threads = std::min({config.max_threads, num_files, hw_threads});
    if (num_threads <= 0) num_threads = 1;

    // Build shared context (heap-allocated, shared among all threads)
    auto ctx = std::make_shared<SharedContext>();
    ctx->config     = config;
    ctx->num_levels = num_levels;
    ctx->num_files  = num_files;
    ctx->wall_start = std::chrono::high_resolution_clock::now();

    // Create shared handle
    ctx->handle = std::make_shared<DecodeHandle>(num_threads, num_files);

    // Try to probe HW acceleration (optional, best-effort)
    const AVCodec* probe_codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (probe_codec) {
        ctx->hw_ctx = probe_hw_accel(probe_codec);
    }

    // Work queue
    ctx->work_queue = std::make_shared<WorkQueue>();
    ctx->work_queue->populate(num_files);

    // Per-level accumulators
    ctx->mem_accums.resize(static_cast<size_t>(num_levels));
    ctx->atlas_builders.resize(static_cast<size_t>(num_levels));
    ctx->disk_accums.resize(static_cast<size_t>(num_levels));

    for (int l = 0; l < num_levels; ++l) {
        const auto& lc = config.levels[static_cast<size_t>(l)];
        if (lc.output == OutputFormat::JPEG_STREAM && lc.in_memory) {
            auto accum = std::make_shared<MemoryLevelAccum>();
            accum->blobs.resize(static_cast<size_t>(num_files));
            accum->metadata.resize(static_cast<size_t>(num_files));
            ctx->mem_accums[static_cast<size_t>(l)] = accum;
        } else if (lc.output == OutputFormat::SPRITE_ATLAS) {
            ctx->atlas_builders[static_cast<size_t>(l)] = std::make_shared<AtlasBuilder>(
                lc.width, lc.height, lc.atlas_columns, num_files, lc.quality);
        } else if (lc.output == OutputFormat::JPEG_STREAM && !lc.in_memory) {
            ctx->disk_accums[static_cast<size_t>(l)] = std::make_shared<DiskLevelAccum>();
        }
    }

    // Per-thread metrics
    ctx->all_metrics.resize(static_cast<size_t>(num_threads));
    for (int t = 0; t < num_threads; ++t) {
        ctx->all_metrics[static_cast<size_t>(t)] = std::make_shared<ThreadMetrics>();
    }

    // Launch worker threads
    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back(worker_func, ctx, t);
    }

    // Detach a manager thread that waits for workers, then finalizes
    auto handle = ctx->handle;
    std::thread manager_thread(manager_func, ctx, std::move(workers));
    manager_thread.detach();

    return handle;
}

// ── batch_decode (synchronous) ──────────────────────────────────────────────

std::pair<std::vector<LevelResult>, DecodeMetrics>
Orchestrator::batch_decode(const BatchConfig& config) {
    auto handle = batch_decode_async(config);
    handle->wait_until_done();
    return handle->get_results();
}

} // namespace vex
