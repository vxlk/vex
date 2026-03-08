#include "metrics.h"

#include <algorithm>
#include <numeric>

namespace vex {

// ── ScopedTimer ─────────────────────────────────────────────────────────────

ScopedTimer::ScopedTimer(int64_t& accumulator)
    : accum_(accumulator), start_(std::chrono::high_resolution_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    accum_ += elapsed;
}

// ── ThreadMetrics ───────────────────────────────────────────────────────────

void ThreadMetrics::init_levels(int num_levels) {
    levels.resize(static_cast<size_t>(num_levels));
}

// ── merge_thread_metrics ────────────────────────────────────────────────────

DecodeMetrics merge_thread_metrics(const std::vector<ThreadMetrics>& thread_metrics, int num_levels,
                                   int64_t wall_us, int threads_used) {
    DecodeMetrics m{};
    m.total_wall_us = wall_us;
    m.threads_used = threads_used;
    m.levels.resize(static_cast<size_t>(num_levels));

    for (const auto& tm : thread_metrics) {
        m.index_scan_us += tm.index_scan_us;
        m.seek_us += tm.seek_us;
        m.decode_us += tm.decode_us;
        m.io_us += tm.io_us;
        m.keyframes_decoded += tm.keyframes_decoded;
        m.keyframes_skipped += tm.keyframes_skipped;
        m.total_output_bytes += tm.bytes_read;

        // Sum per-level metrics
        for (int l = 0; l < num_levels && l < static_cast<int>(tm.levels.size()); ++l) {
            auto& dst = m.levels[static_cast<size_t>(l)];
            const auto& src = tm.levels[static_cast<size_t>(l)];

            // Carry forward width/height/quality/output_format from whichever thread set them
            if (src.width > 0)
                dst.width = src.width;
            if (src.height > 0)
                dst.height = src.height;
            if (src.quality > 0)
                dst.quality = src.quality;
            if (!src.output_format.empty())
                dst.output_format = src.output_format;

            dst.scale_us += src.scale_us;
            dst.encode_us += src.encode_us;
            dst.atlas_composite_us += src.atlas_composite_us;
            dst.atlas_encode_us += src.atlas_encode_us;
            dst.output_bytes += src.output_bytes;
            dst.frame_count += src.frame_count;
        }

        // Concatenate file_stats
        m.file_stats.insert(m.file_stats.end(), tm.file_stats.begin(), tm.file_stats.end());
    }

    // Count distinct files processed
    m.files_processed = static_cast<int>(m.file_stats.size());

    // Compute total output bytes from levels and derived avg_jpeg_bytes
    m.total_output_bytes = 0;
    for (auto& lev : m.levels) {
        m.total_output_bytes += lev.output_bytes;
        if (lev.frame_count > 0) {
            lev.avg_jpeg_bytes = static_cast<double>(lev.output_bytes) / lev.frame_count;
        }
    }

    return m;
}

}  // namespace vex
