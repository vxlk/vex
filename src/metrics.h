#pragma once

#include "types.h"
#include <chrono>
#include <vector>

namespace vex {

// RAII timer — adds elapsed microseconds to accumulator on destruction.
class ScopedTimer {
public:
    explicit ScopedTimer(int64_t& accumulator);
    ~ScopedTimer();
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
private:
    int64_t& accum_;
    std::chrono::high_resolution_clock::time_point start_;
};

// Per-thread metric accumulator. One per worker thread.
struct ThreadMetrics {
    int64_t index_scan_us = 0;
    int64_t seek_us       = 0;
    int64_t decode_us     = 0;
    int64_t io_us         = 0;
    int64_t bytes_read    = 0;

    int keyframes_decoded = 0;
    int keyframes_skipped = 0;

    std::vector<LevelMetrics> levels;    // one per configured level
    std::vector<FileStats>    file_stats;

    void init_levels(int num_levels);
};

// Merge all per-thread metrics into a single DecodeMetrics.
DecodeMetrics merge_thread_metrics(
    const std::vector<ThreadMetrics>& thread_metrics,
    int num_levels,
    int64_t wall_us,
    int threads_used);

} // namespace vex
