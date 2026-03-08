#include "async_handle.h"

#include <algorithm>

namespace vex {

// ── Constructor ─────────────────────────────────────────────────────────────

DecodeHandle::DecodeHandle(int num_threads, int total_files)
    : num_threads_(num_threads), total_files_(total_files) {
    thread_events_.resize(static_cast<size_t>(num_threads));
    drain_cursors_.resize(static_cast<size_t>(num_threads), 0);
    file_blobs_.resize(static_cast<size_t>(total_files), nullptr);
}

// ── Worker API ──────────────────────────────────────────────────────────────

void DecodeHandle::publish_event(int thread_id, FrameEvent evt) {
    auto& te = thread_events_[static_cast<size_t>(thread_id)];
    te.events.push_back(evt);
    te.published.store(static_cast<int>(te.events.size()), std::memory_order_release);
}

void DecodeHandle::increment_keyframes(int count) {
    keyframes_decoded_.fetch_add(count, std::memory_order_relaxed);
}

void DecodeHandle::increment_files(int count) {
    files_completed_.fetch_add(count, std::memory_order_relaxed);
}

void DecodeHandle::set_done() {
    done_.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(done_mutex_);
    done_cv_.notify_all();
}

void DecodeHandle::register_blob(int file_index, const uint8_t* data) {
    std::lock_guard<std::mutex> lock(blobs_mutex_);
    file_blobs_[static_cast<size_t>(file_index)] = data;
}

// ── Consumer API ────────────────────────────────────────────────────────────

std::vector<FrameEvent> DecodeHandle::drain_events() {
    std::vector<FrameEvent> collected;

    for (int i = 0; i < num_threads_; ++i) {
        auto& te = thread_events_[static_cast<size_t>(i)];
        int published = te.published.load(std::memory_order_acquire);
        int cursor = drain_cursors_[static_cast<size_t>(i)];

        if (published > cursor) {
            collected.insert(collected.end(), te.events.begin() + cursor,
                             te.events.begin() + published);
            drain_cursors_[static_cast<size_t>(i)] = published;
        }
    }

    return collected;
}

DecodeHandle::Progress DecodeHandle::progress() const {
    Progress p{};
    p.keyframes_decoded = keyframes_decoded_.load(std::memory_order_relaxed);
    p.files_completed = files_completed_.load(std::memory_order_relaxed);
    p.total_files = total_files_;
    return p;
}

const uint8_t* DecodeHandle::peek_jpeg_data(const FrameEvent& evt, size_t* out_size) const {
    std::lock_guard<std::mutex> lock(blobs_mutex_);
    const uint8_t* blob = file_blobs_[static_cast<size_t>(evt.file_index)];
    if (!blob) {
        if (out_size)
            *out_size = 0;
        return nullptr;
    }
    if (out_size)
        *out_size = evt.jpeg_size;
    return blob + evt.blob_offset;
}

void DecodeHandle::wait_until_done() {
    std::unique_lock<std::mutex> lock(done_mutex_);
    done_cv_.wait(lock, [this]() { return done_.load(std::memory_order_acquire); });
}

void DecodeHandle::set_results(std::vector<LevelResult> results, DecodeMetrics metrics) {
    results_ = std::move(results);
    metrics_ = std::move(metrics);
}

std::pair<std::vector<LevelResult>, DecodeMetrics> DecodeHandle::get_results() {
    return {std::move(results_), std::move(metrics_)};
}

}  // namespace vex
